"""Stage 6 - render: download only the needed seconds, conform every shot (1080p, 30fps,
blurred fill for vertical/4:3 footage, light grade), mix dialogue + music + narration,
burn Hindi subtitles, and export a YouTube-ready MP4 + SRT + thumbnail."""

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import music
from .config import FONT_NAME, FONTS_DIR
from .util import PipelineError, ffmpeg, ffmpeg_has_filter, has_video, probe, srt_ts  # noqa: F401

AUDIO_SR = 48000


# ------------------------------------------------------------------ downloads
def plan_downloads(segments, pad=1.0, merge_gap=6.0):
    """Merge each video's needed ranges into as few downloads as possible."""
    ranges = {}
    for s in segments:
        if s["type"] != "clip":
            continue
        ranges.setdefault(s["video_id"], []).append(
            (max(0.0, s["src_start"] - pad), s["src_start"] + s["dur"] + pad))
    plan = []
    for vid, rs in ranges.items():
        rs.sort()
        merged = [list(rs[0])]
        for a, b in rs[1:]:
            if a <= merged[-1][1] + merge_gap:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        plan.extend({"video_id": vid, "start": round(a, 2), "end": round(b, 2)} for a, b in merged)
    return plan


def download_all(source, segments, out_dir, log, workers=3):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = plan_downloads(segments)
    log(f"Downloading {len(plan)} footage sections (only the seconds we need)...")
    files, failed = {}, []

    def job(r):
        base = out_dir / f"{r['video_id']}_{int(r['start'] * 10)}_{int(r['end'] * 10)}"
        return r, source.download_section(r["video_id"], r["start"], r["end"], base)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(job, r) for r in plan]
        for n, fut in enumerate(as_completed(futures), 1):
            try:
                r, got = fut.result()
                entry = {**r, **got} if isinstance(got, dict) else {**r, "file": got}
                if not has_video(entry["file"]):
                    raise PipelineError(f"{r['video_id']}: downloaded file has no picture")
                files.setdefault(r["video_id"], []).append(entry)
            except Exception as e:  # noqa: BLE001 - drop clips from this section later
                failed.append(str(e)[:200])
                log(f"  download failed: {str(e)[:160]}")
            if n % 5 == 0 or n == len(plan):
                log(f"  downloaded {n}/{len(plan)}")
    return files, failed


def drop_missing(files):
    """Forget downloads whose file is gone (deleted, or a temporary file yt-dlp replaced);
    their shots are then re-edited like any failed download. Returns how many were dropped."""
    dropped = 0
    for vid in list(files):
        keep = [r for r in files[vid] if Path(r["file"]).exists()]
        dropped += len(files[vid]) - len(keep)
        if keep:
            files[vid] = keep
        else:
            del files[vid]
    return dropped


def locate(files, seg, strict=False):
    """File + offset for a shot. Prefers a file that covers the whole shot; unless strict,
    falls back to one that covers its start (the renderer holds the last frame)."""
    ranges = files.get(seg["video_id"], [])
    for r in ranges:
        if r["start"] - 0.01 <= seg["src_start"] and seg["src_start"] + seg["dur"] <= r["end"] + 0.6:
            return r["file"], seg["src_start"] - r["start"]
    if not strict:
        for r in ranges:
            if r["start"] - 0.01 <= seg["src_start"] < r["end"] - 0.5:
                return r["file"], seg["src_start"] - r["start"]
    return None, 0.0


def frame_hash(path, t):
    """64-bit difference hash of one frame (None for dark/flat frames that would all match)."""
    import subprocess
    try:
        raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{max(0, t):.2f}", "-i", str(path),
                              "-frames:v", "1", "-vf", "scale=9:8,format=gray", "-f", "rawvideo", "-"],
                             capture_output=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    if len(raw) < 72:
        return None
    px = list(raw[:72])
    mean = sum(px) / 72
    if mean < 18 or max(px) - min(px) < 25:
        return None
    bits = 0
    for r in range(8):
        for c in range(8):
            bits = (bits << 1) | (px[r * 9 + c] > px[r * 9 + c + 1])
    return bits


def visual_duplicates(segments, files, max_distance=6):
    """Shots (other than people speaking) that show the same picture as an earlier shot,
    e.g. the same CCTV clip re-aired by two channels. Returns {(video_id, src_start)}."""
    seen, dupes = [], set()
    per_beat = {}
    for seg in segments:
        if seg["type"] == "clip":
            per_beat[(seg["act"], seg["beat"])] = per_beat.get((seg["act"], seg["beat"]), 0) + 1
    for seg in segments:
        if seg["type"] != "clip":
            continue
        src, offset = locate(files, seg)
        if not src:
            continue
        h = frame_hash(src, offset + seg["dur"] / 2)
        if h is None:
            continue
        key = (seg["act"], seg["beat"])
        # Only drop a shot when something remains: montage shots, or extra pictures under a
        # narration/text line - never the only picture under a line, never a speaker.
        removable = seg["mode"] == "music" or (seg["mode"] in ("narration", "text")
                                               and per_beat.get(key, 0) > 1)
        if removable and any(bin(h ^ o).count("1") <= max_distance for o in seen):
            dupes.add((seg["video_id"], round(seg["src_start"], 2)))
            per_beat[key] -= 1
            continue
        seen.append(h)
    return dupes


def merge_files(files, more):
    for vid, lst in more.items():
        files.setdefault(vid, []).extend(lst)
    return files


# ------------------------------------------------------------------ segments
def _stream_info(path):
    info = probe(path)
    v = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
    a = any(s["codec_type"] == "audio" for s in info["streams"])
    w, h = (int(v["width"]), int(v["height"])) if v else (0, 0)
    rot = 0
    if v:
        for sd in v.get("side_data_list", []) or []:
            if "rotation" in sd:
                rot = abs(int(sd["rotation"]))
        rot = rot or abs(int((v.get("tags") or {}).get("rotate", 0)))
    if rot in (90, 270):
        w, h = h, w
    return w, h, a


def render_segment(seg, src, offset, out_v, out_a, tl, settings):
    W, H, fps = tl["width"], tl["height"], tl["fps"]
    n, dur = seg["frames"], seg["frames"] / fps
    samples = int(round(dur * AUDIO_SR))
    w, h, has_audio = _stream_info(src)
    if not w:   # no picture at all (should have been caught at download): never fail the film
        src_inputs = ["-f", "lavfi", "-i", f"color=c=black:s={W}x{H}:r={fps}:d={dur + 1:.2f}"]
    aspect = w / h if h else 16 / 9
    if abs(aspect - W / H) < 0.04:
        vchain = f"[0:v]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H}[v0];"
    else:  # vertical / 4:3: blurred copy fills the frame behind the real shot
        vchain = (f"[0:v]split=2[bgs][fgs];[bgs]scale={W}:{H}:force_original_aspect_ratio=increase,"
                  f"crop={W}:{H},boxblur=luma_radius=40:luma_power=2,eq=brightness=-0.12[bg];"
                  f"[fgs]scale={W}:{H}:force_original_aspect_ratio=decrease[fg];"
                  f"[bg][fg]overlay=(W-w)/2:(H-h)/2[v0];")
    fades = ""
    if seg["fade_in"]:
        fades += f",fade=t=in:st=0:d={seg['fade_in']}"
    if seg["fade_out"]:
        fades += f",fade=t=out:st={max(0, dur - seg['fade_out']):.3f}:d={seg['fade_out']}"
    overlay = ""
    if seg.get("overlay"):       # on-screen text over darkened, still-moving footage
        overlay = ",eq=brightness=-0.28:saturation=0.75,gblur=sigma=3"
        if can_draw_text():
            ass = Path(out_v).with_suffix(".ass")
            ass.write_text(ass_header(W, H, int(H * 0.075), 0, outline=int(H * 0.018), box=True) +
                           f"Dialogue: 0,{_ass_ts(0)},{_ass_ts(dur)},Default,,{int(W * 0.1)},{int(W * 0.1)},0,,"
                           f"{{\\an5\\q0\\fad(300,300)\\fscx100\\fscy100\\t(0,{int(dur * 1000)},\\fscx104\\fscy104)}}"
                           f"{_ass_escape(seg['overlay'])}\n", encoding="utf-8")
            overlay += f",{ass_filter(ass)}"
    vchain += (f"[v0]setsar=1,fps={fps},eq=contrast=1.05:saturation=1.08,"
               f"tpad=stop_mode=clone:stop_duration={max(4.0, dur + 1):.1f}{overlay}{fades},format=yuv420p[v]")
    gain = seg["clip_gain"]
    afades = f"afade=t=in:d={max(0.03, seg['fade_in'])}"
    afades += f",afade=t=out:st={max(0, dur - max(0.06, seg['fade_out'])):.3f}:d={max(0.06, seg['fade_out'])}"
    a_in = "[0:a]" if has_audio else "[1:a]"
    achain = (f"{a_in}aresample={AUDIO_SR},aformat=sample_fmts=fltp:channel_layouts=stereo,"
              + ("loudnorm=I=-18:TP=-3:LRA=9," if has_audio else "")
              + f"aresample={AUDIO_SR},volume={gain:.3f},{afades},apad,atrim=end_sample={samples}[a]")
    args = ["-ss", f"{max(0, offset):.3f}", "-t", f"{dur + 0.5:.3f}", "-i", src]
    if not w:
        # input 0 = black picture, input 1 = the file's sound
        args = src_inputs + ["-ss", f"{max(0, offset):.3f}", "-t", f"{dur + 0.5:.3f}", "-i", src]
        a_in = "[1:a]" if has_audio else None
        if not has_audio:
            args += ["-f", "lavfi", "-i", f"anullsrc=r={AUDIO_SR}:cl=stereo"]
            a_in = "[2:a]"
        achain = achain.replace(achain[:achain.index("aresample")], a_in, 1)
    elif not has_audio:
        args += ["-f", "lavfi", "-i", f"anullsrc=r={AUDIO_SR}:cl=stereo"]
    ffmpeg(*args, "-filter_complex", vchain + ";" + achain,
           "-map", "[v]", "-frames:v", str(n), "-an", "-c:v", "libx264",
           "-preset", settings["preset"], "-crf", "18", "-pix_fmt", "yuv420p", "-r", str(fps),
           "-video_track_timescale", str(fps * 1000), str(out_v),
           "-map", "[a]", "-vn", "-c:a", "pcm_s16le", "-ar", str(AUDIO_SR), str(out_a))


def _ass_escape(text):
    return (text or "").replace("{", "(").replace("}", ")").replace("\n", " ")


def _filter_path(path):
    return str(path).replace("\\", "/").replace(":", r"\:").replace("'", r"\'")


def can_draw_text():
    return ffmpeg_has_filter("ass")


def ass_filter(path):
    """libass with *complex* shaping: required for Hindi (e.g. the ि sign in विराट is drawn
    before its consonant). ffmpeg's 'subtitles' filter uses simple shaping and breaks it."""
    return f"ass='{_filter_path(path)}':fontsdir='{_filter_path(FONTS_DIR)}':shaping=complex"


def ass_header(W, H, size, margin_v, outline=3, primary="&H00FFFFFF", bold=-1, box=False):
    return (
        "[Script Info]\nScriptType: v4.00+\nWrapStyle: 0\nScaledBorderAndShadow: yes\n"
        f"PlayResX: {W}\nPlayResY: {H}\n\n[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,{FONT_NAME},{size},{primary},&H000000FF,"
        f"{'&H50000000' if box else '&H00000000'},&H96000000,"
        f"{bold},0,0,0,100,100,0,0,{3 if box else 1},{outline},{0 if box else 1},2,60,60,{margin_v},1\n\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")


def _ass_ts(t):
    cs = int(round(t * 100))
    h, rem = divmod(cs, 360000)
    m, rem = divmod(rem, 6000)
    s, cs = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def render_card(seg, background_png, out_v, out_a, tl, settings, work):
    W, H, fps = tl["width"], tl["height"], tl["fps"]
    dur = seg["frames"] / fps
    ass = Path(work) / f"card_{seg['i']}.ass"
    if seg.get("style") == "text":   # story text card: calmer, smaller, slow push-in
        size, tags = int(H * 0.058), (f"{{\\an5\\q0\\fad(350,350)\\fscx100\\fscy100"
                                      f"\\t(0,{int(dur * 1000)},\\fscx104\\fscy104)}}")
    else:                            # title card
        size, tags = int(H * 0.085), (f"{{\\an5\\fad(500,500)\\fscx108\\fscy108"
                                      f"\\t(0,{int(dur * 1000)},\\fscx100\\fscy100)}}")
    ass.write_text(ass_header(W, H, size, 0, outline=4) +
                   f"Dialogue: 0,{_ass_ts(0)},{_ass_ts(dur)},Default,,{int(W * 0.1)},{int(W * 0.1)},0,,"
                   f"{tags}{_ass_escape(seg['text'])}\n", encoding="utf-8")
    if background_png and Path(background_png).exists():
        inputs = ["-loop", "1", "-framerate", str(fps), "-i", str(background_png)]
        bg = (f"[0:v]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},"
              f"boxblur=luma_radius=25:luma_power=2,eq=brightness=-0.3:saturation=0.8,")
    else:
        inputs = ["-f", "lavfi", "-i", f"color=c=black:s={W}x{H}:r={fps}"]
        bg = "[0:v]"
    text = f"{ass_filter(ass)}," if can_draw_text() else ""
    vf = (f"{bg}setsar=1,{text}"
          f"fade=t=in:st=0:d={seg['fade_in']},fade=t=out:st={dur - seg['fade_out']:.3f}:"
          f"d={seg['fade_out']},format=yuv420p[v]")
    ffmpeg(*inputs, "-filter_complex", vf, "-map", "[v]", "-frames:v", str(seg["frames"]),
           "-c:v", "libx264", "-preset", settings["preset"], "-crf", "18", "-pix_fmt", "yuv420p",
           "-r", str(fps), "-video_track_timescale", str(fps * 1000), str(out_v))
    ffmpeg("-f", "lavfi", "-i", f"anullsrc=r={AUDIO_SR}:cl=stereo", "-af",
           f"atrim=end_sample={int(round(dur * AUDIO_SR))}", "-c:a", "pcm_s16le", str(out_a))


# ------------------------------------------------------------------ outputs
def write_srt(subtitles, path):
    lines = []
    for i, s in enumerate(subtitles, 1):
        lines += [str(i), f"{srt_ts(s['start'])} --> {srt_ts(s['end'])}", s["text"], ""]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def write_subtitle_ass(subtitles, path, W, H):
    body = ass_header(W, H, int(H * 0.048), int(H * 0.07))
    for s in subtitles:
        body += (f"Dialogue: 0,{_ass_ts(s['start'])},{_ass_ts(s['end'])},Default,,0,0,0,,"
                 f"{_ass_escape(s['text'])}\n")
    Path(path).write_text(body, encoding="utf-8")


def thumbnail_text(title):
    """Big YouTube-style text: one line for short titles, two balanced lines otherwise."""
    words = (title or "").split()
    if len(words) <= 3:
        return " ".join(words)
    half = (len(words) + 1) // 2
    return " ".join(words[:half]) + "\\N" + " ".join(words[half:])


def make_thumbnail(frame_png, title, out_jpg, work):
    ass = Path(work) / "thumb.ass"
    text = thumbnail_text(title)
    ass.write_text(ass_header(1280, 720, 92, 40, outline=7, primary="&H0000F0FF") +
                   f"Dialogue: 0,0:00:00.00,0:00:10.00,Default,,0,0,0,,{_ass_escape(text)}\n",
                   encoding="utf-8")
    text = f",{ass_filter(ass)}" if can_draw_text() else ""
    ffmpeg("-i", str(frame_png), "-vf",
           f"scale=1280:720:force_original_aspect_ratio=increase,crop=1280:720,"
           f"eq=contrast=1.15:saturation=1.3:brightness=-0.03{text}",
           "-frames:v", "1", "-q:v", "3", str(out_jpg))


def extract_frame(video, t, out_png):
    ffmpeg("-ss", f"{t:.2f}", "-i", str(video), "-frames:v", "1", str(out_png))
    return out_png


# ------------------------------------------------------------------ main render
def shot_name(seg, tl):
    """File name of a rendered shot. It carries a fingerprint of everything that shapes the
    shot, so a resumed job never reuses a shot rendered for an earlier cut of the film."""
    key = json.dumps([seg, tl["width"], tl["height"], tl["fps"]], sort_keys=True, default=str)
    return f"{seg['i']:04d}_{hashlib.sha1(key.encode()).hexdigest()[:10]}"


def render(tl, files, settings, job_dir, log, progress=None):
    job_dir = Path(job_dir)
    work = job_dir / "work"
    work.mkdir(parents=True, exist_ok=True)
    segs = tl["segments"]
    clip_segs = [s for s in segs if s["type"] == "clip"]
    names = {s["i"]: shot_name(s, tl) for s in segs}
    vpath = lambda s: work / f"v{names[s['i']]}.mp4"   # noqa: E731
    apath = lambda s: work / f"a{names[s['i']]}.wav"   # noqa: E731
    keep = {vpath(s).name for s in segs} | {apath(s).name for s in segs}
    for old in list(work.glob("v*.mp4")) + list(work.glob("a*.wav")):
        if old.name not in keep and old.name not in ("video.mp4",):
            old.unlink(missing_ok=True)       # shots of an earlier cut
    log(f"Rendering {len(clip_segs)} shots...")

    def do(seg):
        out_v, out_a = vpath(seg), apath(seg)
        if out_v.exists() and out_a.exists():
            return seg
        src, offset = locate(files, seg)
        if not src:
            raise PipelineError(f"no footage for segment {seg['i']} ({seg['video_id']})")
        # write under temporary names: a shot interrupted half-way is never mistaken for done
        tmp_v, tmp_a = out_v.with_suffix(".tmp.mp4"), out_a.with_suffix(".tmp.wav")
        render_segment(seg, src, offset, tmp_v, tmp_a, tl, settings)
        os.replace(tmp_a, out_a)
        os.replace(tmp_v, out_v)
        return seg

    workers = max(1, min(4, (os.cpu_count() or 4) // 2))
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for fut in as_completed([pool.submit(do, s) for s in clip_segs]):
            fut.result()
            done += 1
            if progress:
                progress(done / max(1, len(clip_segs)))
            if done % 20 == 0:
                log(f"  {done}/{len(clip_segs)} shots ready")

    # title card + thumbnail use the hottest climax shot as background
    climax = [s for s in clip_segs if s["act"] == "climax"] or clip_segs
    hero = max(climax, key=lambda s: s.get("heat", 0))
    hero_png = extract_frame(vpath(hero), hero["dur"] * 0.4, work / "hero.png")
    for seg in segs:
        if seg["type"] == "card":
            bg = hero_png
            if seg.get("style") == "text":   # background: the shot the card leads into
                near = next((x for x in segs[seg["i"] + 1:] if x["type"] == "clip"), None) or \
                    next((x for x in reversed(segs[:seg["i"]]) if x["type"] == "clip"), None)
                if near:
                    bg = extract_frame(vpath(near), near["dur"] * 0.3,
                                       work / f"bg{seg['i']:04d}.png")
            render_card(seg, bg, vpath(seg), apath(seg),
                        tl, settings, work)

    log("Joining shots...")
    vlist, alist = work / "video.txt", work / "audio.txt"
    vlist.write_text("".join(f"file '{vpath(s).name}'\n" for s in segs))
    alist.write_text("".join(f"file '{apath(s).name}'\n" for s in segs))
    ffmpeg("-f", "concat", "-safe", "0", "-i", str(vlist), "-c", "copy", str(work / "video.mp4"))
    ffmpeg("-f", "concat", "-safe", "0", "-i", str(alist), "-c", "copy", str(work / "clips.wav"))

    log("Mixing music and Hindi narration...")
    bed_files = []
    for i, act in enumerate(tl["acts"]):
        seconds = act["frames"] / tl["fps"]
        path = work / f"bed{i}.wav"
        music.render_act_bed(path, seconds, act["track"], act["mood"], act["gain_points"],
                             act["narration"], seed=i,
                             generate=bool(settings.get("generated_music", False)))
        bed_files.append(path)
    blist = work / "bed.txt"
    blist.write_text("".join(f"file '{p.name}'\n" for p in bed_files))
    ffmpeg("-f", "concat", "-safe", "0", "-i", str(blist), "-c", "copy", str(work / "bed.wav"))

    write_srt(tl["subtitles"], job_dir / "narration_hi.srt")
    final = job_dir / "final.mp4"
    log("Final export (loudness for YouTube, subtitles)...")
    mix = (f"[1:a][2:a]amix=inputs=2:normalize=0:duration=first,"
           f"loudnorm=I={settings['target_lufs']}:TP=-1.5:LRA=11,aresample={AUDIO_SR}[a]")
    burn = settings.get("burn_subtitles") and tl["subtitles"]
    if burn and not can_draw_text():
        log("Your ffmpeg cannot draw text (no libass) - subtitles saved as .srt only. "
            "For burned-in Hindi subtitles and titles: brew install ffmpeg-full")
        burn = False
    if burn:
        sub_ass = work / "subs.ass"
        write_subtitle_ass(tl["subtitles"], sub_ass, tl["width"], tl["height"])
        video_args = ["-filter_complex",
                      f"[0:v]{ass_filter(sub_ass)}[v];" + mix,
                      "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-preset", settings["preset"],
                      "-crf", str(settings["crf"]), "-pix_fmt", "yuv420p"]
    else:
        video_args = ["-filter_complex", mix, "-map", "0:v", "-map", "[a]", "-c:v", "copy"]
    ffmpeg("-i", str(work / "video.mp4"), "-i", str(work / "clips.wav"), "-i", str(work / "bed.wav"),
           *video_args, "-c:a", "aac", "-b:a", "192k", "-ar", str(AUDIO_SR),
           "-movflags", "+faststart", str(final))

    make_thumbnail(hero_png, tl.get("title_hi") or "", job_dir / "thumbnail.jpg", work)
    return final
