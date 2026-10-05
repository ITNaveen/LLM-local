"""Stage 6 - render: download only the needed seconds, conform every shot (1080p, 30fps,
blurred fill for vertical/4:3 footage, light grade), mix dialogue + music + narration,
burn Hindi subtitles, and export a YouTube-ready MP4 + SRT + thumbnail."""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import music
from .config import FONT_NAME, FONTS_DIR
from .util import PipelineError, ffmpeg, ffmpeg_has_filter, probe, srt_ts  # noqa: F401

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
                r, path = fut.result()
                files.setdefault(r["video_id"], []).append({**r, "file": path})
            except Exception as e:  # noqa: BLE001 - drop clips from this section later
                failed.append(str(e)[:200])
                log(f"  download failed: {str(e)[:160]}")
            if n % 5 == 0 or n == len(plan):
                log(f"  downloaded {n}/{len(plan)}")
    return files, failed


def locate(files, seg):
    for r in files.get(seg["video_id"], []):
        if r["start"] - 0.01 <= seg["src_start"] and seg["src_start"] + seg["dur"] <= r["end"] + 0.6:
            return r["file"], seg["src_start"] - r["start"]
    return None, 0.0


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
    vchain += (f"[v0]setsar=1,fps={fps},eq=contrast=1.05:saturation=1.08,"
               f"tpad=stop_mode=clone:stop_duration=4{fades},format=yuv420p[v]")
    gain = seg["clip_gain"]
    afades = f"afade=t=in:d={max(0.03, seg['fade_in'])}"
    afades += f",afade=t=out:st={max(0, dur - max(0.06, seg['fade_out'])):.3f}:d={max(0.06, seg['fade_out'])}"
    a_in = "[0:a]" if has_audio else "[1:a]"
    achain = (f"{a_in}aresample={AUDIO_SR},aformat=sample_fmts=fltp:channel_layouts=stereo,"
              + ("loudnorm=I=-18:TP=-3:LRA=9," if has_audio else "")
              + f"aresample={AUDIO_SR},volume={gain:.3f},{afades},apad,atrim=end_sample={samples}[a]")
    args = ["-ss", f"{max(0, offset):.3f}", "-t", f"{dur + 0.5:.3f}", "-i", src]
    if not has_audio:
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


def ass_header(W, H, size, margin_v, outline=3, primary="&H00FFFFFF", bold=-1):
    return (
        "[Script Info]\nScriptType: v4.00+\nWrapStyle: 0\nScaledBorderAndShadow: yes\n"
        f"PlayResX: {W}\nPlayResY: {H}\n\n[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,{FONT_NAME},{size},{primary},&H000000FF,&H00000000,&H96000000,"
        f"{bold},0,0,0,100,100,0,0,1,{outline},1,2,60,60,{margin_v},1\n\n"
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
def render(tl, files, settings, job_dir, log, progress=None):
    job_dir = Path(job_dir)
    work = job_dir / "work"
    work.mkdir(parents=True, exist_ok=True)
    segs = tl["segments"]
    clip_segs = [s for s in segs if s["type"] == "clip"]
    log(f"Rendering {len(clip_segs)} shots...")

    def do(seg):
        out_v, out_a = work / f"v{seg['i']:04d}.mp4", work / f"a{seg['i']:04d}.wav"
        if out_v.exists() and out_a.exists():
            return seg
        src, offset = locate(files, seg)
        if not src:
            raise PipelineError(f"no footage for segment {seg['i']} ({seg['video_id']})")
        render_segment(seg, src, offset, out_v, out_a, tl, settings)
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
    hero_png = extract_frame(work / f"v{hero['i']:04d}.mp4", hero["dur"] * 0.4, work / "hero.png")
    for seg in segs:
        if seg["type"] == "card":
            bg = hero_png
            if seg.get("style") == "text":   # background: the shot the card leads into
                near = next((x for x in segs[seg["i"] + 1:] if x["type"] == "clip"), None) or \
                    next((x for x in reversed(segs[:seg["i"]]) if x["type"] == "clip"), None)
                if near:
                    bg = extract_frame(work / f"v{near['i']:04d}.mp4", near["dur"] * 0.3,
                                       work / f"bg{seg['i']:04d}.png")
            render_card(seg, bg, work / f"v{seg['i']:04d}.mp4", work / f"a{seg['i']:04d}.wav",
                        tl, settings, work)

    log("Joining shots...")
    vlist, alist = work / "video.txt", work / "audio.txt"
    vlist.write_text("".join(f"file 'v{s['i']:04d}.mp4'\n" for s in segs))
    alist.write_text("".join(f"file 'a{s['i']:04d}.wav'\n" for s in segs))
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
