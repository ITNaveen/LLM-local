"""Stage 6 - render: download only the needed seconds, conform every shot (1080p, 30fps,
blurred fill for vertical/4:3 footage, light grade), mix dialogue + music + narration,
burn Hindi subtitles, and export a YouTube-ready MP4 + SRT + thumbnail."""

import hashlib
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import music, textimg
from .util import PipelineError, ffmpeg, has_video, probe, srt_ts

AUDIO_SR = 48000


# ------------------------------------------------------------------ downloads
def audio_seg(seg):
    """The sound a cutaway shot plays: the speaker's own passage, as a pseudo-segment."""
    a = seg["audio_src"]
    return {"type": "clip", "video_id": a["video_id"], "src_start": a["src_start"], "dur": seg["dur"]}


def footage_needs(segments):
    """Every (picture or sound) range the clips need from the sources."""
    for s in segments:
        if s["type"] == "clip":
            yield s
            if s.get("audio_src"):
                yield audio_seg(s)


def shot_available(files, seg, strict=False):
    return bool(locate(files, seg, strict)[0]) and (
        not seg.get("audio_src") or bool(locate(files, audio_seg(seg), strict)[0]))


def plan_downloads(segments, pad=1.0, merge_gap=6.0):
    """Merge each video's needed ranges into as few downloads as possible."""
    ranges = {}
    for s in footage_needs(segments):
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


def render_segment(seg, src, offset, out_v, out_a, tl, settings, asrc=None, aoffset=0.0):
    """One shot: picture from `src` at `offset`; sound from the same file, or from `asrc` at
    `aoffset` for a cutaway (a speaker keeps talking while we show what they talk about)."""
    W, H, fps = tl["width"], tl["height"], tl["fps"]
    n, dur = seg["frames"], seg["frames"] / fps
    samples = int(round(dur * AUDIO_SR))
    w, h, has_audio = _stream_info(src)
    args, count = [], 0

    def add(*inp):
        nonlocal count
        args.extend(inp)
        count += 1
        return count - 1

    clip_in = ["-ss", f"{max(0, offset):.3f}", "-t", f"{dur + 0.5:.3f}", "-i", str(src)]
    if w:
        v_in = add(*clip_in)
    else:   # no picture at all (should have been caught at download): never fail the film
        v_in = add("-f", "lavfi", "-i", f"color=c=black:s={W}x{H}:r={fps}:d={dur + 1:.2f}")
    if asrc:
        a_has = _stream_info(asrc)[2]
        a_in = add("-ss", f"{max(0, aoffset):.3f}", "-t", f"{dur + 0.5:.3f}", "-i", str(asrc)) if a_has else None
    else:
        a_has = has_audio
        a_in = (v_in if w else add(*clip_in)) if a_has else None
    if a_in is None:
        a_in = add("-f", "lavfi", "-i", f"anullsrc=r={AUDIO_SR}:cl=stereo")

    aspect = w / h if h else 16 / 9
    zoom = 1.12 if seg.get("fx") == "punch" and seg.get("zoom", True) else 1.0
    if abs(aspect - W / H) < 0.04:
        vchain = (f"[{v_in}:v]scale={int(W * zoom) // 2 * 2}:{int(H * zoom) // 2 * 2}:"
                  f"force_original_aspect_ratio=increase,crop={W}:{H}[v0];")
    else:  # vertical / 4:3: blurred copy fills the frame behind the real shot
        vchain = (f"[{v_in}:v]split=2[bgs][fgs];[bgs]scale={W}:{H}:force_original_aspect_ratio=increase,"
                  f"crop={W}:{H},boxblur=luma_radius=40:luma_power=2,eq=brightness=-0.12[bg];"
                  f"[fgs]scale={W}:{H}:force_original_aspect_ratio=decrease[fg];"
                  f"[bg][fg]overlay=(W-w)/2:(H-h)/2[v0];")
    darken = ""
    text_png = None
    if seg.get("overlay") and can_draw_text():
        style = seg.get("overlay_style", "headline")
        text_png = Path(out_v).with_suffix(".text.png")
        (textimg.title if style == "title" else textimg.headline)(text_png, W, H, seg["overlay"])
        darken = (",eq=brightness=-0.22:saturation=0.8" if style == "title"
                  else ",eq=brightness=-0.3:saturation=0.75")
    vchain += (f"[v0]setsar=1,fps={fps},eq=contrast=1.05:saturation=1.08,"
               f"tpad=stop_mode=clone:stop_duration={max(4.0, dur + 1):.1f}{darken}[vb];")
    last = "vb"
    if text_png:
        t_in = add("-i", str(text_png))       # decoded once, repeated by the loop filter
        fade_t = min(0.35, dur / 4)
        vchain += (f"[{t_in}:v]format=rgba,loop=loop={n + 30}:size=1,fps={fps},"
                   f"fade=t=in:st=0:d={fade_t:.2f}:alpha=1,"
                   f"fade=t=out:st={max(0, dur - fade_t):.3f}:d={fade_t:.2f}:alpha=1[tx];"
                   f"[vb][tx]overlay=0:0:shortest=0[vt];")
        last = "vt"
    fades = ""
    if seg.get("fx") == "punch":       # flash-cut: the shot bursts in from white
        fades += ",fade=t=in:st=0:d=0.16:color=white"
    elif seg["fade_in"]:
        fades += f",fade=t=in:st=0:d={seg['fade_in']}"
    if seg["fade_out"]:
        fades += f",fade=t=out:st={max(0, dur - seg['fade_out']):.3f}:d={seg['fade_out']}"
    vchain += f"[{last}]null{fades},format=yuv420p[v]"
    gain = seg["clip_gain"]
    afades = f"afade=t=in:d={max(0.03, seg['fade_in'] if seg.get('fx') != 'punch' else 0.03)}"
    afades += f",afade=t=out:st={max(0, dur - max(0.06, seg['fade_out'])):.3f}:d={max(0.06, seg['fade_out'])}"
    achain = (f"[{a_in}:a]aresample={AUDIO_SR},aformat=sample_fmts=fltp:channel_layouts=stereo,"
              + ("loudnorm=I=-18:TP=-3:LRA=9," if a_has else "")
              + f"aresample={AUDIO_SR},volume={gain:.3f},{afades},apad,atrim=end_sample={samples}[a]")
    ffmpeg(*args, "-filter_complex", vchain + ";" + achain,
           "-map", "[v]", "-frames:v", str(n), "-an", "-c:v", "libx264",
           "-preset", settings["preset"], "-crf", "18", "-pix_fmt", "yuv420p", "-r", str(fps),
           "-video_track_timescale", str(fps * 1000), str(out_v),
           "-map", "[a]", "-vn", "-c:a", "pcm_s16le", "-ar", str(AUDIO_SR), str(out_a))


def can_draw_text():
    """Hindi text is drawn by Pillow (any ffmpeg works); see textimg.py."""
    return textimg.available()


def render_card(seg, background_png, out_v, out_a, tl, settings, work):
    """A text card on a background picture (only used when no footage could be found)."""
    W, H, fps = tl["width"], tl["height"], tl["fps"]
    dur = seg["frames"] / fps
    if background_png and Path(background_png).exists():
        inputs = ["-loop", "1", "-framerate", str(fps), "-i", str(background_png)]
        bg = (f"[0:v]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},"
              f"boxblur=luma_radius=25:luma_power=2,eq=brightness=-0.3:saturation=0.8,setsar=1[bg]")
    else:
        inputs = ["-f", "lavfi", "-i", f"color=c=black:s={W}x{H}:r={fps}"]
        bg = "[0:v]setsar=1[bg]"
    last = "bg"
    if can_draw_text():
        png = Path(work) / f"card_{seg['i']}.png"
        (textimg.title if seg.get("style") == "title" else textimg.headline)(png, W, H, seg["text"])
        inputs += ["-i", str(png)]
        bg += (f";[1:v]format=rgba,loop=loop={seg['frames'] + 30}:size=1,fps={fps}[tx];"
               "[bg][tx]overlay=0:0[bt]")
        last = "bt"
    vf = (f"{bg};[{last}]fade=t=in:st=0:d={seg['fade_in']},"
          f"fade=t=out:st={dur - seg['fade_out']:.3f}:d={seg['fade_out']},format=yuv420p[v]")
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


def subtitle_track(subtitles, W, H, duration, work):
    """Narration subtitles as one transparent video track (Pillow pictures, so it works with
    any ffmpeg), laid over the film in the final export."""
    d = Path(work) / "subs"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    blank = textimg.subtitle(d / "blank.png", W, H, "")
    lines, t = [], 0.0
    for i, sub in enumerate(sorted(subtitles, key=lambda x: x["start"])):
        start = max(t, sub["start"])
        end = max(sub["end"], start + 0.2)
        if start - t > 0.01:
            lines += [f"file '{blank.name}'", f"duration {start - t:.3f}"]
        png = d / f"s{i:04d}.png"
        textimg.subtitle(png, W, H, sub["text"])
        lines += [f"file '{png.name}'", f"duration {end - start:.3f}"]
        t = end
    lines += [f"file '{blank.name}'", f"duration {max(0.1, duration - t + 1):.3f}", f"file '{blank.name}'"]
    (d / "list.txt").write_text("\n".join(lines) + "\n")
    out = d / "subs.mov"
    # one frame per change (variable frame rate): the overlay holds each until the next
    ffmpeg("-f", "concat", "-safe", "0", "-i", str(d / "list.txt"), "-fps_mode", "passthrough",
           "-c:v", "qtrle", "-pix_fmt", "argb", str(out))
    return out


def thumbnail_text(title):
    """Big YouTube-style text: one line for short titles, two balanced lines otherwise."""
    words = (title or "").split()
    if len(words) <= 3:
        return " ".join(words)
    half = (len(words) + 1) // 2
    return " ".join(words[:half]) + "\\N" + " ".join(words[half:])


def make_thumbnail(frame_png, title, out_jpg, work):
    if can_draw_text():
        return textimg.thumbnail(frame_png, out_jpg, title)
    ffmpeg("-i", str(frame_png), "-vf",
           "scale=1280:720:force_original_aspect_ratio=increase,crop=1280:720,"
           "eq=contrast=1.15:saturation=1.3:brightness=-0.03",
           "-frames:v", "1", "-q:v", "3", str(out_jpg))
    return out_jpg


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
        asrc, aoffset = None, 0.0
        if seg.get("audio_src"):          # cutaway: the speaker's voice over other footage
            asrc, aoffset = locate(files, audio_seg(seg))
            if not asrc:
                raise PipelineError(f"no sound for segment {seg['i']} ({seg['audio_src']['video_id']})")
        # write under temporary names: a shot interrupted half-way is never mistaken for done
        tmp_v, tmp_a = out_v.with_suffix(".tmp.mp4"), out_a.with_suffix(".tmp.wav")
        render_segment(seg, src, offset, tmp_v, tmp_a, tl, settings, asrc, aoffset)
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
                             generate=bool(settings.get("builtin_music", True)),
                             effects=act.get("effects", []))
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
        log("Hindi text needs the Pillow package (it installs on the next start) - "
            "subtitles saved as .srt only this time.")
        burn = False
    extra = []
    if burn:
        subs = subtitle_track(tl["subtitles"], tl["width"], tl["height"], tl["duration"], work)
        extra = ["-i", str(subs)]
        video_args = ["-filter_complex",
                      "[0:v][3:v]overlay=0:H-h:eof_action=pass:format=auto,format=yuv420p[v];" + mix,
                      "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-preset", settings["preset"],
                      "-crf", str(settings["crf"]), "-pix_fmt", "yuv420p"]
    else:
        video_args = ["-filter_complex", mix, "-map", "0:v", "-map", "[a]", "-c:v", "copy"]
    ffmpeg("-i", str(work / "video.mp4"), "-i", str(work / "clips.wav"), "-i", str(work / "bed.wav"),
           *extra, *video_args, "-c:a", "aac", "-b:a", "192k", "-ar", str(AUDIO_SR),
           "-movflags", "+faststart", str(final))

    make_thumbnail(hero_png, tl.get("title_hi") or "", job_dir / "thumbnail.jpg", work)
    return final
