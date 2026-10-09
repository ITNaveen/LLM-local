"""Where footage comes from. YouTubeSource uses yt-dlp; FixtureSource is an offline
stand-in (synthetic clips) used by the tests and the UI's demo mode."""

import glob
import hashlib
import html
import json
import random
import re
import threading
from pathlib import Path

from .config import CACHE_DIR, FONT_FILE
from .util import PipelineError, ffmpeg, has_video, read_json, write_json

CAPTION_LANG_PRIORITY = ["hi", "hi-IN", "hi-orig", "en-orig", "en", "en-IN", "en-US", "en-GB"]
_LOCKS, _LOCKS_GUARD = {}, threading.Lock()


def _lock(key):
    """One lock per output file: two download threads must never write the same file."""
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(str(key), threading.Lock())


def finished_download(base):
    """The finished file for a download base: exactly '<base>.mp4/.mkv/.webm', with a picture.
    yt-dlp's in-between files ('<base>.f398.mp4', '<base>.f251.webm', '<base>.temp.mp4',
    '.part') are deleted or renamed by yt-dlp once it merges, so they must never be used."""
    out = []
    for ext in ("mp4", "mkv", "webm"):
        p = Path(f"{base}.{ext}")
        if p.exists() and has_video(p):
            out.append(str(p))
    return out


# ---------------------------------------------------------------- captions
def parse_json3(text):
    data = json.loads(text)
    cues = []
    for ev in data.get("events", []):
        if "segs" not in ev or ev.get("aAppend"):
            continue
        line = "".join(s.get("utf8", "") for s in ev["segs"]).replace("\n", " ").strip()
        if not line:
            continue
        start = ev.get("tStartMs", 0) / 1000
        dur = ev.get("dDurationMs", 0) / 1000
        cues.append({"start": start, "end": start + max(dur, 0.3), "text": line})
    return clean_cues(cues)


_VTT_TIME = re.compile(
    r"(\d+:)?(\d{1,2}):(\d{2})[.,](\d{3})\s*-->\s*(\d+:)?(\d{1,2}):(\d{2})[.,](\d{3})")


def _vtt_seconds(h, m, s, ms):
    return int((h or "0:")[:-1]) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def parse_vtt(text):
    cues = []
    blocks = re.split(r"\n\s*\n", text.replace("\r", ""))
    for block in blocks:
        lines = block.strip().split("\n")
        for i, line in enumerate(lines):
            m = _VTT_TIME.search(line)
            if not m:
                continue
            g = m.groups()
            start = _vtt_seconds(g[0], g[1], g[2], g[3])
            end = _vtt_seconds(g[4], g[5], g[6], g[7])
            body = " ".join(lines[i + 1:])
            body = html.unescape(re.sub(r"<[^>]+>", "", body)).strip()
            if body:
                cues.append({"start": start, "end": end, "text": body})
            break
    return clean_cues(cues)


def clean_cues(cues):
    """Drop YouTube auto-caption 'rolling' duplicates and overlaps."""
    cues = sorted(cues, key=lambda c: c["start"])
    out = []
    for cue in cues:
        text = re.sub(r"\s+", " ", cue["text"]).strip()
        if out:
            prev = out[-1]
            if text == prev["text"] or text in prev["text"]:
                prev["end"] = max(prev["end"], cue["end"])
                continue
            if text.startswith(prev["text"]):
                text = text[len(prev["text"]):].strip()
                if not text:
                    continue
            if prev["end"] > cue["start"]:
                prev["end"] = cue["start"]
        if re.fullmatch(r"\[[^\]]*\]", text):  # [Music], [Applause]
            continue
        out.append({"start": round(cue["start"], 3), "end": round(cue["end"], 3), "text": text})
    return [c for c in out if c["end"] - c["start"] > 0.05]


UNDERSTOOD_LANGS = ("hi", "en")


def base_lang(code):
    return (code or "").replace("-orig", "").split("-")[0].split("_")[0].lower()


def spoken_language(info):
    """The language people actually speak in the video ('' if unknown)."""
    if info.get("language"):
        return base_lang(info["language"])
    for key in (info.get("automatic_captions") or {}):
        if key.endswith("-orig"):
            return base_lang(key)
    return ""


def pick_caption_track(info):
    """Best transcript to *understand* the video: manual Hindi/English first, then the
    auto-captions of the spoken language if that is Hindi/English, then YouTube's
    machine translation to English/Hindi (so foreign videos can still be judged)."""
    spoken = spoken_language(info)
    subs = info.get("subtitles") or {}
    auto = info.get("automatic_captions") or {}
    options = [("subtitles", l) for l in CAPTION_LANG_PRIORITY if l in subs]
    orig = [l for l in auto if l.endswith("-orig")]
    options += [("automatic_captions", l) for l in orig if base_lang(l) in UNDERSTOOD_LANGS]
    options += [("automatic_captions", l) for l in CAPTION_LANG_PRIORITY
                if l in auto and l not in orig]
    options += [("automatic_captions", l) for l in orig if base_lang(l) not in UNDERSTOOD_LANGS]
    options += [("subtitles", l) for l in subs if l not in CAPTION_LANG_PRIORITY]
    for kind, lang in options:
        tracks = info.get(kind) or {}
        for ext in ("json3", "vtt"):
            for f in tracks.get(lang) or []:
                if f.get("ext") == ext and f.get("url"):
                    return {"lang": lang, "ext": ext, "url": f["url"], "kind": kind,
                            "spoken": spoken}
    return None


# ---------------------------------------------------------------- YouTube
class YouTubeSource:
    name = "youtube"

    def __init__(self, settings, log=None):
        self.settings = settings
        self.log = log or (lambda m: None)
        self.details_dir = CACHE_DIR / "details_v2"
        self.details_dir.mkdir(parents=True, exist_ok=True)

    def _opts(self, **extra):
        opts = {"quiet": True, "no_warnings": True, "noprogress": True,
                "socket_timeout": 20, "retries": 3, "ignoreerrors": False}
        if self.settings.get("force_ipv4", True):
            opts["source_address"] = "0.0.0.0"   # IPv6 to YouTube is often very slow at home
        browser = (self.settings.get("cookies_from_browser") or "").strip()
        if browser:
            opts["cookiesfrombrowser"] = (browser,)
        opts.update(extra)
        return opts

    def search(self, query, n):
        import yt_dlp
        with yt_dlp.YoutubeDL(self._opts(extract_flat="in_playlist", skip_download=True)) as ydl:
            info = ydl.extract_info(f"ytsearch{int(n)}:{query}", download=False)
        out = []
        for e in (info or {}).get("entries") or []:
            if not e or not e.get("id") or e.get("live_status") in ("is_live", "is_upcoming"):
                continue
            out.append({
                "id": e["id"],
                "title": e.get("title") or "",
                "channel": e.get("channel") or e.get("uploader") or "",
                "duration": float(e.get("duration") or 0),
                "views": int(e.get("view_count") or 0),
                "description": e.get("description") or "",
                "upload_date": e.get("upload_date") or "",
                "url": f"https://www.youtube.com/watch?v={e['id']}",
            })
        return out

    def details(self, video_id):
        cache = self.details_dir / f"{video_id}.json"
        cached = read_json(cache)
        if cached:
            return cached
        import yt_dlp
        with yt_dlp.YoutubeDL(self._opts(skip_download=True, ignore_no_formats_error=True)) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
            track = pick_caption_track(info)
            captions, lang = [], ""
            if track:
                try:
                    raw = ydl.urlopen(track["url"]).read().decode("utf-8", errors="replace")
                    captions = parse_json3(raw) if track["ext"] == "json3" else parse_vtt(raw)
                    lang = base_lang(track["lang"])
                except Exception as e:  # noqa: BLE001 - captions are optional
                    self.log(f"  captions failed for {video_id}: {e}")
        heatmap = [{"start": h["start_time"], "end": h["end_time"], "value": h["value"]}
                   for h in (info.get("heatmap") or [])]
        details = {
            "id": video_id,
            "title": info.get("title") or "",
            "channel": info.get("channel") or info.get("uploader") or "",
            "duration": float(info.get("duration") or 0),
            "views": int(info.get("view_count") or 0),
            "upload_date": info.get("upload_date") or "",
            "width": info.get("width") or 0,
            "height": info.get("height") or 0,
            "description": (info.get("description") or "")[:1500],
            "chapters": [{"start": c["start_time"], "end": c["end_time"], "title": c["title"]}
                         for c in (info.get("chapters") or [])],
            "heatmap": heatmap,
            "downloadable": bool(info.get("formats")),
            "captions": captions,
            "caption_lang": lang,
            "spoken_lang": spoken_language(info) or (lang if track and track["kind"] == "automatic_captions" else ""),
            "url": f"https://www.youtube.com/watch?v={video_id}",
        }
        write_json(cache, details)
        return details

    def download_section(self, video_id, start, end, out_base):
        """Download only [start, end] seconds of a video.
        Returns the file path, or {"file", "start", "end"} when it had to fall back to
        downloading the whole video (some videos refuse cut-out sections)."""
        import yt_dlp
        from yt_dlp.utils import download_range_func
        h = self.settings.get("height", 1080)
        out_base = Path(out_base)
        full_base = out_base.parent / f"{video_id}_full"
        full_lock = _lock(full_base)

        def clean(base):
            for p in glob.glob(str(base) + ".*"):
                Path(p).unlink(missing_ok=True)

        with _lock(out_base):
            if finished_download(out_base):
                return finished_download(out_base)[0]
            if full_lock.locked():          # another thread is fetching this whole video
                with full_lock:
                    pass
            if finished_download(full_base):
                return self._full_result(video_id, finished_download(full_base)[0])
            url = f"https://www.youtube.com/watch?v={video_id}"
            formats = [f"bv*[height<={h}][ext=mp4]+ba[ext=m4a]/bv*[height<={h}]+ba/b[height<={h}]/bv*+ba/b",
                       "b[height<=720][ext=mp4][vcodec!=none]/b[height<=720][vcodec!=none]/18"]
            errors = []
            for fmt in formats:
                try:
                    with yt_dlp.YoutubeDL(self._opts(
                            format=fmt, download_ranges=download_range_func(None, [(start, end)]),
                            force_keyframes_at_cuts=True, merge_output_format="mp4",
                            outtmpl={"default": str(out_base) + ".%(ext)s"})) as ydl:
                        ydl.download([url])
                    if finished_download(out_base):
                        return finished_download(out_base)[0]
                except Exception as e:  # noqa: BLE001 - try the next way
                    errors.append(str(e)[-120:])
                else:
                    errors.append("downloaded file had no picture")
                clean(out_base)
        # Last resort: the whole video (only if it is not too long), cut locally later.
        # Several sections of one video can end up here at once: only one thread downloads,
        # the others wait and then use its finished file.
        duration = float((read_json(self.details_dir / f"{video_id}.json") or {}).get("duration") or 0)
        if 0 < duration <= 40 * 60:
            with full_lock:
                if finished_download(full_base):
                    return self._full_result(video_id, finished_download(full_base)[0])
                self.log(f"  section download refused for {video_id}; fetching the whole video instead")
                try:
                    with yt_dlp.YoutubeDL(self._opts(
                            format="b[height<=720][ext=mp4][vcodec!=none]/bv*[height<=720]+ba/b[vcodec!=none]",
                            merge_output_format="mp4",
                            outtmpl={"default": str(full_base) + ".%(ext)s"})) as ydl:
                        ydl.download([url])
                    if finished_download(full_base):
                        return self._full_result(video_id, finished_download(full_base)[0])
                    errors.append("whole video had no picture")
                except Exception as e:  # noqa: BLE001
                    errors.append(str(e)[-120:])
        raise PipelineError(f"download of {video_id} [{start:.0f}-{end:.0f}s] failed: {errors[-1] if errors else '?'}")

    def _full_result(self, video_id, path):
        from .util import media_duration
        return {"file": path, "start": 0.0, "end": media_duration(path)}

    def download_full(self, url, out_base, max_height=720):
        """Used by the style learner for a reference video."""
        import yt_dlp
        opts = self._opts(
            format=f"bv*[height<={max_height}]+ba/b[height<={max_height}]/b",
            merge_output_format="mp4",
            outtmpl={"default": str(out_base) + ".%(ext)s"},
        )
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            track = pick_caption_track(info)
            captions = []
            if track:
                try:
                    raw = ydl.urlopen(track["url"]).read().decode("utf-8", errors="replace")
                    captions = parse_json3(raw) if track["ext"] == "json3" else parse_vtt(raw)
                except Exception:  # noqa: BLE001
                    captions = []
        files = finished_download(out_base)
        if not files:
            raise PipelineError("reference video download failed")
        return files[0], {"title": info.get("title") or "", "captions": captions,
                          "heatmap": [{"start": h["start_time"], "end": h["end_time"],
                                       "value": h["value"]} for h in (info.get("heatmap") or [])]}


# ---------------------------------------------------------------- offline fixtures
_FAKE_LINES = [
    "यह पल पूरे देश के लिए ऐतिहासिक था।",
    "हमने कभी हार नहीं मानी और आगे बढ़ते रहे।",
    "जनता का भरोसा हमारी सबसे बड़ी ताकत है।",
    "आज हम दुनिया को दिखा देंगे कि हम क्या कर सकते हैं।",
    "ये जीत हर उस इंसान की है जिसने सपना देखा।",
    "मुश्किलें आईं लेकिन हौसला कभी नहीं टूटा।",
    "this is a historic moment for the whole country.",
    "the crowd is going absolutely wild right now.",
]

# Offline catalogue: mostly good news footage plus the traps a real search returns.
FIXTURE_VIDEOS = [
    ("Election night: full report", "hi"), ("Victory speech full video", "hi"),
    ("Press conference after the result", "en"), ("Ground report from the rally", "hi"),
    ("Parliament debate highlights", "en"), ("Crowd celebrations live", "hi"),
    ("Expert analysis: what changes now", "en"), ("Interview with the party leader", "hi"),
    ("Stand-up comedy special 2025", "hi"),          # off-topic: the AI should reject it
    ("జర్మనీ ఉద్యోగాలు vlog", "te"),                 # regional Indian language: never used
    ("Tagesschau Wahlabend Sondersendung", "de"),   # foreign speech: silent visuals only
    ("Drone footage of the rally", ""),             # no speech/captions: visuals only
]


class FixtureSource:
    """Deterministic synthetic 'YouTube' for tests and demo mode. No network."""

    name = "fixture"
    COLORS = ["0x1d3557", "0x8d0801", "0x2a9d8f", "0xe76f51", "0x6a4c93", "0x264653",
              "0xbc6c25", "0x3a5a40", "0x9d0208", "0x023e8a", "0x7b2cbf", "0x495057"]

    def __init__(self, settings=None, n_videos=12, log=None):
        self.settings = settings or {}
        self.n_videos = n_videos
        self.log = log or (lambda m: None)

    def _rng(self, key):
        return random.Random(int(hashlib.md5(key.encode()).hexdigest()[:8], 16))

    def search(self, query, n):
        out = []
        for i in range(self.n_videos):
            rng = self._rng(f"{query}-{i}")
            vid = f"fx{i:02d}"
            out.append({
                "id": vid,
                "title": f"{query} — {FIXTURE_VIDEOS[i % len(FIXTURE_VIDEOS)][0]}",
                "channel": f"Channel {i % 5}",
                "duration": float(90 + (i * 37) % 150),
                "views": int(10 ** rng.uniform(3.5, 6.8)),
                "description": f"{query} full coverage",
                "upload_date": f"2024{(i % 12) + 1:02d}15",
                "url": f"https://www.youtube.com/watch?v={vid}",
            })
        return out[:n]

    def details(self, video_id):
        i = int(video_id[2:])
        duration = float(90 + (i * 37) % 150)
        rng = self._rng(video_id)
        title, spoken = FIXTURE_VIDEOS[i % len(FIXTURE_VIDEOS)]
        captions, t = [], 3.0
        while spoken and t < duration - 4:
            dur = rng.uniform(2.5, 5.5)
            # every video talks about its own details, like real channels do
            line = rng.choice(_FAKE_LINES).rstrip("।.") + " " + " ".join(
                f"v{i}k{rng.randint(0, 30)}" for _ in range(5)) + "।"
            if spoken == "te":
                line = "ఇది ఒక పరీక్ష వాక్యం."
            captions.append({"start": round(t, 2), "end": round(t + dur, 2), "text": line})
            t += dur + rng.uniform(0.2, 1.5)
        heatmap, steps = [], 100
        peaks = [rng.uniform(0.15, 0.9) * duration for _ in range(2)]
        for k in range(steps):
            s = duration * k / steps
            v = 0.15 + 0.1 * rng.random() + sum(
                0.75 * max(0.0, 1 - abs(s - p) / (duration * 0.05)) for p in peaks)
            heatmap.append({"start": s, "end": s + duration / steps, "value": min(1.0, v)})
        return {
            "id": video_id, "title": title, "channel": f"Channel {i % 5}",
            "duration": duration, "views": 1000 * (i + 1), "upload_date": f"2024{(i % 12) + 1:02d}15",
            "width": 1280, "height": 720 if i % 4 else 960, "description": "",
            "chapters": [], "heatmap": heatmap, "captions": captions,
            "caption_lang": "en" if spoken == "de" else spoken, "spoken_lang": spoken,
            "url": f"https://www.youtube.com/watch?v={video_id}",
        }

    def download_section(self, video_id, start, end, out_base):
        out = Path(str(out_base) + ".mp4")
        if out.exists():
            return str(out)
        i = int(video_id[2:])
        color = self.COLORS[i % len(self.COLORS)]
        w, h = (1280, 720) if i % 4 else (960, 720)  # some 4:3 sources
        dur = max(0.5, end - start)
        freq = 180 + i * 40
        font = str(FONT_FILE).replace(":", r"\:")
        label = (f"drawtext=fontfile='{font}':text='{video_id}  %{{pts\\:hms\\:{start:.3f}}}':"
                 f"fontcolor=white:fontsize=48:x=40:y=40:box=1:boxcolor=black@0.5")
        # each video / scene looks different (real footage does), so duplicate checks work
        angle = ((i * 7 + int(start // 10)) % 12) / 12 * 6.2832
        ffmpeg("-f", "lavfi", "-i", f"testsrc2=s={w}x{h}:r=30:d={dur},rotate=a={angle:.3f}",
               "-f", "lavfi", "-i", f"color=c={color}:s={w}x{h}:r=30:d={dur}",
               "-f", "lavfi", "-i", f"sine=frequency={freq}:sample_rate=48000:duration={dur}",
               "-filter_complex",
               f"[0:v][1:v]blend=all_mode=overlay:all_opacity=0.6,{label}[v];"
               f"[2:a]volume=0.3,aformat=channel_layouts=stereo[a]",
               "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-preset", "ultrafast",
               "-crf", "30", "-c:a", "aac", "-shortest", str(out))
        return str(out)

    def download_full(self, url, out_base, max_height=720):
        raise PipelineError("demo mode cannot download reference videos")
