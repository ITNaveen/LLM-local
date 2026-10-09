"""Small shared helpers: ffmpeg wrappers, JSON io, Hindi-safe text handling."""

import json
import re
import shutil
import subprocess
from pathlib import Path


class PipelineError(RuntimeError):
    pass


def run(cmd, timeout=None, check=True):
    """Run a command, return CompletedProcess. Raises PipelineError with stderr tail."""
    proc = subprocess.run(
        [str(c) for c in cmd], capture_output=True, text=True, timeout=timeout
    )
    if check and proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-1500:]
        raise PipelineError(f"{Path(str(cmd[0])).name} failed: {tail}")
    return proc


def ffmpeg(*args, timeout=None):
    return run(["ffmpeg", "-hide_banner", "-nostdin", "-y", "-loglevel", "error", *args],
               timeout=timeout)


def probe(path):
    out = run(["ffprobe", "-v", "error", "-print_format", "json",
               "-show_format", "-show_streams", path]).stdout
    return json.loads(out)


def media_duration(path):
    info = probe(path)
    try:
        return float(info["format"]["duration"])
    except (KeyError, ValueError):
        durations = [float(s["duration"]) for s in info.get("streams", []) if "duration" in s]
        return max(durations) if durations else 0.0


def has_video(path):
    """True if the file exists and contains a picture (not just sound)."""
    try:
        return any(st.get("codec_type") == "video" and st.get("codec_name") not in ("mjpeg", "png")
                   for st in probe(path).get("streams", []))
    except (PipelineError, ValueError, OSError):
        return False


def has_tool(name):
    return shutil.which(name) is not None


_FILTERS = None


def ffmpeg_has_filter(name):
    """Homebrew's slim 'ffmpeg' can lack libass ('ass' filter); 'ffmpeg-full' has it."""
    global _FILTERS
    if _FILTERS is None:
        try:
            out = run(["ffmpeg", "-hide_banner", "-filters"], check=False).stdout
            _FILTERS = {line.split()[1] for line in out.splitlines()
                        if len(line.split()) > 2 and line.startswith(" ")}
        except OSError:
            _FILTERS = set()
    return name in _FILTERS


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    tmp.replace(path)


# Python's \w misses Devanagari vowel signs (matras), so split on separators instead.
_TOKEN_RE = re.compile(r"[^\s.,!?;:\"'()\[\]{}|/\\\-–—#@&*+=<>~`^%$₹।॥…“”‘’]+")

STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "and", "or", "to", "for", "is", "are", "was",
    "with", "by", "at", "from", "his", "her", "its", "this", "that", "it", "as", "be",
    "full", "video", "videos", "new", "latest", "news", "hindi", "live", "today",
    "का", "की", "के", "है", "में", "और", "को", "से", "पर", "ने", "भी", "हैं", "था", "थी",
    "ये", "यह", "वो", "वह", "एक", "तो", "ही", "कि", "जो", "हुआ", "हुई", "कर",
    # Hinglish written in English letters
    "ki", "ka", "ke", "hai", "hain", "aur", "bhi", "ab", "kal", "aaj", "ye", "yeh", "wo", "woh",
    "se", "me", "mein", "ko", "ne", "par", "hi", "nahi", "kya", "hogi", "hoga", "honge", "tha",
    "thi", "hua", "hui", "kar", "ek", "jo", "ho", "rahe", "raha", "rahi",
}


def tokens(text):
    return [t for t in _TOKEN_RE.findall((text or "").lower()) if len(t) > 1]


def keywords(text):
    return [t for t in tokens(text) if t not in STOPWORDS]


def slugify(text, max_len=40):
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text or "").strip("-").lower()
    return (slug[:max_len].strip("-") or "video")


def fmt_ts(seconds):
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def srt_ts(seconds):
    ms = int(round(seconds * 1000))
    h, rem = divmod(ms, 3600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def hindi_word_count(text):
    return len((text or "").split())


def estimate_speech_seconds(text, words_per_second=2.3):
    """Rough Hindi TTS speaking time; used for planning before audio exists."""
    return max(1.0, hindi_word_count(text) / words_per_second + 0.4)
