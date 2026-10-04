"""Editing style: how long shots are, how the pace moves through the five acts.
A style can be learned from a reference video you love (scene cuts + loudness)."""

import re
import statistics

from .config import STYLES_DIR
from .util import ffmpeg, media_duration, read_json, run, write_json

ACT_KEYS = ["opening", "buildup", "rising", "climax", "ending"]

DEFAULT_STYLE = {
    "name": "cinematic",
    "source": "built-in",
    # share of total runtime per act, and typical shot length (seconds) per act
    "acts": {
        "opening": {"share": 0.10, "shot": 3.2},
        "buildup": {"share": 0.30, "shot": 5.5},
        "rising":  {"share": 0.22, "shot": 3.0},
        "climax":  {"share": 0.25, "shot": 2.4},
        "ending":  {"share": 0.13, "shot": 4.5},
    },
    "dialogue_ratio": 0.45,   # share of runtime where the clip's own voice carries the story
    "hard_cuts": True,
}

THEMES = {
    "auto": "Pick the theme that fits the topic best.",
    "epic": "Epic tribute: pride, power, goosebumps. Big music, heroic narration.",
    "emotional": "Emotional journey: struggle, sacrifice, tears and triumph.",
    "documentary": "Documentary: factual, gripping, explains what happened and why it matters.",
    "thriller": "Thriller: tension, mystery, stakes, a dramatic reveal.",
}

THEME_MOODS = {
    "epic":        ["dark", "emotional", "tense", "epic", "triumphant"],
    "emotional":   ["calm", "emotional", "tense", "emotional", "triumphant"],
    "documentary": ["calm", "calm", "tense", "epic", "calm"],
    "thriller":    ["dark", "tense", "tense", "epic", "dark"],
}


def list_styles():
    styles = [DEFAULT_STYLE]
    if STYLES_DIR.exists():
        for p in sorted(STYLES_DIR.glob("*.json")):
            s = read_json(p)
            if s and "acts" in s:
                styles.append(s)
    return styles


def get_style(name):
    for s in list_styles():
        if s["name"] == name:
            return s
    return DEFAULT_STYLE


# ------------------------------------------------------------ learning a style
def detect_cuts(path, threshold=0.32):
    """Scene-change timestamps using ffmpeg's scene score on a downscaled copy."""
    proc = run(["ffmpeg", "-hide_banner", "-nostdin", "-i", path, "-an",
                "-vf", f"scale=320:-2,select='gt(scene,{threshold})',showinfo",
                "-f", "null", "-"], check=True)
    times = [float(t) for t in re.findall(r"pts_time:([\d.]+)", proc.stderr)]
    cleaned = []
    for t in times:  # ignore flashes: cuts closer than 0.35s
        if not cleaned or t - cleaned[-1] >= 0.35:
            cleaned.append(t)
    return cleaned


def loudness_curve(path):
    """Momentary loudness (LUFS) per second (empty if the video has no sound)."""
    from .util import probe
    if not any(st["codec_type"] == "audio" for st in probe(path)["streams"]):
        return []
    proc = run(["ffmpeg", "-hide_banner", "-nostdin", "-i", path, "-vn",
                "-af", "ebur128=metadata=1,ametadata=print:key=lavfi.r128.M",
                "-f", "null", "-"], check=True)
    vals = {}
    for t, m in re.findall(r"pts_time:([\d.]+)[^\n]*\n[^\n]*lavfi\.r128\.M=(-?[\d.inf]+)",
                           proc.stderr):
        try:
            v = float(m)
        except ValueError:
            continue
        vals.setdefault(int(float(t)), []).append(max(v, -70.0))
    return [round(sum(v) / len(v), 1) for _, v in sorted(vals.items())]


def analyze_reference(path, name, captions=None, title=""):
    duration = media_duration(path)
    cuts = detect_cuts(path)
    bounds = [0.0] + cuts + [duration]
    shots = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)
             if bounds[i + 1] - bounds[i] > 0.05]
    loud = loudness_curve(path)

    # Map the reference's timeline onto our five acts using the default shares.
    acts, t = {}, 0.0
    for key in ACT_KEYS:
        share = DEFAULT_STYLE["acts"][key]["share"]
        lo, hi = t, t + share * duration
        lens = [min(e, hi) - max(s, lo) for s, e in shots if s < hi and e > lo]
        lens = [x for x in lens if x > 0.3]
        shot = statistics.median(lens) if lens else DEFAULT_STYLE["acts"][key]["shot"]
        acts[key] = {"share": share, "shot": round(min(max(shot, 1.2), 12.0), 2)}
        t = hi

    speech = sum(c["end"] - c["start"] for c in captions or []) if captions else None
    style = {
        "name": name,
        "source": title or str(path),
        "acts": acts,
        "dialogue_ratio": round(min(0.8, speech / duration), 2) if speech else 0.45,
        "hard_cuts": True,
        "stats": {
            "duration": round(duration, 1),
            "shots": len(shots),
            "avg_shot": round(duration / max(1, len(shots)), 2),
            "loudness_per_act": _per_act_mean(loud),
        },
    }
    write_json(STYLES_DIR / f"{re.sub(r'[^a-zA-Z0-9_-]+', '-', name)}.json", style)
    return style


def _per_act_mean(curve):
    if not curve:
        return []
    out, n, t = [], len(curve), 0.0
    for key in ACT_KEYS:
        share = DEFAULT_STYLE["acts"][key]["share"]
        seg = curve[int(t * n):max(int(t * n) + 1, int((t + share) * n))]
        out.append(round(sum(seg) / len(seg), 1) if seg else None)
        t += share
    return out


def make_test_video(path, cut_times, duration):
    """Synthetic video with hard cuts at known times (used by tests)."""
    colors = ["red", "blue", "green", "yellow", "purple", "orange", "white", "cyan"]
    bounds = [0.0] + list(cut_times) + [duration]
    inputs, filters = [], []
    for i in range(len(bounds) - 1):
        d = bounds[i + 1] - bounds[i]
        inputs += ["-f", "lavfi", "-i", f"color=c={colors[i % len(colors)]}:s=320x180:r=25:d={d}"]
        filters.append(f"[{i}:v]")
    ffmpeg(*inputs, "-filter_complex",
           "".join(filters) + f"concat=n={len(filters)}:v=1:a=0[v]", "-map", "[v]",
           "-c:v", "libx264", "-preset", "ultrafast", str(path))
