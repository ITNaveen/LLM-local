"""Settings, persisted to data/settings.json and editable from the web UI."""

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("STORYMAKER_DATA", ROOT / "data"))
JOBS_DIR = DATA_DIR / "jobs"
STYLES_DIR = DATA_DIR / "styles"
CACHE_DIR = DATA_DIR / "cache"
SETTINGS_FILE = DATA_DIR / "settings.json"
FONTS_DIR = ROOT / "assets" / "fonts"
FONT_FILE = FONTS_DIR / "NotoSansDevanagari-Bold.ttf"
FONT_NAME = "Noto Sans Devanagari"

DEFAULTS = {
    # Local AI (Ollama)
    "ollama_url": "http://127.0.0.1:11434",
    "llm_model": "qwen3.5:9b",
    "llm_think": False,           # thinking mode: smarter but much slower
    "llm_timeout": 600,
    "embed_models": ["bge-m3", "nomic-embed-text"],  # first installed one wins
    # Research
    "max_candidates": 500,        # videos looked at (metadata only)
    "results_per_query": 50,
    "shortlist_size": 30,         # videos whose transcripts + replay graphs are read
    "min_video_seconds": 45,      # skip Shorts
    "max_video_seconds": 3 * 3600,
    "cookies_from_browser": "",   # e.g. "chrome" or "safari" if YouTube asks you to sign in
    "force_ipv4": True,           # fixes very slow YouTube on many home networks
    "search_workers": 4,          # searches / video reads running at the same time
    "research_minutes": 6,        # stop searching after this long and go with what we have
    # Voice
    "tts_engine": "auto",         # auto | parler | edge | piper | say | silent
    "parler_speaker": "Rohit",    # emotional voice: Rohit / Aman (male), Divya / Rani (female)
    "parler_style": "",           # optional custom description of how the voice should sound
    "edge_voice": "hi-IN-MadhurNeural",
    "edge_rate": "+10%",          # energetic, not a slow news reader
    "edge_pitch": "+0Hz",
    "say_voice": "Lekha",
    "piper_model": "",            # path to a Hindi .onnx voice for fully offline TTS
    # Music
    "music_dir": str(ROOT / "music"),
    "builtin_music": True,        # built-in news-thriller score when the music folder is empty
    # Render
    "width": 1920,
    "height": 1080,
    "fps": 30,
    "crf": 20,
    "preset": "veryfast",
    "burn_subtitles": True,
    "title_card": True,
    "target_lufs": -14.0,         # YouTube loudness
    # Server
    "port": 7777,
}


# Earlier defaults that turned out wrong; saved copies of them are upgraded automatically.
_OLD_DEFAULTS = {"edge_rate": ("-6%", "+8%"), "edge_pitch": ("-4Hz",)}


def load_settings():
    settings = dict(DEFAULTS)
    if SETTINGS_FILE.exists():
        try:
            saved = json.loads(SETTINGS_FILE.read_text())
            settings.update({k: v for k, v in saved.items()
                             if k in DEFAULTS and v not in _OLD_DEFAULTS.get(k, ())})
        except (OSError, ValueError):
            pass
    return settings


def save_settings(updates):
    settings = load_settings()
    for key, value in updates.items():
        if key not in DEFAULTS:
            continue
        default = DEFAULTS[key]
        if isinstance(default, bool):
            value = value in (True, "true", "1", 1, "on")
        elif isinstance(default, int):
            value = int(value)
        elif isinstance(default, float):
            value = float(value)
        elif isinstance(default, list) and isinstance(value, str):
            value = [v.strip() for v in value.split(",") if v.strip()]
        settings[key] = value
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SETTINGS_FILE.write_text(json.dumps(settings, indent=2, ensure_ascii=False))
    return settings


def ensure_dirs():
    for d in (DATA_DIR, JOBS_DIR, STYLES_DIR, CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)
