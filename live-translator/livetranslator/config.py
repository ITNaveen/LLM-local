"""Settings, paths and the model catalogue.

Everything the user can change in the Settings panel lives in `Settings` and is
persisted to ~/LiveTranslator/settings.json (override the folder with LT_HOME).
"""
from __future__ import annotations

import json
import os
import platform
import threading
from dataclasses import asdict, dataclass, fields
from pathlib import Path

APP_NAME = "Live Translator"
SAMPLE_RATE = 16000  # everything after the input stage runs at 16 kHz mono


def home_dir() -> Path:
    p = Path(os.environ.get("LT_HOME") or (Path.home() / "LiveTranslator"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def meetings_dir() -> Path:
    p = home_dir() / "Meetings"
    p.mkdir(parents=True, exist_ok=True)
    return p


def logs_dir() -> Path:
    p = home_dir() / "logs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


# Speech models. "mlx" runs on the Apple GPU (M-series); "faster" is the
# CTranslate2 build used on Windows/Linux/Intel Macs.
ASR_MODELS = {
    "turbo": {
        "label": "Fast + accurate (Whisper large-v3-turbo)",
        "mlx": "mlx-community/whisper-large-v3-turbo",
        "faster": "large-v3-turbo",
    },
    "large-v3": {
        "label": "Maximum accuracy, slower (Whisper large-v3)",
        "mlx": "mlx-community/whisper-large-v3-mlx",
        "faster": "large-v3",
    },
}

# Translation models offered in Settings (any installed Ollama model also works).
RECOMMENDED_LLMS = [
    {"name": "gemma3:4b", "label": "Gemma 3 4B - fast and reliable (recommended)"},
    {"name": "gemma3:12b", "label": "Gemma 3 12B - higher quality, needs ~10 GB free memory"},
    {"name": "qwen2.5:14b", "label": "Qwen 2.5 14B - alternative, needs ~11 GB free memory"},
]
SETTINGS_VERSION = 2

SENSITIVITY = {
    # speech-probability thresholds for the voice detector
    "low": 0.6,      # noisy room, only clear speech
    "normal": 0.5,
    "high": 0.35,    # quiet / far-away speaker
}


@dataclass
class Settings:
    # input
    input_source: str = "mic"            # "mic" or "browser"
    input_device: str = ""               # sounddevice name; "" = system default
    sensitivity: str = "normal"          # key of SENSITIVITY
    pause_ms: int = 500                  # silence that ends a line
    max_line_s: float = 14.0             # long monologues are split around here
    live_preview: bool = True            # grey German text while someone is speaking
    # speech recognition
    asr_backend: str = "auto"            # auto | mlx | faster
    asr_model: str = "turbo"             # key of ASR_MODELS, or an HF repo / local path
    language: str = "de"
    # translation
    ollama_url: str = "http://127.0.0.1:11434"
    llm_model: str = "gemma3:4b"         # fast enough on any M-series Mac, even next to Whisper
    llm_fallback: str = "gemma3:4b"      # used automatically when llm_model is too slow on this machine
    auto_fallback: bool = True
    free_ollama_memory: bool = True      # unload other chat models from Ollama while a meeting runs
    context_lines: int = 8               # previous lines the translator sees (at least; up to +10)
    # help for both models
    glossary: str = ""                   # names, products, abbreviations (comma or newline)
    topic: str = ""                      # optional: what the meeting is about
    # display (also stored server-side so every browser shows the same)
    settings_version: int = SETTINGS_VERSION
    font_scale: float = 1.0
    show_german: bool = True
    theme: str = "auto"

    @classmethod
    def from_dict(cls, d: dict) -> "Settings":
        d = migrate(dict(d or {}))
        known = {f.name for f in fields(cls)}
        s = cls(**{k: v for k, v in d.items() if k in known})
        s.validate()
        return s

    def validate(self) -> None:
        if self.input_source not in ("mic", "browser"):
            self.input_source = "mic"
        if self.sensitivity not in SENSITIVITY:
            self.sensitivity = "normal"
        self.pause_ms = int(min(max(int(self.pause_ms), 200), 2000))
        self.max_line_s = float(min(max(float(self.max_line_s), 6.0), 28.0))
        self.context_lines = int(min(max(int(self.context_lines), 0), 40))
        self.font_scale = float(min(max(float(self.font_scale), 0.6), 2.5))
        if self.asr_backend not in ("auto", "mlx", "faster"):
            self.asr_backend = "auto"
        if self.theme not in ("auto", "dark", "light"):
            self.theme = "auto"

    def to_dict(self) -> dict:
        return asdict(self)

    def glossary_terms(self) -> list[str]:
        raw = self.glossary.replace(";", "\n").replace(",", "\n")
        seen, out = set(), []
        for t in (x.strip() for x in raw.splitlines()):
            if t and t.lower() not in seen:
                seen.add(t.lower())
                out.append(t)
        return out

    def resolved_asr_backend(self) -> str:
        if self.asr_backend != "auto":
            return self.asr_backend
        return "mlx" if is_apple_silicon() else "faster"

    def resolved_asr_model(self) -> str:
        backend = self.resolved_asr_backend()
        entry = ASR_MODELS.get(self.asr_model)
        if entry:
            return entry[backend]
        return self.asr_model  # custom repo id or local folder


def migrate(d: dict) -> dict:
    """Bring settings saved by an older version up to date."""
    v = int(d.get("settings_version") or 1)
    if v < 2:
        # 1.x defaulted to Gemma 3 12B, which on a 24 GB Mac next to Whisper can take a minute
        # to load or not fit at all: the translation never arrived. 4B is the default now.
        if d.get("llm_model") in (None, "gemma3:12b"):
            d["llm_model"] = "gemma3:4b"
        if int(d.get("context_lines") or 8) > 8:
            d["context_lines"] = 8
        d["settings_version"] = 2
    return d


class SettingsStore:
    def __init__(self, path: Path | None = None):
        self.path = path or (home_dir() / "settings.json")
        self._lock = threading.Lock()
        self.settings = self._load()

    def _load(self) -> Settings:
        try:
            raw = json.loads(self.path.read_text("utf-8"))
        except Exception:
            return Settings()
        s = Settings.from_dict(raw)
        if int(raw.get("settings_version") or 1) < SETTINGS_VERSION:
            try:   # persist the migration
                self.path.write_text(json.dumps(s.to_dict(), indent=2, ensure_ascii=False), "utf-8")
            except Exception:  # noqa: BLE001
                pass
        return s

    def update(self, changes: dict) -> Settings:
        with self._lock:
            d = self.settings.to_dict()
            d.update({k: v for k, v in changes.items() if k in d})
            self.settings = Settings.from_dict(d)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.settings.to_dict(), indent=2, ensure_ascii=False), "utf-8")
            os.replace(tmp, self.path)
            return self.settings
