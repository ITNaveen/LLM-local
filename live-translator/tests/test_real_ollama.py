"""Integration test against a REAL Ollama server (skipped when none is running).

    LT_REAL_OLLAMA=http://127.0.0.1:11434 LT_REAL_MODEL=gemma3:270m pytest tests/test_real_ollama.py
"""
import os
import threading
import time

import httpx
import pytest

from livetranslator import simulate as sim
from livetranslator.asr import ScriptedASR
from livetranslator.audio_io import FileSource, write_wav
from livetranslator.config import SettingsStore
from livetranslator.pipeline import Pipeline
from livetranslator.storage import MeetingStore

URL = os.environ.get("LT_REAL_OLLAMA", "http://127.0.0.1:11434")
MODEL = os.environ.get("LT_REAL_MODEL", "gemma3:270m")


def _available() -> bool:
    try:
        tags = httpx.get(URL + "/api/tags", timeout=2, trust_env=False).json()
        return any(m["name"] == MODEL for m in tags.get("models", []))
    except Exception:  # noqa: BLE001
        return False


pytestmark = [pytest.mark.skipif(not _available(), reason=f"no real Ollama with {MODEL} at {URL}"),
              pytest.mark.skipif(sim.tts_available() is None, reason="needs TTS")]


def test_every_line_is_translated_by_real_ollama(tmp_path):
    n = 4
    clips = [sim.synthesize(de, i) for i, (de, _) in enumerate(sim.SENTENCES[:n])]
    audio, _ = sim.make_meeting(clips, sim.SCENARIOS["desk"])
    wav = tmp_path / "m.wav"
    write_wav(str(wav), audio)
    cfg = SettingsStore(tmp_path / "s.json")
    cfg.update({"ollama_url": URL, "llm_model": MODEL, "llm_fallback": "", "live_preview": False})
    events = []
    pipe = Pipeline(cfg, MeetingStore(tmp_path / "M"), events.append,
                    asr_factory=lambda s: ScriptedASR([de for de, _ in sim.SENTENCES[:n]] * 2))
    t0 = time.time()
    while pipe.asr_state["status"] != "ready" or not pipe.llm_state.get("model_ready"):
        assert time.time() - t0 < 30
        time.sleep(0.1)
    done = threading.Event()
    pipe.start(name="real", source=FileSource(str(wav), pipe.push_audio, speed=1.0, on_end=done.set))
    assert done.wait(120)
    pipe.stop(wait=True)
    finals = {e["id"]: e for e in events if e["type"] == "tr" and e.get("final")}
    lines = [e["line"] for e in events if e["type"] == "line"]
    assert len(lines) == n
    assert len(finals) == n, [e["text"] for e in events if e["type"] == "notice"]
    for f in finals.values():
        assert f["ok"] and f["en"].strip()
    # the first line may wait behind the start-up warm-up; after that the prompt cache must keep it quick
    later = [f["lat"]["tr_first_s"] for i, f in sorted(finals.items()) if i > 1]
    assert max(later) < 5, later
    # streamed: English arrived in pieces before the final event
    assert any(e["type"] == "tr" and not e.get("final") for e in events)
    st = pipe.status()["llm"]
    assert st["loaded"] is True and st["tok_s"]
    # saved to disk with the English
    m = pipe.store.list()[0]
    md = (pipe.store.load(m["id"]).folder / "transcript.md").read_text()
    assert md.count("EN: ") == n and "(translation unavailable)" not in md
    pipe.shutdown()
