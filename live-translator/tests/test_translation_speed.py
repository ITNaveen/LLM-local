"""The translator must never leave lines on 'translating…' - whatever Ollama does.

Reproduces the situations that make a 24 GB Mac's Ollama slow or stuck (big model
too slow next to Whisper, model stuck loading, model not fitting into GPU memory)
and checks that every line still gets its English, by switching to the fast model.
"""
import threading
import time

import pytest
from fake_ollama import FakeOllama

from livetranslator import simulate as sim
from livetranslator.asr import ScriptedASR
from livetranslator.audio_io import FileSource, write_wav
from livetranslator.config import SettingsStore
from livetranslator.pipeline import Pipeline
from livetranslator.storage import MeetingStore

pytestmark = pytest.mark.skipif(sim.tts_available() is None, reason="needs TTS")
N = 4


@pytest.fixture(scope="module")
def meeting_wav(tmp_path_factory):
    clips = [sim.synthesize(de, i) for i, (de, _) in enumerate(sim.SENTENCES[:N])]
    audio, _ = sim.make_meeting(clips, sim.SCENARIOS["desk"])
    p = tmp_path_factory.mktemp("w") / "m.wav"
    write_wav(str(p), audio)
    return str(p)


def wait_for(cond, timeout=30.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.05)
    return False


def run_meeting(tmp_path, fake, wav, settings=None, fast_limits=True, before_start=None):
    cfg = SettingsStore(tmp_path / "s.json")
    cfg.update({"ollama_url": fake.url, "llm_model": "gemma3:12b", "llm_fallback": "gemma3:4b",
                "live_preview": False, **(settings or {})})
    events = []
    pipe = Pipeline(cfg, MeetingStore(tmp_path / "M"), events.append,
                    asr_factory=lambda s: ScriptedASR([de for de, _ in sim.SENTENCES[:N]] * 3))
    if fast_limits:   # same logic, shorter clocks so the test runs in seconds
        pipe.WAIT_LOADED_S, pipe.WAIT_LOADING_S, pipe.SLOW_FIRST_S = 1.5, 3.0, 0.4
    assert wait_for(lambda: pipe.asr_state["status"] == "ready")
    if before_start:
        before_start(pipe)
    done = threading.Event()
    pipe.start(name="t", source=FileSource(wav, pipe.push_audio, speed=0, on_end=done.set))
    assert done.wait(60)
    pipe.stop(wait=True)
    finals = {}
    assert wait_for(lambda: len({e["id"]: e for e in events if e["type"] == "tr" and e.get("final")
                                 and e.get("ok")}) >= N, 60), [e for e in events if e["type"] == "notice"]
    for e in events:
        if e["type"] == "tr" and e.get("final"):
            finals[e["id"]] = e
    return pipe, events, finals


def notices(events):
    return [e["text"] for e in events if e["type"] == "notice"]


def test_healthy_model_is_kept(tmp_path, meeting_wav):
    with FakeOllama(models=["gemma3:12b", "gemma3:4b"], token_delay=0.005) as fo:
        pipe, events, finals = run_meeting(tmp_path, fo, meeting_wav)
        assert all(f["ok"] and f["en"] for f in finals.values())
        assert pipe.translator.model == "gemma3:12b"
        assert not fo.chats("gemma3:4b")
        assert [e["en"] for e in sorted(finals.values(), key=lambda e: e["id"])] == [en for _, en in sim.SENTENCES[:N]]
        pipe.shutdown()


def test_slow_model_switches_to_fast_one(tmp_path, meeting_wav):
    beh = {"gemma3:12b": {"first_s": 0.8}}   # English starts 0.8 s late, limit in this test: 0.4 s
    with FakeOllama(models=["gemma3:12b", "gemma3:4b"], token_delay=0.005, behaviour=beh) as fo:
        pipe, events, finals = run_meeting(tmp_path, fo, meeting_wav)
        assert pipe.translator.model == "gemma3:4b"
        assert any("Switched to the faster model gemma3:4b" in t for t in notices(events))
        assert fo.chats("gemma3:4b"), "later lines must be translated by the fast model"
        assert all(f["ok"] and f["en"] for f in finals.values())
        # the slow model was unloaded to give the Mac its memory back
        assert any(r["path"] == "/api/generate" and r["body"].get("keep_alive") == 0
                   and r["body"]["model"] == "gemma3:12b" for r in fo.requests)
        assert pipe.status()["llm"]["fallback_active"] is True
        pipe.shutdown()


def test_model_that_never_answers_does_not_block_lines(tmp_path, meeting_wav):
    beh = {"gemma3:12b": {"hang": True}}
    with FakeOllama(models=["gemma3:12b", "gemma3:4b"], token_delay=0.005, behaviour=beh) as fo:
        t0 = time.time()
        pipe, events, finals = run_meeting(tmp_path, fo, meeting_wav)
        assert all(f["ok"] and f["en"] for f in finals.values())
        assert pipe.translator.model == "gemma3:4b"
        assert time.time() - t0 < 45
        pipe.shutdown()


def test_missing_fast_model_is_downloaded_then_used(tmp_path, meeting_wav):
    beh = {"gemma3:12b": {"first_s": 0.8}}
    with FakeOllama(models=["gemma3:12b"], token_delay=0.005, behaviour=beh) as fo:
        pipe, events, finals = run_meeting(tmp_path, fo, meeting_wav)
        assert any(r["path"] == "/api/pull" and r["body"]["model"] == "gemma3:4b" for r in fo.requests)
        assert wait_for(lambda: pipe.translator.model == "gemma3:4b", 20)
        assert any("Downloading the faster model" in t for t in notices(events))
        pipe.shutdown()


def test_model_not_fitting_gpu_memory_on_mac_falls_back(tmp_path, meeting_wav, monkeypatch):
    import livetranslator.pipeline as pl

    monkeypatch.setattr(pl.platform, "system", lambda: "Darwin")
    beh = {"gemma3:12b": {"gpu_share": 0.55}}
    with FakeOllama(models=["gemma3:12b", "gemma3:4b"], token_delay=0.005, behaviour=beh) as fo:
        pipe, events, finals = run_meeting(tmp_path, fo, meeting_wav)
        assert wait_for(lambda: pipe.translator.model == "gemma3:4b", 20)
        assert any("does not fit in the Mac's GPU memory" in t for t in notices(events))
        pipe.shutdown()


def test_no_switch_when_disabled_but_user_is_told(tmp_path, meeting_wav):
    beh = {"gemma3:12b": {"first_s": 0.8}}
    with FakeOllama(models=["gemma3:12b", "gemma3:4b"], token_delay=0.005, behaviour=beh) as fo:
        pipe, events, finals = run_meeting(tmp_path, fo, meeting_wav, settings={"auto_fallback": False})
        assert pipe.translator.model == "gemma3:12b"
        assert any("Translation is slow" in t for t in notices(events))
        assert all(f["ok"] for f in finals.values())
        pipe.shutdown()


def test_cold_load_is_waited_for(tmp_path, meeting_wav):
    """A model that takes a while to load (first start) is not mistaken for a broken one."""
    beh = {"gemma3:12b": {"load_s": 2.0}}
    with FakeOllama(models=["gemma3:12b", "gemma3:4b"], token_delay=0.005, behaviour=beh) as fo:
        pipe, events, finals = run_meeting(tmp_path, fo, meeting_wav)
        assert pipe.translator.model == "gemma3:12b"
        assert all(f["ok"] and f["en"] for f in finals.values())
        pipe.shutdown()


def test_translator_status_reports_speed(tmp_path, meeting_wav):
    with FakeOllama(models=["gemma3:12b", "gemma3:4b"], token_delay=0.01) as fo:
        pipe, events, finals = run_meeting(tmp_path, fo, meeting_wav)
        llm = pipe.status()["llm"]
        assert llm["tok_s"] and llm["tok_s"] > 10 and llm["loaded"] is True and llm["queue"] == 0
        assert any(f.get("lat", {}).get("tok_s") for f in finals.values())
        pipe.shutdown()
