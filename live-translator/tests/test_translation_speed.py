"""The translator must never leave lines on 'translating…' - whatever Ollama does.

Reproduces the situations that make a 24 GB Mac's Ollama slow or stuck (big model
too slow next to Whisper, model stuck loading, model not fitting into GPU memory)
and checks that every line still gets its English, by switching to the fast model.
"""
import threading
import time

import numpy as np
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


# ----------------------------------------------------------------- 1.1.1: 4B default, no long loads
def test_old_settings_move_from_12b_to_4b(tmp_path):
    import json

    from livetranslator.config import SettingsStore

    p = tmp_path / "settings.json"
    p.write_text(json.dumps({"llm_model": "gemma3:12b", "context_lines": 12, "pause_ms": 650,
                             "input_device": "MacBook Pro Microphone"}))
    s = SettingsStore(p).settings
    assert s.llm_model == "gemma3:4b" and s.context_lines == 8
    assert s.pause_ms == 650 and s.input_device == "MacBook Pro Microphone"   # the rest is kept
    assert json.loads(p.read_text())["settings_version"] == 2                  # persisted
    # a user who picks 12B again afterwards keeps it
    SettingsStore(p).update({"llm_model": "gemma3:12b"})
    assert SettingsStore(p).settings.llm_model == "gemma3:12b"


def test_fresh_install_defaults_to_the_fast_model(tmp_path):
    from livetranslator.config import SettingsStore

    assert SettingsStore(tmp_path / "none.json").settings.llm_model == "gemma3:4b"


def test_slow_loading_big_model_is_abandoned_quickly(tmp_path, meeting_wav):
    """12B chosen, but it takes ages to load (Mac short of memory): switch after WAIT_SWITCH_S."""
    beh = {"gemma3:12b": {"load_s": 6.0, "hang": True}}
    with FakeOllama(models=["gemma3:12b", "gemma3:4b"], token_delay=0.005, behaviour=beh) as fo:
        t0 = time.time()

        def tune(p):   # real clocks: switch after 20 s, wait up to 150 s for a model that is loading
            p.WAIT_SWITCH_S, p.WAIT_LOADING_S, p.RETRY_EVERY_S = 1.0, 10.0, 1.0

        pipe, events, finals = run_meeting(tmp_path, fo, meeting_wav, before_start=tune)
        assert pipe.translator.model == "gemma3:4b"
        assert all(f["ok"] and f["en"] for f in finals.values())
        assert any("not enough free memory" in t for t in notices(events))
        assert time.time() - t0 < 40
        pipe.shutdown()


def test_other_models_are_unloaded_to_make_room(tmp_path, meeting_wav):
    with FakeOllama(models=["gemma3:4b", "qwen2.5:14b", "nomic-embed-text"], token_delay=0.005) as fo:
        fo.loaded["qwen2.5:14b"] = time.monotonic()        # left in memory by another app
        fo.loaded["nomic-embed-text"] = time.monotonic()
        pipe, events, finals = run_meeting(tmp_path, fo, meeting_wav,
                                           settings={"llm_model": "gemma3:4b", "llm_fallback": "gemma3:4b"})
        assert "qwen2.5:14b" not in fo.loaded
        assert "nomic-embed-text" in fo.loaded               # embedding models are tiny: kept
        assert all(f["ok"] for f in finals.values())
        pipe.shutdown()


def test_failed_lines_are_retried_automatically(tmp_path, meeting_wav):
    """Ollama fails for a while (e.g. busy loading), then works: every line still gets its English."""
    with FakeOllama(models=["gemma3:4b"], token_delay=0.005) as fo:
        fo.fail = True
        threading.Timer(6.0, lambda: setattr(fo, "fail", False)).start()

        def tune(p):
            p.RETRY_EVERY_S = 1.0

        cfg_extra = {"llm_model": "gemma3:4b", "llm_fallback": "gemma3:4b"}
        pipe, events, finals = run_meeting(tmp_path, fo, meeting_wav, settings=cfg_extra, before_start=tune)
        pipe.shutdown()


def test_missing_translation_model_is_downloaded_automatically(tmp_path):
    from livetranslator.config import SettingsStore as SS

    with FakeOllama(models=[], token_delay=0.005) as fo:
        cfg = SS(tmp_path / "s.json")
        cfg.update({"ollama_url": fo.url, "llm_model": "gemma3:4b"})
        events = []
        pipe = Pipeline(cfg, MeetingStore(tmp_path / "M"), events.append, asr_factory=lambda s: ScriptedASR())
        assert wait_for(lambda: "gemma3:4b" in fo.models, 20)
        assert wait_for(lambda: pipe.llm_state.get("model_ready"), 20)
        assert any("Downloading the translation model gemma3:4b" in t for t in notices(events))
        pipe.shutdown()


# ----------------------------------------------------------------- review findings (regressions)
def test_settings_change_keeps_the_fast_model(tmp_path, meeting_wav):
    """A+/A- or any other Settings save must not put the slow model back."""
    beh = {"gemma3:12b": {"first_s": 0.8}}
    with FakeOllama(models=["gemma3:12b", "gemma3:4b"], token_delay=0.005, behaviour=beh) as fo:
        pipe, events, finals = run_meeting(tmp_path, fo, meeting_wav)
        assert pipe.translator.model == "gemma3:4b"
        old = pipe.cfg.settings
        pipe.apply_settings(old, pipe.cfg.update({"font_scale": 1.3}))
        assert pipe.translator.model == "gemma3:4b"
        assert pipe.status()["llm"]["fallback_active"] is True
        old = pipe.cfg.settings      # a real model change by the user does take effect
        pipe.apply_settings(old, pipe.cfg.update({"llm_model": "qwen2.5:14b", "font_scale": 1.0}))
        assert pipe.translator.model == "qwen2.5:14b"
        assert pipe.status()["llm"]["fallback_active"] is False
        pipe.shutdown()


def test_far_too_slow_writing_is_not_saved_as_a_translation(tmp_path):
    from livetranslator.translate import OllamaTranslator

    beh = {"gemma3:12b": {"token_s": 0.6}}
    with FakeOllama(models=["gemma3:12b"], behaviour=beh) as fo:
        t = OllamaTranslator(fo.url, "gemma3:12b")
        r = t.translate(sim.SENTENCES[1][0], max_gen_s=1.0)
        assert not r.ok and r.truncated
        assert t.history == []          # never used as context


def test_reload_for_another_context_size_is_not_called_loaded(tmp_path):
    with FakeOllama(models=["gemma3:4b"]) as fo:
        from livetranslator.config import SettingsStore as SS

        cfg = SS(tmp_path / "s.json")
        cfg.update({"ollama_url": fo.url, "llm_model": "gemma3:4b"})
        pipe = Pipeline(cfg, MeetingStore(tmp_path / "M"), lambda e: None, asr_factory=lambda s: ScriptedASR())
        fo.loaded["gemma3:4b"] = time.monotonic()
        fo.loaded_ctx["gemma3:4b"] = 8192      # e.g. another app loaded it with a bigger context
        assert pipe._model_loaded("gemma3:4b") is False
        fo.loaded_ctx["gemma3:4b"] = 4096
        assert pipe._model_loaded("gemma3:4b") is True
        pipe.shutdown()


def test_summary_uses_the_same_context_size(tmp_path):
    from livetranslator.translate import NUM_CTX, OllamaTranslator

    with FakeOllama(models=["gemma3:4b"]) as fo:
        t = OllamaTranslator(fo.url, "gemma3:4b")
        t.summarize("[10:00] DE: Hallo\n           EN: Hello\n" * 900)   # long meeting: condensed in rounds
        ctxs = {r["body"]["options"]["num_ctx"] for r in fo.chats()}
        assert ctxs == {NUM_CTX}
        assert all(len(r["body"]["messages"][-1]["content"]) < 12000 for r in fo.chats())


def test_failed_lines_are_not_translated_twice(tmp_path, meeting_wav):
    with FakeOllama(models=["gemma3:4b"], token_delay=0.005) as fo:
        fo.fail = True
        threading.Timer(4.0, lambda: setattr(fo, "fail", False)).start()

        def tune(p):
            p.RETRY_EVERY_S = 0.5

        pipe, events, finals = run_meeting(tmp_path, fo, meeting_wav,
                                           settings={"llm_model": "gemma3:4b", "llm_fallback": "gemma3:4b"},
                                           before_start=tune)
        time.sleep(2)
        ok_chats = [r for r in fo.chats() if r["t"] > 0]
        per_line = {}
        for r in ok_chats:
            per_line[r["body"]["messages"][-1]["content"]] = per_line.get(r["body"]["messages"][-1]["content"], 0) + 1
        # after Ollama recovered, each line was translated successfully exactly once more at most
        m = pipe.store.load(pipe.store.list()[0]["id"])
        assert all(ln.ok and ln.en for ln in m.lines)
        oks = [e for e in events if e["type"] == "tr" and e.get("final") and e.get("ok")]
        assert len(oks) == len({e["id"] for e in oks}), "a line was translated successfully twice"
        pipe.shutdown()


def test_english_preview_while_a_long_sentence_is_spoken(tmp_path):
    """Long sentence: grey English appears before the speaker pauses, then the real line replaces it."""
    from livetranslator.asr import EchoASR

    long_de = sim.SENTENCES[5][0]
    clip = np.concatenate([sim.synthesize(sim.SENTENCES[1][0], 0), sim.synthesize(sim.SENTENCES[5][0], 0)])
    audio, _ = sim.make_meeting([clip], sim.SCENARIOS["clean"])
    wav = tmp_path / "long.wav"
    write_wav(str(wav), audio)
    with FakeOllama(models=["gemma3:4b"], token_delay=0.005) as fo:
        cfg = SettingsStore(tmp_path / "s.json")
        cfg.update({"ollama_url": fo.url, "llm_model": "gemma3:4b", "live_preview": True, "max_line_s": 20})
        events = []
        pipe = Pipeline(cfg, MeetingStore(tmp_path / "M"), lambda e: events.append({**e, "_t": time.monotonic()}),
                        asr_factory=lambda s: EchoASR(long_de, delay=0.1))
        assert wait_for(lambda: pipe.asr_state["status"] == "ready" and pipe.llm_state.get("loaded"), 30)
        done = threading.Event()
        pipe.start(name="t", source=FileSource(str(wav), pipe.push_audio, speed=1.0, on_end=done.set))
        assert done.wait(60)
        pipe.stop(wait=True)
        prev = [e for e in events if e["type"] == "partial_en"]
        lines = [e for e in events if e["type"] == "line"]
        assert prev, "no English preview during the long sentence"
        assert prev[0]["text"] == sim.SENTENCES[5][1]
        assert prev[0]["_t"] < lines[0]["_t"], "the preview must come before the speaker pauses"
        finals = [e for e in events if e["type"] == "tr" and e.get("final")]
        assert finals and finals[0]["ok"]
        # previews are never kept as conversation context
        assert all(d != long_de or e for d, e in pipe.translator.history)
        assert len(pipe.translator.history) <= len(finals)
        pipe.shutdown()
