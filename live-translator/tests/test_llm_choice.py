"""Translation models: compared on the user's own meeting (nothing changes automatically), and a
model that is still downloading never costs a meeting its English."""
import json
import time

from fake_ollama import FakeOllama

from livetranslator import llmcheck
from livetranslator.asr import ScriptedASR
from livetranslator.audio_io import PushSource
from livetranslator.config import SettingsStore
from livetranslator.pipeline import Pipeline
from livetranslator.storage import Line, MeetingStore


def wait_for(cond, timeout=30.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.05)
    return False


def _meeting(home):
    store = MeetingStore(home / "Meetings")
    m = store.create("Weekly sync")
    for i, de in enumerate(["Guten Morgen zusammen, lassen Sie uns kurz über den aktuellen Stand des Projekts sprechen.",
                            "Das geht so nicht.", "Das Budget beträgt 25.000 Euro."], 1):
        store.add_line(m, Line(id=i, time=f"10:00:0{i}", offset=float(i), de=de))
        store.set_translation(m, i, "x", True)
    store.finish(m)
    return store, m


def test_compare_on_the_last_meeting_side_by_side(tmp_path, monkeypatch, capsys):
    from livetranslator.__main__ import main

    monkeypatch.setenv("LT_HOME", str(tmp_path / "home"))
    store, m = _meeting(tmp_path / "home")
    beh = {"gemma4:e4b": {"token_s": 0.05}}
    answers = {"Das geht so nicht.": "That works like this."}        # gemma3 and gemma4 both lose the negation
    with FakeOllama(models=["gemma3:4b", "gemma4:e4b"], token_delay=0.005, behaviour=beh, answers=answers) as fo:
        SettingsStore().update({"ollama_url": fo.url})
        rc = main(["compare-llm", "--models", "gemma3:4b,gemma4:e4b,translategemma:4b"])
        out = capsys.readouterr().out
    assert rc == 0
    assert "translategemma:4b: not installed" in out
    md = (m.folder / "model-comparison.md").read_text()
    assert "| gemma3:4b |" in md and "| gemma4:e4b |" in md
    assert "DE: Das Budget beträgt 25.000 Euro." in md and "gemma4:e4b: [EN] Das Budget" in md
    assert SettingsStore().settings.llm_model == "gemma3:4b"           # nothing changed automatically


def test_compare_counts_lost_negations():
    store_lines = [Line(id=1, time="10:00:01", offset=0.0, de="Das geht so nicht.")]

    class M:
        lines = store_lines
        title = "t"
    with FakeOllama(models=["gemma3:4b"], token_delay=0.005,
                    answers={"Das geht so nicht.": "That works like this."}) as fo:
        r = llmcheck.compare(M, ["gemma3:4b"], fo.url, progress=lambda m: None)["results"]["gemma3:4b"]
    assert r["ok"] and r["checks"] == 1 and r["first_s"] is not None


def test_fresh_and_old_settings_keep_gemma3(tmp_path):
    assert SettingsStore(tmp_path / "new.json").settings.llm_model == "gemma3:4b"
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"settings_version": 2, "llm_model": "gemma3:4b", "asr_model": "turbo"}))
    s = SettingsStore(p).settings
    assert s.llm_model == "gemma3:4b" and s.asr_model == "turbo-de"


def _pipeline(tmp_path, fo, model):
    cfg = SettingsStore(tmp_path / "s.json")
    cfg.update({"ollama_url": fo.url, "llm_model": model, "llm_fallback": "gemma3:4b", "live_preview": False})
    events = []
    pipe = Pipeline(cfg, MeetingStore(tmp_path / "M"), events.append, asr_factory=lambda s: ScriptedASR())
    return pipe, cfg, events


def test_chosen_model_downloading_meanwhile_the_fallback_translates_then_switches(tmp_path):
    with FakeOllama(models=["gemma3:4b"], token_delay=0.005) as fo:
        pipe, cfg, events = _pipeline(tmp_path, fo, "gemma4:e4b")      # picked in Settings, not downloaded yet
        assert wait_for(lambda: pipe.translator.model == "gemma3:4b")  # no failing lines meanwhile
        assert any(e["type"] == "notice" and "downloads in the background" in e["text"] for e in events)
        pipe.start(name="t", source=PushSource(pipe.push_audio))
        sess = pipe.session
        ln = Line(id=1, time="10:00:00", offset=0.0, de="Wir müssen das Angebot bis Freitag fertig haben.")
        pipe.store.add_line(sess.meeting, ln)
        pipe._tr_q.put(("line", sess, ln, True, None))
        assert wait_for(lambda: ln.done and ln.ok)
        assert any(c["body"]["messages"][-1]["content"] == ln.de for c in fo.chats("gemma3:4b"))
        assert wait_for(lambda: "gemma4:e4b" in fo.models)                # downloaded
        time.sleep(0.5)
        assert pipe.translator.model == "gemma3:4b"                      # never during a meeting
        pipe.stop(wait=True)
        assert wait_for(lambda: pipe.translator.model == "gemma4:e4b", 30)
        assert pipe.llm_state["fallback_active"] is False
        assert cfg.settings.llm_model == "gemma4:e4b"
        pipe.shutdown()
