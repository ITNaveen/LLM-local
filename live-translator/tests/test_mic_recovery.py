"""Microphone selection and recovery from 'digital silence', with a simulated Mac."""
import sys
import threading
import time

import numpy as np
import pytest
from fake_ollama import FakeOllama
from fake_sounddevice import MAC_DEVICES, FakeSoundDevice

from livetranslator import audio_io
from livetranslator.asr import ScriptedASR
from livetranslator.config import SettingsStore
from livetranslator.pipeline import Pipeline
from livetranslator.storage import MeetingStore


@pytest.fixture()
def mac(monkeypatch):
    def make(behaviour):
        fake = FakeSoundDevice(MAC_DEVICES, default_index=1, behaviour=behaviour)
        monkeypatch.setitem(sys.modules, "sounddevice", fake)
        return fake
    return make


def wait_for(cond, timeout=20.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.05)
    return False


def test_system_default_microphone_is_used_not_the_first_device(mac):
    mac({})
    devs = audio_io.list_input_devices()
    assert [d["name"] for d in devs if d["default"]] == ["MacBook Pro Microphone"]
    assert {d["name"]: d["kind"] for d in devs} == {
        "iPhone Microphone": "phone", "MacBook Pro Microphone": "builtin",
        "Microsoft Teams Audio": "virtual", "External USB Mic": "other"}
    src = audio_io.MicSource("", lambda x, r: None)
    assert src.start() == "MacBook Pro Microphone"
    src.stop()


def test_probe_tells_silent_from_working(mac):
    mac({"MacBook Pro Microphone": "zeros"})
    devs = {d["name"]: d for d in audio_io.list_input_devices()}
    assert audio_io.probe_device(devs["MacBook Pro Microphone"], 0.5)["ok"] is False
    r = audio_io.probe_device(devs["External USB Mic"], 0.5)
    assert r["ok"] and r["peak_db"] < -20
    order = [d["name"] for d in audio_io.preferred_order(list(devs.values()))]
    assert order[0] == "MacBook Pro Microphone" and order[-1] == "Microsoft Teams Audio"


def _pipeline(tmp_path, fake_ollama):
    cfg = SettingsStore(tmp_path / "s.json")
    cfg.update({"ollama_url": fake_ollama.url, "input_source": "mic", "live_preview": False})
    events = []
    pipe = Pipeline(cfg, MeetingStore(tmp_path / "M"), events.append, asr_factory=lambda s: ScriptedASR())
    assert wait_for(lambda: pipe.asr_state["status"] == "ready")
    return pipe, cfg, events


def test_silent_default_mic_switches_to_a_working_one(mac, tmp_path):
    fake = mac({"MacBook Pro Microphone": "zeros", "iPhone Microphone": "zeros", "Microsoft Teams Audio": "zeros"})
    with FakeOllama() as fo:
        pipe, cfg, events = _pipeline(tmp_path, fo)
        pipe.start(name="t")
        assert pipe.source_name == "MacBook Pro Microphone"
        assert wait_for(lambda: pipe.source_name == "External USB Mic")
        assert cfg.settings.input_device == "External USB Mic"       # remembered for next time
        assert wait_for(lambda: any(e["type"] == "notice" and "Switched to 'External USB Mic'" in e["text"]
                                    for e in events))
        assert not any(e["type"] == "browser_audio_needed" for e in events)
        # the virtual Teams device was never preferred over real microphones
        assert fake.opened.index("External USB Mic") < (fake.opened.index("Microsoft Teams Audio")
                                                        if "Microsoft Teams Audio" in fake.opened else 99)
        pipe.stop(wait=True)
        pipe.shutdown()


def test_reopening_fixes_late_permission(mac, tmp_path):
    mac({"MacBook Pro Microphone": "zeros_first_open"})
    with FakeOllama() as fo:
        pipe, cfg, events = _pipeline(tmp_path, fo)
        pipe.start(name="t")
        assert wait_for(lambda: any(e["type"] == "notice" and "works now" in e["text"] for e in events))
        assert pipe.source_name == "MacBook Pro Microphone"
        assert cfg.settings.input_device == ""                       # unchanged
        pipe.stop(wait=True)
        pipe.shutdown()


def test_everything_blocked_falls_back_to_the_browser(mac, tmp_path):
    mac({n: "zeros" for n, _, _ in MAC_DEVICES})
    with FakeOllama() as fo:
        pipe, cfg, events = _pipeline(tmp_path, fo)
        pipe.start(name="t")
        assert wait_for(lambda: any(e["type"] == "browser_audio_needed" for e in events))
        assert wait_for(lambda: pipe.status()["source_kind"] == "browser")
        # browser audio now feeds the meeting
        lines_before = len([e for e in events if e["type"] == "line"])
        rng = np.random.default_rng(0)
        silence = (rng.standard_normal(48000) * 0.0005).astype(np.float32)
        assert pipe.source.active
        from livetranslator import simulate as sim
        if sim.tts_available():
            speech48 = np.repeat(sim.synthesize("Guten Morgen zusammen."), 3)
            for chunk in np.array_split(np.concatenate([silence, speech48, silence, silence]), 60):
                pipe.source.push_float32(chunk.astype("<f4").tobytes(), 48000)
            assert wait_for(lambda: len([e for e in events if e["type"] == "line"]) > lines_before)
        pipe.stop(wait=True)
        # next meeting goes straight to the browser - no second round of checks
        events.clear()
        pipe.start(name="t2")
        assert pipe.status()["source_kind"] == "browser"
        assert any(e["type"] == "browser_audio_needed" for e in events)
        pipe.stop(wait=True)
        pipe.shutdown()


def test_stop_during_recovery_is_clean(mac, tmp_path):
    mac({n: "zeros" for n, _, _ in MAC_DEVICES})
    with FakeOllama() as fo:
        pipe, cfg, events = _pipeline(tmp_path, fo)
        pipe.start(name="t")
        assert wait_for(lambda: any(e["type"] == "mic_check" for e in events))
        pipe.stop(wait=True)
        time.sleep(4)   # recovery thread notices the stop and must not reopen anything
        assert pipe.state == "idle" and pipe.source is None
        assert threading.active_count() < 40
        pipe.shutdown()


def test_mictest_cli(mac, capsys):
    mac({"MacBook Pro Microphone": "zeros"})
    from livetranslator.__main__ import main

    assert main(["mictest"]) == 0
    out = capsys.readouterr().out
    assert "MacBook Pro Microphone [system default]" in out and "SILENT" in out
    assert "External USB Mic" in out and "hears sound" in out


# ----------------------------------------------------------------- races found in review
def test_device_rescan_never_kills_an_open_microphone(mac):
    fake = mac({})
    src = audio_io.MicSource("", lambda x, r: None)
    src.start()
    audio_io.list_input_devices(refresh=True)      # e.g. Settings opened while listening
    assert fake.killed == [] and len(fake.live) == 1
    src.stop()
    audio_io.list_input_devices(refresh=True)      # nothing open: re-scan allowed
    assert fake.reinits >= 2


def test_changing_microphone_during_recovery_leaves_nothing_open(mac, tmp_path):
    # recovery would pick "External USB Mic"; the user picks the (working) Teams device meanwhile
    fake = mac({"MacBook Pro Microphone": "zeros", "iPhone Microphone": "zeros"})
    with FakeOllama() as fo:
        pipe, cfg, events = _pipeline(tmp_path, fo)
        pipe.start(name="t")
        assert wait_for(lambda: any(e["type"] == "mic_check" for e in events))
        old = cfg.settings
        pipe.apply_settings(old, cfg.update({"input_device": "Microsoft Teams Audio"}))
        time.sleep(6)                                             # recovery finishes its probes
        assert pipe.source_name == "Microsoft Teams Audio"       # the user's choice wins
        assert cfg.settings.input_device == "Microsoft Teams Audio"
        assert [st.name for st in fake.live] == ["Microsoft Teams Audio"]
        pipe.stop(wait=True)
        time.sleep(0.3)
        assert fake.live == [], [st.name for st in fake.live]     # no orphaned stream after Stop
        pipe.shutdown()


def test_stale_recovery_does_not_touch_the_next_meeting(mac, tmp_path, monkeypatch):
    import livetranslator.pipeline as pl

    fake = mac({"MacBook Pro Microphone": "zeros_first_open"})
    # first run: macOS shows its permission prompt; the user answers after a few seconds
    monkeypatch.setattr(pl, "mic_permission_status", lambda: "not_determined")
    monkeypatch.setattr(pl, "request_mic_permission", lambda timeout=45: (time.sleep(3), "authorized")[1])
    with FakeOllama() as fo:
        pipe, cfg, events = _pipeline(tmp_path, fo)
        pipe.start(name="a")
        assert wait_for(lambda: any(e["type"] == "mic_check" for e in events))
        pipe.stop(wait=True)                 # stop while meeting a's recovery waits for the prompt
        pipe.start(name="b")                 # meeting b opens the (now working) MacBook mic
        time.sleep(5)                        # a's recovery wakes up meanwhile
        assert pipe.state == "listening" and pipe.source_name == "MacBook Pro Microphone"
        assert fake.killed == [], fake.killed
        assert len(fake.live) == 1
        pipe.stop(wait=True)
        pipe.shutdown()


def test_blocked_memory_is_cleared_when_a_microphone_works_again(mac, tmp_path):
    mac({n: "zeros" for n, _, _ in MAC_DEVICES})
    with FakeOllama() as fo:
        pipe, cfg, events = _pipeline(tmp_path, fo)
        pipe.start(name="t")
        assert wait_for(lambda: pipe.status()["source_kind"] == "browser")
        pipe.stop(wait=True)
        assert pipe._mic_blocked
        old = cfg.settings
        pipe.apply_settings(old, cfg.update({"input_device": "External USB Mic"}))
        assert not pipe._mic_blocked
        pipe.shutdown()


def test_settings_change_racing_stop_opens_nothing(mac, tmp_path):
    fake = mac({})
    with FakeOllama() as fo:
        pipe, cfg, events = _pipeline(tmp_path, fo)
        pipe.start(name="t")
        pipe.stop(wait=True)
        old = cfg.settings
        pipe.apply_settings(old, cfg.update({"input_device": "External USB Mic"}))   # arrives just after Stop
        time.sleep(0.5)
        assert fake.live == [] and pipe.source is None
        pipe.shutdown()
