"""The thermal governor: optional live work is reduced when the Mac is warm, on battery, short of
memory or getting slower - and restored when it has cooled down. Real lines are never affected."""
import math
import threading
import time

import pytest
from fake_ollama import FakeOllama

from livetranslator import simulate as sim
from livetranslator.governor import Governor
from livetranslator.macsignals import MacSignals


class Signals:
    def __init__(self, fanless=True):
        self.pressure = 0
        self.state = 0
        self.lpm = False
        self.mem = 1
        self.power = ("ac", 90)
        self._fanless = fanless

    def thermal_pressure(self):
        return self.pressure

    def thermal_state(self):
        return self.state

    def low_power_mode(self):
        return self.lpm

    def memory_pressure(self):
        return self.mem

    def power_source(self):
        return self.power

    def is_fanless(self):
        return self._fanless


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def gov(fanless=True, performance="auto"):
    s, c = Signals(fanless), Clock()
    return Governor(s, performance, clock=c), s, c


def test_cool_mac_runs_everything():
    g, s, c = gov()
    assert not g.tick() and g.plan().name == "full" and g.preview_slowdown() == 1.0


def test_warm_mac_eases_off_at_once_and_back_only_after_cooling_down():
    g, s, c = gov(fanless=True)
    s.pressure = 1
    assert g.tick() and g.plan().name == "eco" and "warm" in g.reason
    s.pressure = 2
    g.tick()
    assert g.plan().name == "cool" and g.plan().en_preview_min_s is None
    s.pressure = 0
    c.t += 60
    g.tick()
    assert g.plan().name == "cool"                      # an Air keeps cool for a while
    c.t += 130
    g.tick()
    assert g.plan().name == "eco"                       # then one step at a time
    c.t += 200
    g.tick()
    assert g.plan().name == "full"


def test_critical_state_keeps_only_the_real_lines():
    g, s, c = gov()
    s.state = 3
    g.tick()
    p = g.plan()
    assert p.name == "minimal" and math.isinf(g.preview_slowdown()) and not p.early and not p.recheck


def test_battery_low_power_and_memory():
    g, s, c = gov(fanless=True)
    s.power = ("battery", 70)
    g.tick()
    assert g.plan().name == "eco" and "battery" in g.reason
    g, s, c = gov(fanless=True)
    s.power = ("battery", 15)
    g.tick()
    assert g.plan().name == "cool"
    g, s, c = gov(fanless=False)               # a MacBook Pro keeps its speed on battery
    s.power = ("battery", 70)
    g.tick()
    assert g.plan().name == "full"
    s.power = ("battery", 20)                  # ... until the battery runs low
    c.t += 61
    g.tick()
    assert g.plan().name == "eco"
    g, s, c = gov(fanless=False)
    s.lpm = True
    g.tick()
    assert g.plan().name == "eco" and "Low Power" in g.reason
    g, s, c = gov(fanless=False)
    s.mem = 4
    g.tick()
    assert g.plan().name == "cool"


def test_max_performance_ignores_battery_but_not_heat():
    g, s, c = gov(fanless=True, performance="max")
    s.power = ("battery", 15)
    g.tick()
    assert g.plan().name == "full"
    s.state = 2
    g.tick()
    assert g.plan().name == "cool"


def test_saver_always_runs_lighter():
    g, s, c = gov(fanless=False, performance="saver")
    g.tick()
    assert g.plan().name == "eco"


def test_getting_slower_is_noticed_before_macos_says_anything():
    g, s, c = gov(fanless=False)
    for _ in range(10):
        g.note_llm(40.0)                     # first minutes: 40 tok/s
    g.tick()
    assert g.plan().name == "full"
    for _ in range(5):
        g.note_llm(22.0)                     # now 45 % slower (throttling)
    for _ in range(2):
        g.tick()
    assert g.plan().name == "full"           # needs 3 bad samples in a row (no flapping)
    g.tick()
    assert g.plan().name == "cool" and "slower" in g.reason
    for _ in range(10):
        g.tick()
    assert g.plan().name == "cool"           # slowness alone never goes to "minimal"
    g.reset_baselines()                      # e.g. the user picked a bigger translation model
    c.t += 400
    for _ in range(3):
        g.tick()
    assert g.plan().name == "eco"            # eases off again (one step per cool-down)


def test_a_new_meeting_forgets_the_last_meetings_slow_speeds():
    g, s, c = gov(fanless=False)
    for _ in range(10):
        g.note_llm(40.0)
    for _ in range(5):
        g.note_llm(22.0)                     # the end of a long meeting: hot and slow
    for _ in range(3):
        g.tick()
    assert g.plan().name == "cool"
    c.t += 3600                              # an hour later the Mac is cold
    g.reset()                                # new meeting
    for _ in range(6):
        g.tick()                             # nothing translated yet
    assert g.plan().name == "full" and g.speed_ratio() == 1.0


def test_after_the_heat_has_gone_it_eases_off_even_if_a_little_slower():
    g, s, c = gov(fanless=True)
    for _ in range(10):
        g.note_llm(40.0)                     # cold Mac, full boost
    s.pressure = 1
    g.tick()
    assert g.plan().name == "eco"
    for _ in range(5):
        g.note_llm(34.0)                     # heat-soaked: 85 % of the boost speed, for the rest of the meeting
    s.pressure = 0
    for _ in range(40):                      # macOS says nominal again
        c.t += 10
        g.tick()
    assert g.plan().name == "full"


def test_slowness_is_noticed_when_the_meeting_starts_on_battery():
    g, s, c = gov(fanless=True)
    s.power = ("battery", 70)                # an Air on battery: eco from the first second
    g.tick()
    assert g.plan().name == "eco"
    for _ in range(10):
        g.note_llm(40.0)
    for _ in range(5):
        g.note_llm(22.0)                     # throttling that macOS does not report (yet)
    for _ in range(3):
        g.tick()
    assert g.plan().name == "cool" and "slower" in g.reason


def test_slowness_holds_its_level_until_the_speed_recovers():
    g, s, c = gov(fanless=False)
    for _ in range(10):
        g.note_llm(40.0)
    for _ in range(5):
        g.note_llm(30.0)                     # 25 % slower
    for _ in range(3):
        g.tick()
    assert g.plan().name == "eco"
    for _ in range(5):
        g.note_llm(34.0)                     # 85 %: better, but not recovered
    c.t += 600
    for _ in range(5):
        g.tick()
    assert g.plan().name == "eco"
    for _ in range(5):
        g.note_llm(39.0)                     # recovered
    c.t += 100
    g.tick()
    assert g.plan().name == "full"


def test_low_power_mode_speeds_are_not_the_baseline():
    g, s, c = gov(fanless=False)
    s.lpm = True                             # the meeting starts in Low Power Mode: slow by itself
    g.tick()
    for _ in range(10):
        g.note_llm(26.0)
    s.lpm = False                            # plugged in, Low Power Mode off: full speed again
    g.tick()
    for _ in range(10):
        g.note_llm(40.0)
    for _ in range(5):
        g.note_llm(24.0)                     # then real throttling
    for _ in range(3):
        g.tick()
    assert g.plan().name == "cool" and "slower" in g.reason


def test_battery_saver_never_does_more_preview_work_than_before():
    g, s, c = gov(fanless=False)
    for _ in range(10):
        g.note_llm(40.0)
    for _ in range(5):
        g.note_llm(22.0)
    for _ in range(3):
        g.tick()
    assert g.plan().name == "cool"
    g.set_performance("saver")               # the user picks "Battery saver" while in Cool mode
    assert g.plan().name == "cool"           # (not "eco" for a minute until the slowness is measured again)
    g.set_performance("max")
    assert g.plan().name == "full"


def test_max_performance_ignores_slowness():
    g, s, c = gov(fanless=False, performance="max")
    for _ in range(10):
        g.note_llm(40.0)
    for _ in range(5):
        g.note_llm(15.0)
    for _ in range(5):
        g.tick()
    assert g.plan().name == "full"


def test_fanless_mac_keeps_to_a_gpu_budget():
    g, s, c = gov(fanless=True)
    for _ in range(12):
        g.note_work(7.0)                     # 84 s of GPU work in the last 2 minutes: 70 % > 50 %
    for _ in range(5):
        g.tick()
    assert g.plan().name == "full" and 2.0 < g.preview_slowdown() <= 4.0
    c.t += 200                               # quiet now
    for _ in range(30):
        g.tick()
    assert g.preview_slowdown() == 1.0
    g2, s2, c2 = gov(fanless=False)          # a MacBook Pro has a fan: no budget
    for _ in range(12):
        g2.note_work(7.0)
    g2.tick()
    assert g2.preview_slowdown() == 1.0


def test_macsignals_never_raise_and_can_be_faked(monkeypatch):
    m = MacSignals()
    for f in (m.thermal_pressure, m.thermal_state, m.low_power_mode, m.memory_pressure, m.power_source):
        f()                                   # None on Linux, real values on a Mac
    monkeypatch.setenv("LT_FAKE_THERMAL", "2")
    monkeypatch.setenv("LT_FAKE_POWER", "battery:20")
    monkeypatch.setenv("LT_FAKE_FANLESS", "1")
    assert m.thermal_pressure() == 2 and m.power_source() == ("battery", 20) and m.is_fanless()


# ----------------------------------------------------------------- in the pipeline
def test_hot_mac_still_gets_every_line(tmp_path):
    if sim.tts_available() is None:
        pytest.skip("needs TTS")
    from test_preview_engine import Engine, whisper_text

    from livetranslator.audio_io import FileSource, write_wav
    from livetranslator.config import SettingsStore
    from livetranslator.pipeline import Pipeline
    from livetranslator.storage import MeetingStore

    clips = [sim.synthesize(de, i) for i, (de, _) in enumerate(sim.SENTENCES[:3])]
    audio, _ = sim.make_meeting(clips, sim.SCENARIOS["clean"], pauses_s=[1.3, 1.3, 1.3])
    wav = tmp_path / "m.wav"
    write_wav(str(wav), audio)
    whisper = Engine("mlx", whisper_text, 0.2)
    with FakeOllama(models=["gemma3:4b"], token_delay=0.005) as fo:
        cfg = SettingsStore(tmp_path / "s.json")
        cfg.update({"ollama_url": fo.url, "llm_model": "gemma3:4b", "llm_fallback": "gemma3:4b", "live_preview": True})
        events = []
        pipe = Pipeline(cfg, MeetingStore(tmp_path / "M"), events.append, asr_factory=lambda s: whisper,
                        preview_factory=lambda s: None)
        sig = Signals()
        sig.state = 3                                     # critical
        pipe.gov.signals = sig
        end = time.time() + 20
        while not (pipe.asr_state["status"] == "ready" and pipe.llm_state.get("loaded")) and time.time() < end:
            time.sleep(0.05)
        done = threading.Event()
        pipe.start(name="t", source=FileSource(str(wav), pipe.push_audio, speed=1.0, on_end=done.set))
        pipe.gov.tick()
        pipe._apply_plan()
        assert done.wait(60)
        pipe.stop(wait=True)
        st = pipe.status()
        pipe.shutdown()
    assert st["governor"]["mode"] == "minimal"
    assert not [e for e in events if e["type"] == "partial" and e["text"]]       # no previews at all
    lines = [e for e in events if e["type"] == "line"]
    finals = [e for e in events if e["type"] == "tr" and e.get("final") and e.get("ok")]
    assert len(lines) == 3 and len(finals) == 3                                   # every line, translated
    assert len(whisper.calls) == 3                                                # one pass per line, nothing else


def test_changing_the_performance_setting_counts_at_once(tmp_path):
    from livetranslator.asr import ScriptedASR
    from livetranslator.audio_io import PushSource
    from livetranslator.config import SettingsStore
    from livetranslator.pipeline import Pipeline
    from livetranslator.storage import MeetingStore

    with FakeOllama(models=["gemma3:4b"], token_delay=0.005) as fo:
        cfg = SettingsStore(tmp_path / "s.json")
        cfg.update({"ollama_url": fo.url, "live_preview": False})
        events = []
        pipe = Pipeline(cfg, MeetingStore(tmp_path / "M"), events.append, asr_factory=lambda s: ScriptedASR())
        pipe.gov.signals = Signals(fanless=False)
        pipe.start(name="t", source=PushSource(pipe.push_audio))
        old = cfg.settings
        pipe.apply_settings(old, cfg.update({"performance": "saver"}))
        assert pipe.gov.plan().name == "eco" and pipe.preview_slowdown > 1.0           # no 5 s wait
        assert any(e.get("governor", {}).get("mode") == "eco" for e in events if e["type"] == "status")
        old = cfg.settings
        pipe.apply_settings(old, cfg.update({"performance": "auto"}))
        assert pipe.gov.plan().name == "full" and pipe.preview_slowdown == 1.0         # no 90 s cool-down
        pipe.stop(wait=True)
        pipe.shutdown()
