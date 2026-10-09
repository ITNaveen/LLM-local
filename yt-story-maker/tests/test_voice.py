"""The emotional voice runs as a separate program; here a stand-in program plays its part."""

import json
import stat
import sys

from storymaker import config, voice
from storymaker.util import media_duration


def _fake_parler(tmp_path, works=True):
    """A stand-in for .venv-voice/bin/python + voice_parler.py: writes a tone per line."""
    script = tmp_path / "fake_python"
    body = f"""#!{sys.executable}
import json, subprocess, sys
job = json.load(open(sys.argv[2]))
print("Loading the Hindi voice model on cpu...", flush=True)
if not {works!r}:
    print("RuntimeError: model is gated", flush=True); sys.exit(1)
for i, (nid, text) in enumerate(job["lines"].items(), 1):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    "sine=frequency=300:duration=2", "-ac", "1",
                    job["out_dir"] + "/" + nid + ".raw.wav"], check=True)
    print(f"PROGRESS {{i}}/{{len(job['lines'])}} {{nid}}", flush=True)
assert "Divya" in job["description"]
"""
    script.write_text(body)
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return script


def test_emotional_voice_renders_all_lines_in_one_run(tmp_path, monkeypatch):
    monkeypatch.setattr(voice, "PARLER_PY", _fake_parler(tmp_path))
    logs = []
    s = dict(config.DEFAULTS, parler_speaker="Divya")
    out = voice.synthesize({"n01": "सोचिए ज़रा!", "n02": "ये है असली खेल।"}, tmp_path / "v", s, logs.append)
    assert set(out) == {"n01", "n02"} and all(r["engine"] == "parler" for r in out.values())
    assert all(1.5 < r["duration"] < 2.5 for r in out.values())
    assert any("voice line 2/2" in line for line in logs)


def test_emotional_voice_failure_falls_back(tmp_path, monkeypatch):
    monkeypatch.setattr(voice, "PARLER_PY", _fake_parler(tmp_path, works=False))
    logs = []
    s = dict(config.DEFAULTS, parler_speaker="Divya")
    out = voice.synthesize({"n01": "नमस्ते दोस्तों"}, tmp_path / "v", s, logs.append)
    assert out["n01"]["engine"] != "parler" and media_duration(out["n01"]["file"]) > 0
    assert any("emotional voice failed" in line and "gated" in line for line in logs)


def test_old_slow_voice_settings_are_upgraded(tmp_path, monkeypatch):
    f = tmp_path / "settings.json"
    f.write_text(json.dumps({"edge_rate": "-6%", "edge_pitch": "-4Hz", "crf": 23}))
    monkeypatch.setattr(config, "SETTINGS_FILE", f)
    s = config.load_settings()
    assert s["edge_rate"] == "+8%" and s["edge_pitch"] == "+0Hz" and s["crf"] == 23


def test_sentence_splitting_for_the_voice_model():
    sys.path.insert(0, str(voice.ROOT / "storymaker"))
    import voice_parler
    parts = voice_parler.split_sentences("पहला वाक्य है। " + " ".join(["शब्द"] * 30) + "! आख़िर क्यों?")
    assert parts[0] == "पहला वाक्य है।" and parts[-1] == "आख़िर क्यों?"
    assert all(len(p.split()) <= 22 for p in parts)
