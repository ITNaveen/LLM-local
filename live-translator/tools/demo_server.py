"""Run the full app with stand-ins for the two models (for UI tests and screenshots).

    python tools/demo_server.py --port 8799 --home /tmp/lt-demo

Speech recognition returns the self-test sentences in order (one per detected
line); translation comes from a fake Ollama. Everything else - audio input,
voice detection, line splitting, saving, the web UI - is the real code.
"""
import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8799)
    ap.add_argument("--home", default="")
    ap.add_argument("--source", default="browser")
    ap.add_argument("--token-delay", type=float, default=0.06)
    ap.add_argument("--fake-mics", default="", help="simulate Mac inputs: all-silent | default-silent | ok")
    ap.add_argument("--slow-llm", action="store_true", help="gemma3:12b answers late (simulated busy Mac)")
    a = ap.parse_args()
    home = a.home or tempfile.mkdtemp(prefix="lt-demo-")
    os.environ["LT_HOME"] = home
    if a.fake_mics:
        from fake_sounddevice import MAC_DEVICES, FakeSoundDevice

        behaviour = {"all-silent": {n: "zeros" for n, _, _ in MAC_DEVICES},
                     "default-silent": {"MacBook Pro Microphone": "zeros"}, "ok": {}}[a.fake_mics]
        sys.modules["sounddevice"] = FakeSoundDevice(MAC_DEVICES, default_index=1, behaviour=behaviour)
    import uvicorn
    from fake_ollama import FakeOllama

    from livetranslator.asr import ScriptedASR
    from livetranslator.config import SettingsStore
    from livetranslator.pipeline import Pipeline
    from livetranslator.server import create_app
    from livetranslator.simulate import SENTENCES
    from livetranslator.storage import MeetingStore

    behaviour = {"gemma3:12b": {"first_s": 4.0}} if a.slow_llm else {}
    fake = FakeOllama(models=["gemma3:12b", "gemma3:4b"], token_delay=a.token_delay, behaviour=behaviour).start()
    settings = SettingsStore(Path(home) / "settings.json")
    settings.update({"ollama_url": fake.url, "input_source": a.source, "live_preview": False,
                     "llm_model": "gemma3:12b" if a.slow_llm else "gemma3:4b", "llm_fallback": "gemma3:4b"})
    texts = [de for de, _ in SENTENCES] * 50

    def factory(st, store, publish):
        p = Pipeline(st, store, publish, asr_factory=lambda s: ScriptedASR(texts, delay=0.25))
        if a.slow_llm:   # same logic, shorter clock than on a real Mac (8 s)
            p.SLOW_FIRST_S = 1.5
        return p

    app = create_app(pipeline_factory=factory, store=MeetingStore(Path(home) / "Meetings"), settings=settings,
                     start_ollama=False)
    print(json.dumps({"home": home, "port": a.port, "ollama": fake.url}), flush=True)
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
