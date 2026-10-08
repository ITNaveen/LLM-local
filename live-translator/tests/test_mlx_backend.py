"""Exercises the Apple-Silicon speech path (mlx-whisper) end to end.

The real model can't be downloaded in CI, so this builds a tiny Whisper with
random weights but the exact large-v3 layout (128 mel bins, 51866-token
multilingual vocabulary) and runs our MLXWhisper backend on it - from a
worker thread, as the app does. The text is gibberish; what is tested is that
every argument we pass and every field we read matches mlx-whisper's API.
"""
import json
import threading

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_whisper")


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    from mlx.utils import tree_flatten
    from mlx_whisper import whisper

    dims = dict(n_mels=128, n_audio_ctx=1500, n_audio_state=64, n_audio_head=2, n_audio_layer=1,
                n_vocab=51866, n_text_ctx=448, n_text_state=64, n_text_head=2, n_text_layer=1)
    model = whisper.Whisper(whisper.ModelDimensions(**dims), mx.float32)
    d = tmp_path_factory.mktemp("tiny-whisper")
    weights = {k: v.astype(mx.float16) for k, v in tree_flatten(model.parameters())}  # like the real repos
    mx.save_safetensors(str(d / "weights.safetensors"), weights)
    (d / "config.json").write_text(json.dumps({"model_type": "whisper", **dims}))
    return str(d)


def test_mlx_backend_runs_in_worker_thread(tiny_model):
    from livetranslator.asr import MLXWhisper, build_prompt

    out = {}

    def worker():
        try:
            asr = MLXWhisper(tiny_model, language="de")
            asr.load()
            out["warm"] = asr.warmup()
            audio = (np.random.default_rng(0).standard_normal(16000 * 3) * 0.05).astype(np.float32)
            r = asr.transcribe(audio, build_prompt(["Müller", "SAP"], "Das ist der vorige Satz."))
            r2 = asr.transcribe(audio, None)
            out["r"], out["r2"] = r, r2
        except Exception as e:  # noqa: BLE001
            out["err"] = e

    t = threading.Thread(target=worker)
    t.start()
    t.join(300)
    assert "err" not in out, repr(out.get("err"))
    for r in (out["r"], out["r2"]):
        assert isinstance(r.text, str)
        assert r.elapsed > 0
        assert np.isfinite(r.avg_logprob) and 0.0 <= r.no_speech_prob <= 1.0


def test_pipeline_with_mlx_backend(tiny_model, tmp_path):
    """Full pipeline (front end + MLX ASR + translator) with the tiny model and a fake Ollama."""
    from fake_ollama import FakeOllama

    from livetranslator import simulate as sim
    from livetranslator.asr import MLXWhisper
    from livetranslator.audio_io import FileSource, write_wav
    from livetranslator.config import SettingsStore
    from livetranslator.pipeline import Pipeline
    from livetranslator.storage import MeetingStore

    if sim.tts_available() is None:
        pytest.skip("needs TTS")
    with FakeOllama(token_delay=0) as f:
        cfg = SettingsStore(tmp_path / "s.json")
        cfg.update({"ollama_url": f.url, "live_preview": True})
        events = []
        pipe = Pipeline(cfg, MeetingStore(tmp_path / "M"), events.append,
                        asr_factory=lambda s: MLXWhisper(tiny_model, "de"))
        for _ in range(600):
            if pipe.asr_state["status"] != "loading":
                break
            threading.Event().wait(0.1)
        assert pipe.asr_state["status"] == "ready", pipe.asr_state
        clips = [sim.synthesize(de, 0) for de, _ in sim.SENTENCES[:2]]
        audio, _ = sim.make_meeting(clips, sim.SCENARIOS["desk"])
        wav = tmp_path / "a.wav"
        write_wav(str(wav), audio)
        done = threading.Event()
        pipe.start(source=FileSource(str(wav), pipe.push_audio, speed=0, on_end=done.set))
        assert done.wait(60)
        pipe.stop(wait=True)
        assert pipe.state == "idle"
        assert not [e for e in events if e.get("type") == "notice" and e.get("level") == "error"]
        pipe.shutdown()
