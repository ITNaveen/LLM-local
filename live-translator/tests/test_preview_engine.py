"""The grey live preview runs on a cheap engine (Parakeet on Apple Silicon); Whisper writes the
saved lines. A preview engine that is missing, fails or hears another language changes nothing
for the meeting itself."""
import json
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from fake_ollama import FakeOllama

from livetranslator import simulate as sim
from livetranslator.asr import ASRResult, BaseASR, looks_german
from livetranslator.audio_io import FileSource, write_wav
from livetranslator.config import SettingsStore
from livetranslator.pipeline import Pipeline
from livetranslator.storage import MeetingStore


def wait_for(cond, timeout=30.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.05)
    return False


class Engine(BaseASR):
    """Test double: fixed text after `delay` seconds; records every call."""

    def __init__(self, name, text, delay, fail_prefetch=False):
        self.name, self.text, self.delay, self.fail_prefetch = name, text, delay, fail_prefetch
        self.calls = []

    def prefetch(self):
        if self.fail_prefetch:
            raise RuntimeError("download blocked")

    def warmup(self):
        return 0.0

    def transcribe(self, audio, prompt=None):
        time.sleep(self.delay)
        self.calls.append(len(audio) / 16000)
        text = self.text(len(audio) / 16000, prompt) if callable(self.text) else self.text
        return ASRResult(text, -0.2, 0.01, 1.0, self.delay)


@pytest.fixture(scope="module")
def meeting(tmp_path_factory):
    if sim.tts_available() is None:
        pytest.skip("needs TTS")
    clips = [sim.synthesize(de, i) for i, (de, _) in enumerate(sim.SENTENCES[:3])]
    audio, _ = sim.make_meeting(clips, sim.SCENARIOS["clean"], pauses_s=[1.3, 1.3, 1.3])
    p = tmp_path_factory.mktemp("w") / "m.wav"
    write_wav(str(p), audio)
    return str(p)


_seen: dict = {}


def whisper_text(secs, prompt=None):
    """The same audio always gives the same sentence; new audio the next sentence (never a repeat
    of the previous line, which would trigger the 'echo of the prompt' check)."""
    from test_fast_lines import sentence_for

    key = round(secs, 3)
    if key not in _seen:
        _seen[key] = len(_seen)
    return sentence_for(_seen[key], prompt)


def run(tmp_path, wav, whisper, preview):
    with FakeOllama(models=["gemma3:4b"], token_delay=0.005) as fo:
        cfg = SettingsStore(tmp_path / "s.json")
        cfg.update({"ollama_url": fo.url, "llm_model": "gemma3:4b", "llm_fallback": "gemma3:4b", "live_preview": True})
        events = []
        pipe = Pipeline(cfg, MeetingStore(tmp_path / "M"), events.append, asr_factory=lambda s: whisper,
                        preview_factory=lambda s: preview)
        assert wait_for(lambda: pipe.asr_state["status"] == "ready" and pipe.llm_state.get("loaded"))
        if preview is not None and not preview.fail_prefetch:
            assert wait_for(lambda: pipe.preview_asr is preview, 10)
        time.sleep(0.3)
        done = threading.Event()
        pipe.start(name="t", source=FileSource(wav, pipe.push_audio, speed=1.0, on_end=done.set))
        assert done.wait(60)
        pipe.stop(wait=True)
        pipe.shutdown()
        return events


def test_cheap_engine_does_the_previews_and_whisper_the_lines(tmp_path, meeting):
    whisper = Engine("mlx", whisper_text, 0.35)
    preview = Engine("parakeet", "Wir sprechen gerade über das Angebot", 0.02)
    events = run(tmp_path, meeting, whisper, preview)
    lines = [e["line"]["de"] for e in events if e["type"] == "line"]
    partials = [e["text"] for e in events if e["type"] == "partial" and e["text"]]
    assert len(lines) == 3 and all(x in [de for de, _ in sim.SENTENCES] for x in lines)
    assert partials and set(partials) == {"Wir sprechen gerade über das Angebot"}
    assert len(preview.calls) >= 10                       # refreshed often: it is cheap
    # Whisper only wrote lines (once each, during the pause; plus a few pauses that were not line ends)
    assert len(whisper.calls) <= 3 + 4, whisper.calls


def test_a_preview_in_another_language_is_not_shown(tmp_path, meeting):
    whisper = Engine("mlx", whisper_text, 0.3)
    preview = Engine("parakeet", "We are talking about the offer today", 0.02)
    events = run(tmp_path, meeting, whisper, preview)
    partials = [e["text"] for e in events if e["type"] == "partial" and e["text"]]
    assert "We are talking about the offer today" not in partials
    assert partials                                       # Whisper took over the preview for those lines
    assert len([e for e in events if e["type"] == "line"]) == 3


def test_no_preview_engine_means_whisper_previews(tmp_path, meeting):
    whisper = Engine("mlx", whisper_text, 0.3)
    preview = Engine("parakeet", "Wir sprechen gerade über das Angebot", 0.02, fail_prefetch=True)
    events = run(tmp_path, meeting, whisper, preview)
    partials = [e["text"] for e in events if e["type"] == "partial" and e["text"]]
    assert partials and "Wir sprechen gerade über das Angebot" not in partials
    assert not preview.calls
    assert len([e for e in events if e["type"] == "line"]) == 3


def test_looks_german():
    assert looks_german("Wir müssen das Angebot bis Freitag fertig haben")
    assert looks_german("Das ist gut so")
    assert looks_german("Okay")                                           # too short to judge
    assert not looks_german("We need to have the offer ready by Friday")
    assert not looks_german("Nous devons préparer l'offre pour vendredi")


def test_parakeet_backend_runs_the_real_package():
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("parakeet_mlx")
    from mlx.utils import tree_flatten
    from parakeet_mlx.utils import from_config

    from livetranslator.asr import ParakeetMLX

    vocab = ["▁der", "▁die", "▁und", "▁Angebot", "s", "en", "▁wir", "▁müssen"]
    cfg = {
        "target": "nemo.collections.asr.models.rnnt_bpe_models.EncDecRNNTBPEModel",
        "model_defaults": {"tdt_durations": [0, 1, 2, 3, 4]},
        "preprocessor": {"sample_rate": 16000, "normalize": "per_feature", "window_size": 0.025,
                         "window_stride": 0.01, "window": "hann", "features": 80, "n_fft": 512, "dither": 0.0},
        "encoder": {"feat_in": 80, "n_layers": 1, "d_model": 32, "n_heads": 2, "ff_expansion_factor": 2,
                    "subsampling_factor": 8, "self_attention_model": "rel_pos", "subsampling": "dw_striding",
                    "conv_kernel_size": 9, "subsampling_conv_channels": 16, "pos_emb_max_len": 5000},
        "decoder": {"blank_as_pad": True, "vocab_size": len(vocab), "prednet": {"pred_hidden": 32, "pred_rnn_layers": 1}},
        "joint": {"num_classes": len(vocab), "vocabulary": vocab, "num_extra_outputs": 5,
                  "jointnet": {"joint_hidden": 32, "activation": "relu", "encoder_hidden": 32, "pred_hidden": 32}},
        "decoding": {"model_type": "tdt", "durations": [0, 1, 2, 3, 4], "greedy": {"max_symbols": 5}},
    }
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "config.json").write_text(json.dumps(cfg))
        mx.save_safetensors(str(d / "model.safetensors"), dict(tree_flatten(from_config(cfg).parameters())))
        a = ParakeetMLX(str(d))
        a.prefetch()
        a.load()
        r = a.transcribe((np.random.default_rng(0).standard_normal(16000 * 2) * 0.1).astype(np.float32))
        assert isinstance(r.text, str) and r.elapsed > 0
        assert a.transcribe(np.zeros(1000, np.float32)).text == ""       # too short: nothing


class Breaks(Engine):
    """Works for the load check, then fails every preview (e.g. a GPU error mid-meeting)."""

    def transcribe(self, audio, prompt=None):
        self.calls.append(len(audio) / 16000)
        if len(self.calls) > 1:
            raise RuntimeError("metal: command buffer failed")
        return ASRResult("", 0.0, 0.0, 1.0, 0.0)


def test_a_preview_engine_that_breaks_never_stops_the_lines(tmp_path, meeting):
    whisper = Engine("mlx", whisper_text, 0.3)
    preview = Breaks("parakeet", "", 0.0)
    events = run(tmp_path, meeting, whisper, preview)
    assert len([e for e in events if e["type"] == "line"]) == 3      # every line still arrives
    assert [e for e in events if e["type"] == "partial" and e["text"]]  # Whisper took over the preview


def test_preview_settings_change_does_not_reload_the_speech_model(tmp_path):
    whisper = Engine("mlx", whisper_text, 0.0)
    loads = []
    with FakeOllama(models=["gemma3:4b"], token_delay=0.005) as fo:
        cfg = SettingsStore(tmp_path / "s.json")
        cfg.update({"ollama_url": fo.url, "live_preview": True})
        pipe = Pipeline(cfg, MeetingStore(tmp_path / "M"), lambda e: None,
                        asr_factory=lambda s: (loads.append(s.asr_model), whisper)[1],
                        preview_factory=lambda s: Engine("parakeet", "Wir sprechen", 0.0) if s.preview_engine == "auto" else None)
        assert wait_for(lambda: pipe.asr_state["status"] == "ready" and pipe.preview_asr is not None)
        old = cfg.settings
        pipe.apply_settings(old, cfg.update({"preview_engine": "whisper"}))
        assert wait_for(lambda: pipe.preview_asr is None)
        assert loads == [cfg.settings.asr_model]                     # the speech model was not loaded again
        pipe.shutdown()
