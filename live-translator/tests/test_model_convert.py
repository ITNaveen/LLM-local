"""German fine-tunes published in transformers format are converted once for mlx-whisper,
and a speech model that cannot be prepared falls back to the next one."""
import json
import time

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_whisper")

from mlx.utils import tree_flatten  # noqa: E402
from mlx_whisper.load_models import load_model  # noqa: E402
from mlx_whisper.whisper import ModelDimensions, Whisper  # noqa: E402

from livetranslator import mlx_convert  # noqa: E402

# OpenAI -> transformers names, as in transformers' convert_openai_to_hf.py
WHISPER_MAPPING = {
    "blocks": "layers", "mlp.0": "fc1", "mlp.2": "fc2", "mlp_ln": "final_layer_norm",
    ".attn.query": ".self_attn.q_proj", ".attn.key": ".self_attn.k_proj", ".attn.value": ".self_attn.v_proj",
    ".attn_ln": ".self_attn_layer_norm", ".attn.out": ".self_attn.out_proj",
    ".cross_attn.query": ".encoder_attn.q_proj", ".cross_attn.key": ".encoder_attn.k_proj",
    ".cross_attn.value": ".encoder_attn.v_proj", ".cross_attn_ln": ".encoder_attn_layer_norm",
    ".cross_attn.out": ".encoder_attn.out_proj", "decoder.ln.": "decoder.layer_norm.",
    "encoder.ln.": "encoder.layer_norm.", "token_embedding": "embed_tokens",
    "decoder.positional_embedding": "decoder.embed_positions.weight", "ln_post": "layer_norm",
}
DIMS = ModelDimensions(n_mels=128, n_audio_ctx=1500, n_audio_state=64, n_audio_head=4, n_audio_layer=2,
                       n_vocab=51866, n_text_ctx=448, n_text_state=64, n_text_head=4, n_text_layer=4)
HF_CONFIG = {"num_mel_bins": 128, "max_source_positions": 1500, "d_model": 64, "encoder_attention_heads": 4,
             "encoder_layers": 2, "vocab_size": 51866, "max_target_positions": 448, "decoder_attention_heads": 4,
             "decoder_layers": 4, "model_type": "whisper"}


def openai_to_hf(key):
    new = key
    for k, v in WHISPER_MAPPING.items():
        if k in key:
            new = new.replace(k, v)
    return "model." + new


def make_hf_checkpoint(folder, dtype=mx.float32, shards=1):
    """A tiny random Whisper in transformers layout; returns the MLX reference weights."""
    rng = np.random.default_rng(0)
    ref = {k: rng.standard_normal(v.shape).astype(np.float32) * 0.02
           for k, v in tree_flatten(Whisper(DIMS).parameters()) if k != "alignment_heads"}
    hf = {}
    for k, a in ref.items():
        ok = k.replace("mlp1", "mlp.0").replace("mlp2", "mlp.2")
        if "conv" in k and a.ndim == 3:
            a = a.transpose(0, 2, 1)                  # torch Conv1d layout
        hf[openai_to_hf(ok)] = mx.array(np.ascontiguousarray(a)).astype(dtype)
    hf["model.encoder.embed_positions.weight"] = mx.zeros((1500, 64), dtype=dtype)
    hf["proj_out.weight"] = hf["model.decoder.embed_tokens.weight"]
    folder.mkdir(parents=True)
    (folder / "config.json").write_text(json.dumps(HF_CONFIG))
    if shards == 1:
        mx.save_safetensors(str(folder / "model.safetensors"), hf)
    else:
        keys = sorted(hf)
        parts = [keys[i::shards] for i in range(shards)]
        wm = {}
        for i, part in enumerate(parts):
            name = f"model-{i + 1:05d}-of-{shards:05d}.safetensors"
            mx.save_safetensors(str(folder / name), {k: hf[k] for k in part})
            wm.update({k: name for k in part})
        (folder / "model.safetensors.index.json").write_text(json.dumps({"weight_map": wm}))
    return ref


@pytest.mark.parametrize("dtype,shards", [(mx.float32, 1), (mx.bfloat16, 1), (mx.float16, 3)])
def test_transformers_checkpoint_converts_to_mlx_whisper(tmp_path, monkeypatch, dtype, shards):
    monkeypatch.setenv("LT_HOME", str(tmp_path / "home"))
    src = tmp_path / "hf"
    ref = make_hf_checkpoint(src, dtype, shards)
    path = mlx_convert.ensure_mlx_whisper(str(src))
    assert mlx_convert.is_converted(str(src))
    cfg = json.loads((tmp_path / "home" / "models").joinpath(path.split("/")[-1], "config.json").read_text())
    assert cfg["n_text_layer"] == 4 and cfg["n_vocab"] == 51866
    m = load_model(path, dtype=mx.float16)
    got = dict(tree_flatten(m.parameters()))
    assert set(got) - {"alignment_heads"} == set(ref)
    tol = 2e-3 if dtype == mx.bfloat16 else 1e-3
    assert max(float(np.max(np.abs(np.array(got[k].astype(mx.float32)) - ref[k]))) for k in ref) < tol
    # second call: already there, nothing is converted again
    t = time.time()
    assert mlx_convert.ensure_mlx_whisper(str(src)) == path
    assert time.time() - t < 0.5


def test_a_wrong_checkpoint_is_refused_and_nothing_half_written(tmp_path, monkeypatch):
    monkeypatch.setenv("LT_HOME", str(tmp_path / "home"))
    src = tmp_path / "hf"
    make_hf_checkpoint(src)
    w = dict(mx.load(str(src / "model.safetensors")))
    del w["model.decoder.layer_norm.weight"]
    mx.save_safetensors(str(src / "model.safetensors"), w)
    with pytest.raises(ValueError, match="unexpected checkpoint layout"):
        mlx_convert.ensure_mlx_whisper(str(src))
    assert not mlx_convert.is_converted(str(src))
    assert not list((tmp_path / "home" / "models").glob("*"))


def test_converted_model_transcribes_through_the_app_backend(tmp_path, monkeypatch):
    from livetranslator.asr import MLXWhisper

    monkeypatch.setenv("LT_HOME", str(tmp_path / "home"))
    src = tmp_path / "hf"
    make_hf_checkpoint(src)
    asr = MLXWhisper(f"convert:{src}", "de")
    asr.load()
    r = asr.transcribe((np.random.default_rng(1).standard_normal(16000 * 2) * 0.05).astype(np.float32),
                       "Besprechung, Projekt")
    assert isinstance(r.text, str) and r.elapsed > 0


def test_speech_model_falls_back_when_the_german_one_cannot_be_prepared(tmp_path):
    from fake_ollama import FakeOllama

    from livetranslator.asr import ScriptedASR
    from livetranslator.config import SettingsStore
    from livetranslator.pipeline import Pipeline
    from livetranslator.storage import MeetingStore

    tried = []

    class Fails(ScriptedASR):
        def load(self):
            raise RuntimeError("download blocked")

    def factory(s):
        tried.append(s.asr_model)
        return Fails() if s.asr_model != "turbo" else ScriptedASR()

    with FakeOllama(models=["gemma3:4b"]) as fo:
        cfg = SettingsStore(tmp_path / "s.json")
        cfg.update({"ollama_url": fo.url, "asr_model": "turbo-de", "asr_backend": "mlx"})
        events = []
        pipe = Pipeline(cfg, MeetingStore(tmp_path / "M"), events.append, asr_factory=factory)
        end = time.time() + 20
        while pipe.asr_state["status"] != "ready" and time.time() < end:
            time.sleep(0.05)
        assert pipe.asr_state["status"] == "ready"
        assert tried == ["turbo-de", "mlx-community/whisper-large-v3-turbo-german-f16", "turbo"]
        assert any(e["type"] == "notice" and "could not be prepared" in e["text"] and "Whisper turbo" in e["text"]
                   for e in events)
        assert cfg.settings.asr_model == "turbo-de"        # tried again next start
        pipe.shutdown()


def test_old_settings_move_to_the_german_model(tmp_path):
    from livetranslator.config import SettingsStore

    p = tmp_path / "s.json"
    p.write_text(json.dumps({"settings_version": 2, "asr_model": "turbo", "llm_model": "gemma3:4b"}))
    assert SettingsStore(p).settings.asr_model == "turbo-de"
    p.write_text(json.dumps({"settings_version": 2, "asr_model": "large-v3"}))
    assert SettingsStore(p).settings.asr_model == "large-v3"           # a deliberate choice is kept
    assert SettingsStore(tmp_path / "new.json").settings.asr_model == "turbo-de"
