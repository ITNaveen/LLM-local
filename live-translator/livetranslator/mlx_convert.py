"""Use Whisper fine-tunes published in Hugging Face "transformers" format with mlx-whisper.

Some of the best German speech models (e.g. primeline/whisper-large-v3-turbo-german) are only
published as transformers checkpoints. This converts one, once, on this Mac into the folder
layout mlx-whisper loads (config.json + weights.safetensors, fp16) - without torch.

The key mapping mirrors ml-explore/mlx-examples whisper/convert.py (hf_to_pt + remap).
Converted models live in ~/LiveTranslator/models/<repo>/ with a marker file naming the source.
"""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

from .config import home_dir

log = logging.getLogger("lt.convert")

PREFIX = "convert:"

# mlx-whisper ModelDimensions field  <-  transformers WhisperConfig field
DIMS_FROM_HF = {
    "n_mels": "num_mel_bins",
    "n_audio_ctx": "max_source_positions",
    "n_audio_state": "d_model",
    "n_audio_head": "encoder_attention_heads",
    "n_audio_layer": "encoder_layers",
    "n_vocab": "vocab_size",
    "n_text_ctx": "max_target_positions",
    "n_text_state": "d_model",
    "n_text_head": "decoder_attention_heads",
    "n_text_layer": "decoder_layers",
}

# applied in this order (as convert.py does)
RENAMES = (
    ("model.", ""),
    (".layers", ".blocks"),
    (".self_attn", ".attn"),
    (".attn_layer_norm", ".attn_ln"),
    (".encoder_attn.", ".cross_attn."),
    (".encoder_attn_layer_norm", ".cross_attn_ln"),
    (".final_layer_norm", ".mlp_ln"),
    (".q_proj", ".query"),
    (".k_proj", ".key"),
    (".v_proj", ".value"),
    (".out_proj", ".out"),
    (".fc1", ".mlp1"),
    (".fc2", ".mlp2"),
    ("embed_positions.weight", "positional_embedding"),
    ("decoder.embed_tokens", "decoder.token_embedding"),
    ("encoder.layer_norm", "encoder.ln_post"),
    ("decoder.layer_norm", "decoder.ln"),
)

SOURCE_FILES = ["config.json", "generation_config.json", "model.safetensors", "model.safetensors.index.json",
                "model-*-of-*.safetensors"]
MARKER = "converted.json"


def hf_key_to_mlx(k: str) -> str:
    for a, b in RENAMES:
        k = k.replace(a, b)
    return k


def models_dir() -> Path:
    p = home_dir() / "models"
    p.mkdir(parents=True, exist_ok=True)
    return p


def target_dir(source: str) -> Path:
    name = source.rstrip("/").replace("\\", "/").split("/")[-2:]
    return models_dir() / ("--".join(name) + "-mlx-fp16")


def is_converted(source: str) -> bool:
    d = target_dir(source)
    return (d / MARKER).is_file() and (d / "weights.safetensors").is_file() and (d / "config.json").is_file()


def _safetensor_files(src: Path) -> list[Path]:
    single = src / "model.safetensors"
    if single.is_file():
        return [single]
    index = src / "model.safetensors.index.json"
    if index.is_file():
        wm = json.loads(index.read_text())["weight_map"]
        return [src / f for f in sorted(set(wm.values()))]
    raise FileNotFoundError(f"no model.safetensors in {src}")


def convert_hf_whisper(src_dir: str | Path, dst_dir: str | Path) -> Path:
    """transformers Whisper checkpoint folder -> mlx-whisper folder (fp16)."""
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_whisper.whisper import ModelDimensions, Whisper

    src, dst = Path(src_dir), Path(dst_dir)
    cfg = json.loads((src / "config.json").read_text())
    dims = {k: int(cfg[v]) for k, v in DIMS_FROM_HF.items()}
    if dims["n_vocab"] < 51865:
        raise ValueError(f"not a multilingual Whisper model (vocabulary {dims['n_vocab']})")

    weights: dict = {}
    for f in _safetensor_files(src):
        weights.update(mx.load(str(f)))       # reads fp32 / fp16 / bf16 lazily

    out = {}
    for k, v in weights.items():
        if k == "proj_out.weight":            # tied to the token embedding
            continue
        k = hf_key_to_mlx(k)
        if k == "encoder.positional_embedding":   # fixed sinusoids: mlx-whisper computes them itself
            continue
        if "conv" in k and v.ndim == 3:       # torch Conv1d (out, in, k) -> MLX (out, k, in)
            v = v.swapaxes(1, 2)
        out[k] = v.astype(mx.float16)
    gen = src / "generation_config.json"
    if gen.is_file():
        heads = json.loads(gen.read_text()).get("alignment_heads")
        if heads:
            out["alignment_heads"] = mx.array(heads, dtype=mx.int32)

    # every parameter of the mlx-whisper model must be there, with the right shape
    ref = dict(tree_flatten(Whisper(ModelDimensions(**dims), mx.float16).parameters()))
    need = {k for k in ref if k != "alignment_heads"}
    missing, extra = need - out.keys(), out.keys() - ref.keys()
    bad = [k for k in need & out.keys() if tuple(ref[k].shape) != tuple(out[k].shape)]
    if missing or extra or bad:
        raise ValueError(f"unexpected checkpoint layout: missing={sorted(missing)[:3]} "
                         f"extra={sorted(extra)[:3]} shape={bad[:3]}")

    tmp = dst.with_name(dst.name + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    try:
        mx.eval(out)
        mx.save_safetensors(str(tmp / "weights.safetensors"), out)
        (tmp / "config.json").write_text(json.dumps({**dims, "model_type": "whisper"}, indent=2))
        shutil.rmtree(dst, ignore_errors=True)
        tmp.replace(dst)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)       # e.g. disk full: never leave gigabytes behind
        raise
    return dst


def ensure_mlx_whisper(source: str, progress=None) -> str:
    """Folder of the converted model for `source` (HF repo id or local transformers folder);
    downloads and converts it the first time."""
    dst = target_dir(source)
    if is_converted(source):
        return str(dst)
    say = progress or (lambda msg: log.info("%s", msg))
    src = Path(source).expanduser()
    revision = "local"
    if not src.is_dir():
        from huggingface_hub import snapshot_download

        say(f"Downloading the speech model {source} (first time only)…")
        src = Path(snapshot_download(repo_id=source, allow_patterns=SOURCE_FILES))
        revision = src.name                   # snapshot folder name = commit hash
    say(f"Preparing the speech model {source} for this Mac (first time only)…")
    convert_hf_whisper(src, dst)
    (dst / MARKER).write_text(json.dumps({"source": source, "revision": revision}, indent=2))
    if revision != "local":
        _drop_download(source)
    return str(dst)


def _drop_download(repo: str) -> None:
    """The original download is not needed any more after converting (saves ~1.6-3 GB)."""
    try:
        from huggingface_hub import scan_cache_dir

        info = scan_cache_dir()
        revs = [r.commit_hash for repo_info in info.repos if repo_info.repo_id == repo for r in repo_info.revisions]
        if revs:
            info.delete_revisions(*revs).execute()
    except Exception:  # noqa: BLE001
        log.info("could not remove the downloaded copy of %s", repo, exc_info=True)
