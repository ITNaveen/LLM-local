"""Speech recognition backends (German speech -> German text).

* MLXWhisper    - Apple Silicon GPU via mlx-whisper (default on the M4 Mac)
* FasterWhisper - CTranslate2; Windows / Linux / Intel Mac (CPU or CUDA)
* ParakeetMLX   - Apple Silicon GPU, only for the grey live preview (cheap per pass)
* ScriptedASR   - test double

All backends must be created, loaded and used from the same thread (the ASR
worker thread) - MLX streams are per-thread.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass

import numpy as np

log = logging.getLogger("lt.asr")


@dataclass
class ASRResult:
    text: str
    avg_logprob: float = 0.0
    no_speech_prob: float = 0.0
    compression_ratio: float = 1.0
    elapsed: float = 0.0


# Whisper's fallback ladder, trimmed for live use: if greedy decoding looks like
# a loop / garbage, retry warmer - at most twice, so a bad line never stalls the feed.
_TEMPERATURES = (0.0, 0.3, 0.6)


def build_prompt(terms: list[str], previous: str = "", max_chars: int = 360) -> str | None:
    """Whisper 'initial prompt': vocabulary first, then the last thing said.

    Names and jargon in the prompt make Whisper spell them correctly; the
    previous sentence gives it the topic and punctuation style.
    """
    parts = []
    if terms:
        vocab = ", ".join(terms)
        parts.append(vocab[:200].rsplit(",", 1)[0] if len(vocab) > 200 else vocab)
    if previous:
        prev = previous.strip()
        if len(prev) > 160:
            prev = prev[-160:]
            prev = prev[prev.find(" ") + 1:]
        parts.append(prev)
    prompt = ". ".join(p.rstrip(". ") for p in parts if p)
    return (prompt + ".")[:max_chars] if prompt else None


class BaseASR:
    name = "base"

    def prefetch(self) -> None:
        """Download what load() needs (any thread; may take minutes the first time)."""

    def load(self) -> None:
        ...

    def transcribe(self, audio: np.ndarray, prompt: str | None = None) -> ASRResult:
        raise NotImplementedError

    def transcribe_preview(self, audio: np.ndarray, prompt: str | None = None) -> ASRResult:
        """A grey live preview: may take shortcuts (it is replaced by the real line anyway)."""
        return self.transcribe(audio, prompt)

    def warmup(self) -> float:
        t = time.perf_counter()
        noise = (np.random.default_rng(0).standard_normal(16000) * 0.001).astype(np.float32)
        try:
            self.transcribe(noise)
        except Exception:
            log.exception("warm-up failed")
        return time.perf_counter() - t


def resolve_hf_model(repo_or_path: str) -> str:
    """Local folder if given, else the cached HF snapshot (downloading if needed)."""
    from pathlib import Path

    p = Path(repo_or_path).expanduser()
    if p.exists():
        return str(p)
    from huggingface_hub import snapshot_download

    try:
        path = snapshot_download(repo_id=repo_or_path, local_files_only=True)
        if (Path(path) / "config.json").exists() and any((Path(path) / f).exists() for f in ("weights.safetensors", "weights.npz")):
            return path
    except Exception:  # noqa: BLE001
        pass
    log.info("downloading speech model %s (first run only)...", repo_or_path)
    patterns = ["config.json", "weights.safetensors", "weights.npz"]
    try:  # some repos carry the weights twice (.npz and .safetensors): fetch only one
        from huggingface_hub import list_repo_files

        files = set(list_repo_files(repo_or_path))
        if "weights.safetensors" in files:
            patterns = ["config.json", "weights.safetensors"]
    except Exception:  # noqa: BLE001
        pass
    return snapshot_download(repo_id=repo_or_path, allow_patterns=patterns)


def resolve_mlx_model(ref: str) -> str:
    """'convert:<repo>' -> converted on this Mac (once); otherwise an mlx-whisper repo / folder."""
    from .mlx_convert import PREFIX, ensure_mlx_whisper

    if ref.startswith(PREFIX):
        return ensure_mlx_whisper(ref[len(PREFIX):])
    return resolve_hf_model(ref)


class MLXWhisper(BaseASR):
    name = "mlx"

    def __init__(self, model: str, language: str = "de"):
        self.model_ref = model
        self.language = language
        self.path: str | None = None

    def load(self) -> None:
        import mlx.core as mx
        from mlx_whisper.transcribe import ModelHolder

        self.path = resolve_mlx_model(self.model_ref)
        # MLX keeps freed GPU buffers cached for reuse; on a 24 GB Mac that memory is
        # needed by Ollama's translation model, so keep the cache small.
        for setter in (getattr(mx, "set_cache_limit", None), getattr(getattr(mx, "metal", None), "set_cache_limit", None)):
            if setter is not None:
                try:
                    setter(512 * 1024 * 1024)
                    break
                except Exception:  # noqa: BLE001
                    continue
        ModelHolder.get_model(self.path, mx.float16)

    def transcribe(self, audio: np.ndarray, prompt: str | None = None, temperatures=_TEMPERATURES) -> ASRResult:
        import mlx_whisper

        t = time.perf_counter()
        r = mlx_whisper.transcribe(
            np.ascontiguousarray(audio, dtype=np.float32),
            path_or_hf_repo=self.path or self.model_ref,
            language=self.language,
            task="transcribe",
            temperature=temperatures,
            # German speech is ~4-5 tokens/s; this only stops runaway loops on noise (seconds of GPU)
            sample_len=min(224, int(len(audio) / 16000 * 10) + 32),
            compression_ratio_threshold=2.4,
            logprob_threshold=-1.0,
            no_speech_threshold=0.6,
            condition_on_previous_text=False,
            initial_prompt=prompt,
            without_timestamps=True,
            fp16=True,
            verbose=None,
        )
        return _from_segments(r.get("segments") or [], r.get("text", ""), time.perf_counter() - t)

    def transcribe_preview(self, audio: np.ndarray, prompt: str | None = None) -> ASRResult:
        # each warmer retry re-runs the whole 30-s encoder: not worth it for a preview
        return self.transcribe(audio, prompt, temperatures=(0.0,))


class FasterWhisper(BaseASR):
    name = "faster"

    def __init__(self, model: str, language: str = "de", device: str = "auto", compute_type: str = "default"):
        self.model_ref = model
        self.language = language
        self.device = device
        self.compute_type = compute_type
        self.model = None

    def load(self) -> None:
        from faster_whisper import WhisperModel

        self.model = WhisperModel(self.model_ref, device=self.device, compute_type=self.compute_type)

    def transcribe(self, audio: np.ndarray, prompt: str | None = None) -> ASRResult:
        t = time.perf_counter()
        segs, _info = self.model.transcribe(
            np.ascontiguousarray(audio, dtype=np.float32),
            language=self.language,
            task="transcribe",
            beam_size=5,
            temperature=list(_TEMPERATURES),
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
            condition_on_previous_text=False,
            initial_prompt=prompt,
            without_timestamps=True,
            vad_filter=False,
        )
        seg_dicts = [
            {"text": s.text, "avg_logprob": s.avg_logprob, "no_speech_prob": s.no_speech_prob,
             "compression_ratio": s.compression_ratio}
            for s in segs
        ]
        return _from_segments(seg_dicts, None, time.perf_counter() - t)


class ScriptedASR(BaseASR):
    """Test double: returns queued texts in order (or a description of the audio)."""

    name = "scripted"

    def __init__(self, texts: list[str] | None = None, delay: float = 0.0):
        self.texts = list(texts or [])
        self.delay = delay
        self.calls: list[tuple[float, str | None]] = []

    def warmup(self) -> float:
        return 0.0

    def transcribe(self, audio: np.ndarray, prompt: str | None = None) -> ASRResult:
        self.calls.append((len(audio) / 16000, prompt))
        if self.delay:
            time.sleep(self.delay)
        if self.texts:
            return ASRResult(self.texts.pop(0), elapsed=self.delay)
        return ASRResult(f"Satz mit {len(audio) / 16000:.1f} Sekunden.", elapsed=self.delay)


class EchoASR(BaseASR):
    """Test double: every call (live preview or final) returns the same text."""

    name = "echo"

    def __init__(self, text: str, delay: float = 0.0):
        self.text = text
        self.delay = delay

    def warmup(self) -> float:
        return 0.0

    def transcribe(self, audio: np.ndarray, prompt: str | None = None) -> ASRResult:
        if self.delay:
            time.sleep(self.delay)
        return ASRResult(self.text, elapsed=self.delay)


def _from_segments(segs: list[dict], text: str | None, elapsed: float) -> ASRResult:
    if not segs:
        return ASRResult((text or "").strip(), elapsed=elapsed, no_speech_prob=1.0 if not text else 0.0)
    joined = "".join(s["text"] for s in segs).strip() if text is None else text.strip()
    return ASRResult(
        text=joined,
        avg_logprob=float(np.mean([s["avg_logprob"] for s in segs])),
        no_speech_prob=float(max(s["no_speech_prob"] for s in segs)),
        compression_ratio=float(max(s["compression_ratio"] for s in segs)),
        elapsed=elapsed,
    )


class ParakeetMLX(BaseASR):
    """NVIDIA Parakeet TDT 0.6B v3 on the Apple GPU - only for the grey live preview.

    Whisper pads every pass to 30 s of audio, so each preview of a 2-s phrase costs as much as a
    full line (~1 s of GPU on an M4) - previews kept the GPU ~45 % busy and the real line often
    waited behind one. Parakeet's cost grows with the audio length (a few tens of ms per preview),
    so previews refresh faster, the final pass never queues behind them and the Mac stays cooler.
    Whisper still writes every saved line (German forced, glossary, dialect strength).
    """

    name = "parakeet"
    MODEL = "mlx-community/parakeet-tdt-0.6b-v3"

    def __init__(self, model: str = MODEL):
        self.model_ref = model
        self.local: str | None = None
        self.model = None

    def prefetch(self) -> None:
        from pathlib import Path

        if Path(self.model_ref).expanduser().is_dir():
            self.local = str(Path(self.model_ref).expanduser())
            return
        from huggingface_hub import snapshot_download

        self.local = snapshot_download(repo_id=self.model_ref, allow_patterns=["config.json", "model.safetensors"])

    def load(self) -> None:
        import mlx.core as mx
        from parakeet_mlx import from_pretrained

        self.model = from_pretrained(self.local or self.model_ref, dtype=mx.bfloat16)

    def transcribe(self, audio: np.ndarray, prompt: str | None = None) -> ASRResult:
        import mlx.core as mx
        from parakeet_mlx.audio import get_logmel

        t = time.perf_counter()
        x = mx.array(np.ascontiguousarray(audio, dtype=np.float32))
        if x.shape[0] < 16000 * 0.3:
            return ASRResult("", elapsed=time.perf_counter() - t)
        res = self.model.generate(get_logmel(x, self.model.preprocessor_config))[0]
        conf = min((s.confidence for s in res.sentences), default=1.0)
        return ASRResult(res.text.strip(), avg_logprob=float(np.log(max(conf, 1e-6))),
                         elapsed=time.perf_counter() - t)


def parakeet_available() -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec("parakeet_mlx") is not None
    except Exception:  # noqa: BLE001
        return False


_DE_WORDS = set("der die das und ist nicht ich wir sie es ein eine zu mit auf für dass den dem auch noch so ja "
                "aber wie was haben sind wird kann oder schon mal jetzt dann im in von bei hier gut".split())


def looks_german(text: str) -> bool:
    """Parakeet detects the language by itself: a preview in another language is not shown."""
    words = re.findall(r"[a-zäöüß]+", text.lower())
    if len(words) < 4 or re.search(r"[äöüß]", text.lower()):
        return True
    return sum(w in _DE_WORDS for w in words) / len(words) >= 0.12


def create_asr(backend: str, model: str, language: str = "de") -> BaseASR:
    if backend == "mlx":
        return MLXWhisper(model, language)
    if backend == "faster":
        return FasterWhisper(model, language)
    raise ValueError(f"unknown ASR backend {backend!r}")
