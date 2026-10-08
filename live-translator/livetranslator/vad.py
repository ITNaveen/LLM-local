"""Silero VAD (v6, MIT licence) via onnxruntime - no PyTorch needed.

Feeds 512-sample (32 ms) frames at 16 kHz and returns the speech probability.
Mirrors silero_vad.utils_vad.OnnxWrapper exactly (64-sample context, state).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

FRAME = 512          # samples per VAD frame at 16 kHz (32 ms)
_CONTEXT = 64
_MODEL = Path(__file__).parent / "assets" / "silero_vad.onnx"


class SileroVAD:
    def __init__(self, model_path: Path | str = _MODEL):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.log_severity_level = 3
        self.session = ort.InferenceSession(str(model_path), sess_options=opts, providers=["CPUExecutionProvider"])
        self._sr = np.array(16000, dtype=np.int64)
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, _CONTEXT), dtype=np.float32)

    def __call__(self, frame: np.ndarray) -> float:
        if frame.shape[-1] != FRAME:
            raise ValueError(f"VAD needs {FRAME} samples, got {frame.shape[-1]}")
        x = np.concatenate([self._context, frame.reshape(1, FRAME).astype(np.float32)], axis=1)
        out, self._state = self.session.run(None, {"input": x, "state": self._state, "sr": self._sr})
        self._context = x[:, -_CONTEXT:]
        return float(out[0, 0])
