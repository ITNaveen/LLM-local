"""Audio front end: any-rate audio in -> finished lines (Segments) out.

resample to 16 kHz -> 80 Hz high-pass -> [slow AGC -> Silero VAD] -> Segmenter
Used by the live pipeline and, unchanged, by the tests and the self-test.
"""
from __future__ import annotations

import numpy as np

from .config import SENSITIVITY, Settings
from .dsp import HighPass, Resampler, VadGain, lin_to_db
from .segmenter import Segment, Segmenter
from .vad import FRAME


class FrontEnd:
    def __init__(self, settings: Settings, vad, agc: bool = True):
        self.vad = vad
        self.seg = Segmenter(threshold=SENSITIVITY[settings.sensitivity], pause_ms=settings.pause_ms,
                             max_line_s=settings.max_line_s)
        self.hp = HighPass()
        self.agc = VadGain() if agc else None
        self._resamplers: dict[int, Resampler] = {}
        self._buf = np.zeros(0, dtype=np.float32)
        self.peak = 0.0
        self._lvl_acc = 0.0
        self._lvl_n = 0
        self.probs: list[float] = []      # kept only when record_probs is set (diagnostics)
        self.record_probs = False

    def configure(self, settings: Settings) -> None:
        self.seg.configure(SENSITIVITY[settings.sensitivity], settings.pause_ms, settings.max_line_s)

    def process(self, x: np.ndarray, rate: int) -> list[Segment]:
        rs = self._resamplers.get(rate)
        if rs is None:
            rs = self._resamplers[rate] = Resampler(rate)
        if x.size:
            self.peak = max(self.peak, float(np.max(np.abs(x))))
        y = self.hp(rs(x))
        buf = np.concatenate([self._buf, y]) if self._buf.size else y
        out: list[Segment] = []
        nfr = buf.size // FRAME
        for i in range(nfr):
            f = buf[i * FRAME:(i + 1) * FRAME]
            p = self.vad(self.agc(f) if self.agc else f)
            if self.record_probs:
                self.probs.append(p)
            self._lvl_acc += float(np.mean(f.astype(np.float64) ** 2))
            self._lvl_n += 1
            out.extend(self.seg.push(f, p))
        self._buf = buf[nfr * FRAME:]
        return out

    def flush(self) -> list[Segment]:
        return self.seg.flush()

    def take_level_db(self) -> float | None:
        if not self._lvl_n:
            return None
        db = lin_to_db(np.sqrt(self._lvl_acc / self._lvl_n))
        self._lvl_acc, self._lvl_n = 0.0, 0
        return db

    def boost_db(self) -> float:
        return lin_to_db(self.agc.gain()) if self.agc else 0.0
