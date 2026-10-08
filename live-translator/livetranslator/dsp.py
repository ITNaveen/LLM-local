"""Audio clean-up: resampling, rumble filter, and the volume leveller.

The microphone hears a laptop speaker across a desk, so levels swing a lot
(the speaker leans back, Teams' own gain control kicks in, someone else talks
quietly). Two stages deal with that:

* `VadGain`   - a slow automatic gain in front of the voice detector so quiet
                speech still registers as speech.
* `level_segment` - per-line loudness normalisation plus gentle compression
                (quiet words are lifted, loud bursts are tamed) right before
                the audio goes to Whisper. It works on the finished line so it
                can look ahead and never "pumps" in the middle of a word.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi

from .config import SAMPLE_RATE


def db_to_lin(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def lin_to_db(x: float) -> float:
    return float(20.0 * np.log10(max(float(x), 1e-12)))


def rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


class Resampler:
    """Streaming resampler from the device rate to 16 kHz (mono float32)."""

    def __init__(self, in_rate: int, out_rate: int = SAMPLE_RATE):
        self.in_rate, self.out_rate = int(in_rate), int(out_rate)
        self._stream = None
        if self.in_rate != self.out_rate:
            import soxr

            self._stream = soxr.ResampleStream(self.in_rate, self.out_rate, 1, dtype="float32", quality="HQ")

    def __call__(self, x: np.ndarray, last: bool = False) -> np.ndarray:
        x = np.ascontiguousarray(x, dtype=np.float32)
        if self._stream is None:
            return x
        return self._stream.resample_chunk(x, last=last)


class HighPass:
    """80 Hz high-pass: removes desk thumps, fan rumble and DC offset."""

    def __init__(self, cutoff: float = 80.0, rate: int = SAMPLE_RATE):
        self.sos = butter(2, cutoff, btype="highpass", fs=rate, output="sos")
        self.zi = sosfilt_zi(self.sos) * 0.0

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if x.size == 0:  # the streaming resampler can return an empty block
            return x.astype(np.float32)
        y, self.zi = sosfilt(self.sos, x, zi=self.zi)
        return y.astype(np.float32)


class VadGain:
    """Slow AGC used only for the voice detector's copy of the signal.

    Tracks the loudness of recent speech and brings it towards -26 dBFS
    (up to +30 dB of boost, never attenuates below unity). The voice detector
    then sees comparable levels whether the boss is close or far away.
    """

    def __init__(self, target_db: float = -26.0, max_gain_db: float = 30.0, frame_s: float = 0.032):
        self.target = db_to_lin(target_db)
        self.max_gain = db_to_lin(max_gain_db)
        self.level = None  # tracked speech level (linear rms)
        # attack fast (loud speech arrives -> back off within ~0.1 s) and recover
        # within ~0.5 s, so a quiet sentence right after a loud one is still caught.
        # (Measured: a 3 s recovery lost 28 s of quiet speech in the volume-swing
        # tests, 0.5 s lost 0.4 s - with zero false lines on noise, typing or hum.)
        self.a_up = 1.0 - np.exp(-frame_s / 0.10)
        self.a_down = 1.0 - np.exp(-frame_s / 0.5)
        self.floor = db_to_lin(-70.0)

    def gain(self) -> float:
        if self.level is None:
            return 1.0
        return float(min(max(self.target / max(self.level, 1e-9), 1.0), self.max_gain))

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        g = self.gain()
        out = frame * g
        r = rms(frame)
        if r > self.floor:
            if self.level is None:
                self.level = r
            else:
                a = self.a_up if r > self.level else self.a_down
                self.level += a * (r - self.level)
        peak = float(np.max(np.abs(out))) if out.size else 0.0
        if peak > 0.99:
            out = out * (0.99 / peak)
        return out.astype(np.float32)


def level_segment(
    x: np.ndarray,
    target_db: float = -20.0,
    max_boost_db: float = 30.0,
    compress_range_db: float = 12.0,
    rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Normalise a finished line for speech recognition.

    1. Overall gain so the speech parts sit at `target_db` RMS.
    2. A smooth gain curve (~0.25 s) lifts quiet stretches and lowers loud ones
       by up to `compress_range_db`, but never amplifies pure background noise.
    3. Peak limiter so nothing clips.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.size < rate // 20:
        return x
    hop = rate // 100            # 10 ms frames
    win = hop * 2
    n = 1 + max(0, (x.size - win) // hop)
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    fr = np.sqrt(np.mean(np.square(x[idx], dtype=np.float64), axis=1) + 1e-12)
    fr_db = 20 * np.log10(fr)

    loud = np.percentile(fr_db, 95)
    noise = np.percentile(fr_db, 10)
    if loud < -85:               # digital silence
        return x
    # frames that are clearly speech (well above the noise floor)
    speech_mask = fr_db > max(noise + 6.0, loud - 30.0)
    speech_db = float(np.mean(fr_db[speech_mask])) if speech_mask.any() else float(loud)

    base_gain_db = min(target_db - speech_db, max_boost_db)

    # local level, smoothed over ~250 ms, only from speech frames
    k = 25
    kernel = np.hanning(k * 2 + 1)
    kernel /= kernel.sum()
    lvl = np.where(speech_mask, fr_db, speech_db)
    lvl_s = np.convolve(np.pad(lvl, k, mode="edge"), kernel, mode="valid")
    # how far each region is from the line's average speech level
    dev = np.clip(speech_db - lvl_s, -compress_range_db, compress_range_db) * 0.75
    # don't lift regions that are basically background noise
    noise_like = lvl_s < noise + 3.0
    dev = np.where(noise_like & (dev > 0), 0.0, dev)
    gain_db = base_gain_db + dev
    gain_db = np.minimum(gain_db, max_boost_db)

    # per-sample gain curve
    centers = hop * np.arange(n) + win // 2
    g = np.interp(np.arange(x.size), centers, 10.0 ** (gain_db / 20.0)).astype(np.float32)
    y = x * g

    peak = float(np.max(np.abs(y)))
    if peak > 0.95:
        # soft limiter: only the top part is squashed
        thr = 0.8
        over = np.abs(y) > thr
        y[over] = np.sign(y[over]) * (thr + (1 - thr) * np.tanh((np.abs(y[over]) - thr) / (1 - thr)))
        y = np.clip(y, -0.99, 0.99)
    return y.astype(np.float32)
