"""Turns a stream of (frame, speech-probability) into finished lines.

A line ends as soon as the speaker pauses for `pause_ms`. People who talk
for a long time without pausing still get regular lines: the longer the
line gets, the shorter the pause that is accepted as a break, and at a hard
limit the line is cut at the quietest point (between words).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .config import SAMPLE_RATE
from .vad import FRAME


@dataclass
class Segment:
    audio: np.ndarray
    start: float            # seconds since the stream started
    end: float
    forced: bool = False    # cut in the middle of speech (no pause found)
    voiced_s: float = 0.0   # seconds of actual speech inside
    detected_at: float = 0.0  # stream time at which the line was closed


@dataclass
class _Cur:
    frames: list = field(default_factory=list)
    probs: list = field(default_factory=list)
    start_frame: int = 0
    last_voiced: int = -1
    voiced: int = 0
    silence_run: int = 0


class Segmenter:
    def __init__(
        self,
        threshold: float = 0.5,
        pause_ms: int = 500,
        max_line_s: float = 14.0,
        min_speech_ms: int = 200,
        pre_pad_ms: int = 300,
        post_pad_ms: int = 200,
        start_frames: int = 2,
    ):
        self.frame_s = FRAME / SAMPLE_RATE
        self.on = float(threshold)
        self.off = max(0.05, self.on - 0.15)           # hysteresis
        self.pause_frames = max(1, round(pause_ms / 1000 / self.frame_s))
        self.max_frames = round(max_line_s / self.frame_s)
        self.hard_frames = round(max_line_s * 1.4 / self.frame_s)
        self.min_voiced = max(1, round(min_speech_ms / 1000 / self.frame_s))
        self.pre_pad = deque(maxlen=max(1, round(pre_pad_ms / 1000 / self.frame_s)))
        self.post_pad = round(post_pad_ms / 1000 / self.frame_s)
        self.start_frames = start_frames
        self.n = 0                  # frames seen so far
        self._pending = []          # candidate start frames (idle state)
        self.cur: _Cur | None = None

    def configure(self, threshold: float, pause_ms: int, max_line_s: float) -> None:
        """Change sensitivity / pause length while running."""
        self.on = float(threshold)
        self.off = max(0.05, self.on - 0.15)
        self.pause_frames = max(1, round(pause_ms / 1000 / self.frame_s))
        self.max_frames = round(max_line_s / self.frame_s)
        self.hard_frames = round(max_line_s * 1.4 / self.frame_s)

    @property
    def current_start(self) -> float | None:
        return None if self.cur is None else self.cur.start_frame * self.frame_s

    # ------------------------------------------------------------------ state
    @property
    def in_speech(self) -> bool:
        return self.cur is not None

    def current_audio(self) -> tuple[np.ndarray, float] | None:
        """Audio of the line being spoken right now (for the live preview)."""
        c = self.cur
        if c is None or not c.frames:
            return None
        return np.concatenate(c.frames), c.start_frame * self.frame_s

    def current_voiced_s(self) -> float:
        return 0.0 if self.cur is None else self.cur.voiced * self.frame_s

    def _required_pause(self, length: int) -> int:
        if length >= int(self.max_frames * 0.85):
            return min(self.pause_frames, 4)     # ~130 ms: any small gap will do
        if length >= int(self.max_frames * 0.55):
            return min(self.pause_frames, 8)     # ~250 ms
        return self.pause_frames

    # ------------------------------------------------------------------- push
    def push(self, frame: np.ndarray, prob: float) -> list[Segment]:
        out: list[Segment] = []
        idx = self.n
        self.n += 1
        if self.cur is None:
            if prob >= self.on:
                self._pending.append((frame, prob))
                if len(self._pending) >= self.start_frames:
                    self._open(idx - len(self._pending) + 1)
                    for f, p in self._pending:
                        self._append(f, p)
                    self._pending = []
            else:
                for f, _ in self._pending:
                    self.pre_pad.append(f)
                self._pending = []
                self.pre_pad.append(frame)
            return out

        self._append(frame, prob)
        c = self.cur
        length = len(c.frames)
        if c.silence_run >= self._required_pause(length):
            out.extend(self._close(c.last_voiced + 1 + self.post_pad, forced=False))
        elif length >= self.hard_frames:
            out.extend(self._force_cut())
        return out

    def flush(self) -> list[Segment]:
        if self.cur is None:
            return []
        return self._close(self.cur.last_voiced + 1 + self.post_pad, forced=False)

    # ---------------------------------------------------------------- helpers
    def _open(self, first_speech_frame: int) -> None:
        pre = list(self.pre_pad)
        self.pre_pad.clear()
        self.cur = _Cur(start_frame=first_speech_frame - len(pre))
        for f in pre:
            self.cur.frames.append(f)
            self.cur.probs.append(0.0)

    def _append(self, frame: np.ndarray, prob: float) -> None:
        c = self.cur
        c.frames.append(frame)
        c.probs.append(prob)
        if prob >= self.off:
            c.last_voiced = len(c.frames) - 1
            c.silence_run = 0
            if prob >= self.on:
                c.voiced += 1
        else:
            c.silence_run += 1

    def _make(self, c: _Cur, upto: int, forced: bool) -> Segment | None:
        upto = max(1, min(upto, len(c.frames)))
        voiced = sum(1 for p in c.probs[:upto] if p >= self.on)
        if voiced < self.min_voiced:
            return None
        audio = np.concatenate(c.frames[:upto])
        start = c.start_frame * self.frame_s
        return Segment(
            audio=audio,
            start=start,
            end=start + upto * self.frame_s,
            forced=forced,
            voiced_s=voiced * self.frame_s,
            detected_at=self.n * self.frame_s,
        )

    def _close(self, upto: int, forced: bool) -> list[Segment]:
        c = self.cur
        seg = self._make(c, upto, forced)
        # whatever trailed the cut is silence: keep it as pre-roll for the next line
        for f in c.frames[upto:]:
            self.pre_pad.append(f)
        self.cur = None
        return [seg] if seg else []

    def _force_cut(self) -> list[Segment]:
        """No pause for too long: cut at the quietest moment of the last part."""
        c = self.cur
        probs = np.asarray(c.probs, dtype=np.float32)
        lo = int(self.max_frames * 0.5)
        hi = len(probs) - 3
        # smooth over ~100 ms so we find a gap between words, not a single dip
        sm = np.convolve(probs, np.ones(3) / 3, mode="same")
        region = sm[lo:hi]
        # prefer later cut points slightly (longer, more complete lines)
        bias = np.linspace(0.05, 0.0, region.size)
        cut = lo + int(np.argmin(region + bias)) + 1
        seg = self._make(c, cut, forced=True)
        rest_frames, rest_probs = c.frames[cut:], c.probs[cut:]
        new = _Cur(start_frame=c.start_frame + cut)
        self.cur = new
        for f, p in zip(rest_frames, rest_probs):
            self._append(f, p)
        return [seg] if seg else []
