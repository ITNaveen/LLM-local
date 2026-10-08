import numpy as np

from livetranslator.segmenter import Segmenter
from livetranslator.vad import FRAME

FS = FRAME / 16000  # 32 ms


def run(probs, **kw):
    seg = Segmenter(**kw)
    out = []
    for i, p in enumerate(probs):
        f = np.full(FRAME, i, dtype=np.float32)  # frame index encoded in samples
        out += seg.push(f, p)
    out += seg.flush()
    return out


def frames(s):
    return int(round(s / FS))


def test_single_utterance_with_pause():
    probs = [0.0] * frames(1) + [0.9] * frames(2) + [0.0] * frames(1)
    segs = run(probs, pause_ms=500)
    assert len(segs) == 1
    s = segs[0]
    # starts ~300 ms before speech (pre-roll), ends ~200 ms after
    assert abs(s.start - (1 - 0.3)) < 0.07
    assert abs(s.end - (3 + 0.2)) < 0.07
    # closed right after the 500 ms pause
    assert abs(s.detected_at - (3 + 0.5)) < 0.07


def test_short_pause_does_not_split():
    probs = [0.0] * 10 + [0.9] * frames(1.5) + [0.1] * frames(0.3) + [0.9] * frames(1.5) + [0.0] * frames(1)
    segs = run(probs, pause_ms=500)
    assert len(segs) == 1


def test_long_pause_splits():
    probs = [0.0] * 10 + [0.9] * frames(1.5) + [0.1] * frames(0.7) + [0.9] * frames(1.5) + [0.0] * frames(1)
    assert len(run(probs, pause_ms=500)) == 2


def test_blips_are_ignored():
    probs = [0.0] * 20 + [0.9] * 3 + [0.0] * 40   # ~100 ms click
    assert run(probs, min_speech_ms=200) == []


def test_monologue_gets_cut_regularly():
    # 40 s of continuous speech with tiny 130 ms dips every ~2.5 s
    probs = [0.0] * 10
    for _ in range(16):
        probs += [0.9] * frames(2.4) + [0.2] * 4
    probs += [0.0] * frames(1)
    segs = run(probs, max_line_s=10.0, pause_ms=500)
    durs = [s.end - s.start for s in segs]
    assert len(segs) >= 4
    assert max(durs) <= 10.0 * 1.4 + 0.5
    # everything is covered: no audio lost between consecutive lines
    for a, b in zip(segs, segs[1:]):
        assert b.start <= a.end + 0.6


def test_hard_cut_without_any_dip():
    probs = [0.0] * 10 + [0.95] * frames(30) + [0.0] * frames(1)
    segs = run(probs, max_line_s=10.0)
    assert len(segs) >= 2
    assert sum(1 for s in segs if s.forced) >= 1
    total = sum(s.end - s.start for s in segs)
    assert total >= 29.5


def test_audio_is_contiguous_and_complete():
    probs = [0.0] * 10 + [0.9] * 50 + [0.0] * 30
    seg = Segmenter(pause_ms=500)
    out = []
    for i, p in enumerate(probs):
        out += seg.push(np.full(FRAME, i, dtype=np.float32), p)
    s = out[0]
    idx = s.audio[::FRAME].astype(int)
    assert list(idx) == list(range(idx[0], idx[0] + len(idx)))  # no gaps / duplicates
    assert idx[0] <= 10 and idx[-1] >= 59


def test_live_reconfigure():
    seg = Segmenter(pause_ms=500)
    seg.configure(0.35, 300, 12.0)
    assert seg.on == 0.35 and seg.pause_frames == round(0.3 / FS)
