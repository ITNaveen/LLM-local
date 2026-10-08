"""Acoustic tests of the listening front end with synthetic German speech.

Simulates the real setup (laptop speaker across a desk: band-limited sound,
room echo, background noise, volume going up and down) and checks that every
sentence becomes exactly one line, nothing is lost, and noise never produces
lines.
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from eval_segmentation import EXTRA, evaluate  # noqa: E402
from livetranslator import simulate as sim  # noqa: E402
from livetranslator.config import Settings  # noqa: E402
from livetranslator.dsp import db_to_lin  # noqa: E402
from livetranslator.frontend import FrontEnd  # noqa: E402
from livetranslator.vad import SileroVAD  # noqa: E402

pytestmark = pytest.mark.skipif(sim.tts_available() is None, reason="needs a German TTS voice (espeak-ng / macOS say)")


@pytest.fixture(scope="module")
def clips():
    return {v: [sim.synthesize(de, v) for de, _ in sim.SENTENCES] for v in (0, 1)}


@pytest.mark.parametrize("scenario", ["clean", "desk", "very_quiet", "noisy", "swings"])
def test_every_sentence_is_heard_once(clips, scenario):
    sc = {**sim.SCENARIOS, **EXTRA}[scenario]
    for v, cl in clips.items():
        audio, spans = sim.make_meeting(cl, sc, seed=v + 1)
        r = evaluate(audio, spans, Settings())
        assert r["false"] == 0, r
        assert r["split"] == 0, r
        assert r["merged"] <= 1, r
        assert r["lost_s"] < 0.5, r
        # line closes ~pause_ms after the speaker stops
        assert np.median(r["lat"]) < 0.8, r


def test_resampled_48k_input_behaves_the_same(clips):
    """The Mac microphone delivers 48 kHz; the front end resamples."""
    from scipy.signal import resample_poly

    audio, spans = sim.make_meeting(clips[0], sim.SCENARIOS["desk"])
    a48 = resample_poly(audio, 3, 1).astype(np.float32)
    fe = FrontEnd(Settings(), SileroVAD())
    segs = []
    for i in range(0, a48.size, 960):  # 20 ms blocks like the mic callback
        segs += fe.process(a48[i:i + 960], 48000)
    segs += fe.flush()
    assert len(segs) == len(spans)


def test_noise_never_becomes_a_line():
    rng = np.random.default_rng(5)
    sr, n = 16000, 16000 * 30
    t = np.arange(n) / sr
    clicks = np.zeros(n, np.float32)
    pos = 0
    while pos < n - 400:
        pos += int(rng.uniform(0.08, 0.35) * sr)
        ln = int(rng.uniform(0.004, 0.02) * sr)
        if pos + ln < n:
            clicks[pos:pos + ln] += rng.standard_normal(ln) * np.hanning(ln)
    cases = [
        sim.pink_noise(n, rng) * db_to_lin(-50),
        sim.pink_noise(n, rng) * db_to_lin(-35),
        clicks / np.max(np.abs(clicks)) * 0.5,
        (np.sin(2 * np.pi * 50 * t) * db_to_lin(-40)).astype(np.float32),
        np.zeros(n, np.float32),
    ]
    for sens in ("normal", "high"):
        for x in cases:
            fe = FrontEnd(Settings(sensitivity=sens), SileroVAD())
            segs = []
            for i in range(0, x.size, 320):
                segs += fe.process(x[i:i + 320].astype(np.float32), sr)
            segs += fe.flush()
            assert segs == []


@pytest.mark.skipif(shutil.which("espeak-ng") is None, reason="espeak-ng")
def test_tts_helper_produces_audio():
    x = sim.synthesize("Hallo zusammen.")
    assert x.size > 8000 and np.max(np.abs(x)) > 0.5
