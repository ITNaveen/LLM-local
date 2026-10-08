import numpy as np

from livetranslator.dsp import HighPass, Resampler, VadGain, level_segment, lin_to_db, rms
from livetranslator.simulate import pink_noise

SR = 16000


def tone_speech(seconds, level_db, rng):
    """Speech-like signal: modulated noise bursts (syllables)."""
    n = int(SR * seconds)
    t = np.arange(n) / SR
    env = (0.5 + 0.5 * np.sin(2 * np.pi * 4 * t)) ** 2
    x = rng.standard_normal(n) * env
    x = x / rms(x) * 10 ** (level_db / 20)
    return x.astype(np.float32)


def test_level_segment_brings_quiet_speech_to_target():
    rng = np.random.default_rng(0)
    x = tone_speech(3, -45, rng)
    y = level_segment(x)
    assert -24 < lin_to_db(rms(y)) < -16


def test_level_segment_evens_out_volume_swings():
    rng = np.random.default_rng(1)
    loud = tone_speech(2, -18, rng)
    quiet = tone_speech(2, -36, rng)
    x = np.concatenate([loud, quiet])
    y = level_segment(x)
    before = lin_to_db(rms(x[:2 * SR])) - lin_to_db(rms(x[2 * SR:]))
    after = lin_to_db(rms(y[:2 * SR])) - lin_to_db(rms(y[2 * SR:]))
    assert before > 17
    assert after < before - 6, (before, after)   # gap reduced substantially
    assert np.max(np.abs(y)) < 1.0


def test_level_segment_does_not_blow_up_noise_only_parts():
    rng = np.random.default_rng(2)
    speech = tone_speech(2, -20, rng)
    noise = pink_noise(SR * 2, rng) * 10 ** (-60 / 20)
    x = np.concatenate([speech, noise])
    y = level_segment(x)
    # noise tail stays far below speech
    assert lin_to_db(rms(y[2 * SR + 4000:])) < lin_to_db(rms(y[:2 * SR])) - 25


def test_level_segment_never_clips():
    rng = np.random.default_rng(3)
    x = np.clip(tone_speech(2, -3, rng), -1, 1)
    assert np.max(np.abs(level_segment(x))) <= 0.99


def test_vad_gain_boosts_quiet_but_not_loud():
    rng = np.random.default_rng(4)
    g = VadGain()
    for _ in range(100):
        g(tone_speech(0.032, -50, rng)[:512])
    assert lin_to_db(g.gain()) > 15
    g2 = VadGain()
    for _ in range(100):
        g2(tone_speech(0.032, -15, rng)[:512])
    assert lin_to_db(g2.gain()) < 1


def test_resampler_48k_to_16k_streaming():
    t = np.arange(48000) / 48000
    x = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    r = Resampler(48000)
    out = np.concatenate([r(x[i:i + 960]) for i in range(0, len(x), 960)] + [r(np.zeros(0, np.float32), last=True)])
    assert abs(len(out) - 16000) < 50
    assert abs(rms(out[1000:15000]) - rms(x)) < 0.02


def test_highpass_removes_rumble():
    t = np.arange(SR) / SR
    hp = HighPass()
    low = hp((0.5 * np.sin(2 * np.pi * 20 * t)).astype(np.float32))
    hp2 = HighPass()
    mid = hp2((0.5 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32))
    assert rms(low[4000:]) < 0.1 * rms(mid[4000:])   # ~ -24 dB at 20 Hz
    assert rms(mid[4000:]) > 0.34
