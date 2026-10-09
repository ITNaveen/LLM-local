"""Music: picks a royalty-free track per act by mood, finds its beat grid (so cuts land
on the beat), and builds the audio bed (music with automatic ducking + narration).

Put tracks in music/<mood>/ (epic, tense, emotional, triumphant, calm, dark). The safest
source for YouTube is the free YouTube Audio Library (studio.youtube.com -> Audio library).
With an empty library a simple generated cinematic pad is used so the pipeline still works."""

import random
import subprocess
import wave
from pathlib import Path

import numpy as np

from .util import ffmpeg, media_duration

SR = 48000
MOODS = ["epic", "tense", "emotional", "triumphant", "calm", "dark"]
AUDIO_EXT = (".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac")
# Moods that can stand in for each other when a folder is empty.
MOOD_FALLBACK = {
    "epic": ["triumphant", "tense", "dark"], "tense": ["dark", "epic"],
    "emotional": ["calm", "triumphant"], "triumphant": ["epic", "emotional"],
    "calm": ["emotional"], "dark": ["tense", "epic"],
}


def scan_library(music_dir):
    lib = {m: [] for m in MOODS}
    root = Path(music_dir)
    if not root.exists():
        return lib
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in AUDIO_EXT:
            continue
        rel = [part.lower() for part in p.relative_to(root).parts]
        mood = next((m for m in MOODS if m in rel[:-1]), None)
        if mood is None:
            mood = next((m for m in MOODS if m in p.stem.lower()), None)
        if mood:
            lib[mood].append(str(p))
    for m in lib:
        lib[m].sort()
    return lib


def choose_tracks(acts, music_dir, seed=0):
    """acts: [{key, mood}] -> {act_key: path or None}."""
    lib = scan_library(music_dir)
    rng = random.Random(seed)
    used, out = set(), {}
    for act in acts:
        options = []
        for mood in [act["mood"]] + MOOD_FALLBACK.get(act["mood"], []):
            options = [t for t in lib.get(mood, []) if t not in used] or options
            if options:
                break
        if not options:  # allow reuse rather than nothing
            options = lib.get(act["mood"], [])
        track = rng.choice(options) if options else None
        if track:
            used.add(track)
        out[act["key"]] = track
    return out


# ------------------------------------------------------------------ decoding / io
def decode(path, seconds=None, sr=SR, channels=2, offset=0.0):
    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error"]
    if offset:
        cmd += ["-ss", f"{offset:.3f}"]
    cmd += ["-i", str(path)]
    if seconds:
        cmd += ["-t", f"{seconds:.3f}"]
    cmd += ["-f", "f32le", "-ac", str(channels), "-ar", str(sr), "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    data = np.frombuffer(raw, dtype=np.float32)
    return data.reshape(-1, channels) if channels > 1 else data


def write_wav(path, data, sr=SR):
    data = np.clip(data, -1.0, 1.0)
    if data.ndim == 1:
        data = data[:, None]
    pcm = (data * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(pcm.shape[1])
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


# ------------------------------------------------------------------ beat grid
def beat_times(path, max_seconds=180):
    """Estimate beat times (seconds) with an onset envelope + autocorrelation tempo."""
    sr = 11025
    try:
        x = decode(path, seconds=max_seconds, sr=sr, channels=1)
    except subprocess.CalledProcessError:
        return []
    return beats_from_signal(x, sr)


def beats_from_signal(x, sr):
    hop = 128
    n = len(x) // hop
    if n < 128:
        return []
    frames = x[: n * hop].reshape(n, hop)
    energy = np.log1p(100 * np.sqrt((frames ** 2).mean(axis=1)))
    onset = np.maximum(0, np.diff(energy, prepend=energy[0]))
    onset -= onset.mean()
    fps = sr / hop
    ac = np.correlate(onset, onset, mode="full")[len(onset) - 1:]
    lo, hi = int(fps * 60 / 180), int(fps * 60 / 60)   # 60-180 BPM
    if hi >= len(ac):
        return []
    lags = np.arange(lo, hi)
    # Prefer tempi near 100 BPM slightly (log-normal weighting) to avoid half/double errors.
    bpm = 60 * fps / lags
    weight = np.exp(-0.5 * (np.log2(bpm / 100) / 0.9) ** 2)
    lag = lags[np.argmax(ac[lo:hi] * weight)]
    # Refine to a fractional period + phase with a comb over the whole track, so the
    # grid does not drift away from the music over several minutes.
    best = (-np.inf, float(lag), 0.0)
    for period in np.arange(lag - 1.0, lag + 1.0, 0.02):
        k = np.arange(int((n - 1) / period))
        for phase in np.arange(0, period, 0.5):
            idx = np.round(phase + k * period).astype(int)
            idx = idx[idx < n]
            score = onset[idx].sum()
            if score > best[0]:
                best = (score, period, phase)
    _, period, phase = best
    return [round(float((phase + i * period) / fps), 3) for i in range(int((n - phase) / period))]


# ------------------------------------------------------------------ built-in score
# Used when the music folder is empty: a generated news-thriller bed in the style of Hindi news
# packages - a dark drone, a ticking pulse, a heartbeat, a pluck ostinato and a low "braam" hit
# every few bars. Everything is plain sine/noise synthesis, so nothing is copyrighted.
MINOR = [[0, 3, 7], [-4, 0, 3], [3, 7, 10], [-2, 2, 5]]       # i - VI - III - VII
MAJOR = [[0, 4, 7], [5, 9, 12], [-3, 0, 4], [7, 11, 14]]      # I - IV - vi - V
MOOD_SYNTH = {
    #              root Hz  chords bpm  drone tick heart ostinato drums pad  braam every
    "tense":      (55.00, MINOR, 112, 0.55, 0.30, 0.70, 0.45, 0.00, 0.00, 8),
    "dark":       (49.00, MINOR, 92, 0.75, 0.18, 0.85, 0.30, 0.00, 0.00, 8),
    "epic":       (55.00, MINOR, 96, 0.50, 0.20, 0.00, 0.50, 0.90, 0.25, 4),
    "triumphant": (65.41, MAJOR, 100, 0.35, 0.20, 0.00, 0.45, 0.75, 0.45, 4),
    "emotional":  (65.41, MINOR, 72, 0.25, 0.00, 0.00, 0.40, 0.00, 0.70, 0),
    "calm":       (65.41, MAJOR, 70, 0.20, 0.00, 0.00, 0.25, 0.00, 0.70, 0),
}


def _place(out, sound, at, gain=1.0):
    """Add a (mono or stereo) sound into out at sample index `at`."""
    if at >= len(out) or at + len(sound) <= 0:
        return
    s0, e = max(0, at), min(len(out), at + len(sound))
    piece = sound[s0 - at:e - at]
    out[s0:e] += gain * (piece[:, None] if piece.ndim == 1 else piece)


def _tone(freq, seconds, decay, harmonics=(1.0, 0.3, 0.12), attack=0.004, sr=SR):
    t = np.arange(int(seconds * sr)) / sr
    env = np.minimum(1.0, t / attack) * np.exp(-t / decay)
    wave = sum(a * np.sin(2 * np.pi * freq * (k + 1) * t) for k, a in enumerate(harmonics))
    return (wave * env).astype(np.float32)


def _thump(seconds=0.45, f0=95.0, f1=42.0, decay=0.16, sr=SR):
    """Kick / heartbeat: a sine that drops in pitch."""
    t = np.arange(int(seconds * sr)) / sr
    freq = f1 + (f0 - f1) * np.exp(-t * 18)
    phase = 2 * np.pi * np.cumsum(freq) / sr
    return (np.sin(phase) * np.exp(-t / decay) * np.minimum(1, t / 0.002)).astype(np.float32)


def _noise(n, rng):
    return rng.standard_normal(n).astype(np.float32)


def _smooth(x, length):
    """Moving average (a gentle low-pass); length may vary per sample."""
    cs = np.concatenate([[0.0], np.cumsum(x, dtype=np.float64)])
    idx = np.arange(1, len(x) + 1)
    L = np.clip(np.asarray(length, dtype=int) if np.ndim(length) else np.full(len(x), int(length)), 1, None)
    lo = np.maximum(0, idx - L)
    return ((cs[idx] - cs[lo]) / (idx - lo)).astype(np.float32)


def _tick(rng, sr=SR):
    n = int(0.05 * sr)
    x = np.diff(_noise(n + 1, rng))                        # high-passed noise: a hi-hat tick
    return (x * np.exp(-np.arange(n) / (0.012 * sr))).astype(np.float32)


def synth_bed(mood, seconds, seed=0):
    """Generated cinematic news bed (stereo float32, about -20 dBFS)."""
    root, chords, bpm, drone, tick, heart, ost, drums, pad, braam_every = \
        MOOD_SYNTH.get(mood, MOOD_SYNTH["tense"])
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    out = np.zeros((n, 2), dtype=np.float32)
    t = np.arange(n) / SR
    beat = 60.0 / bpm
    bar = beat * 4
    n_bars = int(seconds / bar) + 1

    if drone:   # two slightly detuned voices per note, slow swell
        swell = 0.75 + 0.25 * np.sin(2 * np.pi * 0.05 * t + rng.random() * 6)
        for semi, amp in ((0, 1.0), (7, 0.5), (12, 0.35)):
            f = root * 2 ** (semi / 12)
            for ch, det in ((0, 0.998), (1, 1.002)):
                for k, ha in enumerate((1.0, 0.45, 0.22, 0.1)):
                    out[:, ch] += (drone * 0.07 * amp * ha * swell *
                                   np.sin(2 * np.pi * f * det * (k + 1) * t + k)).astype(np.float32)
    for b in range(n_bars):
        bar_start = int(b * bar * SR)
        chord = chords[(b // 2) % len(chords)]
        if pad:     # soft chord pad, one per two bars
            if b % 2 == 0:
                for semi in chord:
                    f = root * 4 * 2 ** (semi / 12)
                    note = _tone(f, bar * 2 + 1.0, decay=bar * 1.6, harmonics=(1.0, 0.2), attack=0.6)
                    _place(out, note, bar_start, pad * 0.05)
        if ost:     # 8th/16th-note pluck ostinato over the chord
            notes = [chord[0], chord[1], chord[2], chord[1] + 12 if bpm > 90 else chord[2]]
            steps = 8 if bpm > 90 else 4
            for k in range(steps):
                semi = notes[k % len(notes)] + 12
                f = root * 4 * 2 ** (semi / 12)
                pl = _tone(f, 0.45, decay=0.11 if bpm > 90 else 0.5, harmonics=(1.0, 0.35, 0.1))
                pan = 0.35 + 0.3 * (k % 2)
                at = bar_start + int(k * bar / steps * SR)
                _place(out, np.stack([pl * (1 - pan), pl * pan], axis=1), at, ost * 0.22)
        if heart:   # lub-dub every bar
            th = _thump(0.4, 80, 40, 0.12)
            _place(out, th, bar_start, heart * 0.30)
            _place(out, th, bar_start + int(beat * 0.45 * SR), heart * 0.18)
        if drums:   # taiko-like hits on 1 and 3, ghost on 4+
            for at_beat, g in ((0, 1.0), (2, 0.8), (3.5, 0.45)):
                hit = _thump(0.6, 120, 48, 0.2) + 0.25 * _smooth(_noise(int(0.6 * SR), rng), 6) * \
                    np.exp(-np.arange(int(0.6 * SR)) / (0.04 * SR))
                _place(out, hit.astype(np.float32), bar_start + int(at_beat * beat * SR), drums * 0.32 * g)
        if tick:    # ticking clock: 8th notes, accented on the beat
            for k in range(8):
                _place(out, _tick(rng), bar_start + int(k * beat / 2 * SR), tick * (0.16 if k % 2 == 0 else 0.09))
        if braam_every and b % braam_every == 0 and b > 0:
            br = sum(_tone(root * m * 2 ** (chord[0] / 12), 3.2, decay=1.1, harmonics=(1.0, 0.6, 0.4, 0.25),
                           attack=0.02) for m in (1, 2, 3))
            _place(out, br, bar_start, 0.07)            # an accent, never louder than a voice
            _place(out, _thump(1.2, 70, 30, 0.45), bar_start, 0.2)
    fade = min(n // 2, int(1.5 * SR))
    if fade:
        out[:fade] *= np.linspace(0, 1, fade)[:, None]
        out[n - fade:] *= np.linspace(1, 0, fade)[:, None]
    return _level(out, 0.10)


def _level(x, rms):
    """Bring to a target loudness, then a soft limiter so hits never clip."""
    cur = float(np.sqrt((x ** 2).mean())) or 1.0
    y = x * (rms / cur)
    return (np.tanh(y * 1.25) / 1.25).astype(np.float32)


# ------------------------------------------------------------------ sound effects
SFX_KINDS = ("whoosh", "hit", "boom", "riser")


def sfx(kind, seed=0):
    """Generated sound effects (stereo float32): whoosh (transition), hit (cut in the teaser),
    boom (title / big reveal), riser (tension build into the climax, ends at its start)."""
    rng = np.random.default_rng(seed + SFX_KINDS.index(kind) * 101)
    if kind == "whoosh":
        n = int(0.75 * SR)
        x = np.linspace(0, 1, n)
        env = np.sin(np.pi * np.clip(x / 0.75, 0, 1)) ** 2 * np.where(x < 0.75, 1.0, 0.0)
        env += np.where(x >= 0.75, np.exp(-(x - 0.75) * 30), 0.0)
        noise = _noise(n, rng)
        sweep = _smooth(noise, 40 - 34 * np.sin(np.pi * x)) - _smooth(noise, 160)
        mono = sweep * env
        pan = x
        st = np.stack([mono * (1 - pan * 0.7), mono * (0.3 + pan * 0.7)], axis=1)
        return (st / (np.abs(st).max() or 1) * 0.6).astype(np.float32)
    if kind in ("hit", "boom"):
        long = kind == "boom"
        n = int((2.6 if long else 1.2) * SR)
        body = np.zeros(n, dtype=np.float32)
        th = _thump(2.4 if long else 1.0, 110, 32, 0.5 if long else 0.28)
        body[:len(th)] += th
        burst = _smooth(_noise(n, rng), 3) * np.exp(-np.arange(n) / (0.05 * SR))
        body += 0.5 * burst
        tail = _smooth(_noise(n, rng), 12) * np.exp(-np.arange(n) / ((0.9 if long else 0.35) * SR)) * 0.25
        st = np.stack([body + tail, body + np.roll(tail, 331)], axis=1)
        return (st / (np.abs(st).max() or 1) * (0.95 if long else 0.8)).astype(np.float32)
    if kind == "riser":
        n = int(3.0 * SR)
        x = np.linspace(0, 1, n)
        noise = _noise(n, rng)
        swish = _smooth(noise, 60 - 57 * x) - _smooth(noise, 200)
        freq = 140 * (9 ** x)
        tone = np.sin(2 * np.pi * np.cumsum(freq) / SR) * 0.35
        mono = (swish * 2.5 + tone) * x ** 2.2
        st = np.stack([mono, np.roll(mono, 240)], axis=1)
        return (st / (np.abs(st).max() or 1) * 0.7).astype(np.float32)
    raise ValueError(kind)


# ------------------------------------------------------------------ bed rendering
def load_track(track, seconds, mood, seed, generate=False):
    """Track audio of exactly `seconds`, looped if short. Without a track: silence, or the
    generated pad if switched on (it is plain, so it is off by default)."""
    n = int(round(seconds * SR))
    if not track:
        if not generate:
            return np.zeros((n, 2), dtype=np.float32)
        return synth_bed(mood, seconds, seed)[:n]
    data = decode(track)
    if len(data) == 0:
        return synth_bed(mood, seconds, seed)[:n]
    reps = int(np.ceil(n / len(data)))
    data = np.tile(data, (reps, 1))[:n] if reps > 1 else data[:n]
    peak = np.abs(data).max() or 1.0
    data = data / peak * 0.8
    fade = min(n // 3, int(1.2 * SR))
    data[:fade] *= np.linspace(0, 1, fade)[:, None]
    data[n - fade:] *= np.linspace(1, 0, fade)[:, None]
    return data.astype(np.float32)


def envelope(n, points, ramp=0.35, rate=1000):
    """points: [(t_start, t_end, gain)] relative to act start -> smoothed gain curve.
    Built at 1 kHz with a moving average, then interpolated to the audio rate."""
    m = int(np.ceil(n * rate / SR)) + 1
    env = np.full(m, 0.5)
    for s, e, g in points:
        env[int(s * rate):int(e * rate)] = g
    if points:
        env[int(points[-1][1] * rate):] = points[-1][2]
    k = max(1, int(ramp * rate))
    padded = np.concatenate([np.full(k, env[0]), env, np.full(k, env[-1])])
    csum = np.cumsum(np.concatenate([[0.0], padded]))
    idx = np.arange(m) + k - k // 2
    smooth = (csum[idx + k] - csum[idx]) / k
    return np.interp(np.arange(n) * (rate / SR), np.arange(m), smooth).astype(np.float32)


def render_act_bed(out_path, act_seconds, track, mood, gain_points, narration, seed=0,
                   generate=True, effects=()):
    """Write one act's bed: ducked music + sound effects + narration placed at their times."""
    n = int(round(act_seconds * SR))
    music = load_track(track, act_seconds, mood, seed, generate)
    if len(music) < n:
        music = np.pad(music, ((0, n - len(music)), (0, 0)))
    bed = music * envelope(n, gain_points)[:, None]
    for i, fx in enumerate(effects):         # not ducked: they punctuate the cut
        sound = sfx(fx["kind"], seed=seed * 31 + i)
        at = int(round(fx["t"] * SR))
        if fx["kind"] == "riser":             # a riser ends exactly on its moment
            at -= len(sound)
        _place(bed, sound[:max(0, n - max(0, at))] if at >= 0 else sound, at, fx.get("gain", 0.5))
    for item in narration:
        voice = decode(item["file"], channels=1)
        s = int(round(item["t"] * SR))
        e = min(n, s + len(voice))
        if e > s:
            bed[s:e, :] += (voice[: e - s] * 0.95)[:, None]
    write_wav(out_path, bed)
    return out_path


def track_duration(track):
    return media_duration(track) if track else 0.0
