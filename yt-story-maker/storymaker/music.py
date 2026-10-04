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


# ------------------------------------------------------------------ fallback music
MOOD_SYNTH = {
    # (root Hz, chord intervals per bar (semitones), bpm, pulse strength, brightness)
    "epic":       (73.42, [[0, 3, 7], [-4, 0, 3], [-9, -5, -2], [-2, 2, 5]], 88, 0.9, 0.7),
    "tense":      (65.41, [[0, 3, 6], [0, 3, 7], [1, 4, 8], [0, 3, 6]], 104, 0.8, 0.5),
    "emotional":  (98.00, [[0, 4, 7], [-3, 0, 4], [-8, -5, -1], [-5, -1, 2]], 72, 0.0, 0.6),
    "triumphant": (87.31, [[0, 4, 7], [5, 9, 12], [-3, 0, 4], [7, 11, 14]], 96, 0.7, 0.9),
    "calm":       (110.0, [[0, 4, 7], [5, 9, 12], [-3, 0, 4], [-5, -1, 2]], 66, 0.0, 0.4),
    "dark":       (55.00, [[0, 3, 7], [1, 5, 8], [0, 3, 7], [-2, 1, 5]], 70, 0.5, 0.3),
}


def synth_bed(mood, seconds, seed=0):
    """A simple generated cinematic pad + pulse. Placeholder until you add real tracks."""
    root, chords, bpm, pulse, bright = MOOD_SYNTH.get(mood, MOOD_SYNTH["epic"])
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    out = np.zeros((n, 2), dtype=np.float32)
    beat = 60.0 / bpm
    bar = beat * 4
    for b in range(int(seconds / bar) + 1):
        s0 = int(b * bar * SR)
        s1 = min(n, int((b + 1) * bar * SR + 0.6 * SR))
        if s0 >= n:
            break
        tt = t[s0:s1] - b * bar
        env = np.minimum(1, tt / 0.8) * np.exp(-np.maximum(0, tt - bar) * 4)
        for semi in chords[b % len(chords)]:
            f = root * 2 ** (semi / 12) * 2
            for k, amp in ((1, 1.0), (2, 0.35 * bright), (3, 0.15 * bright)):
                for ch, det in ((0, 0.997), (1, 1.003)):
                    out[s0:s1, ch] += (0.05 * amp * env *
                                       np.sin(2 * np.pi * f * k * det * tt + rng.random() * 6))
        # sub bass
        out[s0:s1, :] += (0.08 * env * np.sin(2 * np.pi * root * 2 ** (chords[b % len(chords)][0] / 12) * tt))[:, None]
    if pulse:
        k_len = int(0.35 * SR)
        kt = np.arange(k_len) / SR
        kick = (np.sin(2 * np.pi * (45 + 60 * np.exp(-kt * 30)) * kt) * np.exp(-kt * 9)).astype(np.float32)
        for i in range(int(seconds / beat)):
            s = int(i * beat * SR)
            e = min(n, s + k_len)
            out[s:e, :] += (pulse * 0.35 * kick[: e - s])[:, None]
    fade = min(n, int(1.5 * SR))
    out[:fade] *= np.linspace(0, 1, fade)[:, None]
    out[n - fade:] *= np.linspace(1, 0, fade)[:, None]
    peak = np.abs(out).max() or 1.0
    return out / peak * 0.6


# ------------------------------------------------------------------ bed rendering
def load_track(track, seconds, mood, seed):
    """Track audio of exactly `seconds`, looped if short; synth if no track."""
    n = int(round(seconds * SR))
    if not track:
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


def render_act_bed(out_path, act_seconds, track, mood, gain_points, narration, seed=0):
    """Write one act's bed: ducked music + narration placed at their times."""
    n = int(round(act_seconds * SR))
    music = load_track(track, act_seconds, mood, seed)
    if len(music) < n:
        music = np.pad(music, ((0, n - len(music)), (0, 0)))
    bed = music * envelope(n, gain_points)[:, None]
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
