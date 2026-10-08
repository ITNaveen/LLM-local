"""Build realistic test audio: a meeting heard through a laptop speaker across a desk.

Used by the self-test and the automated tests. Starts from clean speech clips
and adds what the real setup does to them: small-speaker band-limiting, room
echo, background noise, and the speaker's volume going up and down.
"""
from __future__ import annotations

import platform
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import butter, fftconvolve, sosfilt

from .audio_io import read_wav
from .dsp import Resampler, db_to_lin, rms

SR = 16000

# Meeting-style German with the things that matter: numbers, dates, names, negations.
SENTENCES = [
    ("Guten Morgen zusammen, lassen Sie uns kurz über den aktuellen Stand des Projekts sprechen.",
     "Good morning everyone, let's talk briefly about the current status of the project."),
    ("Wir müssen das Angebot für den Kunden bis Freitag, den 17. Oktober, fertig haben.",
     "We need to have the offer for the customer ready by Friday, October 17."),
    ("Die Schnittstelle zum SAP-System funktioniert leider noch nicht zuverlässig.",
     "Unfortunately, the interface to the SAP system is not working reliably yet."),
    ("Könnten Sie mir bitte bis morgen Mittag eine kurze Zusammenfassung schicken?",
     "Could you please send me a short summary by tomorrow noon?"),
    ("Das Budget beträgt insgesamt 25.000 Euro, mehr ist dieses Jahr nicht drin.",
     "The budget is 25,000 euros in total; there is no more this year."),
    ("Ich glaube nicht, dass wir den Termin halten können, wenn das Testteam erst nächste Woche anfängt.",
     "I don't think we can meet the deadline if the test team only starts next week."),
    ("Bitte sprechen Sie vorher mit Frau Müller aus der Buchhaltung.",
     "Please talk to Ms. Müller from accounting beforehand."),
    ("Alles klar, dann machen wir das so. Vielen Dank für Ihre Arbeit.",
     "All right, then let's do it that way. Thank you for your work."),
]


# ------------------------------------------------------------------ TTS
def german_voices_mac() -> list[str]:
    try:
        out = subprocess.run(["say", "-v", "?"], capture_output=True, text=True, timeout=20).stdout
    except Exception:  # noqa: BLE001
        return []
    voices = []
    for line in out.splitlines():
        if " de_" in line:
            name = line.split(" de_")[0].strip()
            voices.append(name)
    # prefer the natural-sounding ones
    voices.sort(key=lambda v: (0 if v.split()[0] in ("Anna", "Petra", "Markus", "Yannick", "Helena", "Martin") else 1, v))
    return voices


def tts_available() -> str | None:
    if platform.system() == "Darwin" and shutil.which("say") and shutil.which("afconvert") and german_voices_mac():
        return "say"
    if shutil.which("espeak-ng"):
        return "espeak"
    return None


def synthesize(text: str, voice_index: int = 0) -> np.ndarray:
    """German TTS -> float32 16 kHz mono."""
    engine = tts_available()
    with tempfile.TemporaryDirectory() as td:
        wav = str(Path(td) / "s.wav")
        if engine == "say":
            voices = german_voices_mac()
            v = voices[voice_index % len(voices)]
            aiff = str(Path(td) / "s.aiff")
            subprocess.run(["say", "-v", v, "-o", aiff, text], check=True, timeout=60)
            subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", aiff, wav], check=True, timeout=60)
        elif engine == "espeak":
            mb = ["mb-de2", "mb-de1", "mb-de4", "mb-de5", "mb-de6", "mb-de7", "mb-de3"]
            voice = mb[voice_index % len(mb)]
            r = subprocess.run(["espeak-ng", "-v", voice, "-s", "150", "-w", wav, text], capture_output=True)
            if r.returncode != 0 or not Path(wav).exists() or Path(wav).stat().st_size < 1000:
                subprocess.run(["espeak-ng", "-v", "de", "-s", "150", "-w", wav, text], check=True)
        else:
            raise RuntimeError("No German text-to-speech voice found (macOS: install a German voice in "
                               "System Settings → Accessibility → Spoken Content).")
        x, rate = read_wav(wav)
    if rate != SR:
        x = Resampler(rate, SR)(x, last=True)
    x = x / max(1e-6, float(np.max(np.abs(x)))) * 0.7
    return x.astype(np.float32)


# ------------------------------------------------------------------ acoustics
def pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    white = rng.standard_normal(n)
    f = np.fft.rfft(white)
    scale = 1 / np.sqrt(np.maximum(np.arange(len(f)), 1))
    y = np.fft.irfft(f * scale, n)
    return (y / (rms(y) + 1e-9)).astype(np.float32)


def room_ir(rt60: float, rng: np.random.Generator, length_s: float = 0.6) -> np.ndarray:
    n = int(SR * length_s)
    t = np.arange(n) / SR
    decay = np.exp(-6.9 * t / rt60)
    ir = rng.standard_normal(n) * decay * 0.25
    ir[: int(SR * 0.004)] = 0          # direct path arrives first
    ir[0] = 1.0
    return (ir / np.sqrt(np.sum(ir ** 2))).astype(np.float32)


def laptop_speaker(x: np.ndarray) -> np.ndarray:
    """Tiny laptop speakers: no bass, rolled-off top, a bit of mid boost."""
    sos = butter(3, [250, 6500], btype="bandpass", fs=SR, output="sos")
    return sosfilt(sos, x).astype(np.float32)


@dataclass
class Scenario:
    name: str
    speaker: bool = True        # laptop speaker coloration
    rt60: float = 0.0           # room echo (s)
    noise_db: float = -90.0     # background noise level (dBFS rms)
    levels_db: tuple = (-22.0,)  # per-sentence speech level (cycled) -> volume up/down
    swing_db: float = 0.0       # volume drift inside a sentence (± dB)


SCENARIOS = {
    "clean": Scenario("clean", speaker=False),
    "desk": Scenario("desk", speaker=True, rt60=0.35, noise_db=-55, levels_db=(-24.0, -30.0, -20.0, -34.0)),
    "hard": Scenario("hard", speaker=True, rt60=0.6, noise_db=-46, levels_db=(-26.0, -38.0, -22.0, -42.0, -30.0),
                     swing_db=8.0),
}


def make_meeting(clips: list[np.ndarray], scenario: Scenario, pauses_s: list[float] | None = None,
                 seed: int = 1) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Concatenate clips with pauses and apply the scenario. Returns (audio, [(start, end)] of each clip)."""
    rng = np.random.default_rng(seed)
    pauses_s = pauses_s or [0.9, 0.7, 1.4, 0.6, 1.1, 0.8, 1.6, 0.7]
    parts, spans, t = [np.zeros(int(SR * 1.0), np.float32)], [], 1.0
    for i, c in enumerate(clips):
        c = c.copy()
        if scenario.speaker:
            c = laptop_speaker(c)
        target = scenario.levels_db[i % len(scenario.levels_db)]
        c = c / (rms(c[np.abs(c) > 1e-4]) + 1e-9) * db_to_lin(target)
        if scenario.swing_db:
            env_t = np.linspace(0, 1, c.size)
            phase = rng.uniform(0, 2 * np.pi)
            env_db = scenario.swing_db * np.sin(2 * np.pi * env_t * rng.uniform(0.6, 1.4) + phase)
            c = c * (10 ** (env_db / 20)).astype(np.float32)
        # where speech actually starts/ends inside the clip (TTS adds silence)
        env = np.abs(c) > (np.max(np.abs(c)) * 0.02)
        nz = np.flatnonzero(env)
        v0, v1 = (nz[0], nz[-1]) if nz.size else (0, c.size)
        spans.append((t + v0 / SR, t + v1 / SR))
        parts.append(c)
        t += c.size / SR
        p = pauses_s[i % len(pauses_s)]
        parts.append(np.zeros(int(SR * p), np.float32))
        t += p
    parts.append(np.zeros(int(SR * 1.5), np.float32))
    x = np.concatenate(parts)
    if scenario.rt60 > 0:
        x = fftconvolve(x, room_ir(scenario.rt60, rng))[: x.size].astype(np.float32)
    if scenario.noise_db > -90:
        x = x + pink_noise(x.size, rng) * db_to_lin(scenario.noise_db)
    x = np.clip(x, -1, 1)
    return x.astype(np.float32), spans


# ------------------------------------------------------------------ scoring
def _words(s: str) -> list[str]:
    import re

    s = s.lower().replace("ß", "ss")
    s = re.sub(r"(?<=\d)[.,](?=\d)", "", s)       # 25.000 -> 25000
    s = re.sub(r"[^\w\s]", " ", s)
    return s.split()


def wer(ref: str, hyp: str) -> float:
    r, h = _words(ref), _words(hyp)
    if not r:
        return 0.0 if not h else 1.0
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=np.int32)
    d[:, 0] = np.arange(len(r) + 1)
    d[0, :] = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (r[i - 1] != h[j - 1]))
    return float(d[len(r), len(h)] / len(r))
