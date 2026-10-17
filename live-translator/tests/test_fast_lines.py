"""Lines appear sooner: the speech recogniser starts while the speaker's pause is still running,
and a pause after words that clearly continue ("..., dass", "und") does not cut the sentence."""
import threading
import time

import numpy as np
import pytest
from fake_ollama import FakeOllama

from livetranslator import simulate as sim
from livetranslator.asr import ASRResult, BaseASR
from livetranslator.audio_io import FileSource, write_wav
from livetranslator.config import SettingsStore
from livetranslator.pipeline import Pipeline
from livetranslator.segmenter import Segmenter
from livetranslator.storage import MeetingStore
from livetranslator.textclean import sounds_unfinished
from livetranslator.vad import FRAME


def wait_for(cond, timeout=30.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.05)
    return False


# ----------------------------------------------------------------- pieces
def test_sounds_unfinished():
    for t in ["Wir müssen das Angebot,", "Ich glaube, dass", "Das Budget und", "Wir brauchen das für",
              "Das ist ein Thema für den", "Also ähm", "Wir warten auf …", "Der Termin ist am"]:
        assert sounds_unfinished(t), t
    for t in ["Wir fangen morgen an.", "Das weiß ich.", "Das geht nicht.", "Ist das so?", "Ich mache das.",
              "Kommst du mit?", "Ich weiß nicht, ob das richtig ist.", "Ja.", ""]:
        assert not sounds_unfinished(t), t


def _feed(seg, pattern):
    """pattern: list of (prob, n_frames); returns [(frame index of close, segment)]."""
    out, i = [], 0
    rng = np.random.default_rng(0)
    for p, n in pattern:
        for _ in range(n):
            for s in seg.push((rng.standard_normal(FRAME) * 0.01).astype(np.float32), p):
                out.append((i, s))
            i += 1
    return out


def test_segmenter_advisor_can_hold_and_close():
    speech, pause = (0.9, 40), (0.0, 30)
    plain = _feed(Segmenter(pause_ms=500), [speech, pause])
    assert len(plain) == 1 and plain[0][0] == 40 + 16 - 1          # closes after the 500 ms pause

    held = Segmenter(pause_ms=500)
    held.advisor = lambda s, c, req: "wait" if c.silence_run < 25 else None
    out = _feed(held, [speech, (0.0, 20), speech, pause])           # 640 ms gap is bridged
    assert len(out) == 1

    early = Segmenter(pause_ms=500)
    early.advisor = lambda s, c, req: "close" if c.silence_run >= 3 else None
    out = _feed(early, [speech, pause])
    assert out[0][0] == 40 + early.post_pad - 1                     # never before the trailing pad is there
    assert len(out[0][1].audio) == (40 + early.post_pad) * FRAME


# ----------------------------------------------------------------- in the pipeline
def sentence_for(i, prompt=None):
    """Sentence number i (wrapping round) - skipping the one the prompt ends with."""
    n = len(sim.SENTENCES)
    for k in range(n):
        text = sim.SENTENCES[(i + k) % n][0]
        if not prompt or text[-80:] not in prompt:
            return text
    return text


class TimedASR(BaseASR):
    """Takes `delay` seconds per call; the text depends only on the audio (like a real model)."""
    name = "timed"

    def __init__(self, delay=0.35, texts=None):
        self.delay = delay
        self.texts = texts          # callable(seconds) -> text
        self.calls = []
        self.seen = {}

    def warmup(self):
        return 0.0

    def transcribe(self, audio, prompt=None):
        time.sleep(self.delay)
        secs = len(audio) / 16000
        self.calls.append(secs)
        # different audio -> the next sentence; like a real model it never repeats its prompt (the previous
        # line) - that would add the app's "echo of the prompt" re-check and make the timing random
        if self.texts:
            text = self.texts(secs)
        else:
            key = round(secs, 3)
            self.seen.setdefault(key, len(self.seen))
            text = sentence_for(self.seen[key], prompt)
        return ASRResult(text, -0.2, 0.01, 1.0, self.delay)


@pytest.fixture(scope="module")
def three_lines(tmp_path_factory):
    if sim.tts_available() is None:
        pytest.skip("needs TTS")
    clips = [sim.synthesize(de, i) for i, (de, _) in enumerate(sim.SENTENCES[:3])]
    audio, spans = sim.make_meeting(clips, sim.SCENARIOS["clean"], pauses_s=[1.3, 1.3, 1.3])
    p = tmp_path_factory.mktemp("w") / "m.wav"
    write_wav(str(p), audio)
    return str(p), spans


def run(tmp_path, wav, asr, early, preview=True):
    """A real-time meeting (live preview on by default); `early` = transcribe during the pause."""
    with FakeOllama(models=["gemma3:4b"], token_delay=0.005) as fo:
        cfg = SettingsStore(tmp_path / f"s{early}{preview}.json")
        cfg.update({"ollama_url": fo.url, "llm_model": "gemma3:4b", "llm_fallback": "gemma3:4b",
                    "live_preview": preview})
        events = []
        pipe = Pipeline(cfg, MeetingStore(tmp_path / f"M{early}{preview}"),
                        lambda e: events.append({**e, "_t": time.monotonic()}), asr_factory=lambda s: asr)
        pipe.early_enabled = early
        assert wait_for(lambda: pipe.asr_state["status"] == "ready" and pipe.llm_state.get("loaded"))
        done = threading.Event()
        t0 = time.monotonic()
        pipe.start(name="t", source=FileSource(wav, pipe.push_audio, speed=1.0, on_end=done.set))
        sess = pipe.session
        assert done.wait(60)
        pipe.stop(wait=True)
        stats = sess.stats
        pipe.shutdown()
        lines = [e for e in events if e["type"] == "line"]
        return t0, lines, events, stats


def _lags(tmp_path, three_lines, preview):
    wav, spans = three_lines
    lag = {}
    for early in (False, True):
        asr = TimedASR(delay=0.35)
        t0, lines, events, stats = run(tmp_path, wav, asr, early, preview)
        assert len(lines) == 3, [e["line"]["de"] for e in lines]
        lag[early] = [e["_t"] - (t0 + end) for e, (_, end) in zip(lines, spans)]
        if early:      # every line was transcribed during its pause - no second pass after it closed
            assert stats["early_hits"] == 3, (stats, asr.calls)
    gains = [a - b for a, b in zip(lag[False], lag[True])]
    print("lag", lag, "gains", gains)
    return lag, gains


def test_lines_appear_sooner_when_transcribed_during_the_pause(tmp_path, three_lines):
    # without previews the gain is measured cleanly: normal lines win most of the pause (~0.3 s); the
    # long middle line is near the length limit, closes after a shorter pause and wins little
    lag, gains = _lags(tmp_path, three_lines, preview=False)
    assert gains[0] > 0.15 and gains[2] > 0.15 and gains[1] > -0.08, f"lines were not faster: {lag}"


def test_the_early_pass_never_makes_lines_slower_with_previews(tmp_path, three_lines):
    # a (Whisper) preview pass running when the speaker stops delays the line in both modes alike - by a
    # random part of a pass - so only the average is compared
    lag, gains = _lags(tmp_path, three_lines, preview=True)
    assert np.mean(gains) > 0.0, f"lines got slower: {lag}"


@pytest.mark.parametrize("delay", [0.15, 0.35])      # 0.35 s: the early text is late - the live preview decides
def test_an_unfinished_sentence_is_not_cut_at_a_breath(tmp_path, delay):
    if sim.tts_available() is None:
        pytest.skip("needs TTS")
    a = sim.synthesize("Wir müssen das Angebot für den Kunden und", 0)
    b = sim.synthesize("die Präsentation bis Freitag fertig haben.", 0)
    # the TTS clip ends with ~0.4 s of its own silence: the breath is ~0.65 s - more than the 0.5 s pause
    audio, spans = sim.make_meeting([a, b], sim.SCENARIOS["clean"], pauses_s=[0.25, 1.5])
    wav = tmp_path / "breath.wav"
    write_wav(str(wav), audio)
    first_part_s = spans[0][1] - spans[0][0]

    def texts(secs):
        if secs < first_part_s + 0.6:
            return "Wir müssen das Angebot für den Kunden und"
        return "Wir müssen das Angebot für den Kunden und die Präsentation bis Freitag fertig haben."

    counts = {}
    for early in (False, True):
        _, lines, _, _ = run(tmp_path, str(wav), TimedASR(delay=delay, texts=texts), early)
        counts[early] = [e["line"]["de"] for e in lines]
    assert len(counts[False]) == 2, counts        # without the early text: cut after "und"
    assert len(counts[True]) == 1, counts         # with it: one complete sentence
    assert counts[True][0].endswith("fertig haben.")


def test_a_slow_early_pass_never_joins_two_sentences(tmp_path):
    """A pause just over the normal 0.5 s between two finished sentences: two lines, as without the
    early pass - even when the early text is not ready yet when the pause is over."""
    clips = [sim.synthesize("Das Angebot ist fertig.", 0), sim.synthesize("Wir schicken es morgen.", 0)]
    audio, spans = sim.make_meeting(clips, sim.SCENARIOS["clean"], pauses_s=[0.2, 1.5])     # a 0.6 s gap
    wav = tmp_path / "two.wav"
    write_wav(str(wav), audio)
    counts = {}
    for early in (False, True):
        t0, lines, events, stats = run(tmp_path, str(wav), TimedASR(delay=1.0), early)
        counts[early] = len(lines)
    assert counts[False] == 2 and counts[True] == 2, counts
