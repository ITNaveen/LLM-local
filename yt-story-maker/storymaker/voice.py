"""Stage 4 - Hindi voice-over. Free engines, best first:
  edge  : Microsoft Edge neural voices (hi-IN-MadhurNeural / SwaraNeural). Free, no key,
          needs internet.
  piper : fully offline neural TTS (download a Hindi .onnx voice once).
  say   : macOS built-in 'Lekha' Hindi voice. Offline, robotic.
  silent: placeholder (used in tests / when nothing else works)."""

import asyncio
import os
import platform
import re
from pathlib import Path

from .util import PipelineError, estimate_speech_seconds, ffmpeg, has_tool, media_duration, run

# A YouTuber's mic: trim silence, de-mud, presence, firm compression, level.
VOICE_CHAIN = ("silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.05,"
               "areverse,silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.15,"
               "areverse,highpass=f=75,equalizer=f=250:t=q:w=1:g=-2.5,"
               "equalizer=f=3200:t=q:w=1.2:g=3,equalizer=f=9000:t=q:w=1:g=1.5,"
               "acompressor=threshold=-22dB:ratio=4:attack=5:release=120:makeup=2,"
               "loudnorm=I=-16:TP=-2:LRA=6,aresample=48000")


def speech_parts(text, base_rate="+10%", base_pitch="+0Hz"):
    """Split a line into sentences, each with its own delivery: questions rise, exclamations hit
    harder and faster, '...' leaves a dramatic pause. -> [(sentence, rate, pitch, volume, pause)]."""
    def num(v, default=0):
        m = re.search(r"[-+]?\d+", str(v))
        return int(m.group()) if m else default
    rate, pitch = num(base_rate, 10), num(base_pitch)
    out = []
    for part in [x.strip() for x in re.split(r"(?<=[।!?])\s+|(?<=\.\.\.)\s*", text or "") if re.search(r"\w", x)]:
        if part.endswith("!"):
            r, pt, vol, pause = rate + 8, pitch + 3, 10, 0.12
        elif part.endswith("?"):
            r, pt, vol, pause = rate + 2, pitch + 7, 5, 0.22
        elif part.endswith("..."):
            r, pt, vol, pause = rate - 2, pitch, 0, 0.38
        else:
            r, pt, vol, pause = rate, pitch, 0, 0.10
        out.append((part, f"{r:+d}%", f"{pt:+d}Hz", f"{vol:+d}%", pause))
    return out


def _edge(text, out, settings):
    """Microsoft neural voice, sentence by sentence with varied energy and tight pauses - a
    person talking, not one flat read."""
    import edge_tts
    parts = speech_parts(text, settings.get("edge_rate", "+10%"), settings.get("edge_pitch", "+0Hz"))
    files = [Path(f"{out}.part{i}.mp3") for i in range(len(parts))]

    async def go():
        for (sentence, rate, pitch, volume, _pause), f in zip(parts, files):
            await edge_tts.Communicate(sentence, settings["edge_voice"], rate=rate, pitch=pitch,
                                       volume=volume).save(str(f))
    try:
        asyncio.run(go())
        trim = ("silenceremove=start_periods=1:start_threshold=-42dB:start_silence=0.02,areverse,"
                "silenceremove=start_periods=1:start_threshold=-42dB:start_silence=0.02,areverse")
        chains = ";".join(f"[{i}:a]aresample=48000,{trim},apad=pad_dur={parts[i][4]}[p{i}]"
                          for i in range(len(parts)))
        joined = "".join(f"[p{i}]" for i in range(len(parts)))
        inputs = [x for f in files for x in ("-i", str(f))]
        ffmpeg(*inputs, "-filter_complex", f"{chains};{joined}concat=n={len(parts)}:v=0:a=1[a]",
               "-map", "[a]", "-ac", "1", str(out))
    finally:
        for f in files:
            f.unlink(missing_ok=True)


def _say(text, out, settings):
    run(["say", "-v", settings["say_voice"], "-o", str(out), text], timeout=120)


def _piper(text, out, settings):
    model = settings.get("piper_model")
    if not model or not Path(model).exists():
        raise PipelineError("piper model not configured")
    exe = "piper" if has_tool("piper") else None
    if not exe:
        raise PipelineError("piper not installed (pip install piper-tts)")
    import subprocess
    proc = subprocess.run([exe, "--model", model, "--output_file", str(out)],
                          input=text, text=True, capture_output=True, timeout=180)
    if proc.returncode != 0:
        raise PipelineError(f"piper failed: {proc.stderr[-300:]}")


def _silent(text, out, settings):
    ffmpeg("-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono",
           "-t", f"{estimate_speech_seconds(text):.2f}", str(out))


# ------------------------------------------------------------------ emotional voice (Parler)
ROOT = Path(__file__).resolve().parent.parent
PARLER_PY = ROOT / ".venv-voice" / "bin" / "python"
PARLER_READY = ROOT / ".venv-voice" / "READY"
PARLER_MODEL = "models--ai4bharat--indic-parler-tts"
PARLER_SPEAKERS = {"male": "Rohit", "female": "Divya"}
UNLOCK_HELP = ("The emotional voice is installed but still locked: double-click "
               "install-emotional-voice.command again and follow its 2 steps (accept the model's "
               "free licence, paste a free Hugging Face token). Using the Microsoft voice for now.")


def parler_state():
    """'missing' (not installed), 'locked' (installed, model not downloaded yet - usually the
    Hugging Face licence wasn't accepted) or 'ready'."""
    if not PARLER_PY.exists():
        return "missing"
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub" / PARLER_MODEL
    if PARLER_READY.exists() or any(hub.glob("snapshots/*/*.safetensors")):
        return "ready"
    return "locked"


def parler_available():
    return parler_state() == "ready"


ANGRY = re.compile(r"!|गुस्सा|शर्म|बर्दाश्त|हिम्मत|धमकी|चेतावनी|बस करो|नहीं छोड़")


def parler_description(settings, text=""):
    """How the emotional voice should say this line (the model follows a description)."""
    speaker = settings.get("parler_speaker") or "Rohit"
    if settings.get("parler_style"):
        return settings["parler_style"].format(speaker=speaker)
    if ANGRY.search(text or ""):
        mood = "an angry, intense and powerful"
    elif "?" in (text or ""):
        mood = "a surprised, urgent and questioning"
    else:
        mood = "an excited, energetic and highly expressive"
    return (f"{speaker} speaks in {mood} tone, like a passionate Hindi YouTube news presenter, at "
            "a fast pace, with a very clear, close-sounding recording and no background noise.")


def parler_batch(lines, out_dir, settings, log):
    """Render all lines in one run of the emotional voice model (it loads once)."""
    import json
    import subprocess
    todo = {k: v for k, v in lines.items()
            if not (out_dir / f"{k}.wav").exists() and not (out_dir / f"{k}.raw.wav").exists()}
    if not todo:
        return
    job = out_dir / "parler_job.json"
    job.write_text(json.dumps({"lines": {k: {"text": v, "description": parler_description(settings, v)}
                                         for k, v in todo.items()},
                               "out_dir": str(out_dir), "description": parler_description(settings)},
                              ensure_ascii=False))
    log(f"Recording {len(todo)} lines with the emotional Hindi voice (this can take a while)...")
    proc = subprocess.Popen([str(PARLER_PY), str(ROOT / "storymaker" / "voice_parler.py"), str(job)],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    tail = []
    for line in proc.stdout:
        line = line.strip()
        tail = (tail + [line])[-8:]
        if line.startswith("PROGRESS") or line.startswith("Loading"):
            log(f"  {line.replace('PROGRESS', 'voice line')}")
    if proc.wait() != 0:
        msg = " | ".join(tail)[-300:]
        if re.search(r"gated|401|403|Unauthorized|restricted", msg, re.I):
            log(UNLOCK_HELP)
        raise PipelineError("emotional voice failed: " + msg)
    try:
        PARLER_READY.touch()
    except OSError:
        pass


def _parler(text, out, settings):
    if not out.exists():                    # made by parler_batch()
        raise PipelineError("emotional voice line missing")


ENGINES = {"parler": (_parler, "wav"), "edge": (_edge, "wav"), "piper": (_piper, "wav"),
           "say": (_say, "aiff"), "silent": (_silent, "wav")}


def engine_order(settings):
    choice = settings.get("tts_engine", "auto")
    if choice == "parler" and not parler_available():
        choice = "auto"                     # chosen but not ready: don't waste time loading it
    if choice != "auto":
        return [choice, "silent"]
    order = ["parler", "edge"] if parler_available() else ["edge"]
    if settings.get("piper_model"):
        order.append("piper")
    if platform.system() == "Darwin":
        order.append("say")
    order.append("silent")
    return order


def synthesize(lines, out_dir, settings, log):
    """lines: {narration_id: text}. Returns {id: {"file", "duration", "engine"}}."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results, order = {}, engine_order(settings)
    working = None
    if order[0] == "parler":
        try:
            parler_batch(lines, out_dir, settings, log)
        except Exception as e:  # noqa: BLE001 - fall back to the next voice
            log(f"  emotional voice failed ({str(e)[:200]}); using the Microsoft voice instead.")
            order = [e for e in order if e != "parler"]
    for nid, text in lines.items():
        final = out_dir / f"{nid}.wav"
        if final.exists():
            results[nid] = {"file": str(final), "duration": media_duration(final), "engine": "cached"}
            continue
        engines = [working] if working else order
        for name in engines + [e for e in order if e not in engines]:
            fn, ext = ENGINES[name]
            raw = out_dir / f"{nid}.raw.{ext}"
            try:
                fn(text, raw, settings)
                chain = VOICE_CHAIN if name != "silent" else "aresample=48000"
                ffmpeg("-i", str(raw), "-af", chain, "-ac", "1", str(final))
                raw.unlink(missing_ok=True)
                if working != name:
                    if name == "silent":
                        log("WARNING: no Hindi voice engine worked - narration will be silent. "
                            "Check your internet (Edge voice) or configure Piper in Settings.")
                    else:
                        log(f"Voice engine: {name}")
                working = name
                break
            except Exception as e:  # noqa: BLE001 - try the next engine
                if name != "parler":
                    raw.unlink(missing_ok=True)
                log(f"  voice engine '{name}' failed: {str(e)[:150]}")
        else:
            raise PipelineError("no voice engine could synthesize narration")
        results[nid] = {"file": str(final), "duration": media_duration(final), "engine": working}
    return results
