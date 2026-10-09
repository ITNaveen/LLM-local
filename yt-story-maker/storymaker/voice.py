"""Stage 4 - Hindi voice-over. Free engines, best first:
  edge  : Microsoft Edge neural voices (hi-IN-MadhurNeural / SwaraNeural). Free, no key,
          needs internet.
  piper : fully offline neural TTS (download a Hindi .onnx voice once).
  say   : macOS built-in 'Lekha' Hindi voice. Offline, robotic.
  silent: placeholder (used in tests / when nothing else works)."""

import asyncio
import platform
from pathlib import Path

from .util import PipelineError, estimate_speech_seconds, ffmpeg, has_tool, media_duration, run

# Warm, broadcast-style narration chain: trim silence, de-mud, gentle compression, level.
VOICE_CHAIN = ("silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.05,"
               "areverse,silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.15,"
               "areverse,highpass=f=70,equalizer=f=250:t=q:w=1:g=-2,"
               "equalizer=f=3500:t=q:w=1.2:g=2.5,"
               "acompressor=threshold=-20dB:ratio=3:attack=10:release=150,"
               "loudnorm=I=-16:TP=-2:LRA=7,aresample=48000")


def _edge(text, out, settings):
    import edge_tts

    async def go():
        comm = edge_tts.Communicate(text, settings["edge_voice"], rate=settings["edge_rate"],
                                    pitch=settings["edge_pitch"])
        await comm.save(str(out))
    asyncio.run(go())


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
PARLER_SPEAKERS = {"male": "Rohit", "female": "Divya"}


def parler_available():
    return PARLER_PY.exists()


def parler_description(settings):
    speaker = settings.get("parler_speaker") or "Rohit"
    return (settings.get("parler_style") or
            "{speaker} speaks in an excited, energetic and highly expressive tone, like a passionate "
            "YouTube presenter, at a moderately fast pace, with a very clear, close-sounding "
            "recording and no background noise.").format(speaker=speaker)


def parler_batch(lines, out_dir, settings, log):
    """Render all lines in one run of the emotional voice model (it loads once)."""
    import json
    import subprocess
    todo = {k: v for k, v in lines.items()
            if not (out_dir / f"{k}.wav").exists() and not (out_dir / f"{k}.raw.wav").exists()}
    if not todo:
        return
    job = out_dir / "parler_job.json"
    job.write_text(json.dumps({"lines": todo, "out_dir": str(out_dir),
                               "description": parler_description(settings)}, ensure_ascii=False))
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
        raise PipelineError("emotional voice failed: " + " | ".join(tail)[-300:])


def _parler(text, out, settings):
    if not out.exists():                    # made by parler_batch()
        raise PipelineError("emotional voice line missing")


ENGINES = {"parler": (_parler, "wav"), "edge": (_edge, "mp3"), "piper": (_piper, "wav"),
           "say": (_say, "aiff"), "silent": (_silent, "wav")}


def engine_order(settings):
    choice = settings.get("tts_engine", "auto")
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
