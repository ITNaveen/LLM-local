"""Command line:

    python -m livetranslator serve [--host 127.0.0.1] [--port 8765] [--open]
    python -m livetranslator selftest [--scenario clean,desk,hard]
    python -m livetranslator devices
    python -m livetranslator download          # fetch the speech + translation models
"""
from __future__ import annotations

import argparse
import logging
import logging.handlers
import os
import sys
import tempfile
import threading
import time
import webbrowser
from pathlib import Path


def setup_logging(verbose: bool = False) -> None:
    from .config import logs_dir

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    fh = logging.handlers.RotatingFileHandler(logs_dir() / "server.log", maxBytes=5_000_000, backupCount=3,
                                              encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    sh.setLevel(logging.INFO)
    root.addHandler(sh)
    for noisy in ("httpx", "httpcore", "uvicorn.access", "huggingface_hub", "numba"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ---------------------------------------------------------------- serve
def cmd_serve(a) -> int:
    import uvicorn

    from .server import create_app

    setup_logging(a.verbose)
    if a.token:
        os.environ["LT_TOKEN"] = a.token
    if a.host not in ("127.0.0.1", "localhost", "::1") and not os.environ.get("LT_TOKEN"):
        logging.getLogger("lt").warning(
            "Listening on %s without an access token - anyone on this network can open your transcripts. "
            "Consider --token <secret>.", a.host)
    app = create_app()
    scheme = "https" if a.ssl_certfile else "http"
    url = f"{scheme}://{'127.0.0.1' if a.host in ('0.0.0.0', '::') else a.host}:{a.port}/"
    if a.open:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    print(f"\n  Live Translator is running:  {url}\n  (Ctrl+C to stop)\n", flush=True)
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning", ssl_certfile=a.ssl_certfile,
                ssl_keyfile=a.ssl_keyfile, ws_max_size=4 * 1024 * 1024, timeout_graceful_shutdown=5)
    return 0


# ---------------------------------------------------------------- devices
def cmd_devices(_a) -> int:
    from .audio_io import list_input_devices

    devs = list_input_devices()
    if not devs:
        print("No input devices found.")
        return 1
    for d in devs:
        print(f"{'*' if d['default'] else ' '} [{d['index']}] {d['name']}  ({d['channels']} ch, {d['rate']} Hz)")
    return 0


# ---------------------------------------------------------------- download
def cmd_download(a) -> int:
    from .asr import resolve_hf_model
    from .config import ASR_MODELS, SettingsStore

    s = SettingsStore().settings
    backend = s.resolved_asr_backend()
    keys = [s.asr_model] if not a.all else list(ASR_MODELS)
    for key in keys:
        model = ASR_MODELS.get(key, {}).get(backend, key)
        print(f"Speech model: {model}")
        if backend == "mlx":
            print("  ->", resolve_hf_model(model))
        else:
            from faster_whisper import WhisperModel

            WhisperModel(model, device="auto", compute_type="default")
            print("  -> ok")
    if not a.skip_llm:
        from .server import ensure_ollama
        from .translate import OllamaTranslator

        ensure_ollama(s.ollama_url)
        t = OllamaTranslator(s.ollama_url, s.llm_model)
        h = t.health()
        if not h["running"]:
            print("Ollama is not running - install it from https://ollama.com and run this again.")
            return 1
        if h["model_ready"]:
            print(f"Translation model {s.llm_model}: already installed")
        else:
            print(f"Translation model {s.llm_model}: downloading…")
            last = [""]

            def prog(p):
                tot, done = p.get("total") or 0, p.get("completed") or 0
                msg = f"  {p.get('status', '')} {done * 100 // tot}%" if tot else f"  {p.get('status', '')}"
                if msg != last[0]:
                    last[0] = msg
                    print(msg, flush=True)

            t.pull(s.llm_model, prog)
            print("  -> ok")
    return 0


# ---------------------------------------------------------------- selftest
def cmd_selftest(a) -> int:
    """End-to-end test with synthetic German speech, through the real pipeline at real-time speed."""
    import numpy as np

    from . import simulate as sim
    from .asr import ScriptedASR
    from .audio_io import FileSource, write_wav
    from .config import SettingsStore
    from .pipeline import Pipeline
    from .storage import MeetingStore
    from .translate import OllamaTranslator

    logging.basicConfig(level=logging.WARNING)
    print("Live Translator self-test\n" + "=" * 60)
    if not sim.tts_available():
        print("No German text-to-speech voice available; cannot build test audio.")
        print("macOS: System Settings → Accessibility → Spoken Content → System Voice → Manage Voices → German (Anna).")
        return 1
    n = a.sentences or len(sim.SENTENCES)
    sents = sim.SENTENCES[:n]
    print(f"Synthesising {n} German sentences…", flush=True)
    clips = [sim.synthesize(de, i) for i, (de, _en) in enumerate(sents)]

    tmp = Path(tempfile.mkdtemp(prefix="lt-selftest-"))
    user = SettingsStore()
    cfg = SettingsStore(tmp / "settings.json")
    cfg.update({k: v for k, v in user.settings.to_dict().items() if k not in ("input_source",)})
    if a.llm:
        cfg.update({"llm_model": a.llm})
    if a.asr:
        cfg.update({"asr_model": a.asr})
    s = cfg.settings
    store = MeetingStore(tmp / "Meetings")

    events = []
    first_en_at, final_en_at = {}, {}

    def publish(ev):
        ev = dict(ev)
        ev["_t"] = time.monotonic()
        events.append(ev)
        if ev.get("type") == "tr":
            if ev.get("en") and ev["id"] not in first_en_at:
                first_en_at[ev["id"]] = ev["_t"]
            if ev.get("final"):
                final_en_at[ev["id"]] = ev["_t"]

    asr_factory = None
    if a.fake_asr:  # developer option: exercise the pipeline without a speech model
        cfg.update({"live_preview": False})
        texts = [de for de, _ in sents] * 5
        asr_factory = lambda _s: ScriptedASR(texts)  # noqa: E731
    translator = OllamaTranslator(s.ollama_url, s.llm_model, s.context_lines, s.topic, s.glossary_terms())
    h = translator.health()
    print(f"Speech model     : {s.resolved_asr_backend()} / {s.resolved_asr_model()}")
    print(f"Translation model: {s.llm_model}  (Ollama {'running' if h['running'] else 'NOT RUNNING'}"
          f"{', model installed' if h['model_ready'] else ', MODEL NOT INSTALLED' if h['running'] else ''})")
    pipe = Pipeline(cfg, store, publish, asr_factory=asr_factory, translator=translator)
    print("Loading models…", flush=True)
    t0 = time.monotonic()
    while pipe.asr_state["status"] == "loading" and time.monotonic() - t0 < 1800:
        time.sleep(0.2)
    if pipe.asr_state["status"] != "ready":
        print("Speech model failed:", pipe.asr_state["detail"])
        return 1
    print(f"  speech model ready in {pipe.asr_state.get('load_s')} s (warm-up {pipe.asr_state.get('warm_s')} s)")
    if h["running"] and h["model_ready"]:
        tw = time.monotonic()
        translator.warmup()
        print(f"  translation model ready in {time.monotonic() - tw:.1f} s")

    results = []
    scen_names = [x.strip() for x in a.scenario.split(",") if x.strip()]
    for sname in scen_names:
        scen = sim.SCENARIOS[sname]
        audio, spans = sim.make_meeting(clips, scen)
        wav = tmp / f"{sname}.wav"
        write_wav(str(wav), audio)
        print(f"\n--- scenario '{sname}': {audio.size / 16000:.0f} s of audio, played in real time…", flush=True)
        events.clear()
        first_en_at.clear()
        final_en_at.clear()
        end = threading.Event()
        translator.reset()
        play_t0 = [0.0]
        src = FileSource(str(wav), pipe.push_audio, speed=1.0, on_end=end.set)
        pipe.start(name=f"selftest {sname}", source=src)
        play_t0[0] = time.monotonic()
        end.wait(audio.size / 16000 + 30)
        pipe.stop(wait=True)
        pipe.wait_idle(120)
        lines = [e["line"] for e in events if e.get("type") == "line"]
        line_t = {e["line"]["id"]: e["_t"] for e in events if e.get("type") == "line"}
        finals = {e["id"]: e for e in events if e.get("type") == "tr" and e.get("final")}
        hyp_de = " ".join(ln["de"] for ln in lines)
        ref_de = " ".join(de for de, _ in sents)
        w = sim.wer(ref_de, hyp_de)
        # latency: from the end of each spoken sentence to German / English on screen
        lat_de, lat_en_first, lat_en_done = [], [], []
        for ln in lines:
            # match the line to the sentence whose end is closest after the line's start
            start = ln["offset"]
            cand = [sp for sp in spans if sp[0] - 0.5 <= start <= sp[1]]
            if not cand:
                continue
            spoken_end = play_t0[0] + cand[0][1]
            lat_de.append(line_t[ln["id"]] - spoken_end)
            if ln["id"] in first_en_at:
                lat_en_first.append(first_en_at[ln["id"]] - spoken_end)
            if ln["id"] in final_en_at:
                lat_en_done.append(final_en_at[ln["id"]] - spoken_end)
        print(f"{'':2}{'#':>2}  German heard → English")
        for ln in lines:
            fin = finals.get(ln["id"], {})
            print(f"  {ln['id']:>2}  DE: {ln['de']}")
            print(f"      EN: {fin.get('en') or '(no translation)'}")
        med = lambda v: float(np.median(v)) if v else float("nan")  # noqa: E731
        print(f"\n  German word error rate: {w * 100:.1f}%   ({len(lines)} lines for {len(sents)} sentences)")
        print(f"  Time after speaker stops → German on screen : median {med(lat_de):.2f} s")
        print(f"                           → English starts   : median {med(lat_en_first):.2f} s")
        print(f"                           → English complete : median {med(lat_en_done):.2f} s")
        results.append((sname, w, med(lat_de), med(lat_en_first), med(lat_en_done)))
    pipe.shutdown()

    print("\n" + "=" * 60 + "\nSummary")
    print(f"  {'scenario':<8} {'WER':>6} {'German':>8} {'EN starts':>10} {'EN done':>8}")
    for r in results:
        print(f"  {r[0]:<8} {r[1] * 100:>5.1f}% {r[2]:>7.2f}s {r[3]:>9.2f}s {r[4]:>7.2f}s")
    print(f"\n(Test files: {tmp})")
    ok = all(r[1] < 0.25 for r in results)
    # advice for this machine
    tips = []
    worst = max(results, key=lambda r: r[1]) if results else None
    de_lat = max((r[2] for r in results if r[2] == r[2]), default=0)
    en_lat = max((r[4] for r in results if r[4] == r[4]), default=0)
    if worst and worst[1] > 0.15:
        tips.append(f"Recognition errors are high in '{worst[0]}' ({worst[1] * 100:.0f}%). In real meetings: keep the "
                    "speaker close to this laptop's microphone and add names/terms to the glossary.")
    if de_lat > 1.6:
        tips.append("German lines are slow on this computer: use the 'turbo' speech model and close heavy apps.")
    if en_lat > 3.5:
        tips.append("English takes a while to complete: in Settings choose 'Gemma 3 4B' for ~3x faster translation.")
    if not tips and de_lat and de_lat < 1.0 and en_lat and en_lat < 2.5 and s.asr_model == "turbo":
        tips.append("This Mac has headroom: you can try 'Maximum accuracy (large-v3)' in Settings and run the "
                    "self-test again to compare.")
    if not h["running"] or not h["model_ready"]:
        tips.append("Translation was not tested: start Ollama and install the model (./install.sh), then run again.")
    if tips:
        print("\nAdvice:")
        for t in tips:
            print("  -", t)
    print("\nRESULT:", "PASS" if ok else "CHECK - accuracy lower than expected")
    return 0 if ok else 2


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="livetranslator")
    sub = ap.add_subparsers(dest="cmd")
    sp = sub.add_parser("serve", help="run the app")
    sp.add_argument("--host", default=os.environ.get("LT_HOST", "127.0.0.1"),
                    help="0.0.0.0 to allow other computers (use with --token)")
    sp.add_argument("--port", type=int, default=int(os.environ.get("LT_PORT", "8765")))
    sp.add_argument("--open", action="store_true", help="open the browser")
    sp.add_argument("--token", default=os.environ.get("LT_TOKEN", ""), help="access token for remote use")
    sp.add_argument("--ssl-certfile", default=os.environ.get("LT_SSL_CERT") or None)
    sp.add_argument("--ssl-keyfile", default=os.environ.get("LT_SSL_KEY") or None)
    sp.add_argument("-v", "--verbose", action="store_true")
    sub.add_parser("devices", help="list microphones")
    dp = sub.add_parser("download", help="download the models")
    dp.add_argument("--all", action="store_true", help="also the alternative speech model")
    dp.add_argument("--skip-llm", action="store_true")
    tp = sub.add_parser("selftest", help="measure accuracy and speed on this computer")
    tp.add_argument("--scenario", default="clean,desk,hard")
    tp.add_argument("--sentences", type=int, default=0)
    tp.add_argument("--asr", default="", help="speech model to test (turbo, large-v3, or a repo)")
    tp.add_argument("--llm", default="", help="Ollama model to test")
    tp.add_argument("--fake-asr", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.cmd == "serve":
        return cmd_serve(a)
    if a.cmd == "devices":
        return cmd_devices(a)
    if a.cmd == "download":
        return cmd_download(a)
    if a.cmd == "selftest":
        return cmd_selftest(a)
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
