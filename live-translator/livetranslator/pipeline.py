"""The live engine: audio in -> German lines -> English lines -> screen + disk.

Threads
-------
audio callback   -> _audio_q
processor        resample, rumble filter, voice detection, line splitting
                 -> _asr_q (finished lines)  /  partial slot (live preview)
ASR worker       Whisper; finished lines always go before previews
                 -> store (German saved immediately) -> _tr_q
translator       Ollama, streamed to the screen, then saved
health           watches Ollama every few seconds

Speech recognition of line N+1 runs while line N is being translated.
"""
from __future__ import annotations

import logging
import platform
import queue
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable

import numpy as np

from .asr import BaseASR, build_prompt, create_asr
from .audio_io import FileSource, MicSource, PushSource
from .config import ASR_MODELS, Settings, SettingsStore
from .dsp import level_segment
from .frontend import FrontEnd
from .segmenter import Segment
from .storage import Line, Meeting, MeetingStore
from .textclean import clean_transcript, similarity
from .translate import OllamaTranslator
from .vad import SileroVAD

log = logging.getLogger("lt.pipeline")


@dataclass
class Session:
    meeting: Meeting
    t0_wall: datetime          # wall clock when this audio stream started
    token: int
    offset_base: float = 0.0   # meeting time already recorded before this stream (when continuing)
    prev_text: str = ""
    stats: dict = field(default_factory=lambda: {"lines": 0, "asr_s": [], "tr_first_s": [], "tr_total_s": [],
                                                 "close_lag_s": []})


class Pipeline:
    def __init__(
        self,
        settings: SettingsStore,
        store: MeetingStore,
        publish: Callable[[dict], None],
        asr_factory: Callable[[Settings], BaseASR] | None = None,
        translator: OllamaTranslator | None = None,
        vad_factory: Callable[[], object] = SileroVAD,
    ):
        self.cfg = settings
        self.store = store
        self.publish = publish
        self.asr_factory = asr_factory or (
            lambda s: create_asr(s.resolved_asr_backend(), s.resolved_asr_model(), s.language))
        s = settings.settings
        self.translator = translator or OllamaTranslator(s.ollama_url, s.llm_model, s.context_lines, s.topic,
                                                         s.glossary_terms())
        self.vad_factory = vad_factory

        self.state = "idle"                     # idle | listening | stopping
        self.asr: BaseASR | None = None
        self.asr_state = {"status": "loading", "detail": "Loading speech model…", "model": "", "backend": ""}
        self.llm_state = {"running": False, "model_ready": False, "model": s.llm_model, "models": [], "checked": False}
        self.session: Session | None = None
        self.source = None
        self.source_name = ""
        self.partial = {"text": "", "start": None}
        self.level = {"db": -90.0, "speech": False}

        self._lock = threading.RLock()
        self._token = 0
        self._shutdown = threading.Event()
        self._audio_q: queue.Queue = queue.Queue(maxsize=3000)
        self._asr_q: queue.Queue = queue.Queue()
        self._tr_q: queue.Queue = queue.Queue()
        self._partial_slot = None
        self._partial_lock = threading.Lock()
        self._partial_ema = 0.5
        self._cur_seg_start = None
        self._seg_cfg_version = 0
        self._proc_thread: threading.Thread | None = None
        self._caffeinate = None
        self._llm_fail_noticed = False
        self._summary_busy = False
        self.on_session_end: Callable[[Session], None] | None = None

        self._threads = [
            threading.Thread(target=self._asr_loop, name="asr", daemon=True),
            threading.Thread(target=self._tr_loop, name="translate", daemon=True),
            threading.Thread(target=self._health_loop, name="health", daemon=True),
        ]
        for t in self._threads:
            t.start()

    # ================================================================ public
    @property
    def settings(self) -> Settings:
        return self.cfg.settings

    def status(self) -> dict:
        return {
            "type": "status",
            "state": self.state,
            "asr": dict(self.asr_state),
            "llm": {k: v for k, v in self.llm_state.items()},
            "meeting": self.session.meeting.meta() if self.session else None,
            "source": self.source_name,
        }

    def snapshot(self) -> dict:
        lines = []
        if self.session:
            lines = [self._line_dict(ln) for ln in self.session.meeting.lines]
        return {"type": "snapshot", "status": self.status(), "lines": lines, "partial": self.partial,
                "settings": self.settings.to_dict()}

    def start(self, name: str = "", continue_id: str | None = None, source=None) -> dict:
        with self._lock:
            if self.state != "idle":
                raise RuntimeError("Already listening")
            s = self.settings
            now = datetime.now()
            if continue_id:
                meeting = self.store.load(continue_id)
                if meeting is None:
                    raise RuntimeError("Meeting not found")
                self.store.reopen(meeting)
                if name:
                    self.store.rename(meeting, name)
                last_off = meeting.lines[-1].offset + 1.0 if meeting.lines else 0.0
                offset_base = max((now - meeting.started).total_seconds(), last_off)
                self.translator.seed_history([(ln.de, ln.en) for ln in meeting.lines if ln.ok][-s.context_lines:])
            else:
                meeting = self.store.create(name, now=now, extra={
                    "asr_model": s.resolved_asr_model(), "llm_model": s.llm_model})
                offset_base = 0.0
                self.translator.reset()
            self._token += 1
            sess = Session(meeting=meeting, t0_wall=now, token=self._token, offset_base=offset_base)
            if meeting.lines:
                sess.prev_text = meeting.lines[-1].de
            self.session = sess
            self.partial = {"text": "", "start": None}
            # drop stale audio
            while not self._audio_q.empty():
                try:
                    self._audio_q.get_nowait()
                except queue.Empty:
                    break
            src = source or self._make_source()
            self._proc_thread = threading.Thread(target=self._proc_loop, args=(sess,), name="processor", daemon=True)
            self._proc_thread.start()
            try:
                self.source_name = src.start()
            except Exception as e:
                self._audio_q.put((None, 0))
                self.session = None
                self.store.finish(meeting)
                raise RuntimeError(f"Could not open the microphone: {e}") from e
            self.source = src
            self.state = "listening"
            self._keep_awake(True)
        self.publish({"type": "session", "meeting": meeting.meta(),
                      "lines": [self._line_dict(ln) for ln in meeting.lines]})
        self.publish(self.status())
        self.publish({"type": "meetings_changed"})
        if not self.llm_state.get("model_ready"):
            threading.Thread(target=self._warm_llm, daemon=True).start()
        return meeting.meta()

    def stop(self, wait: bool = False) -> None:
        with self._lock:
            if self.state != "listening" or not self.session:
                return
            sess = self.session
            src, self.source = self.source, None
            self.state = "stopping"
        if src:
            src.stop()
        self._audio_q.put((None, 0))      # processor flushes the last line and exits
        self.publish(self.status())
        done = threading.Event()

        def finish():
            if self._proc_thread:
                self._proc_thread.join(timeout=10)
            barrier = threading.Event()
            self._asr_q.put(("barrier", barrier))
            barrier.wait(timeout=120)
            with self._lock:
                kept = self.store.finish(sess.meeting)
                if self.session is sess:
                    self.session = None
                self.state = "idle"
                self.partial = {"text": "", "start": None}
                self._keep_awake(False)
            self.publish({"type": "session_end", "meeting": sess.meeting.meta() if kept else None,
                          "removed": not kept})
            self.publish(self.status())
            self.publish({"type": "meetings_changed"})
            if self.on_session_end:
                self.on_session_end(sess)
            done.set()

        threading.Thread(target=finish, name="stopper", daemon=True).start()
        if wait:
            done.wait(timeout=180)

    def rename_current(self, name: str) -> dict | None:
        with self._lock:
            if not self.session:
                return None
            self.store.rename(self.session.meeting, name)
            meta = self.session.meeting.meta()
        self.publish({"type": "meeting", "meeting": meta})
        self.publish({"type": "meetings_changed"})
        return meta

    def push_audio(self, samples: np.ndarray, rate: int) -> None:
        try:
            self._audio_q.put_nowait((samples, rate))
        except queue.Full:
            log.warning("audio queue full - dropping audio")

    def apply_settings(self, old: Settings, new: Settings) -> None:
        if (old.asr_model, old.asr_backend, old.language) != (new.asr_model, new.asr_backend, new.language):
            self._asr_q.put(("reload", None))
        self.translator.configure(model=new.llm_model, context_lines=new.context_lines, topic=new.topic,
                                  terms=new.glossary_terms(), base_url=new.ollama_url)
        if old.llm_model != new.llm_model or old.ollama_url != new.ollama_url:
            self.llm_state.update({"model": new.llm_model, "model_ready": False})
            threading.Thread(target=self._refresh_llm, args=(True,), daemon=True).start()
        if (old.sensitivity, old.pause_ms, old.max_line_s) != (new.sensitivity, new.pause_ms, new.max_line_s):
            self._seg_cfg_version += 1
        if self.state == "listening" and (old.input_device != new.input_device or old.input_source != new.input_source):
            self._swap_source()

    def retranslate(self, meeting_id: str, line_id: int) -> bool:
        with self._lock:
            if self.session and self.session.meeting.id == meeting_id:
                sess = self.session
            else:
                m = self.store.load(meeting_id)
                if m is None:
                    return False
                sess = Session(meeting=m, t0_wall=m.started, token=-1)
        ln = next((x for x in sess.meeting.lines if x.id == line_id), None)
        if ln is None:
            return False
        self._tr_q.put(("line", sess, ln, False))
        return True

    def summarize(self, meeting_id: str) -> None:
        if self._summary_busy:
            raise RuntimeError("A summary is already being written")
        if self.session and self.session.meeting.id == meeting_id and self.state != "idle":
            raise RuntimeError("Stop the meeting first - the translator is busy with the live meeting")
        m = self.store.load(meeting_id)
        if m is None:
            raise RuntimeError("Meeting not found")
        if not m.lines:
            raise RuntimeError("This meeting has no lines yet")
        self._summary_busy = True

        def run():
            try:
                text = "\n".join(f"[{ln.time}] DE: {ln.de}\n           EN: {ln.en}" for ln in m.lines)
                last = [0.0]

                def on_delta(t):
                    if time.monotonic() - last[0] > 0.15:
                        last[0] = time.monotonic()
                        self.publish({"type": "summary", "id": meeting_id, "text": t, "done": False})

                out = self.translator.summarize(text, on_delta)
                self.store.save_summary(m, out)
                self.publish({"type": "summary", "id": meeting_id, "text": self.store.summary(m), "done": True})
                self.publish({"type": "meetings_changed"})
            except Exception as e:  # noqa: BLE001
                log.exception("summary failed")
                self.publish({"type": "summary", "id": meeting_id, "error": str(e), "done": True})
            finally:
                self._summary_busy = False

        threading.Thread(target=run, name="summary", daemon=True).start()

    def pull_model(self, model: str) -> None:
        def run():
            last = [0.0]

            def prog(p):
                if time.monotonic() - last[0] > 0.3 or p.get("status") == "success":
                    last[0] = time.monotonic()
                    self.publish({"type": "pull", "model": model, **{k: p.get(k) for k in ("status", "completed", "total")}})

            try:
                self.translator.pull(model, prog)
                self.publish({"type": "pull", "model": model, "status": "success", "done": True})
            except Exception as e:  # noqa: BLE001
                self.publish({"type": "pull", "model": model, "status": "error", "error": str(e), "done": True})
            self._refresh_llm(True)

        threading.Thread(target=run, name="pull", daemon=True).start()

    def shutdown(self) -> None:
        self.stop(wait=True)
        self._shutdown.set()
        self._keep_awake(False)

    def wait_idle(self, timeout: float = 60.0) -> bool:
        """Block until all queued lines have been recognised and translated (tests / self-test)."""
        ev = threading.Event()
        self._asr_q.put(("barrier", ev))
        return ev.wait(timeout)

    # ================================================================ sources
    def _make_source(self):
        s = self.settings
        if s.input_source == "browser":
            return PushSource(self.push_audio)
        return MicSource(s.input_device, self.push_audio)

    def _swap_source(self) -> None:
        with self._lock:
            old = self.source
            if old:
                old.stop()
            src = self._make_source()
            try:
                self.source_name = src.start()
                self.source = src
            except Exception as e:  # noqa: BLE001
                self.source = None
                self._notice("error", f"Could not switch the microphone: {e}")
        self.publish(self.status())

    def _keep_awake(self, on: bool) -> None:
        """macOS: keep the Mac and its screen awake while listening."""
        if platform.system() != "Darwin":
            return
        import os

        if on and self._caffeinate is None:
            try:
                self._caffeinate = subprocess.Popen(["caffeinate", "-di", "-w", str(os.getpid())])
            except Exception:  # noqa: BLE001
                self._caffeinate = None
        elif not on and self._caffeinate is not None:
            self._caffeinate.terminate()
            self._caffeinate = None

    # ================================================================ processor
    def _proc_loop(self, sess: Session) -> None:
        fe = FrontEnd(self.settings, self.vad_factory())
        seg = fe.seg
        cfg_version = self._seg_cfg_version
        last_lvl = 0.0
        last_partial_req = 0.0
        started = time.monotonic()
        silent_warned = False
        got_audio = False
        while True:
            try:
                x, rate = self._audio_q.get(timeout=0.25)
            except queue.Empty:
                if not got_audio and not silent_warned and time.monotonic() - started > 6:
                    silent_warned = True
                    if self.settings.input_source == "browser":
                        self._notice("warn", "Waiting for audio from the browser - click 'Send this browser's audio'.")
                    else:
                        self._notice("error", "No audio is arriving from the microphone.")
                if self.session is not sess:
                    break
                continue
            if x is None:
                break
            if x.size == 0:
                continue
            got_audio = True
            if cfg_version != self._seg_cfg_version:
                cfg_version = self._seg_cfg_version
                fe.configure(self.settings)
            for segment in fe.process(x, rate):
                self._asr_q.put(("final", sess, segment))
            self._cur_seg_start = seg.current_start

            now = time.monotonic()
            if now - last_lvl >= 0.1:
                db = fe.take_level_db()
                if db is not None:
                    self.level = {"db": round(db, 1), "speech": seg.in_speech}
                    self.publish({"type": "level", "db": round(db, 1), "speech": seg.in_speech,
                                  "boost": round(fe.boost_db(), 1)})
                last_lvl = now
            if not silent_warned and now - started > 4 and fe.peak == 0.0:
                silent_warned = True
                self._notice("error", "The microphone is completely silent. On a Mac: System Settings → Privacy & "
                                      "Security → Microphone → allow Terminal, then restart Live Translator.")
            # live preview of the line being spoken
            # (only while words are coming: when a pause starts, the final pass is due soon
            # and must not wait behind a preview)
            if (self.settings.live_preview and seg.in_speech and seg.current_voiced_s() >= 0.6
                    and seg.cur is not None and seg.cur.silence_run <= 2
                    and now - last_partial_req >= max(1.0, 2.2 * self._partial_ema) and self._asr_q.empty()):
                cur = seg.current_audio()
                if cur is not None:
                    audio, start = cur
                    with self._partial_lock:
                        self._partial_slot = (sess, start, audio)
                    last_partial_req = now
        for segment in fe.flush():
            self._asr_q.put(("final", sess, segment))
        self._cur_seg_start = None

    # ================================================================ ASR
    def _load_asr(self) -> None:
        s = self.settings
        model = s.resolved_asr_model()
        backend = s.resolved_asr_backend()
        label = ASR_MODELS.get(s.asr_model, {}).get("label", model)
        self.asr_state = {"status": "loading", "detail": f"Loading speech model ({label})… first start downloads it once.",
                          "model": model, "backend": backend}
        self.publish(self.status())
        t = time.perf_counter()
        try:
            asr = self.asr_factory(s)
            asr.load()
            warm = asr.warmup()
            self.asr = asr
            self.asr_state = {"status": "ready", "detail": f"{label} ready", "model": model, "backend": backend,
                              "load_s": round(time.perf_counter() - t, 1), "warm_s": round(warm, 2)}
            log.info("ASR ready: %s/%s in %.1fs (warm-up %.2fs)", backend, model, time.perf_counter() - t, warm)
        except Exception as e:  # noqa: BLE001
            log.exception("ASR load failed")
            self.asr = None
            self.asr_state = {"status": "error", "detail": f"Speech model failed to load: {e}", "model": model,
                              "backend": backend}
            self._notice("error", self.asr_state["detail"])
        self.publish(self.status())

    def _asr_loop(self) -> None:
        self._load_asr()
        while not self._shutdown.is_set():
            try:
                job = self._asr_q.get(timeout=0.05)
            except queue.Empty:
                job = None
            if job is None:
                with self._partial_lock:
                    pj, self._partial_slot = self._partial_slot, None
                if pj is not None:
                    self._run_partial(*pj)
                continue
            kind = job[0]
            try:
                if kind == "final":
                    self._run_final(job[1], job[2])
                elif kind == "reload":
                    self._load_asr()
                elif kind == "barrier":
                    self._tr_q.put(("barrier", job[1]))
            except Exception:  # noqa: BLE001
                log.exception("ASR job failed")

    def _terms(self) -> list[str]:
        return self.settings.glossary_terms()

    def _run_partial(self, sess: Session, start: float, audio: np.ndarray) -> None:
        if self.asr is None or sess is not self.session or self._cur_seg_start != start:
            return
        terms = self._terms()
        r = self.asr.transcribe(level_segment(audio), build_prompt(terms, sess.prev_text))
        self._partial_ema = 0.7 * self._partial_ema + 0.3 * r.elapsed
        text = clean_transcript(r.text, duration_s=len(audio) / 16000, avg_logprob=r.avg_logprob,
                                no_speech_prob=r.no_speech_prob, prompt_terms=", ".join(terms))
        if sess is self.session and self._cur_seg_start == start and text:
            self.partial = {"text": text, "start": start}
            self.publish({"type": "partial", "text": text, "start": start})

    def _run_final(self, sess: Session, seg: Segment) -> None:
        if self.asr is None:
            if self.asr_state.get("status") == "error":
                return
            # model still loading: this line waits (the loader runs on this thread, so we only get here on error)
            return
        terms = self._terms()
        audio = level_segment(seg.audio)
        dur = len(audio) / 16000
        r = self.asr.transcribe(audio, build_prompt(terms, sess.prev_text))
        text = clean_transcript(r.text, duration_s=dur, avg_logprob=r.avg_logprob,
                                no_speech_prob=r.no_speech_prob, prompt_terms=", ".join(terms))
        asr_s = r.elapsed
        if text and sess.prev_text and similarity(text, sess.prev_text) > 0.85:
            # Whisper may echo its prompt (the previous line); check without it
            r2 = self.asr.transcribe(audio, build_prompt(terms, ""))
            asr_s += r2.elapsed
            text = clean_transcript(r2.text, duration_s=dur, avg_logprob=r2.avg_logprob,
                                    no_speech_prob=r2.no_speech_prob, prompt_terms=", ".join(terms))
        if self.partial.get("start") is not None and self.partial["start"] <= seg.start + 0.05:
            self.partial = {"text": "", "start": None}
            self.publish({"type": "partial", "text": "", "start": None})
        if not text:
            return
        m = sess.meeting
        line_id = (m.lines[-1].id + 1) if m.lines else 1
        wall = sess.t0_wall + timedelta(seconds=seg.start)
        ln = Line(id=line_id, time=f"{wall:%H:%M:%S}", offset=sess.offset_base + seg.start, de=text, forced=seg.forced)
        self.store.add_line(m, ln)
        sess.prev_text = text
        sess.stats["lines"] += 1
        sess.stats["asr_s"].append(asr_s)
        ln_meta = {"asr_s": round(asr_s, 3), "speech_end": seg.end, "closed": seg.detected_at,
                   "t_line": time.monotonic()}
        self.publish({"type": "line", "line": self._line_dict(ln), "lat": {"asr_s": round(asr_s, 2)}})
        self._tr_q.put(("line", sess, ln, True, ln_meta))

    # ================================================================ translation
    def _tr_loop(self) -> None:
        while not self._shutdown.is_set():
            try:
                job = self._tr_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if job[0] == "barrier":
                job[1].set()
                continue
            try:
                self._translate_job(*job[1:])
            except Exception:  # noqa: BLE001
                log.exception("translation job failed")

    def _translate_job(self, sess: Session, ln: Line, record: bool, meta: dict | None = None) -> None:
        last = [0.0]

        def on_delta(t: str) -> None:
            now = time.monotonic()
            if now - last[0] >= 0.04:
                last[0] = now
                self.publish({"type": "tr", "id": ln.id, "meeting": sess.meeting.id, "en": t, "final": False})

        if self.llm_state.get("checked") and not self.llm_state.get("running"):
            # known to be down: don't make every line wait for a time-out; these
            # lines are translated automatically once Ollama is back
            from .translate import TranslationResult

            res = TranslationResult("", False, error="Ollama is not running")
        else:
            res = self.translator.translate(ln.de, on_delta, record=record)
            if not res.ok and not res.text:
                time.sleep(1.0)
                res = self.translator.translate(ln.de, on_delta, record=record)
        if res.ok:
            if self._llm_fail_noticed:
                self._llm_fail_noticed = False
                self._notice("info", "Translation is working again.")
        elif not self._llm_fail_noticed:
            self._llm_fail_noticed = True
            self._notice("error", f"Translation failed: {res.error}. Is Ollama running? German lines are still saved; "
                                  "click a line's ↻ to translate it again later.")
        self.store.set_translation(sess.meeting, ln.id, res.text, res.ok)
        lat = {"tr_first_s": round(res.first_token_s, 2), "tr_total_s": round(res.total_s, 2)}
        if meta:
            sess.stats["tr_first_s"].append(res.first_token_s)
            sess.stats["tr_total_s"].append(res.total_s)
            sess.stats["close_lag_s"].append(meta["closed"] - meta["speech_end"])
            lat["asr_s"] = meta["asr_s"]
        self.publish({"type": "tr", "id": ln.id, "meeting": sess.meeting.id, "en": res.text, "final": True,
                      "ok": res.ok, "lat": lat})

    # ================================================================ Ollama health
    def _warm_llm(self) -> None:
        try:
            self.translator.warmup()
        except Exception:  # noqa: BLE001
            pass
        self._refresh_llm(True)

    def _refresh_llm(self, force_publish: bool = False) -> None:
        h = self.translator.health()
        new = {"running": h["running"], "model_ready": h["model_ready"], "model": self.translator.model,
               "models": h.get("models", []), "checked": True}
        changed = any(self.llm_state.get(k) != new[k] for k in ("running", "model_ready", "model", "models"))
        recovered = new["running"] and new["model_ready"] and not (
            self.llm_state.get("running") and self.llm_state.get("model_ready")) and self.llm_state.get("checked")
        self.llm_state = new
        if recovered and self.session is not None:
            sess = self.session
            for ln in [x for x in sess.meeting.lines if x.done and not x.ok]:
                self._tr_q.put(("line", sess, ln, False))
        if changed or force_publish:
            self.publish(self.status())

    def _health_loop(self) -> None:
        warmed = False
        while not self._shutdown.is_set():
            self._refresh_llm()
            if self.llm_state["running"] and self.llm_state["model_ready"] and not warmed:
                warmed = True
                threading.Thread(target=self._warm_llm, daemon=True).start()
            if not self.llm_state["running"]:
                warmed = False
            self._shutdown.wait(3.0 if self.llm_state["running"] else 2.0)

    # ================================================================ helpers
    def _notice(self, level: str, text: str) -> None:
        log.log(logging.ERROR if level == "error" else logging.INFO, "notice: %s", text)
        self.publish({"type": "notice", "level": level, "text": text})

    @staticmethod
    def _line_dict(ln: Line) -> dict:
        return {"id": ln.id, "time": ln.time, "offset": round(ln.offset, 2), "de": ln.de, "en": ln.en,
                "ok": ln.ok, "done": ln.done, "forced": ln.forced}


def file_source(path: str, pipeline: Pipeline, speed: float = 1.0, on_end=None) -> FileSource:
    return FileSource(path, pipeline.push_audio, speed=speed, on_end=on_end)
