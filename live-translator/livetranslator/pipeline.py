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
from .audio_io import (FileSource, MicSource, PushSource, list_input_devices, mic_permission_status, preferred_order,
                       probe_device, request_mic_permission)
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
        self.llm_state = {"running": False, "model_ready": False, "model": s.llm_model, "models": [], "checked": False,
                          "loaded": False, "gpu_share": None, "tok_s": None, "first_s": None, "speed": "unknown",
                          "queue": 0, "busy": False, "fallback_active": False}
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
        self._partial_tr_slot = None        # (session, line start, German so far) for the English preview
        self._partial_en_last = (None, 0.0, "")   # (line start, time, German) of the last preview
        self._partial_lock = threading.Lock()
        self._partial_ema = 0.5
        self._cur_seg_start = None
        self._seg_cfg_version = 0
        self._proc_thread: threading.Thread | None = None
        self._caffeinate = None
        self._llm_fail_noticed = False
        self._summary_busy = False
        self._recovery_done_at = 0.0
        self._mic_blocked = False   # macOS blocked every microphone this run -> go straight to the browser
        self._source_gen = 0        # bumped whenever the audio source is replaced; cancels a stale recovery
        self._tr_busy = False       # translator is working on a line right now
        self._slow_strikes = 0      # consecutive lines on which the translation model was too slow
        self._fallback_lock = threading.Lock()
        self._retry_at: dict = {}   # (meeting id, line id) -> last automatic retry
        self._queued: set = set()   # (meeting id, line id) queued or being translated by a retry
        self._queued_lock = threading.Lock()
        self._replaced_model = ""   # model replaced by an automatic fallback (unloaded when seen)
        self._auto_pulled = False
        self._last_session: Session | None = None   # just stopped: its missing translations are still filled in
        self._last_session_end = 0.0
        self._pulling: set = set()
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
            "source_kind": self._source_kind(),
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
            self._source_gen += 1
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
            via_browser_fallback = source is None and self.settings.input_source == "mic" and self._mic_blocked
            if self.source is not None:      # defensive: never two sources at once
                stale, self.source = self.source, None
                stale.stop()
            src = source or (PushSource(self.push_audio) if via_browser_fallback else self._make_source())
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
            if via_browser_fallback:
                self.source_name = "This browser's microphone"
            self.state = "listening"
            self._keep_awake(True)
        self.publish({"type": "session", "meeting": meeting.meta(),
                      "lines": [self._line_dict(ln) for ln in meeting.lines]})
        self.publish(self.status())
        if via_browser_fallback:
            self.publish({"type": "browser_audio_needed", "reason": "permission"})
        self.publish({"type": "meetings_changed"})
        if not self.llm_state.get("model_ready") or not self.llm_state.get("loaded"):
            threading.Thread(target=self._warm_llm, daemon=True).start()
        else:
            threading.Thread(target=self._free_ollama_memory, daemon=True).start()
        return meeting.meta()

    def stop(self, wait: bool = False) -> None:
        with self._lock:
            if self.state != "listening" or not self.session:
                return
            sess = self.session
            src, self.source = self.source, None
            self._source_gen += 1
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
                if kept:
                    self._last_session, self._last_session_end = sess, time.monotonic()
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
        model_changed = old.llm_model != new.llm_model or old.ollama_url != new.ollama_url
        # only a real model change replaces the active model (keeps an automatic fallback otherwise)
        self.translator.configure(model=new.llm_model if model_changed else None, context_lines=new.context_lines,
                                  topic=new.topic, terms=new.glossary_terms(), base_url=new.ollama_url)
        if model_changed:
            self._slow_strikes = 0
            self.llm_state.update({"model": new.llm_model, "model_ready": False, "fallback_active": False,
                                   "speed": "unknown", "tok_s": None, "slow_noticed": False, "cpu_noticed": False})
            threading.Thread(target=self._refresh_llm, args=(True,), daemon=True).start()
        if (old.sensitivity, old.pause_ms, old.max_line_s) != (new.sensitivity, new.pause_ms, new.max_line_s):
            self._seg_cfg_version += 1
        if old.input_device != new.input_device or old.input_source != new.input_source:
            self._mic_blocked = False
            if self.state == "listening":
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
        self._tr_q.put(("line", sess, ln, False, None, "user"))
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
        self._pulling.add(model)

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
            finally:
                self._pulling.discard(model)
            self._refresh_llm(True)

        threading.Thread(target=run, name="pull", daemon=True).start()

    def shutdown(self) -> None:
        self.stop(wait=True)
        self._shutdown.set()
        self._keep_awake(False)
        for m in {self.translator.model, self.settings.llm_model, self.settings.llm_fallback}:
            if m:
                self.translator.unload(m)   # give the memory back to the Mac right away

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
            if self.state != "listening" or self.session is None:
                return      # Stop raced with the Settings change: nothing to swap
            self._source_gen += 1
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
        first_audio_t = None
        recovery_started = False
        self._recovery_done_at = 0.0
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
            if first_audio_t is None:
                first_audio_t = time.monotonic()
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
            # Exact digital silence never comes from a working microphone (a quiet room still
            # has noise): the input is blocked by macOS or is a silent/virtual device. Fix it.
            if (not recovery_started and first_audio_t is not None and now - first_audio_t > 2.0
                    and fe.peak == 0.0 and isinstance(self.source, MicSource)):
                recovery_started = True
                threading.Thread(target=self._recover_silent_mic, args=(sess,), name="mic-recovery",
                                 daemon=True).start()
            if (recovery_started and not silent_warned and self._recovery_done_at
                    and time.monotonic() - self._recovery_done_at > 8 and fe.peak == 0.0):
                silent_warned = True
                self._notice("error", "Still no sound from any microphone. Check System Settings → Sound → Input: "
                                      "pick the MacBook microphone and raise the input volume. Then Stop and Start.")
            # live preview of the line being spoken
            # (only while words are coming: when a pause starts, the final pass is due soon
            # and must not wait behind a preview)
            if (self.settings.live_preview and seg.in_speech and seg.current_voiced_s() >= 0.6
                    and seg.cur is not None and seg.cur.silence_run <= 2
                    and now - last_partial_req >= max(1.0, 2.2 * self._partial_ema) and self._asr_q.empty()
                    and not self._tr_busy and self._tr_q.empty()):   # English first: no preview while translating
                cur = seg.current_audio()
                if cur is not None:
                    audio, start = cur
                    with self._partial_lock:
                        self._partial_slot = (sess, start, audio)
                    last_partial_req = now
        for segment in fe.flush():
            self._asr_q.put(("final", sess, segment))
        self._cur_seg_start = None

    def _source_kind(self) -> str:
        src = self.source
        if isinstance(src, PushSource):
            return "browser"
        if isinstance(src, MicSource):
            return "mic"
        return "file" if src is not None else ""

    def _recover_silent_mic(self, sess: Session) -> None:
        """The microphone delivers digital silence. In order:
        1. reopen it (fixes a macOS permission that was granted after recording started),
        2. try the other inputs - built-in microphone first, virtual devices last,
        3. otherwise capture in the browser, which asks for permission itself."""
        with self._lock:
            gen = self._source_gen

        def live() -> bool:
            # still the same meeting, and nobody (Stop, Settings) replaced the source meanwhile
            return self.session is sess and self.state == "listening" and self._source_gen == gen

        with self._lock:
            if not live() or not isinstance(self.source, MicSource):
                return
            old, old_name = self.source, self.source_name
            self.source = None
        old.stop()
        log.warning("microphone %r delivers digital silence - recovering", old_name)
        self.publish({"type": "mic_check", "state": "checking", "device": old_name})
        self._notice("warn", f"No sound from '{old_name}' - checking the microphones…")
        try:
            perm = mic_permission_status()
            if perm == "not_determined":
                perm = request_mic_permission(timeout=45)
            found, tried = None, []
            if not live():
                return
            if perm not in ("denied", "restricted"):
                time.sleep(0.3)
                if not live():
                    return
                devices = list_input_devices(refresh=True)
                order = [d for d in devices if d["name"] == old_name] + preferred_order(devices, exclude=old_name)
                for d in order:
                    if not live():
                        return
                    r = probe_device(d)
                    tried.append(r)
                    log.info("probe %s: ok=%s peak=%s %s", d["name"], r["ok"], r["peak_db"], r["error"])
                    if r["ok"]:
                        found = d
                        break
            with self._lock:
                if not live() or self.source is not None:
                    return      # stopped, or the user picked another input meanwhile
                if found:
                    src = MicSource(found["name"], self.push_audio)
                    try:
                        self.source_name = src.start()
                        self.source = src
                    except Exception as e:  # noqa: BLE001
                        log.warning("could not open %s: %s", found["name"], e)
                        found = None
                if not found:
                    push = PushSource(self.push_audio)
                    push.start()
                    self.source = push
                    self.source_name = "This browser's microphone"
            if found:
                if found["name"] != old_name:
                    self.cfg.update({"input_device": found["name"]})
                    self.publish({"type": "settings", "settings": self.settings.to_dict()})
                    self._notice("ok", f"Switched to '{found['name']}' ('{old_name}' gave no sound). "
                                       "This microphone is now remembered in Settings.")
                else:
                    self._notice("ok", f"'{found['name']}' works now.")
            else:
                all_silent = bool(tried) and not any(r["ok"] for r in tried)
                blocked = perm in ("denied", "restricted") or (perm != "authorized" and all_silent)
                self._mic_blocked = bool(blocked or all_silent)
                self.publish({"type": "browser_audio_needed", "reason": "permission" if blocked else "silent",
                              "permission": perm})
                if perm == "authorized":
                    self._notice("warn", "macOS allows the microphone, but every input is silent (check System "
                                         "Settings → Sound → Input: device and input volume). Using this browser's "
                                         "microphone for now - click Allow if the browser asks.")
                else:
                    self._notice("warn", "macOS gives Live Translator no microphone sound (System Settings → Privacy "
                                         "& Security → Microphone → Terminal). Using this browser's microphone "
                                         "instead - click Allow if the browser asks. To use the Mac microphone: turn "
                                         "Terminal on there, double-click the Desktop icon twice (stop, start).")
        finally:
            self._recovery_done_at = time.monotonic()
            self.publish(self.status())

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
            self._maybe_preview_english(sess, start, text, len(audio) / 16000)

    # English for a line that is still being spoken (long sentences): shown in grey, replaced by
    # the real translation when the speaker pauses. Only when the translator has nothing else to do.
    PREVIEW_EN_MIN_S = 4.0
    PREVIEW_EN_EVERY_S = 2.5

    def _maybe_preview_english(self, sess: Session, start: float, text: str, seconds: float) -> None:
        if seconds < self.PREVIEW_EN_MIN_S or not self.llm_state.get("loaded") or self.llm_state.get("stuck"):
            return
        last_start, last_t, last_text = self._partial_en_last
        now = time.monotonic()
        if last_start == start and (now - last_t < self.PREVIEW_EN_EVERY_S or len(text) - len(last_text) < 12):
            return
        self._partial_en_last = (start, now, text)
        self._partial_tr_slot = (sess, start, text)

    def _run_preview_english(self, sess: Session, start: float, text: str) -> None:
        if sess is not self.session or self._cur_seg_start != start:
            return      # the line is finished already: its real translation is on the way
        res = self.translator.translate(text, record=False, first_token_timeout=8.0, max_gen_s=6.0)
        if res.ok and res.text and sess is self.session and self._cur_seg_start == start:
            self.publish({"type": "partial_en", "text": res.text, "start": start})

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
    def _enqueue(self, sess: Session, ln: Line, origin: str) -> None:
        """Queue a line for (re)translation; a line is never queued twice at the same time."""
        key = (sess.meeting.id, ln.id)
        with self._queued_lock:
            if key in self._queued:
                return
            self._queued.add(key)
        self._tr_q.put(("line", sess, ln, False, None, origin))
        self._safe_set_llm(queue=self._lines_waiting())

    def _tr_loop(self) -> None:
        while not self._shutdown.is_set():
            try:
                job = self._tr_q.get(timeout=0.2)
            except queue.Empty:
                pj, self._partial_tr_slot = self._partial_tr_slot, None
                if pj is not None:
                    try:
                        self._run_preview_english(*pj)
                    except Exception:  # noqa: BLE001
                        log.exception("English preview failed")
                continue
            if job[0] == "barrier":
                job[1].set()
                self._safe_set_llm(queue=self._lines_waiting())
                continue
            self._tr_busy = True
            try:
                self._safe_set_llm(busy=True, queue=self._lines_waiting())
                self._translate_job(*job[1:])
            except Exception:  # noqa: BLE001
                log.exception("translation job failed")
            finally:
                with self._queued_lock:
                    self._queued.discard((job[1].meeting.id, job[2].id))
                self._tr_busy = False
                self._safe_set_llm(busy=False, queue=self._lines_waiting())

    # How long to wait for the first English word before giving up on a request.
    # A model that is already in memory answers in well under a second on an M-series
    # Mac; loading Gemma 12B from disk takes ~5-20 s (much longer when memory is full).
    WAIT_LOADED_S = 25.0
    WAIT_LOADING_S = 150.0
    WAIT_SWITCH_S = 20.0    # a model still not loaded after this, with a fast model available: switch
    # "Too slow": English starts more than this long after the line, or is written slower than
    # this (tokens/s; reading speed is ~4-6), on two lines in a row -> switch to the fast model.
    SLOW_FIRST_S = 8.0
    SLOW_TOK_S = 5.0

    def _translate_job(self, sess: Session, ln: Line, record: bool, meta: dict | None = None,
                       origin: str = "live") -> None:
        from .translate import TranslationResult

        if origin == "auto" and ln.done and ln.ok:
            return      # translated meanwhile: an automatic retry must never overwrite a good line
        last = [0.0]

        def on_delta(t: str) -> None:
            now = time.monotonic()
            if now - last[0] >= 0.04:
                last[0] = now
                self.publish({"type": "tr", "id": ln.id, "meeting": sess.meeting.id, "en": t, "final": False})

        if self.llm_state.get("checked") and not self.llm_state.get("running"):
            # known to be down: don't make every line wait for a time-out; these
            # lines are translated automatically once Ollama is back
            res = TranslationResult("", False, error="Ollama is not running")
        else:
            res = self._translate_with_watchdog(ln.de, on_delta, record)
        if not res.ok and ln.done and ln.ok and origin != "live":
            return      # keep the good translation a line already has
        if res.ok:
            if self._llm_fail_noticed:
                self._llm_fail_noticed = False
                self._notice("info", "Translation is working again.")
            self._safe_set_llm(stuck=False)
        else:
            self._safe_set_llm(stuck=True)
            if not self._llm_fail_noticed:
                self._llm_fail_noticed = True
                hint = ("Ollama did not answer in time." if res.timeout else f"{res.error}.")
                self._notice("error", f"Translation failed: {hint} The German is saved; the app keeps retrying "
                                      "these lines automatically (or click a line's ↻).")
        self.store.set_translation(sess.meeting, ln.id, res.text, res.ok)
        lat = {"tr_first_s": round(res.first_token_s, 2), "tr_total_s": round(res.total_s, 2)}
        if res.stats:
            lat["tok_s"] = res.stats.get("tok_s")
        if meta:
            sess.stats["tr_first_s"].append(res.first_token_s)
            sess.stats["tr_total_s"].append(res.total_s)
            sess.stats["close_lag_s"].append(meta["closed"] - meta["speech_end"])
            lat["asr_s"] = meta["asr_s"]
        self.publish({"type": "tr", "id": ln.id, "meeting": sess.meeting.id, "en": res.text, "final": True,
                      "ok": res.ok, "lat": lat})

    def _translate_with_watchdog(self, german: str, on_delta, record: bool):
        """One translation with a time limit, speed bookkeeping and automatic fallback.
        Every decision is about the model this request actually used."""
        model = self.translator.model
        loaded = self._model_loaded(model)
        limit = (self.WAIT_LOADED_S if loaded
                 else self.WAIT_SWITCH_S if self._fallback_usable(model) else self.WAIT_LOADING_S)
        res = self.translator.translate(german, on_delta, record=record, first_token_timeout=limit, model=model)
        if res.ok:
            self._record_speed(res, loaded, model)
            return res
        if res.timeout or res.truncated:
            # nothing (or almost nothing) came back in time: stuck loading / out of memory / far too slow
            self._slow_strikes = 2
            why = (f"{model} did not answer within {limit:.0f} s"
                   + ("" if loaded else " (still loading - not enough free memory?)")) if res.timeout \
                else f"{model} writes far too slowly on this Mac right now"
            if self._maybe_fallback(why, model):
                return self.translator.translate(german, on_delta, record=record,
                                                 first_token_timeout=self.WAIT_LOADING_S)
            return res
        # an error (e.g. the model runner crashed / out of memory): one retry, then count it as a strike
        time.sleep(1.0)
        cur = self.translator.model
        res = self.translator.translate(german, on_delta, record=record, model=cur,
                                        first_token_timeout=self.WAIT_LOADED_S if self._model_loaded(cur)
                                        else self.WAIT_LOADING_S)
        if res.ok:
            self._record_speed(res, False, cur)
        else:
            self._slow_strikes += 1
            if self._slow_strikes >= 2:
                self._maybe_fallback(f"{cur} keeps failing ({res.error})", cur)
        return res

    def _record_speed(self, res, was_loaded: bool, model: str) -> None:
        if model != self.translator.model:
            return      # the model was replaced meanwhile: its numbers no longer matter
        tok_s = res.stats.get("tok_s") or None
        self._safe_set_llm(tok_s=tok_s, first_s=round(res.first_token_s, 2))
        if not was_loaded or res.stats.get("load_s", 0) > 0.5:
            return   # this request (re)loaded the model: slow by nature, not a verdict on its speed
        slow = res.first_token_s > self.SLOW_FIRST_S or (tok_s is not None and res.stats.get("gen_tokens", 0) >= 8
                                                         and tok_s < self.SLOW_TOK_S)
        self._slow_strikes = self._slow_strikes + 1 if slow else 0
        self._safe_set_llm(speed="slow" if slow else "ok")
        if self._slow_strikes >= 2:
            why = (f"English started {res.first_token_s:.1f} s after the line" if res.first_token_s > self.SLOW_FIRST_S
                   else f"it writes only {tok_s} word-pieces/s")
            self._maybe_fallback(f"{model} is too slow on this Mac right now ({why})", model)

    def _maybe_fallback(self, reason: str, failed_model: str | None = None) -> bool:
        """Switch to the fast translation model (for this run). True if the caller should retry now
        (switched now, or the model had already been replaced since the request started)."""
        cur = failed_model or self.translator.model
        if self.translator.model != cur:
            return True      # switched (or changed in Settings) meanwhile: just retry with the current one
        s = self.settings
        fb = (s.llm_fallback or "").strip()
        if not s.auto_fallback or not fb or fb == cur:
            if not self.llm_state.get("slow_noticed"):
                self._safe_set_llm(slow_noticed=True)
                self._notice("warn", f"Translation is slow: {reason}. Close other heavy apps, or pick a smaller "
                                     "translation model in Settings.")
            return False
        with self._fallback_lock:
            if self.translator.model != cur:
                return True
            if not self._installed(fb):
                if fb not in self._pulling:
                    self._pulling.add(fb)
                    self._notice("warn", f"{reason}. Downloading the faster model {fb} (one time, ~3 GB) and "
                                         "switching to it as soon as it is ready…")
                    threading.Thread(target=self._pull_then_switch, args=(fb, cur), daemon=True).start()
                return False
            self._switch_model(fb, cur, reason)
            return True

    def _switch_model(self, new: str, old: str, reason: str) -> None:
        log.warning("switching translation model %s -> %s: %s", old, new, reason)
        self.translator.configure(model=new)
        self._slow_strikes = 0
        self._replaced_model = old
        self._safe_set_llm(model=new, fallback_active=True, loaded=False, speed="unknown", tok_s=None)
        self._notice("warn", f"{reason}. Switched to the faster model {new} for this session - lines keep coming. "
                             "(Settings → Translation to change the default.)")
        # free the memory the slow model was using (repeated by the health check if it is still loading now)
        threading.Thread(target=self.translator.unload, args=(old,), daemon=True).start()
        for sess in self._sessions_for_retry():
            for ln in [x for x in sess.meeting.lines if x.done and not x.ok]:
                self._enqueue(sess, ln, "auto")

    def _pull_then_switch(self, fb: str, cur: str) -> None:
        try:
            self.translator.pull(fb, lambda p: None)
            self._refresh_llm(True)
            with self._fallback_lock:
                if self.translator.model == cur:
                    self._switch_model(fb, cur, f"{cur} was too slow")
        except Exception as e:  # noqa: BLE001
            self._notice("error", f"Could not download {fb}: {e}")
        finally:
            self._pulling.discard(fb)

    def _safe_set_llm(self, **kw) -> None:
        try:
            self._set_llm(**kw)
        except Exception:  # noqa: BLE001  (status publishing must never kill a worker thread)
            log.exception("status update failed")

    def _lines_waiting(self) -> int:
        with self._tr_q.mutex:
            return sum(1 for j in self._tr_q.queue if j[0] == "line")

    def _fallback_usable(self, model: str) -> bool:
        s = self.settings
        fb = (s.llm_fallback or "").strip()
        return bool(s.auto_fallback and fb and fb != model and self._installed(fb))

    def _free_ollama_memory(self) -> None:
        """Unload other chat models that sit in Ollama's memory (e.g. left there by another app),
        so the translation model loads fast and stays on the GPU. Embedding models are tiny: kept."""
        if not self.settings.free_ollama_memory:
            return
        keep = {self.translator.model, self.settings.llm_fallback}
        keep |= {k + ":latest" for k in keep if k and ":" not in k}
        for m in self.translator.loaded_models():
            name = m["name"]
            if name in keep or "embed" in name.lower() or m["size"] < 1_500_000_000:
                continue
            log.info("unloading %s from Ollama to make room for the translator", name)
            self.translator.unload(name)

    def _installed(self, model: str) -> bool:
        names = set(self.llm_state.get("models") or [])
        return model in names or (":" not in model and model + ":latest" in names)

    def _model_loaded(self, model: str) -> bool:
        from .translate import NUM_CTX

        for m in self.translator.loaded_models():
            if m["name"] in (model, model + ":latest"):
                ctx = m.get("context_length")
                return not ctx or int(ctx) == NUM_CTX
        return False

    def _set_llm(self, **kw) -> None:
        changed = any(self.llm_state.get(k) != v for k, v in kw.items())
        self.llm_state.update(kw)
        if changed:
            self.publish(self.status())

    # ================================================================ Ollama health
    def _warm_llm(self) -> None:
        model = self.translator.model
        self._set_llm(warming=True)
        try:
            if not self._model_loaded(model):
                self._free_ollama_memory()
            res = self.translator.warmup(
                first_token_timeout=self.WAIT_SWITCH_S if self._fallback_usable(model) else self.WAIT_LOADING_S)
            if res.ok:
                self._set_llm(tok_s=res.stats.get("tok_s") or None, loaded=True)
                tok_s = res.stats.get("tok_s") or 0
                if res.stats.get("gen_tokens", 0) >= 5 and 0 < tok_s < self.SLOW_TOK_S * 0.6:
                    self._maybe_fallback(f"{model} writes only {tok_s} word-pieces/s on this Mac")
            elif res.timeout:
                if self._maybe_fallback(f"{model} did not finish loading in time (not enough free memory?)"):
                    self.translator.warmup()      # load the fast model right away
        except Exception:  # noqa: BLE001
            log.exception("warm-up failed")
        finally:
            self._set_llm(warming=False)
        self._refresh_llm(True)

    def _refresh_llm(self, force_publish: bool = False) -> None:
        h = self.translator.health()
        model = self.translator.model
        loaded = [m for m in (self.translator.loaded_models() if h["running"] else [])
                  if m["name"] in (model, model + ":latest")]
        prev = dict(self.llm_state)
        new = {"running": h["running"], "model_ready": h["model_ready"], "model": model,
               "models": h.get("models", []), "checked": True, "loaded": bool(loaded),
               "gpu_share": round(loaded[0]["gpu_share"], 2) if loaded else None}
        changed = any(prev.get(k) != v for k, v in new.items())
        recovered = new["running"] and new["model_ready"] and not (
            prev.get("running") and prev.get("model_ready")) and prev.get("checked")
        self.llm_state.update(new)
        if recovered:
            for sess in self._sessions_for_retry():
                for ln in [x for x in sess.meeting.lines if x.done and not x.ok]:
                    self._enqueue(sess, ln, "auto")
        # a model replaced by the fallback that finished loading anyway: free its memory
        if self._replaced_model and self.llm_state.get("fallback_active") and h["running"]:
            if any(m["name"] in (self._replaced_model, self._replaced_model + ":latest")
                   for m in self.translator.loaded_models()):
                threading.Thread(target=self.translator.unload, args=(self._replaced_model,), daemon=True).start()
        # a model that does not fit in GPU memory runs partly on the CPU: many times slower
        if (loaded and loaded[0]["size"] and loaded[0]["gpu_share"] < 0.9 and platform.system() == "Darwin"
                and not prev.get("cpu_noticed")):
            self.llm_state["cpu_noticed"] = True
            self._maybe_fallback(f"{model} does not fit in the Mac's GPU memory "
                                 f"(only {loaded[0]['gpu_share'] * 100:.0f}% on the GPU)")
        if changed or force_publish:
            self.publish(self.status())

    RETRY_EVERY_S = 10.0

    def _retry_failed_lines(self) -> None:
        """Lines whose translation failed are tried again automatically while the meeting runs."""
        if self._tr_busy or self._lines_waiting() or not self.llm_state.get("running") \
                or not self.llm_state.get("model_ready"):
            return
        now = time.monotonic()
        for sess in self._sessions_for_retry():
            for ln in [x for x in sess.meeting.lines if x.done and not x.ok]:
                key = (sess.meeting.id, ln.id)
                if now - self._retry_at.get(key, 0.0) >= self.RETRY_EVERY_S:
                    self._retry_at[key] = now
                    self._enqueue(sess, ln, "auto")

    def _sessions_for_retry(self) -> list:
        out = [self.session] if self.session is not None else []
        if self._last_session is not None and time.monotonic() - self._last_session_end < 600 \
                and self._last_session is not self.session:
            out.append(self._last_session)
        return out

    def _health_loop(self) -> None:
        warmed = False
        while not self._shutdown.is_set():
            try:
                self._refresh_llm()
                self._retry_failed_lines()
            except Exception:  # noqa: BLE001
                log.exception("Ollama health check failed")
            if (self.llm_state["running"] and not self.llm_state["model_ready"]
                    and self.translator.model not in self._pulling and not self._auto_pulled):
                # the translation model is missing (setup interrupted?): fetch it once, automatically
                self._auto_pulled = True
                self._notice("warn", f"Downloading the translation model {self.translator.model} (one time, ~3 GB)…")
                self.pull_model(self.translator.model)
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
