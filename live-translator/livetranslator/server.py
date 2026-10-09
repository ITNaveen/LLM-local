"""Web server: the UI, a WebSocket with live events, REST for meetings/settings,
and a WebSocket that accepts audio from a browser (for remote / host-name use).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import platform
import shutil
import subprocess
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles

from .audio_io import PushSource, list_input_devices, mic_permission_status, preferred_order, probe_device
from .config import APP_NAME, ASR_MODELS, RECOMMENDED_LLMS, SettingsStore, home_dir, logs_dir
from .pipeline import Pipeline
from .storage import MeetingStore

log = logging.getLogger("lt.server")
STATIC = Path(__file__).parent / "static"


class Hub:
    """Fan-out of pipeline events (published from worker threads) to WebSocket clients."""

    TRANSIENT = {"level", "partial"}

    def __init__(self):
        self.loop: asyncio.AbstractEventLoop | None = None
        self.clients: set[asyncio.Queue] = set()

    def publish(self, ev: dict) -> None:
        loop = self.loop
        if loop is None or loop.is_closed():
            return
        try:
            loop.call_soon_threadsafe(self._fanout, ev)
        except RuntimeError:
            pass

    def _fanout(self, ev: dict) -> None:
        for q in list(self.clients):
            if q.full():
                if ev.get("type") in self.TRANSIENT or (ev.get("type") == "tr" and not ev.get("final")):
                    continue
                try:  # make room by dropping the oldest event
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            q.put_nowait(ev)


def ensure_ollama(url: str) -> None:
    """Start Ollama if it's local and not running yet."""
    import httpx

    def up() -> bool:
        try:
            return httpx.get(url.rstrip("/") + "/api/version", timeout=1.5, trust_env=False).status_code == 200
        except Exception:  # noqa: BLE001
            return False

    if up():
        return
    host = urlparse(url).hostname or ""
    if host not in ("127.0.0.1", "localhost", "::1"):
        return
    try:
        if platform.system() == "Darwin" and Path("/Applications/Ollama.app").exists():
            subprocess.Popen(["open", "-g", "-a", "Ollama"])
        elif shutil.which("ollama"):
            logf = open(logs_dir() / "ollama.log", "ab")
            subprocess.Popen(["ollama", "serve"], stdout=logf, stderr=logf, start_new_session=True)
        else:
            log.warning("Ollama is not installed - translation will not work")
            return
    except Exception:  # noqa: BLE001
        log.exception("could not start Ollama")
        return
    for _ in range(40):
        if up():
            log.info("Ollama started")
            return
        time.sleep(0.5)


def create_app(pipeline_factory=None, store: MeetingStore | None = None, settings: SettingsStore | None = None,
               token: str | None = None, start_ollama: bool = True) -> FastAPI:
    hub = Hub()
    settings = settings or SettingsStore()
    store = store or MeetingStore()
    token = token if token is not None else os.environ.get("LT_TOKEN", "")
    state: dict = {"pipeline": None, "push": None}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        hub.loop = asyncio.get_running_loop()
        if start_ollama:
            threading.Thread(target=ensure_ollama, args=(settings.settings.ollama_url,), daemon=True).start()
        factory = pipeline_factory or Pipeline
        state["pipeline"] = await asyncio.to_thread(factory, settings, store, hub.publish)
        yield
        p = state["pipeline"]
        if p is not None:
            await asyncio.to_thread(p.shutdown)

    app = FastAPI(title=APP_NAME, lifespan=lifespan, docs_url=None, redoc_url=None)

    def P() -> Pipeline:
        p = state["pipeline"]
        if p is None:
            raise HTTPException(503, "Starting up")
        return p

    # ----------------------------------------------------------- access token
    def _authorized(req_token: str | None, cookie: str | None) -> bool:
        return not token or req_token == token or cookie == token

    @app.middleware("http")
    async def auth(request: Request, call_next):
        if token:
            q = request.query_params.get("token")
            if not _authorized(q, request.cookies.get("lt_token")):
                return PlainTextResponse("Access token required: open the page with ?token=...", status_code=401)
            resp = await call_next(request)
            if q == token:
                resp.set_cookie("lt_token", token, httponly=True, samesite="strict", max_age=3600 * 24 * 365)
            return resp
        return await call_next(request)

    # ----------------------------------------------------------- pages
    @app.get("/")
    async def index():
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})

    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    # ----------------------------------------------------------- live events
    @app.websocket("/ws")
    async def ws_events(ws: WebSocket):
        if not _authorized(ws.query_params.get("token"), ws.cookies.get("lt_token")):
            await ws.close(code=4401)
            return
        await ws.accept()
        q: asyncio.Queue = asyncio.Queue(maxsize=4000)
        hub.clients.add(q)
        try:
            p = state["pipeline"]
            if p is not None:
                await ws.send_json(p.snapshot())

            async def reader():
                while True:
                    msg = await ws.receive_text()
                    if msg == "ping":
                        await ws.send_json({"type": "pong"})

            rt = asyncio.create_task(reader())
            try:
                while True:
                    get = asyncio.create_task(q.get())
                    done, _ = await asyncio.wait({get, rt}, return_when=asyncio.FIRST_COMPLETED)
                    if rt in done:
                        get.cancel()
                        break
                    await ws.send_json(get.result())
            finally:
                rt.cancel()
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            hub.clients.discard(q)

    @app.websocket("/ws/audio")
    async def ws_audio(ws: WebSocket):
        """Browser audio: first a JSON text frame {"rate": 48000, "format": "i16"|"f32"}, then binary PCM."""
        if not _authorized(ws.query_params.get("token"), ws.cookies.get("lt_token")):
            await ws.close(code=4401)
            return
        await ws.accept()
        prev = state.get("audio_ws")
        state["audio_ws"] = ws
        if prev is not None:
            try:   # only one browser may send audio: the newest wins
                await prev.close(code=4000)
            except Exception:  # noqa: BLE001
                pass
        rate, fmt = 48000, "i16"
        try:
            while True:
                msg = await ws.receive()
                if msg.get("type") == "websocket.disconnect":
                    break
                if msg.get("text"):
                    cfg = json.loads(msg["text"])
                    rate, fmt = int(cfg.get("rate", rate)), cfg.get("format", fmt)
                    continue
                data = msg.get("bytes")
                if state.get("audio_ws") is not ws:
                    break
                p = state["pipeline"]
                if not data or p is None or not isinstance(p.source, PushSource):
                    continue
                if fmt == "f32":
                    p.source.push_float32(data, rate)
                else:
                    p.source.push_int16(data, rate)
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            if state.get("audio_ws") is ws:
                state["audio_ws"] = None

    # ----------------------------------------------------------- session
    @app.get("/api/state")
    async def get_state():
        return P().snapshot()

    @app.post("/api/start")
    async def start(body: dict | None = None):
        body = body or {}
        try:
            meta = await asyncio.to_thread(P().start, body.get("name", ""), body.get("continue_id"))
        except RuntimeError as e:
            raise HTTPException(409, str(e)) from e
        return meta

    @app.post("/api/stop")
    async def stop():
        P().stop()
        return {"ok": True}

    # ----------------------------------------------------------- settings
    @app.get("/api/settings")
    async def get_settings():
        return settings.settings.to_dict()

    settings_lock = asyncio.Lock()

    @app.post("/api/settings")
    async def set_settings(changes: dict):
        async with settings_lock:   # one change at a time, applied in order
            old = settings.settings
            new = settings.update(changes)
            await asyncio.to_thread(P().apply_settings, old, new)   # may open a microphone: keep the loop free
            cur = settings.settings.to_dict()
        hub.publish({"type": "settings", "settings": cur})
        return cur

    @app.get("/api/devices")
    async def devices():
        # re-scans the devices unless a microphone is open (audio_io refuses to restart PortAudio then)
        return await asyncio.to_thread(list_input_devices, True)

    @app.post("/api/mictest")
    async def mictest():
        """Record ~1 s from every input: which ones actually hear something?"""
        p = P()
        if p.state != "idle":
            raise HTTPException(409, "Stop the meeting first - the microphone is in use")

        def run():
            devs = list_input_devices(refresh=True)
            res = {"permission": mic_permission_status(),
                   "devices": [probe_device(d) for d in preferred_order(devs)]}
            if any(d["ok"] for d in res["devices"]):
                p._mic_blocked = False     # a microphone works again: use it on the next Start
            return res

        return await asyncio.to_thread(run)

    @app.get("/api/models")
    async def models():
        p = P()
        h = await asyncio.to_thread(p.translator.health)
        p.llm_state.update({"running": h["running"], "model_ready": h["model_ready"], "models": h.get("models", [])})
        return {"asr": {k: v["label"] for k, v in ASR_MODELS.items()}, "llm_recommended": RECOMMENDED_LLMS,
                "llm_installed": h.get("models", []), "ollama_running": h["running"]}

    @app.post("/api/models/pull")
    async def pull(body: dict):
        model = (body.get("model") or "").strip()
        if not model:
            raise HTTPException(400, "model required")
        P().pull_model(model)
        return {"ok": True}

    # ----------------------------------------------------------- meetings
    @app.get("/api/meetings")
    async def meetings():
        return await asyncio.to_thread(store.list)

    def _load(mid: str):
        p = P()
        if p.session and p.session.meeting.id == mid:
            return p.session.meeting
        m = store.load(mid)
        if m is None:
            raise HTTPException(404, "Meeting not found")
        return m

    @app.get("/api/meetings/{mid}")
    async def meeting(mid: str):
        m = await asyncio.to_thread(_load, mid)
        return {"meta": m.meta(), "lines": [P()._line_dict(x) for x in m.lines], "summary": store.summary(m)}

    @app.post("/api/meetings/{mid}/rename")
    async def rename(mid: str, body: dict):
        name = body.get("name", "")
        p = P()
        if p.session and p.session.meeting.id == mid:
            return await asyncio.to_thread(p.rename_current, name)
        m = await asyncio.to_thread(_load, mid)
        await asyncio.to_thread(store.rename, m, name)
        hub.publish({"type": "meetings_changed"})
        return m.meta()

    @app.delete("/api/meetings/{mid}")
    async def delete(mid: str):
        p = P()
        if p.session and p.session.meeting.id == mid:
            raise HTTPException(409, "Stop this meeting before deleting it")
        m = await asyncio.to_thread(_load, mid)
        await asyncio.to_thread(store.delete, m)
        hub.publish({"type": "meetings_changed"})
        return {"ok": True}

    @app.get("/api/meetings/{mid}/download")
    async def download(mid: str, fmt: str = "md"):
        m = await asyncio.to_thread(_load, mid)
        base = f"{m.started:%Y-%m-%d %H-%M} {m.name}".strip()
        if fmt == "txt":
            body, mt = m.plain_text(), "text/plain; charset=utf-8"
        elif fmt == "json":
            body = json.dumps({"meta": m.meta(), "lines": [P()._line_dict(x) for x in m.lines]}, ensure_ascii=False, indent=2)
            mt = "application/json"
        else:
            fmt, body, mt = "md", m.markdown(), "text/markdown; charset=utf-8"
        from urllib.parse import quote

        return Response(body, media_type=mt, headers={
            "Content-Disposition": f"attachment; filename*=UTF-8''{quote(base + '.' + fmt)}"})

    @app.post("/api/meetings/{mid}/summary")
    async def summary(mid: str):
        try:
            P().summarize(mid)
        except RuntimeError as e:
            raise HTTPException(409, str(e)) from e
        return {"ok": True}

    @app.post("/api/meetings/{mid}/lines/{line_id}/retranslate")
    async def retranslate(mid: str, line_id: int):
        if not P().retranslate(mid, line_id):
            raise HTTPException(404, "Line not found")
        return {"ok": True}

    @app.post("/api/meetings/{mid}/open-folder")
    async def open_folder(mid: str, request: Request):
        if request.client and request.client.host not in ("127.0.0.1", "::1", "localhost"):
            raise HTTPException(403, "Only available on the computer running Live Translator")
        m = await asyncio.to_thread(_load, mid)
        cmd = {"Darwin": ["open"], "Windows": ["explorer"]}.get(platform.system(), ["xdg-open"])
        subprocess.Popen(cmd + [str(m.folder)])
        return {"ok": True}

    @app.get("/api/info")
    async def info():
        p = state["pipeline"]
        return {"app": APP_NAME, "data_dir": str(home_dir()), "meetings_dir": str(store.root),
                "platform": platform.platform(), "token_required": bool(token), "pid": os.getpid(),
                "state": p.state if p is not None else "starting"}

    @app.exception_handler(HTTPException)
    async def http_err(_req: Request, exc: HTTPException):
        return JSONResponse({"error": exc.detail}, status_code=exc.status_code)

    app.state.hub = hub
    app.state.store = store
    app.state.settings = settings
    app.state.runtime = state
    return app
