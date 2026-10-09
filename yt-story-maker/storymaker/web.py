"""Local web app: http://localhost:7777 . Only listens on this computer."""

import platform
import queue
import shutil
import threading
import time
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request, send_from_directory

from . import pipeline, style as style_mod
from .config import JOBS_DIR, ROOT, STYLES_DIR, CACHE_DIR, ensure_dirs, load_settings, save_settings
from .llm import OllamaLLM
from .music import MOODS, scan_library
from .source import YouTubeSource
from .util import ffmpeg_has_filter, has_tool, read_json

app = Flask(__name__, template_folder=str(ROOT / "templates"), static_folder=str(ROOT / "static"))
work_queue = queue.Queue()
running = {}          # job_id -> Job currently executing
style_tasks = {}      # name -> {"status", "error", "style"}
ALLOWED_FILES = {"final.mp4", "thumbnail.jpg", "narration_hi.srt", "youtube.txt"}


def worker():
    while True:
        job_id = work_queue.get()
        job = pipeline.Job(JOBS_DIR / job_id)
        if not job.state or job.state.get("status") == "cancelled":
            continue
        running[job_id] = job
        try:
            pipeline.run_job(job, load_settings())
        finally:
            running.pop(job_id, None)


def resume_unfinished():
    for j in reversed(pipeline.list_jobs()):
        if j["status"] in ("queued", "running"):
            work_queue.put(j["id"])


# ------------------------------------------------------------------ pages
@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/status")
def status():
    s = load_settings()
    llm = OllamaLLM(s["ollama_url"], s["llm_model"], s["embed_models"])
    models = llm.models()
    lib = scan_library(s["music_dir"])
    try:
        import yt_dlp
        ytv = yt_dlp.version.__version__
    except ImportError:
        ytv = ""
    return jsonify({
        "ffmpeg": has_tool("ffmpeg"),
        "ffmpeg_text": has_tool("ffmpeg") and ffmpeg_has_filter("ass"),
        "yt_dlp": ytv,
        "js_runtime": has_tool("deno") or has_tool("node"),
        "ollama": bool(models),
        "models": models,
        "llm_ready": llm.available(),
        "embed_model": llm.embed_model() if models else "",
        "music": {m: len(lib[m]) for m in MOODS},
        "music_dir": s["music_dir"],
        "mac": platform.system() == "Darwin",
        "queue": work_queue.qsize(),
    })


@app.get("/api/settings")
def get_settings():
    return jsonify(load_settings())


@app.post("/api/settings")
def post_settings():
    return jsonify(save_settings(request.get_json(force=True) or {}))


# ------------------------------------------------------------------ jobs
@app.get("/api/jobs")
def jobs():
    return jsonify(pipeline.list_jobs())


@app.post("/api/jobs")
def create_job():
    data = request.get_json(force=True) or {}
    topic = (data.get("topic") or "").strip()
    if not topic:
        return jsonify({"error": "Please enter a topic."}), 400
    minutes = float(data.get("minutes") or 10)
    if not data.get("demo"):
        minutes = min(15.0, max(8.0, minutes))
    req = {
        "topic": topic[:200],
        "description": (data.get("description") or "").strip()[:3000],
        "minutes": minutes,
        "theme": data.get("theme") if data.get("theme") in style_mod.THEMES else "sensational",
        "narration": data.get("narration") if data.get("narration") in ("none", "light", "medium") else "light",
        "voice": data.get("voice") if data.get("voice") in pipeline.VOICES else "male",
        "style": data.get("style") or "cinematic",
        "review": bool(data.get("review")),
        "demo": bool(data.get("demo")),
        "resolution": "720p" if data.get("resolution") == "720p" else "1080p",
    }
    job = pipeline.Job.create(req)
    work_queue.put(job.id)
    return jsonify({"id": job.id})


def _job_or_404(job_id):
    if "/" in job_id or ".." in job_id:
        abort(404)
    job = pipeline.Job(JOBS_DIR / job_id)
    if not job.state:
        abort(404)
    return job


@app.get("/api/jobs/<job_id>")
def job_detail(job_id):
    job = _job_or_404(job_id)
    st = read_json(job.path("story.json"))
    tl = read_json(job.path("timeline.json"))
    mom = read_json(job.path("moments.json")) or {}
    return jsonify({
        "state": job.state,
        "story": st,
        "rejected": mom.get("rejected"),
        "chapters": (tl or {}).get("chapters"),
        "youtube": read_json(job.path("youtube.json")),
        "research": _research_summary(job),
    })


def _research_summary(job):
    res = read_json(job.path("research.json"))
    if not res:
        return None
    return {"total": res.get("total_candidates"), "queries": res["plan"]["queries"],
            "shortlist": [{"id": c["id"], "title": c["title"], "channel": c.get("channel"),
                           "views": c.get("views"), "score": c.get("score")}
                          for c in res["shortlist"]]}


@app.post("/api/jobs/<job_id>/cancel")
def cancel(job_id):
    job = _job_or_404(job_id)
    if job_id in running:
        running[job_id].cancel_requested = True
    else:
        job.set(status="cancelled")
    return jsonify({"ok": True})


@app.post("/api/jobs/<job_id>/approve")
def approve(job_id):
    job = _job_or_404(job_id)
    body = request.get_json(force=True) or {}
    pipeline.apply_review_edits(job, body.get("edits") or {}, body.get("remove") or [])
    job.set(approved=True, status="queued")
    work_queue.put(job_id)
    return jsonify({"ok": True})


@app.post("/api/jobs/<job_id>/retry")
def retry(job_id):
    job = _job_or_404(job_id)
    if job_id not in running:
        job.set(status="queued", error="")
        work_queue.put(job_id)
    return jsonify({"ok": True})


@app.delete("/api/jobs/<job_id>")
def delete_job(job_id):
    job = _job_or_404(job_id)
    if job_id in running:
        return jsonify({"error": "Cancel the job first."}), 409
    shutil.rmtree(job.dir, ignore_errors=True)
    return jsonify({"ok": True})


@app.get("/jobs/<job_id>/<name>")
def job_file(job_id, name):
    job = _job_or_404(job_id)
    if name not in ALLOWED_FILES:
        abort(404)
    return send_from_directory(job.dir, name, conditional=True,
                               as_attachment=request.args.get("download") == "1")


# ------------------------------------------------------------------ styles
@app.get("/api/styles")
def styles():
    return jsonify({"styles": style_mod.list_styles(), "tasks": style_tasks,
                    "themes": style_mod.THEMES})


@app.post("/api/styles")
def learn_style():
    data = request.get_json(force=True) or {}
    url = (data.get("url") or "").strip()
    name = (data.get("name") or "").strip() or "my-style"
    if not url:
        return jsonify({"error": "Paste a YouTube link or a local video path."}), 400
    style_tasks[name] = {"status": "running", "error": ""}

    def go():
        try:
            path, meta = url, {"captions": None, "title": ""}
            if url.startswith("http"):
                base = CACHE_DIR / "reference" / f"{int(time.time())}"
                base.parent.mkdir(parents=True, exist_ok=True)
                path, meta = YouTubeSource(load_settings()).download_full(url, base)
            elif not Path(url).expanduser().exists():
                raise FileNotFoundError(url)
            st = style_mod.analyze_reference(str(Path(path).expanduser()), name,
                                             meta.get("captions"), meta.get("title") or url)
            style_tasks[name] = {"status": "done", "error": "", "style": st}
        except Exception as e:  # noqa: BLE001
            style_tasks[name] = {"status": "failed", "error": str(e)[:400]}

    threading.Thread(target=go, daemon=True).start()
    return jsonify({"ok": True})


@app.delete("/api/styles/<name>")
def delete_style(name):
    for p in STYLES_DIR.glob("*.json"):
        if (read_json(p) or {}).get("name") == name:
            p.unlink()
    style_tasks.pop(name, None)
    return jsonify({"ok": True})


def main():
    ensure_dirs()
    settings = load_settings()
    threading.Thread(target=worker, daemon=True).start()
    resume_unfinished()
    port = int(settings["port"])
    print(f"\n  StoryMaker running at  http://localhost:{port}\n")
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
