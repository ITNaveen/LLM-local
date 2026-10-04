"""Runs the whole editor, stage by stage. Every stage saves its result in the job folder,
so a failed or edited job resumes from where it stopped instead of starting over."""

import copy
import json
import threading
import time
import traceback
import zlib
from datetime import datetime
from pathlib import Path

from . import moments as moments_mod
from . import music, publish, render, research, story, timeline, voice
from .config import JOBS_DIR
from .llm import NoLLM, OllamaLLM
from .source import FixtureSource, YouTubeSource
from .style import get_style
from .util import read_json, slugify, write_json

STAGES = [  # name, share of the progress bar
    ("research", 0.10), ("moments", 0.15), ("story", 0.10), ("voice", 0.05),
    ("edit", 0.05), ("timeline", 0.02), ("download", 0.20), ("render", 0.30), ("publish", 0.03),
]
VOICES = {"male": "hi-IN-MadhurNeural", "female": "hi-IN-SwaraNeural"}


class Cancelled(Exception):
    pass


class Job:
    def __init__(self, job_dir):
        self.dir = Path(job_dir)
        self.state = read_json(self.dir / "job.json", {})
        self._lock = threading.Lock()
        self.cancel_requested = False

    @classmethod
    def create(cls, request):
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        job_id = f"{stamp}-{slugify(request.get('topic', 'video'), 30)}"
        job_dir = JOBS_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        job = cls(job_dir)
        job.state = {"id": job_id, "request": request, "status": "queued", "stage": "",
                     "progress": 0.0, "error": "", "created": time.time(), "updated": time.time(),
                     "log": []}
        job.save()
        return job

    @property
    def id(self):
        return self.state["id"]

    def save(self):
        with self._lock:
            self.state["updated"] = time.time()
            write_json(self.dir / "job.json", self.state)

    def log(self, msg):
        line = f"{datetime.now().strftime('%H:%M:%S')}  {msg}"
        print(f"[{self.id}] {msg}", flush=True)
        with self._lock:
            self.state.setdefault("log", []).append(line)
            self.state["log"] = self.state["log"][-400:]
        self.save()
        if self.cancel_requested:
            raise Cancelled()

    def set(self, **kw):
        self.state.update(kw)
        self.save()

    def path(self, name):
        return self.dir / name


def make_llm(settings, log):
    llm = OllamaLLM(settings["ollama_url"], settings["llm_model"], settings["embed_models"],
                    think=settings["llm_think"], timeout=settings["llm_timeout"], log=log)
    if llm.available():
        emb = llm.embed_model() or "none (keyword matching)"
        log(f"Local AI: {settings['llm_model']} | embeddings: {emb}")
        return llm
    installed = llm.models()
    if installed:
        log(f"Model '{settings['llm_model']}' not installed (have: {', '.join(installed[:6])}). "
            "Using built-in rules. Pick a model in Settings.")
    else:
        log("Ollama is not running - using built-in rules (start Ollama for AI-written stories).")
    return NoLLM()


def make_source(request, settings, log):
    if request.get("demo"):
        log("DEMO MODE: using synthetic offline footage (no YouTube).")
        return FixtureSource(settings, log=log)
    return YouTubeSource(settings, log=log)


def run_job(job, settings, source=None, llm=None):
    req = job.state["request"]
    settings = dict(settings)
    if req.get("voice") in VOICES:
        settings["edge_voice"] = VOICES[req["voice"]]
    if req.get("resolution") == "720p":
        settings["width"], settings["height"] = 1280, 720
    job.set(status="running", error="")
    done_share = 0.0

    def stage(name):
        nonlocal done_share
        share = dict(STAGES)[name]
        job.set(stage=name, progress=round(done_share, 3))
        def sub(frac):
            job.set(progress=round(done_share + share * min(1.0, frac), 3))
        return share, sub

    def finish(share):
        nonlocal done_share
        done_share += share
        job.set(progress=round(done_share, 3))

    try:
        log = job.log
        source = source or make_source(req, settings, log)
        llm = llm or make_llm(settings, log)
        topic, desc = req["topic"].strip(), (req.get("description") or "").strip()
        minutes = float(req.get("minutes") or 10)
        style = get_style(req.get("style") or "cinematic")

        # 1. research
        share, _ = stage("research")
        res = read_json(job.path("research.json"))
        if not res:
            log(f"Topic: {topic}")
            res = research.research(source, llm, topic, desc, settings, log)
            write_json(job.path("research.json"), res)
        finish(share)

        # 2. moments
        share, _ = stage("moments")
        mom = read_json(job.path("moments.json"))
        if not mom:
            log("Reading transcripts and 'most replayed' graphs...")
            mom = moments_mod.gather(source, res["shortlist"], log)
            write_json(job.path("moments.json"), mom)
            log(f"Found {len(mom['moments'])} candidate moments in {len(mom['videos'])} videos.")
        finish(share)

        # 3. story outline
        share, _ = stage("story")
        outline = read_json(job.path("outline.json"))
        if not outline:
            log("Writing the story...")
            outline = story.plan_outline(llm, topic, desc, req.get("theme", "auto"), minutes,
                                         req.get("narration", "light"), mom["moments"], log)
            story.assign_narration_ids(outline)
            write_json(job.path("outline.json"), outline)
        finish(share)

        # 4. narration voice
        share, _ = stage("voice")
        lines = {b["narration_id"]: b["narration"] for a in outline["acts"] for b in a["beats"]
                 if b.get("narration_id") and b["narration"]}
        vo = read_json(job.path("voice.json"))
        if vo is None or set(vo) != set(lines):
            if lines:
                log(f"Recording {len(lines)} Hindi narration lines...")
            vo = voice.synthesize(lines, job.path("voice"), settings, log)
            write_json(job.path("voice.json"), vo)
        finish(share)

        # 5. edit decision: which seconds of which video go where
        share, _ = stage("edit")
        st = read_json(job.path("story.json"))
        if not st:
            log("Choosing the best seconds of footage for every beat...")
            filler = story.Filler(mom["moments"], style, minutes * 60, topic, llm=llm, log=log)
            st = filler.fill(copy.deepcopy(outline),
                             {k: v["duration"] for k, v in vo.items()})
            write_json(job.path("story.json"), st)
        finish(share)

        if req.get("review") and not job.state.get("approved"):
            job.set(status="awaiting_review", stage="review")
            log("Story ready for your review. Edit narration if you like, then press Render.")
            return job

        # 6. music + timeline
        share, _ = stage("timeline")
        tracks = music.choose_tracks(st["acts"], settings["music_dir"], seed=zlib.crc32(job.id.encode()) % 1000)
        if not any(tracks.values()):
            log("Music library is empty - using the generated background score. Add royalty-free "
                "tracks to the music/ folder for a much better result.")
        grids = {k: music.beat_times(t) if t else [] for k, t in tracks.items()}
        tl = timeline.build(st, vo, tracks, grids, settings)
        tl["title_hi"] = st.get("title_hi", topic)
        write_json(job.path("timeline.json"), tl)
        log(f"Timeline: {len(tl['segments'])} shots, {tl['duration'] / 60:.1f} minutes.")
        finish(share)

        # 7. download only what we use
        share, sub = stage("download")
        files, failed = render.download_all(source, tl["segments"], job.path("downloads"), log)
        clip_segs = [s for s in tl["segments"] if s["type"] == "clip"]
        missing = [s for s in clip_segs if not render.locate(files, s)[0]]
        if len(missing) == len(clip_segs):
            raise RuntimeError("No footage could be downloaded. " + "; ".join(failed[:2]))
        if missing:
            # Edit around unavailable shots for this render only; the saved story keeps them,
            # so a temporary network problem never loses footage from the plan.
            log(f"{len(missing)} shots unavailable - re-editing around them.")
            gone = {(s["video_id"], round(s["src_start"], 2)) for s in missing}
            pruned = copy.deepcopy(st)
            for act in pruned["acts"]:
                for b in act["beats"]:
                    b["clips"] = [c for c in b["clips"] if (c["video_id"], round(c["start"], 2)) not in gone]
            tl = timeline.build(pruned, vo, tracks, grids, settings)
            tl["title_hi"] = st.get("title_hi", topic)
            write_json(job.path("timeline.json"), tl)
        finish(share)

        # 8. render
        share, sub = stage("render")
        final = render.render(tl, files, settings, job.dir, log, progress=lambda f: sub(f * 0.85))
        finish(share)

        # 9. upload kit
        share, _ = stage("publish")
        used_ids = {s["video_id"] for s in tl["segments"] if s["type"] == "clip"}
        videos_used = [v for v in mom["videos"] if v["id"] in used_ids]
        meta = publish.build_metadata(llm, topic, desc, st, tl, videos_used,
                                      list(tracks.values()), log)
        write_json(job.path("youtube.json"), meta)
        publish.write_upload_kit(meta, job.path("youtube.txt"))
        finish(share)

        job.set(status="done", stage="done", progress=1.0,
                outputs={"video": final.name, "duration": tl["duration"],
                         "srt": "narration_hi.srt", "thumbnail": "thumbnail.jpg",
                         "youtube": "youtube.txt"})
        log(f"Done! {tl['duration'] / 60:.1f}-minute video ready.")
    except Cancelled:
        job.set(status="cancelled")
    except Exception as e:  # noqa: BLE001 - surface every failure in the UI
        job.state.setdefault("log", []).append(traceback.format_exc()[-1500:])
        job.set(status="failed", error=str(e)[:600])
    return job


def apply_review_edits(job, edits):
    """edits: {narration_id: new Hindi text}. Re-records changed lines, re-edits footage."""
    outline = read_json(job.path("outline.json"))
    changed = False
    for act in outline["acts"]:
        for b in act["beats"]:
            nid = b.get("narration_id")
            if nid in edits:
                new = story.clean_narration(edits[nid])
                if new and new != b["narration"]:
                    b["narration"] = new
                    (job.path("voice") / f"{nid}.wav").unlink(missing_ok=True)
                    changed = True
    if changed:
        write_json(job.path("outline.json"), outline)
        for name in ("voice.json", "story.json", "timeline.json"):
            job.path(name).unlink(missing_ok=True)
    return changed


def list_jobs():
    jobs = []
    if JOBS_DIR.exists():
        for d in sorted(JOBS_DIR.iterdir(), reverse=True):
            state = read_json(d / "job.json")
            if state:
                jobs.append({k: state.get(k) for k in
                             ("id", "status", "stage", "progress", "error", "created", "outputs")}
                            | {"topic": state.get("request", {}).get("topic", "")})
    return jobs


def dumps(obj):
    return json.dumps(obj, ensure_ascii=False)
