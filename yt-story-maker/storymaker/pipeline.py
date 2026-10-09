"""Runs the whole editor, stage by stage. Every stage saves its result in the job folder,
so a failed or edited job resumes from where it stopped instead of starting over."""

import json
import threading
import time
import traceback
import zlib
from datetime import datetime
from pathlib import Path

from . import editor, music, publish, render, research, timeline, voice
from .config import JOBS_DIR
from .llm import NoLLM, OllamaLLM
from .source import FixtureSource, YouTubeSource
from .style import get_style
from .util import keywords, read_json, slugify, write_json

STAGES = [  # name, share of the progress bar
    ("research", 0.08), ("moments", 0.22), ("story", 0.10), ("voice", 0.05),
    ("edit", 0.02), ("timeline", 0.01), ("download", 0.20), ("render", 0.29), ("publish", 0.03),
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
        if req["voice"] == "female" and settings.get("parler_speaker") in ("Rohit", "Aman", None, ""):
            settings["parler_speaker"] = "Divya"
        if req["voice"] == "male" and settings.get("parler_speaker") in ("Divya", "Rani"):
            settings["parler_speaker"] = "Rohit"
    if not render.can_draw_text():
        settings["title_card"] = False      # a title card without text is just a blurred frame
    if req.get("resolution") == "720p":
        settings["width"], settings["height"] = 1280, 720
    job.set(status="running", error="", started=time.time())
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
        share, sub = stage("research")
        res = read_json(job.path("research.json"))
        if not res:
            log(f"Topic: {topic}")
            res = research.research(source, llm, topic, desc, settings, log, progress=sub,
                                    outline=req.get("outline", ""))
            write_json(job.path("research.json"), res)
        finish(share)

        # 2. read every video, throw out what doesn't belong, understand what is said
        share, sub = stage("moments")
        mom = read_json(job.path("moments.json"))
        if not mom:
            mom = understand_videos(source, llm, topic, desc, res, settings, log, sub)
            write_json(job.path("moments.json"), mom)
        finish(share)

        # 3. the story: scenes, links, editor review
        share, _ = stage("story")
        outline = read_json(job.path("outline.json"))
        if not outline:
            log("Planning the film scene by scene...")
            can_text = render.can_draw_text()
            if not can_text:
                log("Hindi text can't be drawn yet (the Pillow package installs on the next "
                    "start), so on-screen lines are spoken by the narrator this time.")
            outline = editor.plan_story(llm, topic, desc, req.get("theme", "auto"), minutes,
                                        req.get("narration", "light"), mom["passages"],
                                        mom["videos"], log, can_text=can_text,
                                        user_outline=req.get("outline", ""),
                                        brief=editor.brief_of(res.get("plan"), topic, desc))
            write_json(job.path("outline.json"), outline)
        finish(share)

        # 4. narration voice
        share, _ = stage("voice")
        lines = editor.narration_lines(outline)
        vo = read_json(job.path("voice.json"))
        if vo is None or set(vo) != set(lines):
            if lines:
                log(f"Recording {len(lines)} Hindi narration lines...")
            vo = voice.synthesize(lines, job.path("voice"), settings, log)
            write_json(job.path("voice.json"), vo)
        finish(share)

        # 5. exact clips for every scene, fitted to the requested length
        share, _ = stage("edit")
        st = read_json(job.path("story.json"))
        if not st:
            log("Cutting the scenes...")
            theme = req.get("theme", "auto")
            st = editor.Assembler(mom["passages"], mom["videos"], mom["visuals"], style,
                                  minutes * 60, log).build(
                outline, {k: v["duration"] for k, v in vo.items()},
                theme if theme in ("epic", "emotional", "documentary", "thriller", "sensational")
                else "sensational")
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
            log("Music folder is empty - using the built-in news score. Royalty-free tracks in the "
                "music/ folder sound even better." if settings.get("builtin_music", True) else
                "Music folder is empty - the film uses the footage's own sound.")
        grids = {k: music.beat_times(t) if t else [] for k, t in tracks.items()}
        tl = timeline.build(st, vo, tracks, grids, settings)
        tl["title_hi"] = st.get("title_hi", topic)
        write_json(job.path("timeline.json"), tl)
        log(f"Timeline: {len(tl['segments'])} shots, {tl['duration'] / 60:.1f} minutes.")
        finish(share)

        # 7. download only what we use
        share, sub = stage("download")
        files, failed = render.download_all(source, tl["segments"], job.path("downloads"), log)
        gone_keys = set()
        dupes = render.visual_duplicates(tl["segments"], files)
        if dupes:
            log(f"Dropping {len(dupes)} shots that repeat footage already shown.")
        for _round in range(3):
            if render.drop_missing(files):
                log("Some downloaded files disappeared - re-editing around them.")
            clip_segs = [s for s in tl["segments"] if s["type"] == "clip"]
            gone = [s for s in clip_segs if not render.shot_available(files, s)
                    or (s["video_id"], round(s["src_start"], 2)) in dupes]
            if len(gone) == len(clip_segs):
                raise RuntimeError("No footage could be downloaded. " + "; ".join(failed[:2]))
            if gone:
                # Edit around unavailable shots for this render only; the saved story keeps
                # them, so a temporary network problem never loses footage from the plan.
                log(f"{len(gone)} shots unavailable - re-editing around them.")
                gone_keys |= {(s["video_id"], round(s["src_start"], 2)) for s in gone}
                pruned = editor.without_clips(st, gone_keys)
                tl = timeline.build(pruned, vo, tracks, grids, settings)
                tl["title_hi"] = st.get("title_hi", topic)
                write_json(job.path("timeline.json"), tl)
            # Re-cutting can lengthen a neighbouring shot: fetch those extra seconds.
            short = [s for s in tl["segments"] if s["type"] == "clip"
                     and render.shot_available(files, s) and not render.shot_available(files, s, strict=True)]
            if not gone and not short:
                break
            if short:
                log(f"Fetching a few extra seconds for {len(short)} re-cut shots...")
                more, _ = render.download_all(source, short, job.path("downloads"), log)
                render.merge_files(files, more)
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


def apply_review_edits(job, edits, remove=()):
    """edits: {scene_id: new Hindi text} for narration/text scenes; remove: scene ids to drop.
    Re-records changed lines and re-cuts the film; the AI planning is kept."""
    outline = read_json(job.path("outline.json"))
    remove = set(remove or ())
    changed = False
    kept = []
    for sc in outline["scenes"]:
        if sc.get("id") in remove:
            changed = True
            continue
        if sc.get("id") in edits and sc["type"] in ("narration", "text", "voiceover"):
            new = editor.clean_text(edits[sc["id"]], 30)
            if new and new != sc["text"]:
                sc["text"] = new
                if sc.get("narration_id"):
                    (job.path("voice") / f"{sc['narration_id']}.wav").unlink(missing_ok=True)
                changed = True
        kept.append(sc)
    if changed:
        outline["scenes"] = kept
        write_json(job.path("outline.json"), outline)
        for name in ("voice.json", "story.json", "timeline.json"):
            job.path(name).unlink(missing_ok=True)
    return changed


def reset_edit(job):
    """Forget the plan, voice, cut and render (keep research, transcripts and downloads) so the
    job is edited again from the story stage on."""
    import shutil
    for name in ("outline.json", "voice.json", "story.json", "timeline.json", "final.mp4",
                 "thumbnail.jpg", "narration_hi.srt", "youtube.json", "youtube.txt"):
        job.path(name).unlink(missing_ok=True)
    shutil.rmtree(job.path("voice"), ignore_errors=True)
    job.set(status="queued", error="", approved=False, outputs=None, progress=0.0)
    job.log("Re-editing with the current editor (research and transcripts kept)...")


def understand_videos(source, llm, topic, desc, res, settings, log, sub):
    log("Reading the shortlisted videos (language, transcript, replay graph)...")
    details = editor.read_videos(source, res["shortlist"], log, progress=lambda f: sub(0.3 * f),
                                 workers=int(settings.get("search_workers", 4)))
    log("Checking every video: on-topic? understandable language?")
    kept, rejected = editor.screen(llm, topic, desc, res.get("plan"), details, log,
                                   progress=lambda f: sub(0.3 + 0.2 * f),
                                   trust_topic=isinstance(source, FixtureSource) and not llm.available(),
                                   avoid=settings.get("avoid_channels") or ())
    if not kept:
        raise RuntimeError("None of the videos found were on-topic and in Hindi/English. "
                           "Try a more specific topic or description.")
    speech = [v for v in kept if v["role"] == "speech"]
    brief = editor.brief_of(res.get("plan"), topic, desc)
    topic_words = keywords(" ".join([topic, desc, " ".join(brief.get("entities") or []),
                                     " ".join(b["name"] + " " + b.get("about", "") for b in brief["beats"])]))
    passages, visuals = [], []
    for v in kept:
        visuals += editor.visual_moments(v)
    for i, v in enumerate(speech, 1):
        log(f"Understanding what is said ({i}/{len(speech)}): {v['title'][:60]}")
        allp = editor.build_passages(v)
        chosen = {p["id"] for p in editor.preselect(allp, topic_words)}
        annotated = {p["id"]: p for p in editor.annotate(
            llm, topic, desc, v, [p for p in allp if p["id"] in chosen], log,
            brief=brief)}
        for p in allp:
            p = annotated.get(p["id"], p)
            p.setdefault("video_title", v.get("title", ""))
            p.setdefault("channel", v.get("channel", ""))
            passages.append(p)
        sub(0.5 + 0.5 * i / max(1, len(speech)))
    usable = sum(1 for p in passages if p.get("use"))
    log(f"{usable} strong on-topic passages from {len(speech)} videos, "
        f"{len(visuals)} visual moments from {len(kept)} videos.")
    if not usable:
        raise RuntimeError("The on-topic videos had no usable Hindi/English speech. "
                           "Try a broader topic or a more specific description.")
    videos = [{k: val for k, val in v.items() if k not in ("captions", "heatmap")} for v in kept]
    return {"videos": videos, "rejected": rejected, "passages": passages, "visuals": visuals}


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
