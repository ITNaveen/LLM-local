"""The editor brain. Works like a human documentary editor:

1. screen   - throw out videos that are off-topic, comedy/vlogs/reactions, or in a language
              the audience can't follow (Indian regional languages are never used; foreign
              speech is only allowed as silent visuals).
2. passages - cut every usable transcript into passages: complete thoughts of 10-35 s.
3. annotate - the local AI reads each passage: what is said, which sub-topic, how strong.
4. story    - the AI builds the film scene by scene. A dialogue scene plays ONE complete
              passage from ONE source. Every scene must state why it follows the previous one.
5. critic   - a second pass reviews the whole edit like a senior editor: removes scenes that
              break the flow, reorders, adds bridges, scores the result.
6. assemble - turns scenes into exact clips, fits the requested length by letting speakers
              continue (next passage of the same video) rather than adding random shots.
"""

import re

from .llm import LLMError
from .moments import JUNK, SENTENCE_END, build_moments, heat_at, mean_heat
from .style import ACT_KEYS, THEME_MOODS, THEMES
from .util import clamp, estimate_speech_seconds, keywords

ACT_NAMES_HI = {"opening": "शुरुआत", "buildup": "कहानी", "rising": "तूफ़ान से पहले",
                "climax": "चरम", "ending": "अंजाम"}
SCENE_TYPES = ("hook", "text", "narration", "dialogue", "montage")
NARRATION_LIMITS = {"none": 0, "light": 6, "medium": 12}
SPEECH_LANGS = {"hi", "en", "ur"}
INDIAN_REGIONAL = {"te", "ta", "kn", "ml", "bn", "mr", "gu", "pa", "or", "as", "ne", "sd",
                   "si", "kok", "mai", "bho", "raj"}
# Bengali .. Malayalam/Sinhala blocks (Devanagari, used by Hindi, is not included).
REGIONAL_SCRIPT = re.compile("[ঀ-෿]")
OFF_TOPIC_TITLE = re.compile(
    r"stand-?up|comedy|comedian|roast|vlog|prank|\breact(s|ion|ing)?\b|eurovision|song contest|"
    r"music video|lyrics|trailer|gameplay|gaming|unboxing|best of 20\d\d|#shorts|meme", re.I)
KINDS_OK = {"news", "speech", "interview", "debate", "documentary", "explainer",
            "ground_report", "footage", "press_conference", "analysis"}


def clean_text(text, max_words=28):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"[#*_`\"“”]|\(.*?\)|\[.*?\]", "", text)
    text = re.sub(r"(?i)subscribe|सब्सक्राइब", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return " ".join(text.split()[:max_words])


def _num(value, default=None):
    """First whole number in a value: 7, "7", "7/10", "P12", "V3" -> 7, 7, 7, 12, 3."""
    m = re.search(r"-?\d+", str(value)) if value is not None else None
    return int(m.group()) if m else default


# =================================================================== 0. reading
def read_videos(source, shortlist, log, progress=None, workers=4):
    """Full details (transcript, replay graph, language) for every shortlisted video."""
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def one(cand):
        t0 = time.time()
        return cand, source.details(cand["id"]), time.time() - t0

    got = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(one, c) for c in shortlist]
        for n, fut in enumerate(as_completed(futures), 1):
            if progress:
                progress(n / max(1, len(shortlist)))
            try:
                cand, d, took = fut.result()
            except Exception as e:  # noqa: BLE001 - skip unavailable / age-restricted videos
                log(f"  [{n}/{len(shortlist)}] skipped a video: {str(e)[:120]}")
                continue
            d["score"] = cand.get("score", 0.5)
            got[cand["id"]] = d
            log(f"  [{n}/{len(shortlist)}] {d['title'][:60]} - language '{d.get('spoken_lang') or '?'}', "
                f"{len(d.get('captions') or [])} caption lines ({took:.0f}s)")
    return [got[c["id"]] for c in shortlist if c["id"] in got]


# =================================================================== 1. screening
def language_role(video):
    """-> ('speech'|'visual'|'reject', reason)."""
    spoken = (video.get("spoken_lang") or "").lower()
    title = video.get("title") or ""
    has_text = bool(video.get("captions")) and (video.get("caption_lang") or "") in SPEECH_LANGS
    if spoken in INDIAN_REGIONAL:
        return "reject", f"spoken in a regional language ({spoken})"
    if REGIONAL_SCRIPT.search(title) and spoken not in SPEECH_LANGS:
        return "reject", "regional-language title"
    if spoken in SPEECH_LANGS or (not spoken and has_text):
        return ("speech", "") if has_text else ("visual", "no transcript - visuals only")
    if spoken:
        return "visual", f"spoken in '{spoken}' - used only as silent visuals"
    return "visual", "no speech/transcript - visuals only"


SCREEN_SYSTEM = """You are a strict researcher for a serious Hindi news-documentary channel.
Decide if a YouTube video is genuinely ABOUT the given topic and usable as source footage.
Say relevant=false for: comedy/stand-up, music, vlogs, study/jobs/travel-abroad videos,
reaction videos, memes, gaming, unrelated news, or videos that only mention the topic in
passing. If you are not sure, say false. JSON only."""

SCREEN_USER = """Topic: {topic}
What the film is about: {description}

Video title: {title}
Channel: {channel}
Length: {minutes} min
Video description: {vdesc}
Transcript excerpt: {excerpt}

Return JSON: {{"relevant": true or false,
 "kind": "news|speech|interview|debate|documentary|explainer|ground_report|press_conference|footage|comedy|vlog|reaction|other",
 "reason": "max 12 words"}}"""


def _excerpt(captions, chars=1400):
    text = " ".join(c["text"] for c in captions or [])
    if len(text) <= chars:
        return text
    half = chars // 2
    mid = len(text) // 2
    return text[:half] + " ... " + text[mid:mid + half]


def screen(llm, topic, description, plan, videos, log, progress=None, trust_topic=False):
    """trust_topic: skip the topic check (demo mode's synthetic clips can't be on-topic)."""
    kept, rejected = [], []
    must = [w.lower() for w in (plan or {}).get("must_keywords") or keywords(topic)[:3]]
    for i, v in enumerate(videos, 1):
        if progress:
            progress(i / max(1, len(videos)))
        role, why = language_role(v)
        if role == "reject":
            rejected.append({"id": v["id"], "title": v["title"], "reason": why})
            continue
        if OFF_TOPIC_TITLE.search(v.get("title") or ""):
            rejected.append({"id": v["id"], "title": v["title"], "reason": "comedy/vlog/reaction format"})
            continue
        verdict = {"relevant": True, "kind": "news", "reason": "demo"} if trust_topic else None
        if verdict is None and llm.available():
            try:
                verdict = llm.chat_json(SCREEN_SYSTEM, SCREEN_USER.format(
                    topic=topic, description=description or "-", title=v["title"],
                    channel=v.get("channel", ""), minutes=int(v.get("duration", 0) // 60),
                    vdesc=(v.get("description") or "")[:300].replace("\n", " "),
                    excerpt=_excerpt(v.get("captions")) or "(no transcript)"), temperature=0.1,
                    max_tokens=200)
            except LLMError as e:
                log(f"  AI check failed for '{v['title'][:50]}' ({e}); using keywords")
        if not isinstance(verdict, dict) or "relevant" not in verdict:
            text = " ".join([v.get("title", ""), v.get("description", ""),
                             _excerpt(v.get("captions"), 4000)]).lower()
            in_title = any(w in (v.get("title") or "").lower() for w in must)
            hits = sum(text.count(w) for w in must)
            verdict = {"relevant": in_title or hits >= 3, "kind": "news",
                       "reason": "keyword match" if (in_title or hits >= 3) else "topic not mentioned"}
        relevant = verdict.get("relevant") in (True, "true", "yes", 1)
        kind = str(verdict.get("kind") or "other").lower().replace(" ", "_")
        if relevant and kind not in KINDS_OK and kind != "other":
            relevant = False
        if not relevant:
            rejected.append({"id": v["id"], "title": v["title"],
                             "reason": f"not relevant: {str(verdict.get('reason', ''))[:80]}"})
            continue
        v["role"], v["kind"], v["role_note"] = role, kind, why
        kept.append(v)
    for r in rejected:
        log(f"  ✗ {r['title'][:60]} — {r['reason']}")
    log(f"Kept {len(kept)} videos ({sum(v['role'] == 'speech' for v in kept)} with usable speech), "
        f"rejected {len(rejected)}.")
    return kept, rejected


# =================================================================== 2. passages
def build_passages(video, min_len=10.0, max_len=35.0):
    """Complete thoughts from the transcript, with per-cue replay heat."""
    vid, dur = video["id"], float(video.get("duration") or 0)
    lo = min(8.0, dur * 0.04)
    hi = dur - min(15.0, dur * 0.05)
    heat = video.get("heatmap") or []
    cues = [c for c in video.get("captions") or [] if lo <= c["start"] and c["end"] <= hi]
    out, cur = [], []

    def flush():
        if cur and cur[-1]["end"] - cur[0]["start"] >= 6.0:
            text = " ".join(c["text"] for c in cur)
            if not JUNK.search(text):
                out.append({
                    "id": f"{vid}#{len(out)}", "video_id": vid, "index": len(out),
                    "start": round(cur[0]["start"], 2), "end": round(cur[-1]["end"], 2),
                    "text": text,
                    "cues": [{"s": c["start"], "e": c["end"], "t": c["text"],
                              "h": round(heat_at(heat, (c["start"] + c["end"]) / 2), 3)} for c in cur],
                    "heat": round(mean_heat(heat, cur[0]["start"], cur[-1]["end"]), 3),
                })
        cur.clear()

    for i, cue in enumerate(cues):
        if cur and cue["start"] - cur[-1]["end"] > 2.5:   # long silence = new thought
            flush()
        cur.append(cue)
        span = cue["end"] - cur[0]["start"]
        gap = cues[i + 1]["start"] - cue["end"] if i + 1 < len(cues) else 99
        if span >= max_len or (span >= min_len and (gap >= 0.6 or SENTENCE_END.search(cue["text"]))):
            flush()
    flush()
    return out


def preselect(passages, topic_words, limit=18):
    """Cheap pre-filter before the AI reads them: on-topic words + replay heat."""
    def score(p):
        text = p["text"].lower()
        hits = sum(1 for w in topic_words if w in text)
        return 0.5 * min(1.0, hits / 2) + 0.5 * p["heat"]
    best = sorted(passages, key=score, reverse=True)[:limit]
    return sorted(best, key=lambda p: p["start"])


ANNOTATE_SYSTEM = """You help a top Hindi documentary editor. You read transcript passages from
one source video and judge each one as raw material for a film on the topic. Be strict:
keep only passages that clearly say something about the topic. JSON only."""

ANNOTATE_USER = """Topic: {topic}
Film description: {description}
Source video: {title} ({channel})

Passages:
{listing}

For EVERY passage return one item:
{{"passages": [{{"n": 1, "use": true or false,
  "summary": "max 14 words, English: who says what",
  "topic": "1-3 word sub-topic, reuse the same label for the same thread",
  "strength": 1-5 (5 = powerful, emotional, quotable; 1 = filler),
  "standalone": true if it makes sense without what came before}}]}}"""


def annotate(llm, topic, description, video, passages, log):
    if not passages:
        return []
    results = {}
    if llm.available():
        for b in range(0, len(passages), 18):
            batch = passages[b:b + 18]
            listing = "\n".join(f"{k + 1}. [{int(p['end'] - p['start'])}s] {p['text'][:420]}"
                                for k, p in enumerate(batch))
            try:
                raw = llm.chat_json(ANNOTATE_SYSTEM, ANNOTATE_USER.format(
                    topic=topic, description=description or "-", title=video["title"],
                    channel=video.get("channel", ""), listing=listing), temperature=0.2,
                    max_tokens=2600)
                for item in raw.get("passages") or []:
                    n = _num(item.get("n")) if isinstance(item, dict) else None
                    if n and 1 <= n <= len(batch):
                        results[batch[n - 1]["id"]] = item
            except (LLMError, AttributeError) as e:
                log(f"  AI could not read passages of '{video['title'][:40]}' ({e})")
    out = []
    topic_words = keywords(f"{topic} {description}")
    for p in passages:
        item = results.get(p["id"])
        if item:
            use = item.get("use") in (True, "true", "yes", 1)
            strength = clamp(_num(item.get("strength"), 2) or 2, 1, 5)
            p.update(use=use, summary=str(item.get("summary") or p["text"][:90])[:160],
                     subtopic=str(item.get("topic") or "general").lower()[:40],
                     strength=strength, standalone=item.get("standalone") not in (False, "false"),
                     ai=True)
        else:
            hits = sum(1 for w in topic_words if w in p["text"].lower())
            p.update(use=hits > 0 or p["heat"] > 0.5, summary=p["text"][:100],
                     subtopic=(video.get("title") or "general").lower()[:30],
                     strength=int(clamp(1 + round(4 * p["heat"]), 1, 5)), standalone=True, ai=False)
        p["video_title"] = video.get("title", "")
        p["channel"] = video.get("channel", "")
        p["upload_date"] = video.get("upload_date", "")
        out.append(p)
    return out


def visual_moments(video):
    """B-roll candidates: the most replayed seconds of a relevant video."""
    ms = build_moments(video, per_video=25)
    return [{"video_id": m["video_id"], "start": m["start"], "end": m["end"], "peak": m["peak"],
             "video_duration": m["video_duration"], "video_title": m["video_title"],
             "channel": m["channel"]} for m in ms]


# =================================================================== 3. story (architect)
ARCHITECT_SYSTEM = """You are the lead editor of India's most-watched Hindi documentary YouTube
channel. Your films feel like a movie: every scene grows out of the previous one and leads into
the next. Viewers never feel a random jump. Rules you never break:
1. ONE THREAD AT A TIME. Finish a sub-topic before starting the next. Never ping-pong between
   sub-topics, speakers or countries.
2. EVERY SCENE CONNECTS. For every scene write "link": the concrete reason it follows the
   previous scene (cause -> effect, claim -> reaction, question -> answer, before -> after).
   If you cannot write a real link, do not use that scene.
3. A dialogue scene plays ONE passage completely. Never use the same speaker in fragments;
   if a speaker continues, use the next passage of the same source right after.
4. When the source, place or sub-topic changes, put a bridge first: a short Hindi narration
   line or an on-screen text card that tells the viewer where we are going and why.
5. Only use passages that are clearly about the topic. Fewer good scenes beat many weak ones.
6. Structure: opening = 1-2 hook lines (the most powerful quotes) + text cards that set the
   context; buildup = who, what, why it matters; rising = conflict and stakes grow;
   climax = the strongest passages + a montage; ending = consequence and a final thought.
7. Hindi text is natural spoken Hindi in Devanagari, short and cinematic (6-20 words).
JSON only."""

ARCHITECT_USER = """Topic: {topic}
What the creator wants: {description}
Theme: {theme}
Target length: about {minutes} minutes (dialogue passages give most of the runtime)
Narration: {narration_rule}

DIALOGUE PASSAGES (id | source | sub-topic | seconds | strength 1-5 | what is said):
{passages}

VISUAL SOURCES for montages / background (id | what it is):
{visuals}

Scene types:
- "hook": a short powerful quote from a passage (use: passage id). Opening only, max 2.
- "text": an on-screen Hindi text card (text). Use for context, dates, facts, bridges.
- "narration": a Hindi voice-over line (text). Bridges and context.
- "dialogue": play a passage completely (use: passage id).
- "montage": music + visuals, no words (use: list of visual ids, seconds: 8-25).

Return JSON:
{{"title_hi": "powerful Hindi title",
  "theme": "one line",
  "chapters": {{"opening": "Hindi chapter name", "buildup": "...", "rising": "...", "climax": "...", "ending": "..."}},
  "scenes": [{{"act": "opening|buildup|rising|climax|ending", "type": "...", "use": "P3 or [V1, V2]",
              "text": "Hindi, only for text/narration", "seconds": 12, "link": "why this follows"}}]}}
Use about {n_dialogue} dialogue scenes."""

CRITIC_SYSTEM = """You are a ruthless senior editor reviewing a junior's edit before it goes to
millions of Indian viewers. You check that the film flows like a movie: each scene must
connect to the one before and after it, one thread at a time, no off-topic material, no
repetition, no abrupt jumps between speakers/countries/sub-topics without a bridge, a strong
hook, rising tension and a satisfying ending. JSON only."""

CRITIC_USER = """Topic: {topic}
Film description: {description}

DRAFT EDIT (number | act | type | source | content | editor's link):
{draft}

Fix it. Return JSON:
{{"order": [scene numbers in the best final order; leave out scenes that are off-topic,
            repetitive or break the flow],
  "bridges": [{{"before": scene number, "type": "narration or text", "text": "Hindi line that connects the previous scene to this one"}}],
  "score": 1-10 (how gripping and well-connected the film is after your fixes),
  "issues": ["max 5 short notes on what is still weak"]}}"""


EXTEND_SYSTEM = """You are the lead editor of a top Hindi documentary YouTube channel. The film
is too short. Add more scenes WITHOUT breaking the flow: put a passage right after a scene it
continues (same speaker or same sub-topic), and add a short Hindi bridge (narration or text)
whenever the source or sub-topic changes. Never add off-topic material. JSON only."""

EXTEND_USER = """Topic: {topic}
The film runs about {have} minutes and must reach about {want} minutes.

CURRENT FILM (number | act | type | source | content):
{draft}

UNUSED PASSAGES (id | source | sub-topic | seconds | strength | what is said):
{unused}

Return JSON: {{"insert": [{{"after": scene number, "type": "dialogue|narration|text",
  "use": "P id for dialogue", "text": "Hindi for narration/text", "link": "why it belongs here"}}]}}"""


def estimate_seconds(scenes, cat):
    total = 0.0
    for s in scenes:
        if s["type"] == "dialogue":
            total += cat[s["pid"]]["end"] - cat[s["pid"]]["start"]
        elif s["type"] == "hook":
            total += 7
        elif s["type"] == "narration":
            total += estimate_speech_seconds(s["text"]) + 0.9
        elif s["type"] == "text":
            total += clamp(1.8 + len(s["text"]) / 14, 2.8, 6.5)
        else:
            total += s.get("seconds", 12)
    return total


def _draft_lines(scenes, cat, vcat):
    lines = []
    for i, s in enumerate(scenes, 1):
        if s["type"] in ("hook", "dialogue"):
            p = cat[s["pid"]]
            content, src = f"[{p['subtopic']}] {p['summary']}", _source_label(p)
        elif s["type"] == "montage":
            content, src = "music montage", ", ".join(_source_label(vcat[v])[:30] for v in s["vids"]) or "-"
        else:
            content, src = s["text"], "-"
        lines.append(f"{i} | {s['act']} | {s['type']} | {src} | {content} | {s.get('link', '')}")
    return lines


def _extend(llm, topic, outline, cat, vcat, minutes, log):
    for _round in range(2):
        have = estimate_seconds(outline["scenes"], cat)
        if have >= minutes * 60 * 0.85:
            return outline
        used = {s.get("pid") for s in outline["scenes"]}
        unused = {k: p for k, p in cat.items() if k not in used}
        if not unused:
            return outline
        ulines = "\n".join(f"{pid} | {_source_label(p)} | {p['subtopic']} | {int(p['end'] - p['start'])}s | "
                           f"{p['strength']} | {p['summary']}" for pid, p in unused.items())
        try:
            raw = llm.chat_json(EXTEND_SYSTEM, EXTEND_USER.format(
                topic=topic, have=round(have / 60, 1), want=minutes,
                draft="\n".join(_draft_lines(outline["scenes"], cat, vcat)), unused=ulines),
                temperature=0.4, max_tokens=4000)
        except LLMError as e:
            log(f"Could not extend the story ({e}).")
            return outline
        inserts = {}
        for item in raw.get("insert") or [] if isinstance(raw, dict) else []:
            if not isinstance(item, dict):
                continue
            after = _num(item.get("after"))
            if after is None or not 0 <= after <= len(outline["scenes"]):
                continue
            new = normalize_scenes([{**item, "act": (outline["scenes"][after - 1]["act"]
                                                    if after else "opening")}], cat, vcat)
            if new and new[0]["type"] in ("dialogue", "narration", "text"):
                if new[0]["type"] == "dialogue" and new[0]["pid"] in used:
                    continue
                inserts.setdefault(after, []).append(new[0])
                used.add(new[0].get("pid"))
        if not inserts:
            return outline
        scenes = []
        for i, sc in enumerate([None] + outline["scenes"]):
            if sc is not None:
                scenes.append(sc)
            scenes.extend(inserts.get(i, []))
        log(f"Story extended with {sum(len(v) for v in inserts.values())} connected scenes.")
        outline["scenes"] = scenes
    return outline


def make_catalog(passages, limit=70, per_video=8):
    usable = [p for p in passages if p.get("use")]
    by_video = {}
    for p in sorted(usable, key=lambda p: (p["strength"], p["heat"]), reverse=True):
        lst = by_video.setdefault(p["video_id"], [])
        if len(lst) < per_video:
            lst.append(p)
    chosen = sorted((p for lst in by_video.values() for p in lst),
                    key=lambda p: (p["strength"], p["heat"]), reverse=True)[:limit]
    chosen.sort(key=lambda p: (p["video_id"], p["start"]))  # sources together, in time order
    return {f"P{i}": p for i, p in enumerate(chosen, 1)}


def make_visual_catalog(videos, limit=16):
    visual_first = sorted(videos, key=lambda v: (v.get("role") != "visual", -v.get("score", 0)))
    return {f"V{i}": v for i, v in enumerate(visual_first[:limit], 1)}


def _source_label(v_or_p):
    title = (v_or_p.get("video_title") or v_or_p.get("title") or "")[:48]
    ch = (v_or_p.get("channel") or "")[:22]
    return f"{ch}: {title}" if ch else title


def plan_story(llm, topic, description, theme, minutes, narration, passages, videos, log):
    cat = make_catalog(passages)
    vcat = make_visual_catalog(videos)
    outline = None
    if llm.available() and cat:
        outline = _architect(llm, topic, description, theme, minutes, narration, cat, vcat, log)
        if outline:
            outline = _extend(llm, topic, outline, cat, vcat, minutes, log)
            outline = _critic(llm, topic, description, outline, cat, vcat, log)
    if not outline:
        if cat:
            log("Using the built-in story builder (start Ollama for an AI-edited story).")
        outline = fallback_story(topic, theme, cat, vcat, minutes)
    outline["scenes"] = enforce_rules(outline["scenes"], cat, narration, log)
    outline["catalog"] = {k: p["id"] for k, p in cat.items()}
    outline["visual_catalog"] = {k: v["id"] for k, v in vcat.items()}
    assign_ids(outline)
    return outline


def _architect(llm, topic, description, theme, minutes, narration, cat, vcat, log):
    rule = {"none": "NO voice-over at all. Use text cards for bridges and context.",
            "light": f"Voice-over only where needed, max {NARRATION_LIMITS['light']} lines. "
                     "Prefer letting the footage speak.",
            "medium": f"Voice-over is welcome, max {NARRATION_LIMITS['medium']} lines."}[
        narration if narration in NARRATION_LIMITS else "light"]
    plines = "\n".join(
        f"{pid} | {_source_label(p)} | {p['subtopic']} | {int(p['end'] - p['start'])}s | "
        f"{p['strength']} | {p['summary']}" for pid, p in cat.items())
    vlines = "\n".join(
        f"{vid} | {_source_label(v)}{' (foreign language, silent)' if v.get('role') == 'visual' else ''}"
        for vid, v in vcat.items())
    avg = sum(p["end"] - p["start"] for p in cat.values()) / max(1, len(cat))
    n_dialogue = int(clamp(minutes * 60 * 0.65 / max(10, avg), 6, 30))
    prompt = ARCHITECT_USER.format(topic=topic, description=description or "-",
                                   theme=THEMES.get(theme, THEMES["auto"]), minutes=minutes,
                                   narration_rule=rule, passages=plines, visuals=vlines or "(none)",
                                   n_dialogue=n_dialogue)
    for attempt in range(2):
        try:
            raw = llm.chat_json(ARCHITECT_SYSTEM, prompt, temperature=0.6, max_tokens=7000)
        except LLMError as e:
            log(f"AI story planning failed ({e}).")
            continue
        scenes = normalize_scenes(raw.get("scenes") if isinstance(raw, dict) else None, cat, vcat)
        if sum(s["type"] == "dialogue" for s in scenes) >= 3:
            log(f"AI planned {len(scenes)} scenes: \"{clean_text(raw.get('title_hi'), 14) or topic}\"")
            return {"title_hi": clean_text(raw.get("title_hi"), 14) or topic,
                    "theme": str(raw.get("theme") or theme)[:200],
                    "chapters": _chapters(raw.get("chapters")), "scenes": scenes, "source": "ai"}
        log("AI story was incomplete, retrying..." if attempt == 0 else "AI story still incomplete.")
    return None


def _chapters(raw):
    raw = raw if isinstance(raw, dict) else {}
    return {k: clean_text(raw.get(k), 8) or ACT_NAMES_HI[k] for k in ACT_KEYS}


def normalize_scenes(raw_scenes, cat, vcat):
    if not isinstance(raw_scenes, list):
        return []
    scenes, act_idx = [], 0
    for s in raw_scenes:
        if not isinstance(s, dict):
            continue
        typ = str(s.get("type", "")).lower().strip()
        if typ not in SCENE_TYPES:
            continue
        act = str(s.get("act", "")).lower().strip()
        if act in ACT_KEYS:
            act_idx = max(act_idx, ACT_KEYS.index(act))   # acts only move forward
        scene = {"act": ACT_KEYS[act_idx], "type": typ, "link": str(s.get("link") or "")[:200]}
        use = s.get("use")
        if typ in ("hook", "dialogue"):
            pid = f"P{_num(use[0] if isinstance(use, list) and use else use, 0)}"
            if pid not in cat:
                continue
            scene["pid"] = pid
        elif typ in ("text", "narration"):
            scene["text"] = clean_text(s.get("text"), 24)
            if not scene["text"]:
                continue
        else:  # montage
            ids = use if isinstance(use, list) else [use]
            scene["vids"] = [f"V{_num(v, 0)}" for v in ids if f"V{_num(v, 0)}" in vcat][:6]
            scene["seconds"] = clamp(_num(s.get("seconds"), 12) or 12, 6, 25)
        scenes.append(scene)
    return scenes


def _critic(llm, topic, description, outline, cat, vcat, log):
    scenes = outline["scenes"]
    lines = _draft_lines(scenes, cat, vcat)
    try:
        raw = llm.chat_json(CRITIC_SYSTEM, CRITIC_USER.format(
            topic=topic, description=description or "-", draft="\n".join(lines)),
            temperature=0.3, max_tokens=3000)
    except LLMError as e:
        log(f"Editor review skipped ({e}).")
        return outline
    order = [n for n in (_num(x) for x in (raw.get("order") or [])) if n and 1 <= n <= len(scenes)]
    order = list(dict.fromkeys(order))
    report = {"score": clamp(_num(raw.get("score"), 0) or 0, 0, 10),
              "issues": [str(x)[:160] for x in (raw.get("issues") or [])][:5]}
    if len(order) >= max(3, len(scenes) // 2):
        bridges = {}
        for b in raw.get("bridges") or []:
            if isinstance(b, dict) and _num(b.get("before")) in order and clean_text(b.get("text"), 24):
                typ = "text" if str(b.get("type")).lower() == "text" else "narration"
                bridges[_num(b.get("before"))] = {"type": typ, "text": clean_text(b.get("text"), 24),
                                                  "link": "bridge added by the editor review"}
        new, dropped = [], len(scenes) - len(order)
        for n in order:
            s = scenes[n - 1]
            if n in bridges and (not new or new[-1]["type"] not in ("text", "narration")):
                new.append({**bridges[n], "act": s["act"]})
            new.append(s)
        outline["scenes"] = new
        log(f"Editor review: score {report['score']}/10, removed {dropped} scenes, "
            f"added {len(new) - len(order)} bridges.")
    else:
        log(f"Editor review: score {report['score']}/10 (kept the original order).")
    for issue in report["issues"]:
        log(f"  • {issue}")
    outline["report"] = report
    return outline


# =================================================================== rules + fallback
def enforce_rules(scenes, cat, narration, log=lambda m: None):
    """Hard guarantees, whatever the AI wrote."""
    out, used, hooks = [], set(), 0
    limit = NARRATION_LIMITS.get(narration, 6)
    n_narr = 0
    for s in scenes:
        s = dict(s)
        if s["type"] == "hook":
            if hooks >= 2 or s["act"] != "opening":
                s["type"] = "dialogue"
            else:
                hooks += 1
        if s["type"] == "dialogue":
            if s["pid"] in used:
                continue                          # a passage plays once
            used.add(s["pid"])
        if s["type"] == "narration":
            if n_narr >= limit:
                s["type"] = "text"                # over the voice budget: show it as text
            else:
                n_narr += 1
        out.append(s)
    # Within an act, one speaker's passages always play in their original order
    # (never "later sentence, then earlier sentence" from the same interview).
    for act in ACT_KEYS:
        slots = {}
        for i, s in enumerate(out):
            if s["act"] == act and s["type"] == "dialogue":
                slots.setdefault(cat[s["pid"]]["video_id"], []).append(i)
        for idx in slots.values():
            ordered = sorted((out[i] for i in idx), key=lambda s: cat[s["pid"]]["start"])
            for i, s in zip(idx, ordered):
                out[i] = s
    return out


FALLBACK_TEXT = {
    "opening": "ये कहानी है {topic} की।",
    "buildup": "सब कुछ यहीं से शुरू हुआ।",
    "rising": "और फिर हालात बदलने लगे।",
    "climax": "और फिर आया वो पल।",
    "ending": "ये सिर्फ़ एक ख़बर नहीं, एक नया अध्याय है।",
}


def fallback_story(topic, theme, cat, vcat, minutes):
    """No AI: whole passages, one source at a time, strongest material in the climax."""
    passages = list(cat.items())
    scenes = []
    if passages:
        hook = max(passages, key=lambda kv: (kv[1]["strength"], kv[1]["heat"]))
        scenes.append({"act": "opening", "type": "hook", "pid": hook[0], "link": "strongest line first"})
    scenes.append({"act": "opening", "type": "text", "text": FALLBACK_TEXT["opening"].format(topic=topic),
                   "link": "sets the topic"})
    by_video = {}
    for pid, p in passages:
        by_video.setdefault(p["video_id"], []).append((pid, p))
    groups = sorted(by_video.values(), key=lambda g: -max(p["strength"] for _, p in g))
    acts = ["buildup", "buildup", "rising", "rising", "climax", "climax", "ending"]
    for gi, group in enumerate(groups[:len(acts)]):
        act = acts[gi]
        if not scenes or scenes[-1]["act"] != act:
            scenes.append({"act": act, "type": "text", "text": FALLBACK_TEXT[act], "link": "new chapter"})
        for pid, _p in sorted(group, key=lambda kv: kv[1]["start"])[:3]:
            scenes.append({"act": act, "type": "dialogue", "pid": pid, "link": "same speaker continues"})
    if vcat:
        scenes.append({"act": "climax", "type": "montage", "vids": list(vcat)[:4], "seconds": 15,
                       "link": "visual peak"})
    scenes.append({"act": "ending", "type": "text", "text": FALLBACK_TEXT["ending"], "link": "closing"})
    return {"title_hi": topic, "theme": theme, "chapters": _chapters({}), "scenes": scenes,
            "source": "template"}


def assign_ids(outline):
    n = 0
    for i, s in enumerate(outline["scenes"], 1):
        s["id"] = f"s{i:02d}"
        s.pop("narration_id", None)
        if s["type"] == "narration":
            n += 1
            s["narration_id"] = f"n{n:02d}"
    return outline


def narration_lines(outline):
    return {s["narration_id"]: s["text"] for s in outline["scenes"]
            if s["type"] == "narration" and s.get("narration_id")}


# =================================================================== 4. assemble
class Assembler:
    def __init__(self, passages, videos, visuals, style, total_seconds, log=None):
        self.passages = {p["id"]: p for p in passages}
        self.by_video = {}
        for p in passages:
            self.by_video.setdefault(p["video_id"], []).append(p)
        for lst in self.by_video.values():
            lst.sort(key=lambda p: p["start"])
        self.videos = {v["id"]: v for v in videos}
        self.visuals = sorted(visuals, key=lambda m: m["peak"], reverse=True)
        self.style = style
        self.total = total_seconds - 4.0          # title card
        self.log = log or (lambda m: None)
        self.used = {}

    # -- footage bookkeeping
    def free(self, vid, s, e, pad=0.5):
        return all(e + pad <= a or s - pad >= b for a, b in self.used.get(vid, []))

    def take(self, vid, s, e):
        self.used.setdefault(vid, []).append((s, e))

    def clip(self, vid, s, e, text="", heat=0.0):
        v = self.videos.get(vid, {})
        return {"video_id": vid, "start": round(max(0.0, s), 2), "end": round(e, 2), "text": text[:300],
                "heat": heat, "video_title": v.get("title", ""), "channel": v.get("channel", "")}

    def hook_span(self, p, lo=3.5, hi=9.0):
        """Best 3.5-9 s run of whole caption lines inside a passage (highest replay heat)."""
        cues, best = p["cues"], None
        for i in range(len(cues)):
            for j in range(i, len(cues)):
                d = cues[j]["e"] - cues[i]["s"]
                if d > hi:
                    break
                if d >= lo:
                    h = sum(c["h"] for c in cues[i:j + 1]) / (j - i + 1)
                    if best is None or h > best[0]:
                        best = (h, cues[i]["s"], cues[j]["e"], " ".join(c["t"] for c in cues[i:j + 1]))
        if best is None:
            return p["start"], min(p["end"], p["start"] + hi), p["text"]
        return best[1], best[2], best[3]

    def broll(self, seconds, prefer=(), avoid_vid=None, shot=3.0):
        clips, total, last = [], 0.0, avoid_vid
        pool = [m for m in self.visuals if m["video_id"] in prefer] + \
               [m for m in self.visuals if m["video_id"] not in prefer]
        for m in pool:
            if total >= seconds - 0.3:
                break
            length = min(shot, seconds - total) if seconds - total < shot * 1.5 else shot
            if length < 1.0:
                break
            if m["video_id"] == last and len(pool) > 3:
                continue
            centre = (m["start"] + m["end"]) / 2
            s = clamp(centre - length / 2, 0.5, max(0.5, m["video_duration"] - length - 0.5))
            if not self.free(m["video_id"], s, s + length):
                continue
            self.take(m["video_id"], s, s + length)
            clips.append(self.clip(m["video_id"], s, s + length, heat=m["peak"]))
            total += length
            last = m["video_id"]
        return clips

    def lead_in(self, next_p, seconds):
        """Visuals for a bridge: the moments just before the next speaker starts (J-cut feel)."""
        if next_p:
            e = next_p["start"] - 0.3
            s = e - seconds
            if s >= 0.5 and self.free(next_p["video_id"], s, e):
                self.take(next_p["video_id"], s, e)
                return [self.clip(next_p["video_id"], s, e)]
        prefer = (next_p["video_id"],) if next_p else ()
        return self.broll(seconds, prefer=prefer)

    # -- main
    def build(self, outline, narration_seconds, theme):
        cat = {k: self.passages.get(v) for k, v in outline.get("catalog", {}).items()}
        vcat = outline.get("visual_catalog", {})
        scenes = outline["scenes"]
        moods = THEME_MOODS.get(theme) or THEME_MOODS["epic"]
        beats_by_act = {k: [] for k in ACT_KEYS}

        # reserve dialogue footage first so bridges never steal it
        for s in scenes:
            if s["type"] == "dialogue" and cat.get(s["pid"]):
                p = cat[s["pid"]]
                self.take(p["video_id"], p["start"] - 0.2, p["end"] + 0.4)

        for i, s in enumerate(scenes):
            nxt = next((cat.get(x["pid"]) for x in scenes[i + 1:] if x["type"] in ("dialogue", "hook")
                        and cat.get(x["pid"])), None)
            beat = {"scene_id": s["id"], "kind": s["type"], "idea": s.get("link", ""),
                    "narration": "", "clips": [], "said": "", "source": ""}
            if s["type"] in ("hook", "dialogue"):
                p = cat.get(s["pid"])
                if not p:
                    continue
                if s["type"] == "hook":
                    a, b, text = self.hook_span(p)
                else:
                    a, b, text = p["start"] - 0.15, p["end"] + 0.35, p["text"]
                beat.update(audio="original", said=text[:400], source=_source_label(p),
                            passage_id=p["id"], strength=p.get("strength", 3),
                            clips=[self.clip(p["video_id"], a, b, text, p["heat"])])
            elif s["type"] == "narration":
                d = narration_seconds.get(s.get("narration_id"), estimate_speech_seconds(s["text"])) + 0.9
                beat.update(audio="narration", narration=s["text"], narration_id=s.get("narration_id"),
                            clips=self.lead_in(nxt, d))
            elif s["type"] == "text":
                beat.update(audio="text", narration=s["text"],
                            seconds=round(clamp(1.8 + len(s["text"]) / 14, 2.8, 6.5), 2))
            else:
                prefer = tuple(vcat.get(v) for v in s.get("vids", []) if vcat.get(v))
                beat.update(audio="music", clips=self.broll(
                    s.get("seconds", 12), prefer=prefer,
                    shot=self.style["acts"][s["act"]]["shot"]))
                if not beat["clips"]:
                    continue
            beats_by_act[s["act"]].append(beat)

        acts = []
        for ai, key in enumerate(ACT_KEYS):
            if beats_by_act[key]:
                acts.append({"key": key, "title_hi": outline.get("chapters", {}).get(key, ACT_NAMES_HI[key]),
                             "mood": moods[ai], "beats": beats_by_act[key]})
        story = {"title_hi": outline.get("title_hi", ""), "theme": outline.get("theme", ""),
                 "source": outline.get("source", ""), "report": outline.get("report"),
                 "acts": acts, "warnings": []}
        self.fit(story)
        for act in story["acts"]:
            for b in act["beats"]:
                b["seconds"] = round(beat_seconds(b), 2)
        return story

    # -- length
    def fit(self, story):
        def total():
            return sum(beat_seconds(b) for a in story["acts"] for b in a["beats"])

        # Too short: let speakers continue, strongest scenes first. The next passage of the
        # same video either flows on directly or follows as a jump cut (same speaker, later).
        guard = 0
        while total() < self.total * 0.95 and guard < 300:
            guard += 1
            grew = False
            dialogue = sorted((b for a in story["acts"] for b in a["beats"] if b["kind"] == "dialogue"),
                              key=lambda b: (len(b["clips"]), -b.get("strength", 3)))
            for b in dialogue:
                c = b["clips"][-1]
                if sum(x["end"] - x["start"] for x in b["clips"]) > 60:
                    continue
                later = [p for p in self.by_video.get(c["video_id"], [])
                         if p["end"] > c["end"] + 1 and -0.8 <= p["start"] - c["end"] < 90
                         and p.get("use", True)]
                nxt = later[0] if later else None
                # the start of a touching passage overlaps this scene's own padding, so only
                # check that no *other* scene uses it
                if not nxt or not self.free(c["video_id"], max(nxt["start"], c["end"]) + 0.45,
                                            nxt["end"], pad=0.0):
                    continue
                self.take(c["video_id"], nxt["start"], nxt["end"])
                if nxt["start"] - c["end"] < 4:      # flows on: one longer continuous clip
                    c["end"] = round(nxt["end"] + 0.35, 2)
                    c["text"] = (c["text"] + " " + nxt["text"])[:300]
                else:                                # jump cut within the same speaker
                    b["clips"].append(self.clip(c["video_id"], nxt["start"] - 0.15, nxt["end"] + 0.35,
                                                nxt["text"], nxt["heat"]))
                b["said"] = (b["said"] + " … " + nxt["text"])[:600]
                grew = True
                break
            if not grew:
                break
        if total() < self.total * 0.95:
            for b in (b for a in story["acts"] for b in a["beats"] if b["kind"] == "montage"):
                need = min(25.0 - beat_seconds(b), self.total * 0.95 - total())
                if need > 2:
                    b["clips"] += self.broll(need, avoid_vid=b["clips"][-1]["video_id"])
        if total() < self.total * 0.85:
            msg = (f"Only {total() / 60:.1f} min of strong, on-topic material was found, so the film "
                   f"is shorter than {self.total / 60 + 0.07:.0f} min rather than padded with weak clips. "
                   "A more specific description or a broader topic gives more material.")
            story["warnings"].append(msg)
            self.log(msg)

        # Too long: drop the weakest build-up/rising dialogue (with its bridge), never hook/climax.
        while total() > self.total * 1.08:
            cands = [(a, i, b) for a in story["acts"] if a["key"] in ("buildup", "rising", "ending")
                     for i, b in enumerate(a["beats"]) if b["kind"] == "dialogue"]
            if len(cands) <= 2:
                break
            act, i, b = min(cands, key=lambda t: (t[2].get("strength", 3), -beat_seconds(t[2])))
            remove = [b]
            if i > 0 and act["beats"][i - 1]["kind"] in ("narration", "text"):
                remove.append(act["beats"][i - 1])
            act["beats"] = [x for x in act["beats"] if x not in remove]


def beat_seconds(beat):
    if beat.get("audio") == "text":
        return beat.get("seconds", 3.0)
    return sum(c["end"] - c["start"] for c in beat.get("clips", []))
