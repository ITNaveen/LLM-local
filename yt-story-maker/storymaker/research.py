"""Stage 1 - research: turn a topic into many searches, collect ~500 candidates
(metadata only), score them, and shortlist the best videos."""

import math
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeout

from .llm import LLMError
from .util import keywords, tokens

QUERY_SYSTEM = """You are the producer of a viral Hindi news YouTube channel. The creator
describes the video they want. Turn it into the production brief the whole team follows: the
argument and the side the film takes, the story beats in order, and exactly which YouTube
coverage to search for each beat. Everything must be about THIS story - no generic material
(a story about Musk vs India needs Musk-India coverage, not Tesla launches). Searches name the
specific people, companies and events, the way Indians search (Hindi, Hinglish, English). If the
story is about violence, search for the clashes; if it is about a statement or a policy, search
for that statement, the reactions, the press conferences and the debates on Indian channels.
JSON only."""

QUERY_USER = """Video title: {topic}
What the creator wants:
{description}

Return JSON:
{{
  "angle": "2-3 sentences: what the film argues and whose side it takes",
  "tone": "a few words",
  "entities": ["the 4-10 key people, companies, countries and events of this story, each ALSO in Devanagari, e.g. 'Elon Musk', 'एलन मस्क'"],
  "beats": [{{"name": "short English name", "about": "one sentence: what this part shows or argues",
             "queries": ["3 YouTube searches for news coverage / footage of exactly this"]}}],
  "broll": [{{"what": "a visual the creator asked for, e.g. Starlink satellites in orbit",
             "query": "a YouTube search for that footage"}}],
  "avoid": ["claims the film must NOT make (e.g. allegations the creator warned about)"],
  "closing_line": "the creator's final punchline if they wrote one: the same words, in Devanagari (English terms may stay English); else empty",
  "must_keywords": ["2-4 words almost every relevant video title contains"],
  "years": ["relevant years as strings"]
}}
5 to 7 beats in story order (trigger -> context -> conflict -> the other side -> climax -> ending),
0 to 4 broll items."""

RERANK_SYSTEM = """You pick the source videos for a Hindi news YouTube film with a clear angle.
Keep only videos that cover one of the film's beats: news coverage, statements, press
conferences, interviews, debates and footage of exactly these events. Reject: other stories,
general content about one of the people, commentators who argue AGAINST the film's angle,
reaction videos, podcasts, compilations, clickbait. Make sure every beat has some videos. JSON only."""

RERANK_USER = """Film: {topic}
Angle: {angle}
Beats:
{beats}

Candidate videos (number | beat it was found for | title | channel | length | views):
{listing}

Return JSON: {{"keep": [numbers of up to {k} videos to study, best first]}}"""

SUFFIXES = ["", "full video", "highlights", "speech", "news", "hindi", "best moments",
            "crowd reaction", "interview", "ground report"]


def _strs(xs, n=20, size=160):
    return [str(x).strip()[:size] for x in (xs or []) if isinstance(x, (str, int, float)) and str(x).strip()][:n]


def fallback_brief(topic, description="", outline=""):
    """The brief without the AI: beats from the outline lines (or just the topic)."""
    lines = [re.sub(r"^\s*(\d+[.)]|[-*•])\s*", "", ln).strip() for ln in (outline or "").splitlines()]
    lines = [ln for ln in lines if len(ln.split()) >= 2][:8]
    base = topic.strip()
    if len(base.split()) > 6:
        base = " ".join(keywords(f"{description} {topic}")[:4]) or base
    beats = [{"name": ln[:60], "about": ln, "queries": [ln[:80]]} for ln in lines] or \
        [{"name": base[:60], "about": description[:200] or base, "queries": [base]}]
    return {"angle": (description or topic)[:400], "tone": "", "entities": keywords(topic)[:6],
            "beats": beats, "broll": [], "avoid": [], "closing_line": ""}


def make_queries(llm, topic, description, log, outline=""):
    """The production brief (angle, beats, footage, things to avoid) and the searches for it.
    Everything the creator wrote - description and outline - goes in."""
    text = "\n".join(x for x in ((description or "").strip(), (outline or "").strip()) if x) or "-"
    plan = None
    if llm.available():
        try:
            plan = llm.chat_json(QUERY_SYSTEM, QUERY_USER.format(topic=topic, description=text),
                                 temperature=0.4, max_tokens=3500)
        except LLMError as e:
            log(f"AI brief failed, using keyword searches ({e})")
    if not isinstance(plan, dict):
        plan = {}
    brief = fallback_brief(topic, description, outline)
    beats = []
    for b in plan.get("beats") or []:
        if isinstance(b, dict) and str(b.get("name") or "").strip():
            beats.append({"name": str(b["name"]).strip()[:80], "about": str(b.get("about") or "")[:300],
                          "queries": _strs(b.get("queries"), 4, 100)})
    if len(beats) >= 2:
        brief.update(beats=beats[:8])
    for key, n in (("entities", 12), ("avoid", 8)):
        if plan.get(key):
            brief[key] = _strs(plan[key], n)
    for key in ("angle", "tone", "closing_line"):
        if isinstance(plan.get(key), str) and plan[key].strip():
            brief[key] = plan[key].strip()[:600]
    brief["broll"] = [{"what": str(b.get("what") or "")[:100], "query": str(b.get("query") or "")[:100]}
                      for b in plan.get("broll") or [] if isinstance(b, dict) and b.get("query")][:4]

    # searches: round-robin over the beats, so a time limit never leaves a beat uncovered
    tagged, seen = [], set()

    def add(q, beat, broll=False):
        if q and q.lower() not in seen:
            seen.add(q.lower())
            tagged.append({"q": q, "beat": beat, "broll": broll})
    for i in range(4):
        for n, b in enumerate(brief["beats"], 1):
            if i < len(b["queries"]):
                add(b["queries"][i], n)
    for b in brief["broll"]:
        add(b["query"], 0, broll=True)
    kw = keywords(f"{topic} {description}")
    if len(tagged) < 8:                               # no AI: generic searches around the topic
        base = brief["beats"][0]["queries"][0]
        for suffix in SUFFIXES:
            add(f"{base} {suffix}".strip(), 1)
    years = re.findall(r"\b(?:19|20)\d{2}\b", f"{topic} {description} {outline}")
    must = [w.lower() for w in plan.get("must_keywords") or [] if isinstance(w, str)]
    ent_words = keywords(" ".join(brief["entities"]))
    return {
        "queries": [t["q"] for t in tagged][:20],
        "tagged": tagged[:20],
        "brief": brief,
        "must_keywords": must or keywords(topic)[:3],
        "nice_keywords": (ent_words or kw)[:16],
        "years": [str(y) for y in (plan.get("years") or years)][:4],
    }


BAD_TITLE = re.compile(
    r"\b(reaction|reacts|reacting|podcast|status|whatsapp|#shorts|shorts|ringtone|"
    r"karaoke|lyrics|meme|roast|edit audio|fan ?edit|trailer reaction)\b", re.I)
GOOD_TITLE = re.compile(
    r"\b(full|speech|highlights|live|press conference|interview|exclusive|ground report|"
    r"celebration|moment|historic|भाषण|हाइलाइट्स|पूरा|लाइव|ऐतिहासिक)\b", re.I)


def score_candidate(c, plan):
    title = (c.get("title") or "").lower()
    text = f"{title} {(c.get('description') or '').lower()} {(c.get('channel') or '').lower()}"
    title_tok = set(tokens(title))
    must = plan["must_keywords"]
    must_hits = sum(1 for w in must if w in title or w in title_tok)
    nice_hits = sum(1 for w in plan["nice_keywords"] if w in text)
    year_hit = any(y in text for y in plan.get("years") or [])
    relevance = (0.55 * (must_hits / max(1, len(must)))
                 + 0.3 * min(1.0, nice_hits / 4) + 0.15 * year_hit)
    popularity = min(1.0, math.log10(c.get("views", 0) + 10) / 7)
    dur = c.get("duration") or 0
    shape = 1.0
    if dur and dur < 60:
        shape = 0.2      # Shorts: vertical and tiny
    elif dur > 2 * 3600:
        shape = 0.6      # very long streams are slow to mine
    quality = 1.0
    if BAD_TITLE.search(title):
        quality = 0.25
    elif GOOD_TITLE.search(title):
        quality = 1.15
    return round((0.65 * relevance + 0.35 * popularity) * shape * quality, 4)


def research(source, llm, topic, description, settings, log, progress=None, outline=""):
    plan = make_queries(llm, topic, description, log, outline)
    brief = plan["brief"]
    log(f"Story brief: {brief['angle'][:200]}")
    for n, b in enumerate(brief["beats"], 1):
        log(f"  beat {n}: {b['name']}")
    queries = plan["queries"]
    beat_of = {t["q"]: t for t in plan.get("tagged", [])}
    avoid = [a.lower() for a in settings.get("avoid_channels") or [] if a.strip()]
    workers = max(1, int(settings.get("search_workers", 4)))
    budget = float(settings.get("research_minutes", 6)) * 60
    log(f"Searching YouTube with {len(queries)} queries ({workers} at a time)...")
    candidates = {}
    limit = settings["max_candidates"]
    started = time.time()

    def one(q):
        t0 = time.time()
        return q, source.search(q, settings["results_per_query"]), time.time() - t0

    pool = ThreadPoolExecutor(max_workers=workers)
    futures = [pool.submit(one, q) for q in queries]
    done_count = 0
    try:
        for fut in as_completed(futures, timeout=budget):
            done_count += 1
            if progress:
                progress(done_count / len(queries))
            try:
                q, results, took = fut.result()
            except Exception as e:  # noqa: BLE001 - one failed query should not stop research
                log(f"  search failed: {str(e)[:200]}")
                continue
            new = 0
            tag = beat_of.get(q, {"beat": 1, "broll": False})
            for r in results:
                if any(a in (r.get("channel") or "").lower() for a in avoid):
                    continue                          # a channel the creator never wants
                if r["id"] not in candidates:
                    candidates[r["id"]] = dict(r, beats=[], broll=False)
                    new += 1
                c = candidates[r["id"]]
                if tag["broll"]:
                    c["broll"] = c["broll"] or not c["beats"]
                elif tag["beat"] not in c["beats"]:
                    c["beats"].append(tag["beat"])
                    c["broll"] = False
            log(f"  '{q}': {len(results)} results, {new} new (total {len(candidates)}) in {took:.0f}s")
            if took > 60:
                log("  (YouTube is answering slowly - see 'force IPv4' in Settings)")
            if len(candidates) >= limit:
                break
    except FuturesTimeout:
        log(f"  Search time limit reached ({budget / 60:.0f} min) - continuing with "
            f"{len(candidates)} videos.")
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    log(f"Search finished in {time.time() - started:.0f}s.")
    if not candidates:
        raise RuntimeError("YouTube search returned nothing. Check your internet connection, "
                           "update yt-dlp, or set 'cookies from browser' in Settings.")

    min_s, max_s = settings["min_video_seconds"], settings["max_video_seconds"]
    pool = []
    for c in candidates.values():
        if c["duration"] and not (min_s <= c["duration"] <= max_s):
            continue
        c["score"] = score_candidate(c, plan)
        pool.append(c)
    pool.sort(key=lambda c: c["score"], reverse=True)

    k = settings["shortlist_size"]
    shortlist = balanced_top(pool, k * 2, len(brief["beats"]))
    if llm.available() and shortlist:
        shortlist = _llm_rerank(llm, topic, brief, shortlist, k, log)
    shortlist = balanced_top(shortlist, k, len(brief["beats"]))
    log(f"Shortlisted {len(shortlist)} of {len(candidates)} videos.")
    return {"plan": plan, "total_candidates": len(candidates),
            "candidates": pool[:200], "shortlist": shortlist}


def _diverse_top(pool, k, per_channel=3):
    counts, out = {}, []
    for c in pool:
        ch = c.get("channel") or c["id"]
        if counts.get(ch, 0) >= per_channel:
            continue
        counts[ch] = counts.get(ch, 0) + 1
        out.append(c)
        if len(out) >= k:
            break
    return out


def balanced_top(pool, k, n_beats, per_channel=3, broll_slots=4):
    """Best videos, taken in turn from every beat (and a few for the requested visuals), so
    each part of the story has material. Pool order = preference."""
    lanes = {n: [c for c in pool if n in (c.get("beats") or [])] for n in range(1, n_beats + 1)}
    lanes[0] = [c for c in pool if c.get("broll") and not c.get("beats")][:broll_slots]
    untagged = [c for c in pool if not c.get("beats") and not c.get("broll")]
    counts, out, ids = {}, [], set()

    def take(c):
        ch = c.get("channel") or c["id"]
        if c["id"] in ids or counts.get(ch, 0) >= per_channel:
            return False
        counts[ch] = counts.get(ch, 0) + 1
        ids.add(c["id"])
        out.append(c)
        return True
    while len(out) < k and any(lanes.values()):
        for n in list(lanes):
            while lanes[n] and len(out) < k:
                if take(lanes[n].pop(0)):
                    break
    for c in untagged:
        if len(out) >= k:
            break
        take(c)
    return out


def _llm_rerank(llm, topic, brief, shortlist, k, log):
    listing = "\n".join(
        f"{i + 1} | {'visual' if c.get('broll') else ','.join(map(str, c.get('beats') or [])) or '-'} | "
        f"{c['title'][:100]} | {c.get('channel', '')[:28]} | {int(c['duration'] // 60)} min | "
        f"{c.get('views', 0):,} views" for i, c in enumerate(shortlist))
    beats = "\n".join(f"{n}. {b['name']}: {b['about']}" for n, b in enumerate(brief["beats"], 1))
    try:
        res = llm.chat_json(RERANK_SYSTEM, RERANK_USER.format(
            topic=topic, angle=brief["angle"], beats=beats, listing=listing, k=k), temperature=0.2)
        keep = [int(x) - 1 for x in res.get("keep", []) if str(x).strip().isdigit()]
        keep = [i for i in dict.fromkeys(keep) if 0 <= i < len(shortlist)]
        if len(keep) >= max(4, k // 3):
            log(f"  AI picked {len(keep)} videos that cover the story beats.")
            return [shortlist[i] for i in keep]      # the rejected ones are NOT used as filler
        log("  AI picked too few videos - keeping the search ranking.")
    except (LLMError, AttributeError, TypeError, ValueError) as e:
        log(f"  AI re-ranking skipped ({e})")
    return shortlist
