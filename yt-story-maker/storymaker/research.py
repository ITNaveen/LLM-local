"""Stage 1 - research: turn a topic into many searches, collect ~500 candidates
(metadata only), score them, and shortlist the best videos."""

import math
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeout

from .llm import LLMError
from .util import keywords, tokens

QUERY_SYSTEM = """You are the research assistant of a viral Hindi news YouTube channel. Given a
topic and a story description, produce YouTube search queries that find the most DRAMATIC raw
material for this story: clashes, lathicharge (लाठीचार्ज), stone pelting (पथराव), हंगामा, बवाल,
emotional interviews of victims and families, angry statements, threats and warnings, press
conferences, heated debates, viral videos, ground reports, CCTV, and the big Hindi channels'
coverage (Aaj Tak, ABP News, Zee News, News18 India, TV9 Bharatvarsh, NDTV India, The Lallantop).
Cover the whole story: the background and earlier incidents too. Mix Hindi (Devanagari),
Hinglish and English the way Indians actually search. Never just repeat a slogan from the topic -
search for the events and people behind it. Return JSON only."""

QUERY_USER = """Topic: {topic}
Story description: {description}

Return JSON:
{{
  "queries": ["12 to 14 diverse YouTube search queries: at least half in Hindi/Hinglish, several aimed at dramatic footage"],
  "must_keywords": ["2-4 words that a relevant video title almost always contains"],
  "nice_keywords": ["6-12 related words: people, places, events, years"],
  "years": ["relevant years as strings, e.g. 2024"]
}}"""

RERANK_SYSTEM = """You select raw footage for a cinematic Hindi YouTube story video. Prefer
original footage (speeches, highlights, news coverage, crowd moments) that directly matches
the story. Reject reaction videos, compilations of unrelated topics, podcasts, explainers
with only a talking head, clickbait that does not match, and anything off-topic. JSON only."""

RERANK_USER = """Topic: {topic}
Story: {description}

Candidate videos:
{listing}

Return JSON: {{"keep": [list of the numbers of the {k} best videos, best first]}}"""

SUFFIXES = ["", "full video", "highlights", "speech", "news", "hindi", "best moments",
            "crowd reaction", "interview", "ground report"]


def make_queries(llm, topic, description, log):
    plan = None
    if llm.available():
        try:
            plan = llm.chat_json(QUERY_SYSTEM, QUERY_USER.format(
                topic=topic, description=description or "-"), temperature=0.5)
        except LLMError as e:
            log(f"LLM query planning failed, using keyword queries ({e})")
    kw = keywords(f"{topic} {description}")
    years = re.findall(r"\b(?:19|20)\d{2}\b", f"{topic} {description}")
    if not isinstance(plan, dict):
        plan = {}
    queries = [q.strip() for q in plan.get("queries") or [] if isinstance(q, str) and q.strip()]
    base = topic.strip()
    detail = " ".join(kw[:6])
    fallback = []
    if len(base.split()) > 6:        # a long slogan finds nothing on YouTube: search its keywords
        fallback, base = [base], " ".join(keywords(f"{description} {topic}")[:4]) or base
    fallback += [f"{base} {s}".strip() for s in SUFFIXES]
    if detail and detail.lower() != base.lower():
        fallback += [f"{base} {detail}", detail]
    seen, merged = set(), []
    for q in queries + fallback:
        key = q.lower()
        if key not in seen:
            seen.add(key)
            merged.append(q)
    must = [w.lower() for w in plan.get("must_keywords") or [] if isinstance(w, str)]
    nice = [w.lower() for w in plan.get("nice_keywords") or [] if isinstance(w, str)]
    topic_kw = keywords(topic)
    return {
        "queries": merged[:16],
        "must_keywords": must or topic_kw[:3],
        "nice_keywords": nice or kw[:12],
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


def research(source, llm, topic, description, settings, log, progress=None):
    plan = make_queries(llm, topic, description, log)
    queries = plan["queries"]
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
            for r in results:
                if r["id"] not in candidates:
                    candidates[r["id"]] = r
                    new += 1
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
    shortlist = _diverse_top(pool, k * 2)
    if llm.available() and shortlist:
        shortlist = _llm_rerank(llm, topic, description, shortlist, k, log)
    shortlist = _diverse_top(shortlist, k)
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


def _llm_rerank(llm, topic, description, shortlist, k, log):
    listing = "\n".join(
        f"{i + 1}. {c['title'][:110]} | {c.get('channel', '')[:30]} | "
        f"{int(c['duration'] // 60)} min | {c.get('views', 0):,} views"
        for i, c in enumerate(shortlist))
    try:
        res = llm.chat_json(RERANK_SYSTEM, RERANK_USER.format(
            topic=topic, description=description or "-", listing=listing, k=k), temperature=0.2)
        keep = [int(x) - 1 for x in res.get("keep", []) if str(x).strip().isdigit()]
        keep = [i for i in dict.fromkeys(keep) if 0 <= i < len(shortlist)]
        if len(keep) >= max(4, k // 3):
            chosen = [shortlist[i] for i in keep]
            rest = [c for i, c in enumerate(shortlist) if i not in set(keep)]
            log(f"  AI picked {len(chosen)} videos as the most relevant.")
            return chosen + rest
    except (LLMError, AttributeError, TypeError, ValueError) as e:
        log(f"  AI re-ranking skipped ({e})")
    return shortlist
