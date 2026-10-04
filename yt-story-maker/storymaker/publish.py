"""Stage 7 - YouTube upload kit: title, description (with chapters + footage credits),
tags. You still upload yourself; this just saves the busy-work."""

from pathlib import Path

from .llm import LLMError
from .util import fmt_ts

META_SYSTEM = """You write YouTube metadata for a popular Indian Hindi channel. Titles are
emotional and curiosity-driven but honest (no fake claims), 60-90 characters, Hindi in
Devanagari mixed with a few English keywords Indians search for. JSON only."""

META_USER = """Video topic: {topic}
Story: {description}
Video's Hindi title on screen: {title}
Theme: {theme}

Return JSON:
{{"youtube_title": "...", "description_hi": "3-4 lines in Hindi describing the video",
  "tags": ["15 search tags, Hindi and English"]}}"""

DISCLAIMER = (
    "Copyright Disclaimer under Section 107 of the Copyright Act 1976: allowance is made for "
    "'fair use' for purposes such as criticism, comment, news reporting, teaching, scholarship, "
    "and research. All footage belongs to its respective owners (credited above).")


def build_metadata(llm, topic, description, story, timeline, videos_used, music_used, log):
    meta = {}
    if llm.available():
        try:
            meta = llm.chat_json(META_SYSTEM, META_USER.format(
                topic=topic, description=description or "-", title=story.get("title_hi", ""),
                theme=story.get("theme", "")), temperature=0.6)
        except LLMError as e:
            log(f"AI metadata skipped ({e})")
    on_screen = story.get("title_hi") or topic
    fallback_title = on_screen if on_screen.strip().lower() == topic.strip().lower() else f"{on_screen} | {topic}"
    title = str(meta.get("youtube_title") or f"{fallback_title} | पूरी कहानी")[:100]
    desc_hi = str(meta.get("description_hi") or f"{topic} की पूरी कहानी — शुरुआत से अंजाम तक।")
    tags = [str(t)[:40] for t in (meta.get("tags") or []) if str(t).strip()][:20] or \
        [topic, f"{topic} hindi", f"{topic} story", "hindi documentary"]

    chapters = timeline.get("chapters") or []
    chapter_lines = [f"{fmt_ts(c['t'])} {c['title']}" for c in chapters]
    credits = []
    for v in videos_used:
        credits.append(f"• {v['channel'] or 'Unknown'} — {v['title'][:70]} — {v['url']}")
    music_lines = [f"• {Path(m).stem}" for m in music_used if m] or ["• Generated background score"]

    description_text = "\n".join([
        desc_hi, "",
        *(["Chapters:", *chapter_lines, ""] if len(chapter_lines) >= 3 else []),
        "Footage credits:", *credits, "",
        "Music:", *music_lines, "",
        DISCLAIMER,
    ])
    return {"title": title, "description": description_text, "tags": tags}


def write_upload_kit(meta, path):
    text = (f"TITLE\n{meta['title']}\n\nDESCRIPTION\n{meta['description']}\n\n"
            f"TAGS\n{', '.join(meta['tags'])}\n")
    Path(path).write_text(text, encoding="utf-8")
