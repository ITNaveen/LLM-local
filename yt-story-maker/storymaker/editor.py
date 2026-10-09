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

from .llm import LLMError, tfidf_similarity
from .moments import JUNK, SENTENCE_END, build_moments, heat_at, mean_heat
from .style import ACT_KEYS, THEME_MOODS, THEMES
from .util import clamp, estimate_speech_seconds, keywords, tokens

ACT_NAMES_HI = {"opening": "शुरुआत", "buildup": "कहानी", "rising": "तूफ़ान से पहले",
                "climax": "चरम", "ending": "अंजाम"}
SCENE_TYPES = ("hook", "text", "narration", "dialogue", "montage")
# Heard by the audience as-is. Everything else (e.g. English) is retold by the Hindi narrator.
DIALOGUE_LANGS = {"hi", "ur"}
MAX_PER_VIDEO, MAX_PER_CHANNEL = 3, 4
# The narrator only sets the base (one line as each chapter opens), asks the hook question,
# reveals a twist now and then and closes the film. The Hindi clips carry the story.
NARRATION_LIMITS = {"none": 0, "light": 5, "medium": 8}
VOICEOVER_LIMITS = {"none": 0, "light": 2, "medium": 3}     # English speakers retold in Hindi
NARRATION_WORDS, VOICEOVER_WORDS = 22, 30
# How the narrator talks: a passionate Delhi YouTuber, not a news reader.
VOICE_STYLE = ("Write it the way a passionate young Delhi YouTuber talks to his audience: natural "
               "Hinglish in Devanagari (keep common English words like visa, IT, company, job, "
               "shock, plan, game as people say them), short punchy sentences, emotion, "
               "exclamations and questions ('सोचिए ज़रा!', 'ये है असली खेल...', 'आख़िर क्यों?'). "
               "Never sound like a slow, formal news reader.")
CLIP_CAP = 20.0   # Hindi news style: clips are punchy evidence cut to their sharpest part
CTA_LINE = "आपको क्या लगता है? कमेंट में ज़रूर बताइए।"

# English spoken by a Hindi channel's guest is often captioned in Devanagari ("ही वाज़
# अनकॉन्शियस एट द..."). These are English function words written in Devanagari.
EN_IN_DEVANAGARI = set(
    "द दि एट वाज़ वाज इज़ इज ऑफ ऑफ़ एंड फॉर फ़ॉर दिस दैट विद हैव हैज़ हैज बट नॉट देयर वी यू "
    "शी इट ऑन आर वर बीन विल वुड कैन हू व्हाट व्हेन व्हेयर व्हाई हाउ ऑल अबाउट फ्रॉम इनटू आफ्टर "
    "बिफोर बिकॉज़ बिकॉज दैन देम दीज़ दोज़ एवरी एव्री पीपल थिंग गोइंग डू डिड हैड वेरी जस्ट ओनली "
    "आल्सो ईवन वेल यस माई योर अवर दे'र अ एन आर्म वेयर व्हिच वेन देन".split())
HI_MARKERS = set("है हैं का की के में को ने से और भी था थी थे हो रहा रही रहे गया गई गए कि यह वह "
                 "ये वो नहीं पर तो हम आप उन इस उस जो कर".split())
DEVANAGARI_WORD = re.compile(r"[ऀ-ॿ]+")
# The AI can't see the footage: lines that describe the picture ("देख रहे हो ये लाइटें?") are
# invented. They are removed.
VISUAL_DEIXIS = re.compile(
    r"देख\s*(रहे|रही)\s*(हो|हैं|हैं\?)|(ये|इन|इस)\s+(सिर्फ\s+)?(लाइट|लाइटें|लाइटों|रोशनी|फ्लैश|चमक|"
    r"तस्वीर|तस्वीरें|तस्वीरों|नज़ारा|नजारा|दृश्य|फुटेज|वीडियो)|स्क्रीन\s+पर|आप\s+देख\s+सकते|(दिख|नज़र आ|नजर आ)\s*(रहा|रही|रहे)\s*(है|हैं)")
# Formal words a Delhi YouTuber never says, and what he says instead.
COLLOQUIAL = [("अत्यधिक", "बहुत ज़्यादा"), ("अत्यंत", "बेहद"), ("तत्पश्चात", "उसके बाद"),
              ("पश्चात", "बाद"), ("किंतु", "लेकिन"), ("परंतु", "लेकिन"), ("अतः", "इसलिए"),
              ("हेतु", "के लिए"), ("उपरोक्त", "ये"), ("आवश्यक", "ज़रूरी"), ("सुनिश्चित", "पक्का"),
              ("वर्तमान में", "अभी"), ("दर्शाती", "दिखाती"), ("दर्शाता", "दिखाता"), ("प्रदान", "दिया")]
# Big names / institutions the AI likes to drag in. A line may only name one if our footage does.
AUTHORITIES = {
    "प्रधानमंत्री": ("प्रधानमंत्री", "पीएम", "prime minister", " pm ", "modi", "मोदी"),
    "मोदी": ("मोदी", "modi", "प्रधानमंत्री", "prime minister"),
    "गृह मंत्री": ("गृह मंत्री", "गृहमंत्री", "home minister", "amit shah", "अमित शाह"),
    "अमित शाह": ("अमित शाह", "amit shah", "शाह"),
    "राष्ट्रपति": ("राष्ट्रपति", "president"),
    "सुप्रीम कोर्ट": ("सुप्रीम कोर्ट", "supreme court", "सर्वोच्च न्यायालय", "apex court"),
    "हाई कोर्ट": ("हाई कोर्ट", "high court", "उच्च न्यायालय"),
    "राहुल गांधी": ("राहुल", "rahul"), "केजरीवाल": ("केजरीवाल", "kejriwal"),
    "योगी": ("योगी", "yogi"), "मुख्यमंत्री": ("मुख्यमंत्री", "सीएम", "chief minister"),
    "सेना": ("सेना", "army", "फौज"), "पाकिस्तान": ("पाकिस्तान", "pakistan"),
    "चीन": ("चीन", "china"), "अमेरिका": ("अमेरिका", "america", "usa", "united states", "trump", "ट्रंप"),
}
DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
# Words that make a soundbite gripping (for picking teaser bites without the AI).
DRAMA = re.compile(
    r"घायल|मौत|मार|हिंसा|पत्थर|पथराव|खून|रो|परिवार|पापा|माँ|हमला|धमकी|चेतावनी|गिरफ्तार|खुलासा|"
    r"सच|साज़िश|साजिश|बवाल|हंगामा|आग|लाठी|ज़िंदा|जिंदा|जान|डर|शर्म|गुस्सा|बर्दाश्त|इजाज़त|इजाजत|"
    r"नहीं छोड़|बदला|ज़ुल्म|जुल्म|बेरहमी|टूट|बचा", re.I)
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


def speech_language(text, default="hi"):
    """'en' when a Devanagari transcript is really English speech written in Hindi letters."""
    words = DEVANAGARI_WORD.findall(text or "")
    if len(words) < 5:
        return default
    en = sum(w in EN_IN_DEVANAGARI for w in words) / len(words)
    hi = sum(w in HI_MARKERS for w in words) / len(words)
    return "en" if en >= 0.12 and en > 1.5 * hi else default


def trim_sentences(text, max_words):
    """Whole sentences only, up to max_words - never 'मेट्रो के 16' cut off mid-thought.
    A first sentence that is too long is cut at a comma, or dropped ('')."""
    parts = [p.strip() for p in re.split(r"(?<=[।!?])\s+|(?<=\.\.\.)\s+", text or "") if p.strip()]
    if len(parts) > 1 and not re.search(r"[।!?.]$", parts[-1]):
        parts = parts[:-1]              # a trailing fragment was cut off mid-thought
    out = []
    for p in parts:
        if len(" ".join(out + [p]).split()) > max_words:
            break
        out.append(p)
    if not out and parts:
        words = parts[0].split()
        cut = max((i for i, w in enumerate(words[:max_words]) if w.endswith(",")), default=-1)
        if cut + 1 >= max(4, max_words // 2):
            out = [" ".join(words[:cut + 1]).rstrip(",") + "..."]
    line = " ".join(out)
    if line and not re.search(r"[।!?.]$", line):
        line += "।"
    return line


def polish_line(text, max_words=NARRATION_WORDS):
    """Narrator-ready line: no picture descriptions, spoken (not formal) Hindi, whole sentences."""
    text = clean_text(text, 200)
    if VISUAL_DEIXIS.search(text):
        return ""          # describes a picture it can't see; the rest of the line leans on it
    for formal, spoken in COLLOQUIAL:
        text = text.replace(formal, spoken)
    return trim_sentences(text, max_words)


def _numbers(text):
    return {n.replace(",", "") for n in NUMBER.findall((text or "").translate(DEVANAGARI_DIGITS))
            if len(n.replace(",", "").split(".")[0]) >= 2}


def unsupported_claim(line, corpus):
    """What a narration line states that our footage never says (a number or a big name), or ''.
    corpus: lower-cased text of everything the sources say."""
    have = _numbers(corpus)
    for n in _numbers(line):
        if n not in have:
            return f"number {n}"
    padded = f" {corpus} "
    for name, aliases in AUTHORITIES.items():
        if name in line and not any(a in padded for a in aliases):
            return name
    return ""


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
    if video.get("downloadable") is False:
        return "reject", "YouTube won't serve this video (live / members-only / blocked)"
    if spoken in INDIAN_REGIONAL:
        return "reject", f"spoken in a regional language ({spoken})"
    if REGIONAL_SCRIPT.search(title) and spoken not in SPEECH_LANGS:
        return "reject", "regional-language title"
    if spoken in SPEECH_LANGS or (not spoken and has_text):
        return ("speech", "") if has_text else ("visual", "no transcript - visuals only")
    if spoken:
        return "visual", f"spoken in '{spoken}' - used only as silent visuals"
    return "visual", "no speech/transcript - visuals only"


SCREEN_SYSTEM = """You are the researcher of a viral Hindi news YouTube channel. Decide if a
YouTube video is about the SAME news story as the topic and usable as source footage.
ANY angle of the same story counts and is valuable: the background and earlier incidents that
led to it, victims and their families, police, protesters, politicians, courts, statements,
reactions, ground reports, raw/viral videos of the incident (kind=footage), debates.
Say relevant=false only for: a different story, comedy/stand-up, music, vlogs, gaming, memes,
reaction channels, study/jobs/travel videos, or videos that only mention the topic in passing.
JSON only."""

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
        v["lang"] = (v.get("spoken_lang") or v.get("caption_lang") or "hi").lower()
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
                lang = (video.get("lang") or video.get("spoken_lang")
                        or video.get("caption_lang") or "hi").lower()
                out.append({
                    "id": f"{vid}#{len(out)}", "video_id": vid, "index": len(out),
                    "lang": speech_language(text, lang) if lang in DIALOGUE_LANGS else lang,
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


ANNOTATE_SYSTEM = """You help the lead editor of a viral Hindi news YouTube channel. You read
transcript passages from one source video and judge each one as raw material for a film on the
topic: what is said, how strong it is, and its single most gripping line. Be strict: keep only
passages that clearly say something about the story. JSON only."""

ANNOTATE_USER = """Topic: {topic}
Film description: {description}
Source video: {title} ({channel})

Passages:
{listing}

For EVERY passage return one item:
{{"passages": [{{"n": 1, "use": true or false,
  "summary": "max 14 words, English: who says what",
  "topic": "1-3 word sub-topic, reuse the same label for the same thread",
  "strength": 1-5 (5 = powerful, emotional, shocking, quotable; 1 = filler),
  "emotion": "anger|grief|fear|shock|pride|defiance|threat|neutral",
  "punch": "the single most dramatic phrase of the passage, copied EXACTLY word for word, 4-14 words",
  "standalone": true if it makes sense without what came before{hindi_field}}}]}}"""

HINDI_FIELD = """,
  "hindi": "this passage is in English and our audience only hears Hindi: the Hindi
            voice-over (Devanagari) of its GIST in 1-2 punchy sentences, max 28 words, naming
            the speaker (e.g. 'अमेरिका के उपराष्ट्रपति JD Vance ने साफ़-साफ़ कह दिया...').
            Natural Hinglish like a passionate Delhi YouTuber. Faithful to what is said: no
            invented facts, numbers or quotes"""


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
                english = any(p.get("lang", "hi") not in DIALOGUE_LANGS for p in batch)
                raw = llm.chat_json(ANNOTATE_SYSTEM, ANNOTATE_USER.format(
                    topic=topic, description=description or "-", title=video["title"],
                    channel=video.get("channel", ""), listing=listing,
                    hindi_field=HINDI_FIELD if english else ""), temperature=0.2,
                    max_tokens=4200 if english else 2600)
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
                     emotion=str(item.get("emotion") or "neutral").lower()[:12], ai=True)
            if p.get("lang", "hi") not in DIALOGUE_LANGS:
                p["hindi"] = polish_line(item.get("hindi"), VOICEOVER_WORDS)
            elif isinstance(item.get("punch"), str):
                span = locate_phrase(p, item["punch"])
                if span:
                    p["punch"] = span
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


def _cue_windows(p, lo, hi, max_cues=3):
    cues = p.get("cues") or []
    for i in range(len(cues)):
        for j in range(i, min(len(cues), i + max_cues)):
            d = cues[j]["e"] - cues[i]["s"]
            if d > hi:
                break
            if d >= lo:
                yield cues[i]["s"], cues[j]["e"], " ".join(c["t"] for c in cues[i:j + 1]), cues[i:j + 1]


def locate_phrase(p, phrase, lo=1.5, hi=8.0):
    """Where in the passage the AI's quoted punch line is said: whole caption lines."""
    want = set(tokens(phrase))
    if len(want) < 2:
        return None
    best = None
    for s, e, text, _c in _cue_windows(p, lo, hi):
        have = set(tokens(text))
        score = len(want & have) / len(want) - 0.01 * (e - s)
        if best is None or score > best[0]:
            best = (score, s, e, text)
    if best is None or best[0] < 0.55:
        return None
    return {"s": round(best[1], 2), "e": round(best[2], 2), "t": best[3]}


def best_punch(p, lo=2.0, hi=6.5):
    """The punch line chosen by the AI, else the most gripping 2-6 s of Hindi in the passage."""
    if p.get("punch"):
        return p["punch"]
    best = None
    for s, e, text, cues in _cue_windows(p, lo, hi):
        if speech_language(text) != "hi" or len(text.split()) < 4:
            continue
        score = (sum(c.get("h", 0.3) for c in cues) / len(cues) + 0.35 * len(DRAMA.findall(text))
                 + 0.15 * bool(re.search(r"[!?]", text)))
        if best is None or score > best[0]:
            best = (score, s, e, text)
    return {"s": round(best[1], 2), "e": round(best[2], 2), "t": best[3]} if best else None


def visual_moments(video):
    """B-roll candidates: the most replayed seconds of a relevant video, with how much of the
    window is someone talking (studio anchors/graphics talk; action footage mostly doesn't)."""
    caps = video.get("captions") or []
    ms = build_moments(video, per_video=25)
    out = []
    for m in ms:
        span = max(0.1, m["end"] - m["start"])
        talk = sum(max(0.0, min(c["end"], m["end"]) - max(c["start"], m["start"])) for c in caps
                   if c["start"] < m["end"] and c["end"] > m["start"]) / span
        out.append({"video_id": m["video_id"], "start": m["start"], "end": m["end"],
                    "peak": m["peak"], "talk": round(min(1.0, talk), 2),
                    "video_duration": m["video_duration"], "video_title": m["video_title"],
                    "channel": m["channel"]})
    return out


# =================================================================== 3. story (architect)
ARCHITECT_SYSTEM = """You are the lead editor of India's most-watched Hindi news YouTube channel
(the style of the biggest Hindi news packages: drama, emotion, a clear side, rising tension).
The film is built from REAL Hindi soundbites - anchors, reporters, victims, police, leaders. They
carry the story. Our narrator is only the glue. Rules:
1. LET THE CLIPS TALK. Put Hindi dialogue scenes back-to-back; most of the film is people in the
   footage speaking. NEVER put narration between two Hindi clips of the same thread.
2. THE NARRATOR SPEAKS ONLY: one hook question in the opening; ONE short line that sets the base
   at the start of each act; at most one twist/reveal line inside an act; one closing line.
3. ONE THREAD AT A TIME, ESCALATING: buildup = what happened and who is involved; rising = the
   anger and conflict grow, stakes get personal (victims, families, threats); climax = the most
   explosive, emotional soundbites; ending = consequence + a strong closing line.
4. EVERY SCENE CONNECTS: "link" = the concrete reason it follows the previous scene
   (cause -> effect, claim -> reaction, question -> answer). No link, no scene.
5. VARIETY: many channels and voices; max 3 passages per video, 4 per channel. Never repeat a
   fact, claim or clip.
6. HINDI AUDIO ONLY. Passages marked EN are English and need our narrator to retell them - use
   EN passages only when they add a fact nobody says in Hindi (max 3 in the film).
7. NARRATION LINES: 8-20 words, a passionate young Delhi YouTuber - natural Hinglish in
   Devanagari, short punchy sentences, emotion, questions to the viewer ('सोचिए ज़रा!', 'आख़िर
   क्यों?', 'और यहीं से कहानी पलट गई...'). Never a formal news reader. NEVER describe what is on
   screen (you cannot see the footage - no 'देख रहे हो', no lights, no pictures).
8. Never invent facts, numbers, names or quotes: use only what the passages say. Dramatise HOW
   you tell it, not WHAT happened. Take the side the creator wants.
JSON only."""

ARCHITECT_USER = """Topic: {topic}
What the creator wants: {description}
Theme: {theme}
Target length: about {minutes} minutes (dialogue passages give most of the runtime)
Narration: {narration_rule}

DIALOGUE PASSAGES (id | language | source | sub-topic | seconds | strength 1-5 | emotion | what is said):
{passages}

VISUAL SOURCES for montages (id | what it is):
{visuals}

Scene types:
- "hook": the single most explosive Hindi line (use: passage id). Exactly one, first.
- "narration": a Hindi voice-over line (text) - only where rule 2 allows.
- "text": a short on-screen Hindi headline (text), max one per act.
- "dialogue": play a passage (use: passage id).
- "montage": music + action visuals, no words (use: list of visual ids, seconds: 8-12), max one, in the climax.

Return JSON:
{{"title_hi": "explosive Hindi YouTube title, max 9 words",
  "theme": "one line",
  "chapters": {{"opening": "Hindi chapter name", "buildup": "...", "rising": "...", "climax": "...", "ending": "..."}},
  "scenes": [{{"act": "opening|buildup|rising|climax|ending", "type": "...", "use": "P3 or [V1, V2]",
              "text": "Hindi, only for text/narration", "seconds": 10, "link": "why this follows"}}]}}
Use about {n_dialogue} dialogue scenes.{outline_rule}"""

CRITIC_SYSTEM = """You are the ruthless senior editor of a viral Hindi news YouTube channel,
reviewing a junior's edit before millions watch it. Viewers leave in the first 15 seconds
unless it grips them. You check: the film opens with fire; Hindi clips carry the story and
the narrator does NOT talk between clips of the same thread; each scene connects to the one
before; one thread at a time and the tension RISES act by act; nothing off-topic; NO
REPETITION (the same fact, claim, speaker or clip twice is cut); many channels and voices;
an emotional climax; a short, strong ending. You do not add narration - you cut and reorder.
JSON only."""

CRITIC_USER = """Topic: {topic}
Film description: {description}

DRAFT EDIT (number | act | type | source | content | editor's link):
{draft}

Fix it. Return JSON:
{{"order": [scene numbers in the best final order; leave out scenes that are off-topic,
            repetitive, weak or break the flow],
  "bridges": [{{"before": number of the FIRST scene of an act that needs a base-setting line,
               "type": "narration", "text": "Hindi line, 8-18 words, Delhi YouTuber Hinglish, only facts from the draft"}}],
  "score": 1-10 (how gripping and well-connected the film is after your fixes),
  "issues": ["max 5 short notes on what is still weak"]}}"""


EXTEND_SYSTEM = """You are the lead editor of a viral Hindi news YouTube channel. The film is
too short. Add more Hindi dialogue scenes WITHOUT breaking the flow: put a passage right after a
scene it continues (same speaker or same sub-topic) so the clips carry the story. Never add
off-topic material and never add narration. JSON only."""

EXTEND_USER = """Topic: {topic}
The film runs about {have} minutes and must reach about {want} minutes.

CURRENT FILM (number | act | type | source | content):
{draft}

UNUSED PASSAGES (id | source | sub-topic | seconds | strength | what is said):
{unused}

Return JSON: {{"insert": [{{"after": scene number, "type": "dialogue",
  "use": "P id", "link": "why it belongs here"}}]}}"""

TEASER_SYSTEM = """You cut the first 15 seconds of a viral Hindi news video - the part that
decides whether a viewer stays. From the candidate soundbites pick the 3 or 4 most gripping
(shock, anger, pain, a threat, a dramatic claim; each must make sense on its own) and order
them so the tension rises, strongest last. Then write the narrator's HOOK LINE that follows
them: max 20 words, Hinglish in Devanagari like a passionate Delhi YouTuber, a burning question
or bold claim that makes the viewer need the answer. Only facts from the bites and the topic;
never describe the pictures. JSON only."""

TEASER_USER = """Topic: {topic}
What the creator wants: {description}

CANDIDATE SOUNDBITES (id | channel | emotion | what is said):
{bites}

Return JSON: {{"bites": ["B3", "B1", "B7"], "hook_line": "Hindi hook line"}}"""

FACT_SYSTEM = """You are the fact-checker and script doctor of a Hindi news YouTube channel.
You get the narrator's lines and the ONLY facts we have (what is said in our footage). For each
line: if it states anything that is NOT in the facts (a person, a statement, a number, an event)
or describes what is on screen (lights, pictures, 'देख रहे हो'), rewrite it so it only uses the
facts. Keep the voice: a passionate Delhi YouTuber, Hinglish in Devanagari, short punchy
sentences, max {max_words} words. A line that is already fine is returned unchanged. If a line
cannot be saved, return "". JSON only."""

FACT_USER = """Topic: {topic}

FACTS (from our footage):
{facts}

NARRATOR LINES:
{lines}

Return JSON: {{"lines": [{{"n": 1, "text": "the checked line"}}]}}"""


def estimate_seconds(scenes, cat, clip_cap=None):
    total = 0.0
    for s in scenes:
        if s["type"] == "dialogue":
            length = cat[s["pid"]]["end"] - cat[s["pid"]]["start"]
            total += min(length, clip_cap) if clip_cap else length
        elif s["type"] == "hook":
            total += 7
        elif s["type"] in ("narration", "voiceover"):
            total += estimate_speech_seconds(s["text"]) + 0.9
        elif s["type"] == "text":
            total += clamp(1.8 + len(s["text"]) / 14, 2.8, 6.5)
        else:
            total += s.get("seconds", 12)
    return total


def _draft_lines(scenes, cat, vcat):
    lines = []
    for i, s in enumerate(scenes, 1):
        if s["type"] in ("hook", "dialogue", "voiceover"):
            p = cat[s["pid"]]
            content, src = f"[{p['subtopic']}] {p['summary']}", _source_label(p)
        elif s["type"] == "montage":
            content, src = "music montage", ", ".join(_source_label(vcat[v])[:30] for v in s["vids"]) or "-"
        else:
            content, src = s["text"], "-"
        lines.append(f"{i} | {s['act']} | {s['type']} | {src} | {content} | {s.get('link', '')}")
    return lines


def _extend(llm, topic, outline, cat, vcat, minutes, log):
    for _round in range(3):
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
            if new and new[0]["type"] == "dialogue":       # the clips carry the story
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


def make_catalog(passages, limit=70, per_video=4):
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


def plan_story(llm, topic, description, theme, minutes, narration, passages, videos, log,
               can_text=True, user_outline=""):
    for p in passages:     # English captioned in Hindi letters (also in jobs read before this check)
        if (p.get("lang") or "hi") in DIALOGUE_LANGS:
            p["lang"] = speech_language(p.get("text", ""), p.get("lang") or "hi")
    cat = make_catalog(passages, limit=90)
    vcat = make_visual_catalog(videos)
    clip_cap = CLIP_CAP if theme == "sensational" else None
    outline = None
    if llm.available() and cat:
        outline = _architect(llm, topic, description, theme, minutes, narration, cat, vcat, log,
                             user_outline)
        if outline:
            # rules first (they drop repeats / over-used sources), then let the AI fill the gap
            outline["scenes"] = enforce_rules(outline["scenes"], cat, narration, log, can_text)
            outline = _extend(llm, topic, outline, cat, vcat, minutes, log)
            outline = _critic(llm, topic, description, outline, cat, vcat, log)
    if not outline:
        if cat:
            log("Using the built-in story builder (start Ollama for an AI-edited story).")
        outline = fallback_story(topic, theme, cat, vcat, minutes)
    outline["scenes"] = enforce_rules(outline["scenes"], cat, narration, log, can_text)
    outline["scenes"] = fill_to_length(outline["scenes"], cat, minutes, clip_cap, log,
                                       vo_limit=VOICEOVER_LIMITS.get(narration, 2))
    outline = add_teaser(llm, topic, description, outline, cat, passages, log, narration=narration)
    outline["scenes"] = add_closing(outline["scenes"], narration)
    outline = check_facts(llm, topic, description, outline, cat, passages, videos, log)
    outline["scenes"] = space_out_narrator(outline["scenes"])
    outline["catalog"] = {k: p["id"] for k, p in cat.items()}
    outline["clip_cap"] = clip_cap
    outline["visual_catalog"] = {k: v["id"] for k, v in vcat.items()}
    assign_ids(outline)
    return outline


def _architect(llm, topic, description, theme, minutes, narration, cat, vcat, log, user_outline=""):
    rule = {"none": "NO voice-over at all. Use short on-screen headlines instead (max one per act).",
            "light": f"Narrator only for the hook and a few act openings: max {NARRATION_LIMITS['light']} "
                     "lines in the whole film.",
            "medium": f"Narrator for the hook, each act opening and one or two reveals: max "
                      f"{NARRATION_LIMITS['medium']} lines in the whole film."}[
        narration if narration in NARRATION_LIMITS else "light"]
    plines = "\n".join(
        f"{pid} | {'HI' if _speaks_hindi(p) else 'EN'} | {_source_label(p)} | {p['subtopic']} | "
        f"{int(p['end'] - p['start'])}s | {p['strength']} | {p.get('emotion', '-')} | {p['summary']}"
        for pid, p in cat.items())
    vlines = "\n".join(
        f"{vid} | {_source_label(v)}{' (foreign language, silent)' if v.get('role') == 'visual' else ''}"
        for vid, v in vcat.items())
    avg = min(CLIP_CAP, sum(p["end"] - p["start"] for p in cat.values()) / max(1, len(cat)))
    n_dialogue = int(clamp(minutes * 60 * 0.8 / max(10, avg), 6, 40))
    prompt = ARCHITECT_USER.format(topic=topic, description=description or "-",
                                   theme=THEMES.get(theme, THEMES["auto"]), minutes=minutes,
                                   narration_rule=rule, passages=plines, visuals=vlines or "(none)",
                                   n_dialogue=n_dialogue, outline_rule=(
                                       "\n\nTHE CREATOR'S OWN SCENE OUTLINE - follow it in this exact "
                                       "order, one or more scenes per point, using the passages that "
                                       "fit each point:\n" + user_outline.strip()) if user_outline.strip() else "")
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
            scene["seconds"] = clamp(_num(s.get("seconds"), 12) or 12, 6, 15)
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
            # A bridge only sets the base as an act opens - never between clips of a thread.
            opens_act = not new or new[-1]["act"] != s["act"]
            if n in bridges and opens_act and s["type"] not in ("text", "narration"):
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
def _speaks_hindi(p):
    return (p.get("lang") or "hi") in DIALOGUE_LANGS


class _Repeats:
    """'Does this passage say what an earlier one already said?' (two channels, same story).
    Word weights come from ALL passages, so words every passage shares (the topic itself:
    police, protest, ...) count for little and specific facts (numbers, names) count a lot."""

    def __init__(self, cat, threshold=0.45):
        self.keys = list(cat)
        texts = [cat[k].get("text", "") or "-" for k in self.keys]
        self.sim = tfidf_similarity(texts, texts) if texts else None
        self.index = {k: i for i, k in enumerate(self.keys)}
        self.threshold = threshold

    def __call__(self, pid, earlier):
        if self.sim is None or not earlier:
            return False
        i = self.index[pid]
        return max(float(self.sim[i, self.index[e]]) for e in earlier) >= self.threshold


def _same_line(line, earlier, threshold=0.6):
    words = set(tokens(line))
    for e in earlier:
        other = set(tokens(e))
        if words and other and len(words & other) / len(words | other) >= threshold:
            return True
    return False


def enforce_rules(scenes, cat, narration, log=lambda m: None, can_text=True):
    """Hard guarantees, whatever the AI wrote:
    - Hindi audio only: English passages become a short Hindi voice-over, and only the few
      strongest of them; the rest are left out.
    - The narrator never talks over the story: one line to set the base as an act opens, at
      most one reveal inside an act, never two lines in a row, a small total budget.
    - Variety caps per video and channel, nothing said twice, a Hindi hook first."""
    out, used, hooks, lines = [], [], 0, []
    per_video, per_channel, acts = {}, {}, {}
    limit = NARRATION_LIMITS.get(narration, NARRATION_LIMITS["light"])
    english = [s["pid"] for s in scenes if s["type"] in ("dialogue", "voiceover") and s.get("pid") in cat
               and not _speaks_hindi(cat[s["pid"]]) and cat[s["pid"]].get("hindi")]
    english = sorted(dict.fromkeys(english),
                     key=lambda k: (-cat[k].get("strength", 3), -cat[k].get("heat", 0)))
    allowed_en = set(english[:VOICEOVER_LIMITS.get(narration, 2)])
    n_narr = 0
    dropped = {"repeat": 0, "variety": 0, "english": 0, "narration": 0}
    repeats = _Repeats(cat)
    for s in scenes:
        s = dict(s)
        t = s["type"]
        act = acts.setdefault(s["act"], {"dialogue": 0, "lines": 0, "reveal": 0, "text": 0})
        if t == "hook":
            p = cat[s["pid"]]
            if s["act"] == "opening" and hooks < 1 and _speaks_hindi(p) and s["pid"] not in used:
                hooks += 1
                used.append(s["pid"])                 # the hook line is never replayed later
                out.append(s)
                continue
            if s["act"] == "opening" and not _speaks_hindi(p):
                continue                              # never open the film in English
            s["type"] = t = "dialogue"
        if t in ("dialogue", "voiceover"):
            p = cat[s["pid"]]
            if s["pid"] in used:
                continue                              # a passage plays once
            vid, ch = p["video_id"], p.get("channel") or p["video_id"]
            if per_video.get(vid, 0) >= MAX_PER_VIDEO or per_channel.get(ch, 0) >= MAX_PER_CHANNEL:
                dropped["variety"] += 1
                continue
            if repeats(s["pid"], used):
                dropped["repeat"] += 1
                continue
            if not _speaks_hindi(p):
                text = polish_line(p.get("hindi"), VOICEOVER_WORDS) if s["pid"] in allowed_en else ""
                if not text:
                    dropped["english"] += 1
                    continue
                s = {**s, "type": "voiceover", "text": text}
            elif t == "voiceover":
                s = {k: v for k, v in s.items() if k != "text"} | {"type": "dialogue"}
            used.append(s["pid"])
            per_video[vid] = per_video.get(vid, 0) + 1
            per_channel[ch] = per_channel.get(ch, 0) + 1
            act["dialogue"] += 1
            out.append(s)
            continue
        if t in ("text", "narration"):
            s["text"] = polish_line(s.get("text"), NARRATION_WORDS)
            if not s["text"] or _same_line(s["text"], lines):
                dropped["repeat" if s["text"] else "narration"] += 1
                continue
            if t == "text" and not can_text:
                s["type"] = t = "narration"           # can't draw text: the narrator says it
            if t == "narration":
                # Where the narrator may speak: as an act opens (sets the base), the hook question
                # and closing line, and one reveal inside an act. Never two lines in a row.
                reveal = act["dialogue"] > 0 and s["act"] not in ("opening", "ending")
                allowed = (n_narr < limit and (not out or out[-1]["type"] != "narration")
                           and act["lines"] < 2 and not (reveal and act["reveal"] >= 1))
                if not allowed:
                    if can_text and act["text"] < 1 and (not out or out[-1]["type"] not in ("text", "narration")):
                        s["type"] = t = "text"        # keep the fact as a short on-screen headline
                    else:
                        dropped["narration"] += 1
                        continue
                else:
                    n_narr += 1
                    act["lines"] += 1
                    act["reveal"] += reveal
            if t == "text":
                if act["text"] >= 1 or (out and out[-1]["type"] == "text"):
                    continue                          # max one headline per act, never two in a row
                act["text"] += 1
            lines.append(s["text"])
        out.append(s)

    # Open with fire: a Hindi hook first, never a text card. Prefer a line not used
    # elsewhere; otherwise the hook takes that scene's place (nothing is shown twice).
    if not out or out[0]["type"] != "hook":
        hindi = [(k, p) for k, p in cat.items() if _speaks_hindi(p)]
        if hindi:
            fresh = [kv for kv in hindi if kv[0] not in used] or hindi
            pid, _p = max(fresh, key=lambda kv: (kv[1]["strength"], kv[1]["heat"]))
            out = [s for s in out if not (s["type"] == "dialogue" and s.get("pid") == pid)]
            out.insert(0, {"act": "opening", "type": "hook", "pid": pid,
                           "link": "the most explosive line first"})
    if any(dropped.values()):
        log(f"Rules: dropped {dropped['repeat']} repeated, {dropped['variety']} over-used-source, "
            f"{dropped['english']} English and {dropped['narration']} extra narrator scenes.")

    # Within an act, one speaker's passages always play in their original order
    # (never "later sentence, then earlier sentence" from the same interview).
    for act in ACT_KEYS:
        slots = {}
        for i, s in enumerate(out):
            if s["act"] == act and s["type"] in ("dialogue", "voiceover"):
                slots.setdefault(cat[s["pid"]]["video_id"], []).append(i)
        for idx in slots.values():
            ordered = sorted((out[i] for i in idx), key=lambda s: cat[s["pid"]]["start"])
            for i, s in zip(idx, ordered):
                out[i] = s
    return out


def fill_to_length(scenes, cat, minutes, clip_cap=None, log=lambda m: None, vo_limit=3):
    """Guarantee the requested length: add unused strong passages where they belong - after
    the same speaker's earlier passage, else after the same sub-topic, else before the climax.
    Hindi passages play as clips first; English ones (retold by the narrator) only while the
    film has fewer than vo_limit voice-overs."""
    target = minutes * 60 * 0.92 - 4
    have = estimate_seconds(scenes, cat, clip_cap)
    if have >= target or not cat:
        return scenes
    scenes = list(scenes)
    used = [s["pid"] for s in scenes if s.get("pid")]
    per_video, per_channel = {}, {}
    for s in scenes:
        if s["type"] in ("dialogue", "voiceover") and s.get("pid") in cat:
            p = cat[s["pid"]]
            per_video[p["video_id"]] = per_video.get(p["video_id"], 0) + 1
            ch = p.get("channel") or p["video_id"]
            per_channel[ch] = per_channel.get(ch, 0) + 1
    repeats = _Repeats(cat)
    n_vo = sum(s["type"] == "voiceover" for s in scenes)
    pool = sorted((k for k in cat if k not in used),
                  key=lambda k: (not _speaks_hindi(cat[k]), -cat[k].get("strength", 3), -cat[k].get("heat", 0)))
    added = 0
    for cap_v, cap_c in ((MAX_PER_VIDEO, MAX_PER_CHANNEL), (MAX_PER_VIDEO + 1, MAX_PER_CHANNEL + 2)):
        for pid in pool:
            if have >= target:
                break
            if pid in used:
                continue
            p = cat[pid]
            vid, ch = p["video_id"], p.get("channel") or p["video_id"]
            if per_video.get(vid, 0) >= cap_v or per_channel.get(ch, 0) >= cap_c or repeats(pid, used):
                continue
            if _speaks_hindi(p):
                new = {"type": "dialogue", "pid": pid, "link": "continues this thread"}
            elif n_vo < vo_limit and polish_line(p.get("hindi"), VOICEOVER_WORDS):
                new = {"type": "voiceover", "pid": pid, "text": polish_line(p["hindi"], VOICEOVER_WORDS),
                       "link": "continues this thread"}
                n_vo += 1
            else:
                continue
            idx = None
            for i, sc in enumerate(scenes):          # after this speaker's earlier passage
                q = cat.get(sc.get("pid"))
                if q and sc["act"] != "opening" and q["video_id"] == vid and q["start"] < p["start"]:
                    idx = i
            if idx is None:                          # after the same thread
                for i, sc in enumerate(scenes):
                    q = cat.get(sc.get("pid"))
                    if q and sc["act"] != "opening" and q.get("subtopic") == p.get("subtopic"):
                        idx = i
            if idx is None:                          # before the climax
                idx = next((i - 1 for i, sc in enumerate(scenes) if sc["act"] in ("climax", "ending")),
                           len(scenes) - 1)
                idx = max(idx, 0)
            act = scenes[idx]["act"] if scenes else "buildup"
            new["act"] = "buildup" if act == "opening" else act
            scenes.insert(idx + 1, new)
            have += estimate_seconds([new], cat, clip_cap)
            used.append(pid)
            per_video[vid] = per_video.get(vid, 0) + 1
            per_channel[ch] = per_channel.get(ch, 0) + 1
            added += 1
    if added:
        log(f"Length: added {added} more scenes from unused material (about {have / 60:.1f} min planned).")
    return scenes


EMOTIONAL = {"anger", "grief", "shock", "fear", "threat", "defiance"}
ACTION_WORDS = re.compile(
    r"clash|violen|lathi|stone|pelt|injur|chaos|attack|fire|riot|protest|march|barricad|police|"
    r"crowd|viral|caught on camera|cctv|ground report|हंगाम|बवाल|लाठी|पथराव|झड़प|हिंसा|घायल|भगदड़|"
    r"आग|धक्का|प्रदर्शन|पुलिस|भीड़|वायरल|ग्राउंड", re.I)


def add_teaser(llm, topic, description, outline, cat, passages, log, max_bites=4, narration="light"):
    """The cold open: 3-4 of the most gripping Hindi soundbites from different channels, hard
    cut with flashes and hits, then the narrator's hook question. Replaces the single hook."""
    scenes = outline["scenes"]
    hook_id = next((cat[s["pid"]]["id"] for s in scenes if s["type"] == "hook" and s.get("pid") in cat), None)
    used = {cat[s["pid"]]["id"] for s in scenes if s.get("pid") in cat and s["type"] != "hook"}
    cands = []
    for p in passages:
        if not p.get("use") or not _speaks_hindi(p) or p["id"] in used:
            continue
        span = best_punch(p)
        if not span or speech_language(span["t"]) != "hi" or not 1.5 <= span["e"] - span["s"] <= 8.0:
            continue
        score = (p.get("strength", 3) + 1.5 * p.get("heat", 0.3) + 0.6 * len(DRAMA.findall(span["t"]))
                 + (1.0 if p.get("emotion") in EMOTIONAL else 0.0) + (2.0 if p["id"] == hook_id else 0.0))
        cands.append((score, p, span))
    cands.sort(key=lambda c: -c[0])
    pool, videos, channels = [], set(), {}
    for c in cands:                                   # one bite per video, max 2 per channel
        ch = c[1].get("channel") or c[1]["video_id"]
        if c[1]["video_id"] in videos or channels.get(ch, 0) >= 2:
            continue
        if any(_same_quote(c[2]["t"], x[2]["t"]) for x in pool):
            continue                                  # the same quote re-aired by another channel
        videos.add(c[1]["video_id"])
        channels[ch] = channels.get(ch, 0) + 1
        pool.append(c)
        if len(pool) >= 10:
            break
    if len(pool) < 2:
        return outline
    order, hook_line = [], ""
    if llm.available():
        listing = "\n".join(f"B{i} | {(p.get('channel') or '-')[:24]} | {p.get('emotion', '-')} | {span['t'][:170]}"
                            for i, (_sc, p, span) in enumerate(pool, 1))
        try:
            raw = llm.chat_json(TEASER_SYSTEM, TEASER_USER.format(
                topic=topic, description=description or "-", bites=listing), temperature=0.4, max_tokens=700)
            picks = [_num(x) for x in (raw.get("bites") or [])] if isinstance(raw, dict) else []
            order = [pool[n - 1] for n in dict.fromkeys(picks) if n and 1 <= n <= len(pool)]
            hook_line = polish_line(raw.get("hook_line"), NARRATION_WORDS) if isinstance(raw, dict) else ""
        except LLMError as e:
            log(f"AI could not pick the opening soundbites ({e}); using the strongest ones.")
    if len(order) < 2:
        order = sorted(pool[:3], key=lambda c: c[0])  # strongest last
    order = order[:max_bites]
    outline["teaser"] = [{"passage_id": p["id"], "video_id": p["video_id"], "start": span["s"],
                          "end": span["e"], "text": span["t"]} for _sc, p, span in order]
    scenes = [s for s in scenes if s["type"] != "hook"]
    if hook_line and narration == "none":             # no narrator: the hook is a headline
        scenes.insert(0, {"act": "opening", "type": "text", "text": hook_line,
                          "link": "the hook question after the cold open"})
    elif hook_line:
        first = scenes[0] if scenes else None
        if first and first["act"] == "opening" and first["type"] == "narration":
            scenes[0] = {**first, "text": hook_line, "link": "the hook question after the cold open"}
        else:
            scenes.insert(0, {"act": "opening", "type": "narration", "text": hook_line,
                              "link": "the hook question after the cold open"})
    hook_at = 0 if scenes and scenes[0].get("link") == "the hook question after the cold open" else -1
    for i, sc in enumerate(scenes):
        if sc["act"] == "opening" and i != hook_at:
            scenes[i] = {**sc, "act": "buildup"}
    outline["scenes"] = scenes
    log(f"Cold open: {len(order)} soundbites from {len({p['video_id'] for _s, p, _x in order})} "
        f"videos{', then the hook line' if hook_line else ''}.")
    return outline


def _same_quote(a, b, share=0.5):
    """Two soundbites carry the same quote (one may have a few more words around it)."""
    ka, kb = set(keywords(a)), set(keywords(b))
    return bool(ka and kb) and len(ka & kb) / min(len(ka), len(kb)) >= share


def space_out_narrator(scenes):
    """The narrator is never heard twice in a row (a line and then a retelling, or two
    retellings): the extra retelling goes, the clips carry the story."""
    talk = ("narration", "voiceover")
    out = []
    for i, s in enumerate(scenes):
        nxt = scenes[i + 1]["type"] if i + 1 < len(scenes) else None
        if s["type"] == "voiceover" and ((out and out[-1]["type"] in talk) or nxt in talk):
            continue
        if s["type"] == "narration" and out and out[-1]["type"] == "narration":
            if CTA_LINE in s.get("text", ""):
                out[-1] = s
            continue
        if s["type"] == "text" and ((out and out[-1]["type"] == "narration") or nxt == "narration"):
            continue                                  # the narrator is setting the base anyway
        out.append(s)
    return out


def add_closing(scenes, narration):
    """Indian YouTube closes by asking the viewer: the last narrator line ends with the
    comment call, or it gets its own short line."""
    if narration == "none" or not scenes:
        return scenes
    last = max((i for i, s in enumerate(scenes) if s["type"] == "narration"), default=None)
    if last is not None and scenes[last]["act"] == "ending" and CTA_LINE not in scenes[last]["text"]:
        if len((scenes[last]["text"] + " " + CTA_LINE).split()) <= NARRATION_WORDS + 8:
            scenes[last] = {**scenes[last], "text": scenes[last]["text"] + " " + CTA_LINE}
            return scenes
    if not any(CTA_LINE in s.get("text", "") for s in scenes) and scenes[-1]["type"] != "narration":
        scenes.append({"act": "ending", "type": "narration", "text": CTA_LINE, "link": "asks the viewer"})
    return scenes


def check_facts(llm, topic, description, outline, cat, passages, videos, log):
    """Every narrator line is checked against what our footage actually says: the AI rewrites
    lines that add facts or describe the picture, then lines that still name a number or a big
    name the sources never mention are cut."""
    corpus = " ".join([topic, description or ""] + [
        f"{p.get('text', '')} {p.get('hindi', '')} {p.get('summary', '')} {p.get('video_title', '')}"
        for p in passages] + [v.get("title", "") for v in videos]).lower()
    scenes = outline["scenes"]
    idx = [i for i, s in enumerate(scenes) if s["type"] in ("narration", "text")]
    if llm.available() and idx:
        used = [cat[s["pid"]] for s in scenes if s.get("pid") in cat]
        extra = sorted((p for p in cat.values() if p not in used), key=lambda p: -p.get("strength", 3))
        facts = "\n".join(f"- {p.get('summary', '')}: {p['text'][:200]}" for p in (used + extra)[:45])
        listing = "\n".join(f"{k}. {scenes[i]['text']}" for k, i in enumerate(idx, 1))
        try:
            raw = llm.chat_json(FACT_SYSTEM.format(max_words=NARRATION_WORDS), FACT_USER.format(
                topic=topic, facts=facts, lines=listing), temperature=0.2, max_tokens=2500)
            fixed = 0
            for item in raw.get("lines") or [] if isinstance(raw, dict) else []:
                k = _num(item.get("n")) if isinstance(item, dict) else None
                if not k or not 1 <= k <= len(idx) or not isinstance(item.get("text"), str):
                    continue
                i = idx[k - 1]
                had_cta = CTA_LINE in scenes[i]["text"]
                new = polish_line(item["text"], NARRATION_WORDS + (8 if had_cta else 0))
                if had_cta and CTA_LINE not in new:
                    new = polish_line(new, NARRATION_WORDS) + " " + CTA_LINE if new else CTA_LINE
                if new != scenes[i]["text"]:
                    scenes[i] = {**scenes[i], "text": new}
                    fixed += 1
            if fixed:
                log(f"Fact check: {fixed} narrator lines rewritten to stick to what the footage says.")
        except LLMError as e:
            log(f"Fact check by the AI skipped ({e}).")
    keep = []
    for s in scenes:
        if s["type"] in ("narration", "text", "voiceover") and s.get("text"):
            why = unsupported_claim(s["text"], corpus)
            if why:
                log(f"  cut a narrator line that the footage doesn't support ({why}): {s['text'][:60]}")
                continue
        if s["type"] in ("narration", "text") and not s.get("text"):
            continue
        keep.append(s)
    outline["scenes"] = keep
    return outline


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
        if s["type"] in ("narration", "voiceover"):
            n += 1
            s["narration_id"] = f"n{n:02d}"
    return outline


def narration_lines(outline):
    return {s["narration_id"]: s["text"] for s in outline["scenes"]
            if s["type"] in ("narration", "voiceover") and s.get("narration_id")}


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
        for m in self.visuals:      # how much a moment looks like action footage (not a studio)
            v = self.videos.get(m["video_id"], {})
            title = f"{m.get('video_title', '')} {v.get('title', '')}"
            m["action"] = round(0.6 * min(3, len(ACTION_WORDS.findall(title))) / 3
                                + (0.3 if v.get("kind") in ("footage", "ground_report") or v.get("role") == "visual" else 0)
                                + (0.2 if m.get("talk", 0) < 0.4 else 0) + 0.2 * m["peak"], 3)
        self.style = style
        self.total = total_seconds - 4.0          # title card
        self.log = log or (lambda m: None)
        self.used = {}
        self.clip_cap = None

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
        """Best lo-hi s run of whole caption lines inside a passage (highest replay heat)."""
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

    def broll(self, seconds, prefer=(), avoid_vid=None, shot=3.0, action=False, exclude=()):
        clips, total, last = [], 0.0, avoid_vid
        if action:      # clashes, crowds, barricades first; studio talk last
            quiet = sorted(self.visuals, key=lambda m: -m.get("action", 0))
        else:
            quiet = sorted(self.visuals, key=lambda m: (m.get("talk", 0) > 0.4, -m["peak"]))
        quiet = [m for m in quiet if m["video_id"] not in exclude]
        pool = [m for m in quiet if m["video_id"] in prefer] + \
               [m for m in quiet if m["video_id"] not in prefer]
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

    def lead_in(self, next_p, seconds, single=False):
        """Visuals for a bridge: the moments just before the next speaker starts (J-cut feel)."""
        if next_p:
            e = next_p["start"] - 0.3
            s = e - seconds
            if s >= 0.5 and self.free(next_p["video_id"], s, e):
                self.take(next_p["video_id"], s, e)
                return [self.clip(next_p["video_id"], s, e)]
        prefer = (next_p["video_id"],) if next_p else ()
        return self.broll(seconds, prefer=prefer, shot=seconds if single else 3.0, action=True)

    # -- main
    def build(self, outline, narration_seconds, theme):
        self.clip_cap = outline.get("clip_cap")
        cat = {k: self.passages.get(v) for k, v in outline.get("catalog", {}).items()}
        vcat = outline.get("visual_catalog", {})
        scenes = outline["scenes"]
        moods = THEME_MOODS.get(theme) or THEME_MOODS["epic"]
        beats_by_act = {k: [] for k in ACT_KEYS}

        # reserve dialogue footage first so bridges never steal it
        for s in scenes:
            if s["type"] in ("dialogue", "voiceover") and cat.get(s["pid"]):
                p = cat[s["pid"]]
                self.take(p["video_id"], p["start"] - 0.2, p["end"] + 0.4)

        # Cold open: the soundbites, hard cuts with a flash, a punch-in and a hit on each.
        teaser = outline.get("teaser") or []
        for k, bite in enumerate(teaser):
            p = self.passages.get(bite["passage_id"])
            if not p:
                continue
            a, b = bite["start"] - 0.12, bite["end"] + 0.3
            self.take(p["video_id"], a, b)
            c = self.clip(p["video_id"], a, b, bite["text"], p.get("heat", 0.5))
            c.update(fx="punch", zoom=k % 2 == 1)
            beats_by_act["opening"].append({
                "scene_id": f"t{k + 1}", "kind": "teaser", "idea": "cold open: a gripping line first",
                "narration": "", "audio": "original", "said": bite["text"], "source": _source_label(p),
                "passage_id": p["id"], "strength": p.get("strength", 3), "clips": [c], "sfx": "hit"})
        if teaser:
            self.total += 4.0      # the title is a real beat (counted), not a separate card

        for i, s in enumerate(scenes):
            nxt = next((cat.get(x["pid"]) for x in scenes[i + 1:]
                        if x["type"] in ("dialogue", "hook", "voiceover") and cat.get(x["pid"])), None)
            beat = {"scene_id": s["id"], "kind": s["type"], "idea": s.get("link", ""),
                    "narration": "", "clips": [], "said": "", "source": ""}
            if s["type"] in ("hook", "dialogue"):
                p = cat.get(s["pid"])
                if not p:
                    continue
                if s["type"] == "hook":
                    a, b, text = self.hook_span(p)
                elif self.clip_cap and p["end"] - p["start"] > self.clip_cap + 2:
                    # Hindi-news style: the sharpest 10-20 s of the passage, whole sentences
                    a, b, text = self.hook_span(p, lo=10.0, hi=self.clip_cap)
                    a, b = a - 0.15, b + 0.35
                else:
                    a, b, text = p["start"] - 0.15, p["end"] + 0.35, p["text"]
                beat.update(audio="original", said=text[:400], source=_source_label(p),
                            passage_id=p["id"], strength=p.get("strength", 3),
                            clips=[self.clip(p["video_id"], a, b, text, p["heat"])])
            elif s["type"] == "voiceover":
                # English speaker on screen, our Hindi narrator says what they said.
                p = cat.get(s["pid"])
                if not p:
                    continue
                d = narration_seconds.get(s.get("narration_id"), estimate_speech_seconds(s["text"])) + 0.9
                a = p["start"] - 0.15
                beat.update(audio="narration", narration=s["text"], narration_id=s.get("narration_id"),
                            said=p["text"][:400], source=_source_label(p) + " (EN, retold in Hindi)",
                            passage_id=p["id"], strength=p.get("strength", 3),
                            clips=[self.clip(p["video_id"], a, a + d, p["text"], p["heat"])])
            elif s["type"] == "narration":
                d = narration_seconds.get(s.get("narration_id"), estimate_speech_seconds(s["text"])) + 0.9
                beat.update(audio="narration", narration=s["text"], narration_id=s.get("narration_id"),
                            clips=(self.broll(d, action=True) if s["act"] == "opening"
                                   else self.lead_in(nxt, d)))
            elif s["type"] == "text":
                # Text over moving, darkened footage of what comes next - never a frozen frame.
                secs = round(clamp(1.8 + len(s["text"]) / 14, 2.8, 6.5), 2)
                beat.update(audio="text", narration=s["text"], seconds=secs,
                            clips=self.lead_in(nxt, secs, single=True))
            else:
                prefer = tuple(vcat.get(v) for v in s.get("vids", []) if vcat.get(v))
                secs = min(s.get("seconds", 12), 10 if s["act"] == "ending" else 15)
                beat.update(audio="music", clips=self.broll(
                    secs, prefer=prefer, shot=self.style["acts"][s["act"]]["shot"]))
                if not beat["clips"]:
                    continue
            beats_by_act[s["act"]].append(beat)

        if teaser and outline.get("title_hi"):
            # Title sting: the film's name in big letters over action footage, with a boom.
            shots = self.broll(3.2, action=True, shot=3.2)
            if shots:
                beats_by_act["opening"].append({
                    "scene_id": "title", "kind": "title", "idea": "title sting", "audio": "text",
                    "narration": outline["title_hi"], "overlay_style": "title", "seconds": 3.2,
                    "clips": shots, "said": "", "source": "", "sfx": "boom"})
        for key in ACT_KEYS[1:]:     # a whoosh carries the viewer into every new chapter
            if beats_by_act[key]:
                beats_by_act[key][0].setdefault("sfx", "whoosh")

        acts = []
        for ai, key in enumerate(ACT_KEYS):
            if beats_by_act[key]:
                acts.append({"key": key, "title_hi": outline.get("chapters", {}).get(key, ACT_NAMES_HI[key]),
                             "mood": moods[ai], "beats": beats_by_act[key]})
        story = {"title_hi": outline.get("title_hi", ""), "theme": outline.get("theme", ""),
                 "source": outline.get("source", ""), "report": outline.get("report"),
                 "acts": acts, "warnings": []}
        self.fit(story)
        if theme in ("sensational", "thriller", "documentary", "epic"):
            self.cutaways(story)
        for act in story["acts"]:
            for b in act["beats"]:
                b["seconds"] = round(beat_seconds(b), 2)
        return story

    def cutaways(self, story, min_len=9.0):
        """News-package editing: during a long soundbite the speaker stays on screen first, then
        we cut to action footage while their voice keeps going, then back to them."""
        for act in story["acts"]:
            for b in act["beats"]:
                p = self.passages.get(b.get("passage_id"))
                if b["kind"] != "dialogue" or (p and p.get("emotion") in ("grief", "fear")):
                    continue                    # pain and fear stay on the face
                new = []
                for c in b["clips"]:
                    length = c["end"] - c["start"]
                    if length < min_len or c.get("audio"):
                        new.append(c)
                        continue
                    head = max(3.5, 0.25 * length)
                    span = min(length - head, 0.55 * length)
                    if length - head - span < 1.5:
                        span = length - head        # don't return to the speaker for a blink
                    shots = self.broll(span, shot=3.0, action=True, exclude=(c["video_id"],))
                    if not shots:
                        new.append(c)
                        continue
                    t = round(c["start"] + head, 2)
                    new.append({**c, "end": t})
                    for sh in shots:
                        new.append({**sh, "audio": {"video_id": c["video_id"], "start": t}})
                        t = round(t + sh["end"] - sh["start"], 2)
                    if c["end"] - t > 0.3:
                        new.append({**c, "start": t, "text": ""})
                b["clips"] = new

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
                if sum(x["end"] - x["start"] for x in b["clips"]) > (
                        1.6 * self.clip_cap if self.clip_cap else 60):
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
                if self.clip_cap:
                    # Hindi-news style: the speaker continues as one more sharp cut (max 2 per scene)
                    if len(b["clips"]) >= 2:
                        continue
                    a, e, text = self.hook_span(nxt, lo=6.0, hi=self.clip_cap)
                    self.take(c["video_id"], nxt["start"], nxt["end"])
                    b["clips"].append(self.clip(c["video_id"], a - 0.15, e + 0.35, text, nxt["heat"]))
                    b["said"] = (b["said"] + " … " + text)[:600]
                    grew = True
                    break
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
                need = min(15.0 - beat_seconds(b), self.total * 0.95 - total())
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


def without_clips(story, gone):
    """A copy of the story without the shots whose footage is unavailable (gone: {(video_id,
    start)}). A cutaway whose picture is missing shows the speaker instead, so their words
    never jump."""
    import copy
    out = copy.deepcopy(story)
    for act in out["acts"]:
        for b in act["beats"]:
            kept = []
            for c in b["clips"]:
                if (c["video_id"], round(c["start"], 2)) not in gone:
                    kept.append(c)
                elif c.get("audio") and (c["audio"]["video_id"], round(c["audio"]["start"], 2)) not in gone:
                    a = c["audio"]
                    kept.append({**{k: v for k, v in c.items() if k != "audio"}, "video_id": a["video_id"],
                                 "start": a["start"], "end": round(a["start"] + c["end"] - c["start"], 2)})
            b["clips"] = kept
    return out


def beat_seconds(beat):
    if beat.get("audio") == "text" and not beat.get("clips"):
        return beat.get("seconds", 3.0)
    return sum(c["end"] - c["start"] for c in beat.get("clips", []))
