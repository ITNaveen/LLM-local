"""Stage 3 - story: the 'editor brain'. Writes a five-act Hindi story (opening, build-up,
rising tension, climax, ending), decides for every beat who speaks (narrator, the clip
itself, or just music), then picks the exact seconds of footage for each beat."""

import random
import re

from .llm import LLMError, Similarity
from .style import ACT_KEYS, THEME_MOODS, THEMES
from .util import clamp, estimate_speech_seconds, keywords

AUDIO_MODES = ("narration", "original", "music")
ACT_NAMES_HI = {"opening": "शुरुआत", "buildup": "कहानी", "rising": "तूफ़ान से पहले",
                "climax": "चरम", "ending": "अंजाम"}
ACT_BEATS = {"opening": (2, 3), "buildup": (4, 6), "rising": (3, 4),
             "climax": (3, 4), "ending": (2, 3)}
NARRATION_LIMITS = {"none": 0, "light": 7, "medium": 12}

OUTLINE_SYSTEM = """You are India's best YouTube story-video editor and Hindi scriptwriter.
You turn raw YouTube footage into a gripping 8-15 minute mini-movie that Indian audiences
watch till the end. Rules:
- Five acts: opening (hook + atmosphere), buildup (set up the story scene by scene),
  rising (tension, stakes, pace increases), climax (the peak moments, goosebumps), ending
  (resolution, emotional close).
- Every beat says who carries the audio:
  "original" = the footage's own voice (speech, commentary, crowd) tells the story, no narrator;
  "narration" = a short Hindi voice-over line connects scenes or explains context;
  "music" = no words, pure visuals on music (montages, the climax).
- Narration only where it is needed to connect or explain. Let strong footage speak.
  The climax is mostly "original" and "music".
- Narration is spoken Hindi in Devanagari: short, powerful, cinematic sentences, like a
  movie trailer narrator. 8-22 words per line. No hashtags, no emojis, no "subscribe".
- Use the numbered footage moments that fit each beat, in story order. Strong speech
  moments go to "original" beats. High-replay moments go to the climax.
Return JSON only."""

OUTLINE_USER = """Topic: {topic}
Story description from the creator: {description}
Theme: {theme}
Target length: {minutes} minutes
Narration amount: {narration} (at most {max_lines} narration lines in total)

Footage moments available (number | type | replay heat 0-100 | video | what is said):
{moments}

Return JSON exactly in this shape:
{{
  "title_hi": "short powerful Hindi title for the video",
  "theme": "one line describing the emotional theme",
  "acts": [
    {{"key": "opening", "title_hi": "Hindi chapter name", "goal": "what this act does",
      "beats": [
        {{"idea": "what the viewer sees/feels in this beat (English, specific)",
          "audio": "music|original|narration",
          "narration": "Hindi line if audio is narration, else empty",
          "moments": [moment numbers that fit, in order]}}
      ]}},
    {{"key": "buildup", ...}}, {{"key": "rising", ...}}, {{"key": "climax", ...}}, {{"key": "ending", ...}}
  ]
}}
Beats per act: opening 2-3, buildup 4-6, rising 3-4, climax 3-4, ending 2-3."""


# ------------------------------------------------------------------ outline
def moment_menu(moments, limit=44, per_video=4):
    """Strongest, diverse moments for the LLM to see (numbered from 1)."""
    by_score = sorted(moments, key=lambda m: m["score"], reverse=True)
    counts, menu = {}, []
    for m in by_score:
        if counts.get(m["video_id"], 0) >= per_video:
            continue
        counts[m["video_id"]] = counts.get(m["video_id"], 0) + 1
        menu.append(m)
        if len(menu) >= limit:
            break
    return menu


def format_menu(menu):
    lines = []
    for i, m in enumerate(menu, 1):
        text = (m.get("text") or "(no speech)").replace("\n", " ")[:110]
        lines.append(f"{i} | {m['kind']} | {int(m['peak'] * 100)} | "
                     f"{m['video_title'][:45]} | {text}")
    return "\n".join(lines)


def plan_outline(llm, topic, description, theme, minutes, narration, moments, log):
    menu = moment_menu(moments)
    outline = None
    if llm.available():
        prompt = OUTLINE_USER.format(
            topic=topic, description=description or "-",
            theme=THEMES.get(theme, THEMES["auto"]), minutes=minutes, narration=narration,
            max_lines=NARRATION_LIMITS.get(narration, 7), moments=format_menu(menu))
        for attempt in range(2):
            try:
                raw = llm.chat_json(OUTLINE_SYSTEM, prompt, temperature=0.75, max_tokens=6000)
                outline = normalize_outline(raw, menu, topic, theme, narration)
                if outline:
                    log(f"AI wrote the story: \"{outline['title_hi']}\" ({_count_beats(outline)} beats).")
                    break
                log("AI story was incomplete, retrying..." if attempt == 0 else
                    "AI story still incomplete.")
            except LLMError as e:
                log(f"AI story writing failed ({e}).")
    if not outline:
        log("Using the built-in story template (start Ollama for a smarter, AI-written story).")
        outline = fallback_outline(topic, description, theme, narration, menu)
    return outline


def _count_beats(outline):
    return sum(len(a["beats"]) for a in outline["acts"])


def normalize_outline(raw, menu, topic, theme, narration):
    """Validate/repair LLM output into our schema. Returns None if unusable."""
    if not isinstance(raw, dict) or not isinstance(raw.get("acts"), list):
        return None
    by_key = {}
    for i, act in enumerate(raw["acts"]):
        if not isinstance(act, dict):
            continue
        key = str(act.get("key", "")).lower().strip()
        if key not in ACT_KEYS and i < len(ACT_KEYS):
            key = ACT_KEYS[i]
        if key in ACT_KEYS and key not in by_key:
            by_key[key] = act
    if len(by_key) < 4:
        return None
    max_lines = NARRATION_LIMITS.get(narration, 7)
    lines_used = 0
    moods = THEME_MOODS.get(theme) or THEME_MOODS["epic"]
    acts = []
    for ai, key in enumerate(ACT_KEYS):
        src = by_key.get(key, {"beats": []})
        beats = []
        for b in src.get("beats") or []:
            if not isinstance(b, dict):
                continue
            audio = str(b.get("audio", "music")).lower().strip()
            audio = audio if audio in AUDIO_MODES else "music"
            text = clean_narration(b.get("narration"))
            if audio == "narration" and (not text or lines_used >= max_lines):
                audio = "music" if key in ("climax", "opening") else "original"
                text = ""
            if audio != "narration":
                text = ""
            else:
                lines_used += 1
            nums = b.get("moments") or []
            if not isinstance(nums, list):
                nums = [nums]
            ids = []
            for n in nums:
                try:
                    idx = int(str(n).strip()) - 1
                except ValueError:
                    continue
                if 0 <= idx < len(menu) and menu[idx]["id"] not in ids:
                    ids.append(menu[idx]["id"])
            beats.append({"idea": str(b.get("idea") or "")[:300], "audio": audio,
                          "narration": text, "moment_ids": ids[:8]})
        lo, hi = ACT_BEATS[key]
        beats = beats[:hi + 1]
        while len(beats) < lo:
            beats.append({"idea": f"{topic} {key}", "audio": "music" if key == "climax" else "original",
                          "narration": "", "moment_ids": []})
        acts.append({"key": key, "title_hi": str(src.get("title_hi") or ACT_NAMES_HI[key])[:60],
                     "goal": str(src.get("goal") or "")[:300], "mood": moods[ai], "beats": beats})
    title = clean_narration(raw.get("title_hi")) or topic
    return {"title_hi": title[:90], "theme": str(raw.get("theme") or theme)[:200],
            "acts": acts, "source": "ai"}


def clean_narration(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"[#*_`\"“”]|\(.*?\)|\[.*?\]", "", text)
    text = re.sub(r"(?i)subscribe|सब्सक्राइब", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    return " ".join(words[:28])


# ------------------------------------------------------------------ fallback
FALLBACK_LINES = {
    "opening": ["कुछ कहानियाँ सिर्फ़ सुनी नहीं जातीं, महसूस की जाती हैं। ये कहानी है {topic} की।"],
    "buildup": ["सब कुछ यहीं से शुरू हुआ। हालात आसान नहीं थे, लेकिन इरादे मज़बूत थे।",
                "हर गुज़रते दिन के साथ उम्मीदें बढ़ती गईं, और दबाव भी।"],
    "rising":  ["और फिर आया वो वक़्त, जब पूरे देश की नज़रें एक ही जगह टिकी थीं।"],
    "climax":  [],
    "ending":  ["ये सिर्फ़ एक पल नहीं था। ये इतिहास था। और इतिहास हमेशा याद रखा जाता है।"],
}
FALLBACK_SHAPES = {
    "opening": ["music", "narration"],
    "buildup": ["narration", "original", "original", "narration", "original"],
    "rising":  ["music", "narration", "original", "original"],
    "climax":  ["music", "original", "music", "original"],
    "ending":  ["narration", "original", "music"],
}


def fallback_outline(topic, description, theme, narration, menu):
    theme = theme if theme in THEME_MOODS else "epic"
    max_lines = NARRATION_LIMITS.get(narration, 7)
    sentences = [s.strip() for s in re.split(r"[.!?।,;\n]+", description or "") if len(s.strip()) > 3]
    lines_used, acts = 0, []
    n_beats = len(ACT_KEYS) * 4
    for ai, key in enumerate(ACT_KEYS):
        lines = list(FALLBACK_LINES[key])
        beats = []
        for bi, audio in enumerate(FALLBACK_SHAPES[key]):
            text = ""
            if audio == "narration":
                if lines and lines_used < max_lines:
                    text = lines.pop(0).format(topic=topic)
                    lines_used += 1
                else:
                    audio = "original" if key in ("buildup", "rising") else "music"
            # walk through the creator's description in order, act by act
            pos = (ai * len(FALLBACK_SHAPES[key]) + bi) * len(sentences) // max(1, n_beats)
            idea_src = sentences[min(len(sentences) - 1, pos)] if sentences else topic
            beats.append({"idea": f"{topic}: {idea_src}", "audio": audio,
                          "narration": text, "moment_ids": []})
        acts.append({"key": key, "title_hi": ACT_NAMES_HI[key], "goal": "",
                     "mood": THEME_MOODS[theme][ai], "beats": beats})
    return {"title_hi": topic, "theme": theme, "acts": acts, "source": "template"}


# ------------------------------------------------------------------ filling
class Filler:
    """Turns beats into concrete clips (video id + seconds) that fit the time budget."""

    def __init__(self, moments, style, total_seconds, topic, llm=None, seed=7, log=None):
        self.moments = {m["id"]: m for m in moments}
        self.order = sorted(moments, key=lambda m: m["score"], reverse=True)
        self.style = style
        self.total = total_seconds - 4.0   # leave room for the title card
        self.topic = topic
        self.llm = llm
        self.rng = random.Random(seed)
        self.log = log or (lambda m: None)
        self.used = {}   # video_id -> list of (start, end)

    # -- bookkeeping
    def free(self, vid, start, end, pad=1.0):
        return all(end + pad <= s or start - pad >= e for s, e in self.used.get(vid, []))

    def take(self, vid, start, end):
        self.used.setdefault(vid, []).append((start, end))

    # -- shot extraction
    def shot_from(self, m, length):
        """A shot of `length` seconds centred on the moment's hottest part."""
        dur = m.get("video_duration") or (m["end"] + 30)
        centre = (m["start"] + m["end"]) / 2
        start = clamp(centre - length * 0.45, 0.5, max(0.5, dur - length - 0.5))
        return start, start + length

    def speech_clip(self, m):
        start = max(0.2, m["start"] - 0.15)
        end = min(m.get("video_duration") or m["end"] + 1, m["end"] + 0.3)
        if end - start > 16:
            end = start + 16
        return start, end

    def clip(self, m, mode, length=None):
        if mode == "original":
            start, end = self.speech_clip(m)
        else:
            start, end = self.shot_from(m, length)
        return {"moment_id": m["id"], "video_id": m["video_id"], "start": round(start, 2),
                "end": round(end, 2), "kind": m["kind"], "text": m.get("text", "")[:200],
                "heat": m["peak"], "video_title": m["video_title"], "channel": m["channel"]}

    # -- main
    def fill(self, outline, narration_seconds):
        acts = outline["acts"]
        beat_texts, beat_refs = [], []
        for act in acts:
            for b in act["beats"]:
                beat_texts.append(f"{self.topic}. {b['idea']}. {b['narration']}")
                beat_refs.append(b)
        docs = [f"{m['video_title']}. {m.get('text', '')}" for m in self.order]
        sims = Similarity(self.llm or _NullEmbed(), self.log).matrix(beat_texts, docs)
        sim_of = {id(b): sims[i] for i, b in enumerate(beat_refs)}

        # Cold open: flash-forward teaser of the hottest moments (allowed to repeat later).
        hottest = sorted(self.order, key=lambda m: m["peak"], reverse=True)
        teaser = []
        seen_videos = set()
        for m in hottest:
            if m["video_id"] in seen_videos:
                continue
            seen_videos.add(m["video_id"])
            teaser.append(self.clip(m, "music", 1.6 + self.rng.random() * 0.6))
            if len(teaser) >= 4:
                break

        for act in acts:
            budget = self.total * self.style["acts"][act["key"]]["share"]
            if act["key"] == "opening":
                budget = max(0.0, budget - sum(c["end"] - c["start"] for c in teaser))
            self.fill_act(act, budget, sim_of, narration_seconds)
            if act["key"] == "opening" and teaser:
                act["beats"].insert(0, {"idea": "Cold open: flash-forward to the biggest moments",
                                        "audio": "music", "narration": "", "moment_ids": [],
                                        "clips": teaser, "teaser": True})
        self.fit_total(outline)
        for act in acts:
            for b in act["beats"]:
                b["seconds"] = round(beat_seconds(b), 2)
        return outline

    def fill_act(self, act, budget, sim_of, narration_seconds):
        key = act["key"]
        shot = self.style["acts"][key]["shot"]
        beats = act["beats"]
        # Share the act's budget between beats: narration beats need their voice length.
        fixed = {}
        for i, b in enumerate(beats):
            if b["audio"] == "narration":
                fixed[i] = narration_seconds.get(b.get("narration_id", ""), estimate_speech_seconds(b["narration"])) + 0.9
        flexible = [i for i in range(len(beats)) if i not in fixed]
        rest = max(0.0, budget - sum(fixed.values()))
        weights = {i: (1.4 if beats[i]["audio"] == "original" else 1.0) for i in flexible}
        wsum = sum(weights.values()) or 1.0
        for i, b in enumerate(beats):
            target = fixed[i] if i in fixed else max(shot * 1.5, rest * weights[i] / wsum)
            b["target"] = round(target, 2)
            b["clips"] = self.pick(b, key, target, shot, sim_of[id(b)])

    def rank(self, beat, act_key, sims):
        mode = beat["audio"]
        heat_w = 0.45 if act_key in ("climax", "rising") else 0.3
        sim_w = 0.6 - heat_w + 0.15
        scored = []
        for i, m in enumerate(self.order):
            if mode == "original" and (m["kind"] == "visual" or len(m.get("text", "")) < 12):
                continue
            s = sim_w * float(sims[i]) + heat_w * m["peak"] + 0.25 * m["score"]
            scored.append((s, m))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored]

    def pick(self, beat, act_key, target, shot, sims, strict_done=False):
        mode = beat["audio"]
        clips, total, last_vid = [], 0.0, None
        preferred = [self.moments[i] for i in beat.get("moment_ids", []) if i in self.moments]
        ranked = preferred + [m for m in self.rank(beat, act_key, sims) if m not in preferred]
        for m in ranked:
            if total >= target - 0.5:
                break
            remaining = target - total
            if mode == "original":
                if m["kind"] == "visual" or not m.get("text"):
                    continue
                c = self.clip(m, mode)
                length = c["end"] - c["start"]
                # keep whole sentences, but don't let one long speech blow the beat's budget
                if length > max(remaining + 4, 7.0) and not (strict_done and total == 0):
                    continue
            else:
                length = shot * (0.8 + 0.4 * self.rng.random())
                if remaining < length * 1.5:
                    length = remaining
                if length < 1.0:
                    break
                c = self.clip(m, mode, length)
            if m["video_id"] == last_vid and mode != "original" and len(ranked) > 3:
                continue
            if not self.free(c["video_id"], c["start"], c["end"]):
                continue
            self.take(c["video_id"], c["start"], c["end"])
            clips.append(c)
            total += c["end"] - c["start"]
            last_vid = c["video_id"]
        if mode == "original" and not clips:
            if not strict_done:  # nothing short enough: accept one longer sentence
                return self.pick(beat, act_key, target, shot, sims, strict_done=True)
            beat["audio"] = "music"   # no usable speech left: turn into a montage
            return self.pick(beat, act_key, target, shot, sims)
        return clips

    def fit_total(self, outline):
        """Top up (or trim) so the full video lands close to the requested length."""
        def total():
            return sum(beat_seconds(b) for a in outline["acts"] for b in a["beats"])
        guard = 0
        while total() < self.total * 0.97 and guard < 200:
            guard += 1
            added = False
            for key in ("buildup", "rising", "climax"):
                act = next(a for a in outline["acts"] if a["key"] == key)
                beat = next((b for b in act["beats"] if b["audio"] == "music"), None)
                if beat is None:
                    beat = {"idea": f"{self.topic} montage", "audio": "music", "narration": "",
                            "moment_ids": [], "clips": []}
                    act["beats"].insert(len(act["beats"]) // 2 + 1, beat)
                shot = self.style["acts"][key]["shot"]
                for m in self.order:
                    c = self.clip(m, "music", shot)
                    if self.free(c["video_id"], c["start"], c["end"]):
                        self.take(c["video_id"], c["start"], c["end"])
                        beat["clips"].append(c)
                        added = True
                        break
                if total() >= self.total * 0.97:
                    break
            if not added:
                self.log("Ran out of fresh footage; the video will be shorter than requested.")
                break
        # Too long: drop shots from the most padded beats (never narration or the cold open).
        while total() > self.total * 1.06:
            beats = [b for a in outline["acts"] if a["key"] in ("buildup", "rising", "ending", "climax")
                     for b in a["beats"]
                     if b["audio"] != "narration" and not b.get("teaser") and b.get("clips")]
            multi = [b for b in beats if len(b["clips"]) > 1]
            if multi:
                max(multi, key=beat_seconds)["clips"].pop()
                continue
            removable = [(a, b) for a in outline["acts"] for b in a["beats"]
                         if b in beats and len(a["beats"]) > 2]
            if not removable:
                break
            act, beat = max(removable, key=lambda ab: beat_seconds(ab[1]))
            act["beats"].remove(beat)


def beat_seconds(beat):
    return sum(c["end"] - c["start"] for c in beat.get("clips", []))


class _NullEmbed:
    def embed_model(self):
        return ""


def assign_narration_ids(outline):
    n = 0
    for act in outline["acts"]:
        for b in act["beats"]:
            if b["audio"] == "narration" and b["narration"]:
                n += 1
                b["narration_id"] = f"n{n:02d}"
    return outline


def story_keywords(outline):
    return keywords(" ".join(b["idea"] for a in outline["acts"] for b in a["beats"]))[:30]
