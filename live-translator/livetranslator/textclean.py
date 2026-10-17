"""Clean Whisper output: drop known hallucinations, sound tags and loops."""
from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

# Whisper's German training data contains lots of TV subtitles, so on noise or
# silence it "hears" the credits. These are never real meeting speech.
_ALWAYS_FAKE = [
    "untertitel im auftrag des zdf",
    "untertitelung im auftrag des zdf",
    "untertitel im auftrag des zdf für funk",
    "untertitel der amara.org-community",
    "untertitel von stephanie geiges",
    "untertitelung aufgrund der amara.org-community",
    "copyright wdr",
    "swr 2020",
    "swr 2021",
    "ard text im auftrag von funk",
    "untertitel: ",
    "amara.org",
    "zdf 2020",
    "zdf 2021",
    "© br",
    "mehr infos auf",
    "abonniere den kanal",
    "abonnieren nicht vergessen",
    "thank you for watching",
    "thanks for watching",
    "please subscribe",
]
# Plausible as real speech, but typical hallucinations when the audio is weak.
_SUSPICIOUS = [
    "vielen dank fürs zuschauen",
    "vielen dank für's zuschauen",
    "danke fürs zuschauen",
    "bis zum nächsten mal",
    "tschüss",
    "das war's für heute",
    "wir sehen uns im nächsten video",
    "vielen dank",
    "danke",
    "ich danke ihnen",
    "untertitel",
]
# A line consisting only of one of these is a sound description, not speech.
_EXACT_FAKE = {"musik", "applaus", "lachen", "stille", "untertitel", "music", "applause", "silence", "you", "ähm", "äh", "hm", "hmm"}

# a line that is only hesitation sounds ("Uuuuhh.", "Ähm, äh…", "Hmm") carries nothing to translate
_FILLER_RE = re.compile(r"^(?:(?:u+h+|ä+h*m*|e+h*m*|h+m+|m+h+|a+h+|o+h+)[\s,.!?…-]*)+$", re.I)
_TAG_RE = re.compile(r"(\[[^\]]{0,40}\]|\((?:musik|applaus|lachen|lacht|gelächter|husten|räuspern|seufzt|stille|unverständlich|music|laughter|applause|silence)[^)]{0,20}\)|\*[^*]{0,30}\*|♪+|♫+)", re.I)
_WS_RE = re.compile(r"\s+")


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).lower().strip()
    s = re.sub(r"[\"'“”„«»]", "", s)
    s = re.sub(r"[.!?…,;:\-–—]+$", "", s).strip()
    return s


def collapse_repeats(text: str, max_repeats: int = 2) -> str:
    """'und dann und dann und dann und dann' -> 'und dann und dann'."""
    words = text.split()
    if len(words) < 4:
        return text
    changed = True
    while changed:
        changed = False
        for n in range(1, 9):
            i = 0
            while i + n * (max_repeats + 1) <= len(words):
                unit = [w.lower().strip(",.") for w in words[i:i + n]]
                j = i + n
                reps = 1
                while j + n <= len(words) and [w.lower().strip(",.") for w in words[j:j + n]] == unit:
                    reps += 1
                    j += n
                if reps > max_repeats:
                    words = words[:i + n * max_repeats] + words[j:]
                    changed = True
                else:
                    i += 1
    return " ".join(words)


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, _norm(a), _norm(b)).ratio()


def clean_transcript(
    text: str,
    *,
    duration_s: float = 0.0,
    avg_logprob: float = 0.0,
    no_speech_prob: float = 0.0,
    prompt_terms: str = "",
) -> str:
    """Return the cleaned line, or "" if it should be discarded."""
    if not text:
        return ""
    t = _TAG_RE.sub(" ", text)
    t = _WS_RE.sub(" ", t).strip()
    t = t.strip("-–— ")
    if not t or not re.search(r"\w", t):
        return ""
    n = _norm(t)
    for fake in _ALWAYS_FAKE:
        if fake in n:
            # remove the fake part; keep real speech around it if any
            t2 = re.sub(re.escape(fake), " ", n, flags=re.I)
            if len(re.sub(r"\W", "", t2)) < 4:
                return ""
            # rebuild from original casing where possible: drop sentences with the fake
            parts = re.split(r"(?<=[.!?])\s+", t)
            t = " ".join(p for p in parts if fake not in _norm(p)).strip()
            if not t:
                return ""
            n = _norm(t)
    if n in _EXACT_FAKE or _FILLER_RE.match(n):
        return ""
    weak = no_speech_prob > 0.5 or avg_logprob < -0.9
    # polite closings are what Whisper (and its German fine-tune) "hears" in noise: need clear speech
    if n in _SUSPICIOUS and (weak or no_speech_prob > 0.3 or avg_logprob < -0.6):
        return ""
    if weak and no_speech_prob > 0.8 and avg_logprob < -0.7:
        return ""
    # Whisper sometimes just repeats the prompt (glossary) back
    if prompt_terms and len(n) > 12 and _norm(prompt_terms).find(n) >= 0:
        return ""
    t = collapse_repeats(t)
    return fix_eszett(t)


# The German fine-tuned model writes "ss" for "ß" (a known quirk). Only unambiguous words are fixed.
_ESZETT = {
    "strasse": "straße", "strassen": "straßen", "gross": "groß", "grosse": "große", "grossen": "großen",
    "grosser": "großer", "grosses": "großes", "grösse": "größe", "grössen": "größen", "grösser": "größer",
    "grösste": "größte", "grössten": "größten", "heisst": "heißt", "heissen": "heißen", "weiss": "weiß",
    "schliesslich": "schließlich", "ausserdem": "außerdem", "ausser": "außer", "äusserst": "äußerst",
    "massstab": "maßstab", "massnahme": "maßnahme", "massnahmen": "maßnahmen", "gruss": "gruß",
    "grüsse": "grüße", "grüssen": "grüßen", "fuss": "fuß", "spass": "spaß", "gemäss": "gemäß",
    "regelmässig": "regelmäßig", "mässig": "mäßig", "grossartig": "großartig", "draussen": "draußen",
    "begrüssen": "begrüßen", "abschliessen": "abschließen", "schliessen": "schließen"
}
_ESZETT_RE = re.compile(r"\b(" + "|".join(sorted((k for k in _ESZETT if "ss" in k), key=len, reverse=True)) + r")\b",
                        re.I)


# nouns are capitalised anywhere; any other listed word capitalised mid-sentence is a name (Herr Gross, Frau Weiss)
_ESZETT_NOUNS = {"strasse", "strassen", "grösse", "grössen", "massstab", "massnahme", "massnahmen", "gruss", "grüsse",
                 "fuss", "spass"}


def fix_eszett(text: str) -> str:
    def sub(m: re.Match) -> str:
        w = m.group(0)
        if w[0].isupper() and w.lower() not in _ESZETT_NOUNS:
            before = text[:m.start()].rstrip()
            if before and before[-1] not in ".!?:„\"(":
                return w                       # mid-sentence capital: a surname, leave it
        fixed = _ESZETT[w.lower()]
        if w.isupper():
            return fixed.upper()
        return fixed[0].upper() + fixed[1:] if w[0].isupper() else fixed
    return _ESZETT_RE.sub(sub, text) if "ss" in text.lower() else text


# --------------------------------------------------------------------- end of a line?
# Words after which a German sentence cannot be over (the speaker is just taking a breath):
# conjunctions, articles and contracted prepositions. Verbs, pronouns, "nicht" and separable
# particles ("an", "auf", "mit", ...) are left out on purpose - German sentences often end with them.
_OPEN_END_WORDS = {
    "und", "oder", "aber", "sondern", "dass", "daß", "weil", "wenn", "ob", "obwohl", "falls", "bevor",
    "nachdem", "sodass", "sowie", "beziehungsweise", "bzw", "sowohl", "entweder", "weder",
    "eine", "einen", "einem", "einer", "eines", "des", "dem", "den",
    "zum", "zur", "beim", "vom", "im", "am", "für", "von", "ähm", "äh", "ehm", "öhm",
}
_LAST_WORD_RE = re.compile(r"[a-zäöüß]+", re.I)


def sounds_unfinished(text: str) -> bool:
    """True when the words so far clearly continue (trailing comma, "und", "dass", "für", ...)."""
    t = (text or "").strip()
    if not t:
        return False
    if t.endswith((",", ";", ":", "-", "–", "—", "…", "...")):
        return True
    words = _LAST_WORD_RE.findall(t.lower())
    return bool(words) and words[-1] in _OPEN_END_WORDS
