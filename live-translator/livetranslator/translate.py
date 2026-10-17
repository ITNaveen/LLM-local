"""German -> English translation with a local LLM (Ollama).

Why an LLM and not a phrase translator (DeepL & co): live speech arrives in
fragments ("...dass wir das Angebot bis Freitag" / "nicht mehr schaffen.").
German puts the verb - and often the negation - at the end, so a translator
that sees one fragment at a time guesses wrong. The LLM sees the recent
conversation, knows the topic and the glossary, and fixes obvious
speech-recognition slips from context.

Speed tricks:
* conversation is sent as chat turns that only ever get *appended* (and trimmed
  in big steps), so Ollama re-uses its prompt cache and only has to read the
  new line -> first English words appear a fraction of a second after the pause;
* the English is streamed token by token to the screen;
* the model is pre-loaded and kept in memory for the whole meeting.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

import httpx

log = logging.getLogger("lt.translate")

KEEP_ALIVE = "8h"
NUM_CTX = 4096     # enough for the system prompt + ~15 lines of conversation


def _ollama_stats(done: dict) -> dict:
    """Ollama's own timings from the final stream message (durations are in nanoseconds)."""
    ns = 1e9
    gen_tokens = int(done.get("eval_count") or 0)
    gen_s = float(done.get("eval_duration") or 0) / ns
    return {
        "load_s": round(float(done.get("load_duration") or 0) / ns, 3),
        "prompt_tokens": int(done.get("prompt_eval_count") or 0),
        "prompt_s": round(float(done.get("prompt_eval_duration") or 0) / ns, 3),
        "gen_tokens": gen_tokens,
        "gen_s": round(gen_s, 3),
        "tok_s": round(gen_tokens / gen_s, 1) if gen_s > 0 and gen_tokens >= 3 else 0.0,
    }

SYSTEM_PROMPT = """You are an expert German-to-English interpreter working live in a business meeting.

You receive the meeting one line at a time, exactly as automatic speech recognition (ASR) wrote it down. Reply to each line with the English translation of that line only.

Rules:
- Be faithful and complete. Keep every name, number, amount, date, time, unit and condition exactly as said.
- Negations are critical: nicht, kein, nie, nichts, niemand - and the dialect forms net, ned, nit, nöd, koa, nix - all negate. Never drop or add a negation.
- Numbers and times: 1.500,50 = 1,500.50; Milliarde = billion; Billion = trillion; zwo = two; halb drei = 2:30; viertel nach drei = 3:15; dreiviertel drei = 2:45; KW 12 = calendar week 12; Jänner = January.
- Speakers have regional accents and dialects (Bavarian, Swabian, Saxon, Franconian, Austrian, Swiss). Translate the meaning into neutral English: mir/mia as subject = we, i = I, des = das, heuer = this year, passt/basst = OK, gell/gelle/ne/oder at the end = "right?"; particles like fei, halt, eben, doch, ja, schon usually need no translation.
- Business false friends: aktuell = currently, eventuell = possibly, Termin = appointment or deadline, Abstimmung = alignment, Chef = boss, Handy = mobile phone, bekommen = get, Angebot = quote or offer, Freigabe = approval.
- Speech recognition makes mistakes. If a word makes no sense, translate what the speaker most likely said, judging from the previous lines and the meeting topic. If you cannot tell, translate it literally. Never invent content.
- A line may be an unfinished sentence cut at a pause; it continues in the next line. Translate only what is there and use the previous lines to understand it. Do not complete it.
- Leave out pure fillers (äh, ähm, also, sozusagen) unless they carry meaning.
- Keep names of people, companies, products, systems and abbreviations unchanged.
- Never answer, comment, explain or add notes. Output only the English translation."""


def build_system_prompt(topic: str = "", terms: list[str] | None = None) -> str:
    extra = []
    if topic.strip():
        extra.append(f"Meeting context: {topic.strip()}")
    if terms:
        extra.append("Names and terms that may come up (spell them like this): " + ", ".join(terms))
    return SYSTEM_PROMPT + ("\n\n" + "\n".join(extra) if extra else "")


_PREFIX_RE = re.compile(r"^\s*(?:\*\*)?(?:english|en|translation|übersetzung|englisch)(?:\s*translation)?\s*[:：]\s*(?:\*\*)?\s*", re.I)
_NOTE_RE = re.compile(r"\s*\((?:note|translator'?s? note|nb)[:\s][^)]*\)\s*$", re.I)
_THINK_RE = re.compile(r"<think>.*?</think>", re.S)


def clean_translation(text: str, source: str = "") -> str:
    t = _THINK_RE.sub("", text)
    if "<think>" in t:  # unfinished reasoning block
        t = t.split("<think>")[0]
    t = t.strip()
    t = _PREFIX_RE.sub("", t)
    # models sometimes add a second line with notes/alternatives
    lines = [ln for ln in t.splitlines() if ln.strip()]
    if len(lines) > 1 and "\n" not in source.strip():
        keep = [lines[0]]
        for ln in lines[1:]:
            if re.match(r"^\s*(\(|note|alternative|or:|literally|\*|—|-{2,})", ln, re.I):
                break
            keep.append(ln)
        t = " ".join(x.strip() for x in keep)
    t = _NOTE_RE.sub("", t).strip()
    if len(t) >= 2 and t[0] == t[-1] and t[0] in "\"'“”" and not source.strip().startswith(("\"", "„", "“")):
        t = t[1:-1].strip()
    if t.startswith("“") and t.endswith("”") and not source.strip().startswith(("„", "“")):
        t = t[1:-1].strip()
    return t


def is_translategemma(model: str) -> bool:
    return model.lower().startswith("translategemma")


# TranslateGemma is trained on exactly this instruction (no system prompt; two blank lines before
# the text). It takes no glossary or topic, so the general models stay the default.
TG_PREFIX = ("You are a professional German (de) to English (en) translator. Your goal is to accurately convey the "
             "meaning and nuances of the original German text while adhering to English grammar, vocabulary, and "
             "cultural sensitivities.\nProduce only the English translation, without any additional explanations "
             "or commentary. Please translate the following German text into English:\n\n\n")
TG_HISTORY = 6

# standard negations (any case) and dialect forms (lower case, not ".NET 8" / "Net Zero" / "Koalition")
_NEG_DE = re.compile(r"\b(nicht|nichts|kein\w*|nie|niemals|niemand|nirgends|weder|nix)\b", re.I)
_NEG_DIALECT = re.compile(r"(?<![.\w])(net|ned|nit|nöd|koa|koan|koane|koana|koans|koam)\b")   # lower case only
# a tag question ("…, oder nicht?", "gell, ned?", "nicht wahr?") asks for agreement - it negates nothing
# (also after a comma at the end of a line without the "?", and with a name after it: "…, oder nicht, Herr
# Müller?" - but "Das ist nicht wahr." and "ob wir das schaffen oder nicht" are real negations)
_TAG_WORDS = r"(?:(?:oder|oda|odr|gell|ne)(?:\s*,)?\s+(?:etwa\s+)?(?:nicht|net|ned|nit|nöd)|nicht wahr)"
_TAG_END = r"(?=\s*(?:,\s*[\w.]+(?:\s+[\w.]+){0,2}\s*)?[?.!]?\s*$)"
_TAG_Q = re.compile(rf",\s*{_TAG_WORDS}\b{_TAG_END}|,?\s*\b{_TAG_WORDS}\s*\?", re.I)
_NEG_EN = re.compile(r"\b(not|no|never|nobody|nothing|none|neither|nor|cannot|without|nowhere|no-one|unable)\b|n[’']t\b",
                     re.I)
# German times and dates are written differently in English (15 Uhr = 3 pm, 15.3. = March 15)
_TIME_RE = re.compile(r"\b\d{1,2}[:.]\d{2}\b|\b\d{1,2}\.\d{1,2}\.(?:\d{2,4})?(?!\d)")
_UHR_RE = re.compile(r"\b(\d{1,2})\s*Uhr(?:\s*(\d{1,2})\b)?", re.I)
_NUM_RE = re.compile(r"\d+(?:[.,]\d+)*")
_WORDS = {0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight",
          9: "nine", 10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen", 15: "fifteen",
          16: "sixteen", 17: "seventeen", 18: "eighteen", 19: "nineteen", 20: "twenty", 30: "thirty", 40: "forty",
          50: "fifty", 60: "sixty", 70: "seventy", 80: "eighty", 90: "ninety", 100: "hundred", 1000: "thousand"}
_ORDINALS = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth", 7: "seventh", 8: "eighth",
             9: "ninth", 10: "tenth", 11: "eleventh", 12: "twelfth"}


def _num_de(n: str) -> str:
    """German number spelling -> plain digits (1.500,50 -> 1500.50; 1.500 -> 1500)."""
    if "," in n or re.fullmatch(r"\d{1,3}(\.\d{3})+", n):
        return n.replace(".", "").replace(",", ".")
    return n


def check_translation(german: str, english: str) -> list[str]:
    """Cheap checks for the two mistakes that hurt most: a lost negation and a changed number.
    Returns what to keep (empty = looks fine)."""
    problems = []
    de_neg = _TAG_Q.sub("?", german)
    if (_NEG_DE.search(de_neg) or _NEG_DIALECT.search(de_neg)) and not _NEG_EN.search(english):
        problems.append("the negation")
    en_nums = {x.replace(",", "") for x in _NUM_RE.findall(english)}
    en_low = english.lower()
    missing = []
    for m in _UHR_RE.finditer(german):                    # 15 Uhr = 3 pm; 14 Uhr 30 = 2:30
        h = int(m.group(1))
        if not ({str(h), str(h - 12) if h > 12 else str(h + 12)} & en_nums):
            missing.append(m.group(0))
    de = _TIME_RE.sub(" ", _UHR_RE.sub(" ", german))
    for n in _NUM_RE.findall(de):
        v = _num_de(n)
        if v in en_nums or ("." in v and v.rstrip("0").rstrip(".") in en_nums):
            continue
        if v.isdigit() and ((int(v) in _WORDS and re.search(rf"\b{_WORDS[int(v)]}", en_low))
                            or (int(v) in _ORDINALS and re.search(rf"\b{_ORDINALS[int(v)]}\b", en_low))):
            continue
        missing.append(n)
    if missing:
        problems.append("the numbers " + ", ".join(missing))
    return problems


_DE_COMMON = {"und", "der", "die", "das", "nicht", "ist", "wir", "ich", "sie", "mit", "auf", "für", "dass", "noch",
              "auch", "haben", "werden", "wird", "sind", "ein", "eine", "den", "dem", "zu"}


def _untranslated(german: str, english: str) -> bool:
    """The model gave the German back. "Okay." -> "Okay.", a name or a list of names ("Müller,
    Schäfer, Köhler?") is a correct translation, so only clearly German text (several words with
    German function words) counts as an echo."""
    a = re.sub(r"\W+", " ", german.lower()).strip()
    b = re.sub(r"\W+", " ", english.lower()).strip()
    de_words = a.split()
    if a and a == b:
        lower_umlaut = any(re.search(r"[äöüß]", t) for t in re.findall(r"\w+", german)[1:] if t.islower())
        return len(de_words) >= 3 and (any(x in _DE_COMMON for x in de_words) or lower_umlaut)
    # lower-case German function words in the English (names keep their capitals: "Die Mobiliar")
    w = re.findall(r"\w+", english)
    return len(w) >= 6 and sum(x in _DE_COMMON for x in w) / len(w) > 0.25


@dataclass
class TranslationResult:
    text: str
    ok: bool
    first_token_s: float = 0.0
    total_s: float = 0.0
    error: str = ""
    timeout: bool = False       # Ollama sent nothing within the time limit
    truncated: bool = False     # generation was cut off (far too slow) - not a usable translation
    stats: dict = field(default_factory=dict)   # Ollama's own timings (load, prompt, generation)

    @property
    def tok_s(self) -> float:
        return float(self.stats.get("tok_s") or 0.0)


class OllamaError(RuntimeError):
    pass


class OllamaTranslator:
    def __init__(self, base_url: str, model: str, context_lines: int = 12, topic: str = "", terms: list[str] | None = None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.context_lines = context_lines
        self.system = build_system_prompt(topic, terms)
        self.history: list[tuple[str, str]] = []
        self._gen = 0              # bumped by reset() / seed_history(): a new meeting's context
        self._lock = threading.Lock()
        self._client = httpx.Client(timeout=httpx.Timeout(60.0, connect=3.0), trust_env=False)

    # ------------------------------------------------------------- settings
    def configure(self, *, model: str | None = None, context_lines: int | None = None, topic: str | None = None,
                  terms: list[str] | None = None, base_url: str | None = None) -> None:
        with self._lock:
            if model:
                self.model = model
            if context_lines is not None:
                self.context_lines = context_lines
            if base_url:
                self.base_url = base_url.rstrip("/")
            if topic is not None or terms is not None:
                self.system = build_system_prompt(topic or "", terms or [])

    def reset(self) -> None:
        with self._lock:
            self.history = []
            self._gen += 1

    def seed_history(self, pairs: list[tuple[str, str]]) -> None:
        with self._lock:
            self.history = [(d, e) for d, e in pairs if d and e][-self.context_lines:]
            self._gen += 1

    # ------------------------------------------------------------- service
    def health(self) -> dict:
        """{'running': bool, 'models': [...], 'model_ready': bool}"""
        try:
            r = self._client.get(f"{self.base_url}/api/tags", timeout=2.0)
            r.raise_for_status()
            models = [m.get("name", "") for m in r.json().get("models", [])]
        except Exception as e:  # noqa: BLE001
            return {"running": False, "models": [], "model_ready": False, "error": str(e)}
        want = self.model if ":" in self.model else self.model + ":latest"
        return {"running": True, "models": models, "model_ready": want in models or self.model in models}

    def warmup(self, model: str | None = None, first_token_timeout: float = 180.0) -> TranslationResult:
        """Load the model into memory and prime the prompt cache with the system prompt."""
        return self.translate("Guten Morgen zusammen.", record=False, model=model,
                              first_token_timeout=first_token_timeout)

    def pull(self, model: str, on_progress: Callable[[dict], None]) -> None:
        with self._client.stream("POST", f"{self.base_url}/api/pull", json={"model": model, "stream": True},
                                 timeout=httpx.Timeout(None, connect=3.0)) as r:
            for line in r.iter_lines():
                if not line:
                    continue
                obj = json.loads(line)
                if obj.get("error"):
                    raise OllamaError(obj["error"])
                on_progress(obj)

    # ------------------------------------------------------------- translate
    def _messages(self, german: str, model: str, hint: str = "", history: list | None = None) -> list[dict]:
        history = self.history if history is None else history
        if is_translategemma(model):
            msgs = []                    # its template has no system role
            for de, en in history[-TG_HISTORY:]:
                msgs.append({"role": "user", "content": TG_PREFIX + de})
                msgs.append({"role": "assistant", "content": en})
            msgs.append({"role": "user", "content": TG_PREFIX + german})
            return msgs
        msgs = [{"role": "system", "content": self.system}]
        for de, en in history:
            msgs.append({"role": "user", "content": de})
            msgs.append({"role": "assistant", "content": en})
        msgs.append({"role": "user", "content": german + hint})
        return msgs

    MAX_HISTORY_CHARS = 6000   # ~1.8k tokens: system prompt + history + line + answer stay well inside NUM_CTX

    def _trim_history(self) -> None:
        # Trim in big steps: the conversation prefix then stays identical for
        # many lines in a row, which is what makes Ollama's prompt cache hit.
        keep = self.context_lines
        if keep <= 0:
            self.history = []
            return
        # Every trim makes Ollama re-read the whole conversation once (seconds on a big model),
        # so trim rarely: let it grow by 10 lines, then cut back to `keep`.
        if len(self.history) > keep + 10:
            self.history = self.history[-keep:]
        # hard size cap (long monologue lines): drop the oldest half until it fits
        while len(self.history) > 1 and sum(len(d) + len(e) for d, e in self.history) > self.MAX_HISTORY_CHARS:
            self.history = self.history[len(self.history) // 2:]

    @staticmethod
    def plausible(german: str, english: str) -> bool:
        """A translation that is far longer than its German (rambling, an echoed prompt)
        must not become context for the next lines - it would derail and slow them."""
        return 0 < len(english) <= 3 * len(german) + 80

    CHECK_WITHIN_S = 3.0     # a line that took longer is not asked again (the translator is busy)
    recheck = True           # the thermal governor switches the second question off when the Mac is hot

    def translate(self, german: str, on_delta: Callable[[str], None] | None = None, record: bool = True,
                  first_token_timeout: float = 60.0, max_gen_s: float = 45.0,
                  model: str | None = None) -> TranslationResult:
        """Translate one line. Gives up if Ollama sends nothing for `first_token_timeout`
        seconds (model stuck loading / machine out of memory) instead of waiting forever.
        A real line (record=True) whose English lost a negation or a number is asked once more."""
        german = german.strip()
        if not german:
            return TranslationResult("", True)
        model = model or self.model
        gen = self._gen
        res = self._request(german, on_delta, first_token_timeout, max_gen_s, model)
        if res.ok and record and is_translategemma(model) and _untranslated(german, res.text):
            # TranslateGemma sometimes echoes the German: once more, without context
            again = self._request(german, on_delta, first_token_timeout, max_gen_s, model, history=[])
            if again.ok and not _untranslated(german, again.text):
                res = again
            else:
                log.warning("translate [%s]: the model repeated the German: %r", model, german)
                return TranslationResult(res.text, False, res.first_token_s, res.total_s,
                                         "the model repeated the German instead of translating")
        if (res.ok and record and self.recheck and not is_translategemma(model)
                and res.total_s < self.CHECK_WITHIN_S):
            problems = check_translation(german, res.text)
            if problems:
                hint = "\n\n(Check your translation: keep " + " and ".join(problems) + ".)"
                log.info("translation check: %s missing in %r -> asking again", " / ".join(problems), res.text)
                again = self._request(german, on_delta, first_token_timeout, max_gen_s, model, hint)
                if again.ok:
                    again.text = re.sub(r"\s*\(?check your translation[^)]*\)?\s*$", "", again.text, flags=re.I).strip()
                if (again.ok and again.text and "check your translation" not in again.text.lower()
                        and len(check_translation(german, again.text)) < len(problems)):
                    again.first_token_s = res.first_token_s
                    again.total_s += res.total_s
                    res = again
        if record and res.ok and res.text and self.plausible(german, res.text):
            with self._lock:
                if self._gen == gen:     # a new meeting started meanwhile: not its context
                    self.history.append((german, res.text))
                    self._trim_history()
        return res

    def _request(self, german: str, on_delta, first_token_timeout: float, max_gen_s: float, model: str,
                 hint: str = "", history: list | None = None) -> TranslationResult:
        with self._lock:
            msgs = self._messages(german, model, hint, history)
        body = {
            "model": model,
            "messages": msgs,
            "stream": True,
            "keep_alive": KEEP_ALIVE,
            # always off: a thinking model would write its reasoning first (seconds before any English);
            # Ollama accepts "think": false for every model
            "think": False,
            "options": {"temperature": 0.0, "top_p": 0.9, "num_ctx": NUM_CTX,
                        "num_predict": max(64, min(600, len(german) * 2))},
        }
        t0 = time.perf_counter()
        first = 0.0
        raw = ""
        stats: dict = {}
        timeout = httpx.Timeout(connect=3.0, read=first_token_timeout, write=10.0, pool=10.0)
        try:
            with self._client.stream("POST", f"{self.base_url}/api/chat", json=body, timeout=timeout) as r:
                if r.status_code != 200:
                    r.read()
                    try:
                        msg = r.json().get("error", r.text)
                    except Exception:  # noqa: BLE001
                        msg = r.text
                    raise OllamaError(f"Ollama {r.status_code}: {msg}")
                for line in r.iter_lines():
                    if not line:
                        continue
                    obj = json.loads(line)
                    if obj.get("error"):
                        raise OllamaError(obj["error"])
                    piece = (obj.get("message") or {}).get("content") or ""
                    if piece:
                        if not first:
                            first = time.perf_counter() - t0
                        raw += piece
                        if on_delta:
                            on_delta(clean_translation(raw, german))
                    if obj.get("done"):
                        stats = _ollama_stats(obj)
                        break
                    if first and time.perf_counter() - t0 - first > max_gen_s:
                        # writing one line for this long: the machine is overloaded - give up on this attempt
                        log.warning("translate [%s]: still writing after %.0fs - cut off", model, max_gen_s)
                        return TranslationResult(clean_translation(raw, german), False, first,
                                                 time.perf_counter() - t0, "translation far too slow",
                                                 truncated=True)
        except httpx.TimeoutException as e:
            secs = time.perf_counter() - t0
            log.warning("translate [%s]: no answer from Ollama after %.1fs (%s)", model, secs, type(e).__name__)
            return TranslationResult(clean_translation(raw, german), False, first, secs,
                                     f"no answer from Ollama within {first_token_timeout:.0f} s", timeout=True)
        except (httpx.HTTPError, OllamaError, json.JSONDecodeError) as e:
            log.warning("translate [%s] failed: %s", model, e)
            return TranslationResult(clean_translation(raw, german), False, first, time.perf_counter() - t0, str(e))
        text = clean_translation(raw, german)
        total = time.perf_counter() - t0
        log.info("translate [%s]: first word %.2fs, done %.2fs (load %.2fs, read %s tok in %.2fs, wrote %s tok @ %.1f tok/s)",
                 model, first, total, stats.get("load_s", 0), stats.get("prompt_tokens", "?"), stats.get("prompt_s", 0),
                 stats.get("gen_tokens", "?"), stats.get("tok_s", 0))
        return TranslationResult(text, True, first, total, stats=stats)

    def loaded_models(self) -> list[dict]:
        """What Ollama has in memory right now, and how much of it sits on the GPU."""
        try:
            r = self._client.get(f"{self.base_url}/api/ps", timeout=2.0)
            r.raise_for_status()
            out = []
            for m in r.json().get("models", []):
                size, vram = int(m.get("size") or 0), int(m.get("size_vram") or 0)
                out.append({"name": m.get("name") or m.get("model", ""), "size": size, "size_vram": vram,
                            "gpu_share": (vram / size) if size else 0.0, "context_length": m.get("context_length")})
            return out
        except Exception:  # noqa: BLE001
            return []

    def version(self) -> str:
        try:
            return self._client.get(f"{self.base_url}/api/version", timeout=2.0).json().get("version", "")
        except Exception:  # noqa: BLE001
            return ""

    # ------------------------------------------------------------- summary
    SUMMARY_CHUNK_CHARS = 8000   # ~2k tokens: fits NUM_CTX together with the instructions and the answer

    @classmethod
    def _chunks(cls, text: str) -> list[str]:
        chunks, cur = [], ""
        for line in text.splitlines(keepends=True):
            if len(cur) + len(line) > cls.SUMMARY_CHUNK_CHARS and cur:
                chunks.append(cur)
                cur = ""
            cur += line
        if cur:
            chunks.append(cur)
        return chunks

    def summarize(self, transcript: str, on_delta: Callable[[str], None] | None = None) -> str:
        """Meeting notes in English from the bilingual transcript. Long meetings are condensed in
        rounds (parts -> notes -> notes of notes) so every request fits the model's context."""
        instr = ("Below is the transcript of a business meeting (German original with English translation). "
                 "Write concise meeting notes in English for the listener, a non-native German speaker:\n"
                 "1. **Summary** - 3-6 bullet points.\n2. **Decisions**.\n3. **Action items / what is expected from me** "
                 "(who, what, deadline).\n4. **Open questions**.\nOnly use what is in the transcript. Keep names, numbers and dates exact.")
        body = transcript
        for _round in range(4):
            chunks = self._chunks(body)
            if len(chunks) <= 1:
                break
            notes = []
            for i, c in enumerate(chunks, 1):
                notes.append(self._chat_once(
                    f"Part {i} of {len(chunks)} of a meeting record. List every important point, decision, number, "
                    f"date and action item from this part as short English bullet points.\n\n{c}"))
            body = "Notes from consecutive parts of the meeting:\n\n" + "\n\n".join(notes)
        return self._chat_once(f"{instr}\n\n---\n{body[:self.SUMMARY_CHUNK_CHARS + 2000]}", on_delta=on_delta)

    # the same context size as live translation: a different one makes Ollama reload the model
    def _chat_once(self, prompt: str, num_ctx: int = NUM_CTX, on_delta: Callable[[str], None] | None = None) -> str:
        body = {"model": self.model, "messages": [{"role": "user", "content": prompt}], "stream": True,
                "keep_alive": KEEP_ALIVE, "think": False, "options": {"temperature": 0.2, "num_ctx": num_ctx}}
        out = ""
        with self._client.stream("POST", f"{self.base_url}/api/chat", json=body,
                                 timeout=httpx.Timeout(600.0, connect=3.0)) as r:
            if r.status_code != 200:
                r.read()
                raise OllamaError(f"Ollama {r.status_code}: {r.text}")
            for line in r.iter_lines():
                if not line:
                    continue
                obj = json.loads(line)
                if obj.get("error"):
                    raise OllamaError(obj["error"])
                out += (obj.get("message") or {}).get("content") or ""
                if on_delta:
                    on_delta(_THINK_RE.sub("", out))
                if obj.get("done"):
                    break
        return _THINK_RE.sub("", out).strip()

    def unload(self, model: str | None = None) -> None:
        """Tell Ollama to drop the model from memory now (instead of after KEEP_ALIVE)."""
        try:
            self._client.post(f"{self.base_url}/api/generate", json={"model": model or self.model, "keep_alive": 0},
                              timeout=5.0)
        except Exception:  # noqa: BLE001
            pass

    def close(self) -> None:
        self._client.close()
