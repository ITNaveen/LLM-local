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

You receive the meeting one line at a time, exactly as speech recognition heard it. Reply to each line with the English translation of that line only.

Rules:
- Be faithful and complete. Keep every number, date, time, amount, name, negation (nicht, kein, nie) and condition exactly as said.
- Write natural, clear English, the way a native speaker would say it in a meeting.
- Lines are cut at the speaker's pauses, so a line may be an unfinished sentence that continues in the next line. Translate what is there and use the previous lines to understand it. Do not invent the missing part.
- Speech recognition makes mistakes. If a word was clearly misheard, translate what the speaker most likely meant, given the context.
- Drop pure filler words (ähm, äh, also, ne) unless they carry meaning.
- Keep names of people, companies, products, systems and abbreviations as they are.
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


def _thinking_model(name: str) -> bool:
    n = name.lower()
    return any(k in n for k in ("qwen3", "deepseek-r1", "qwq", "magistral"))


@dataclass
class TranslationResult:
    text: str
    ok: bool
    first_token_s: float = 0.0
    total_s: float = 0.0
    error: str = ""
    timeout: bool = False       # Ollama sent nothing within the time limit
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

    def seed_history(self, pairs: list[tuple[str, str]]) -> None:
        with self._lock:
            self.history = [(d, e) for d, e in pairs if d and e][-self.context_lines:]

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
    def _messages(self, german: str) -> list[dict]:
        msgs = [{"role": "system", "content": self.system}]
        for de, en in self.history:
            msgs.append({"role": "user", "content": de})
            msgs.append({"role": "assistant", "content": en})
        msgs.append({"role": "user", "content": german})
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

    def translate(self, german: str, on_delta: Callable[[str], None] | None = None, record: bool = True,
                  first_token_timeout: float = 60.0, max_total_s: float = 90.0,
                  model: str | None = None) -> TranslationResult:
        """Translate one line. Gives up if Ollama sends nothing for `first_token_timeout`
        seconds (model stuck loading / machine out of memory) instead of waiting forever."""
        german = german.strip()
        if not german:
            return TranslationResult("", True)
        with self._lock:
            msgs = self._messages(german)
            model = model or self.model
        body = {
            "model": model,
            "messages": msgs,
            "stream": True,
            "keep_alive": KEEP_ALIVE,
            "options": {"temperature": 0.1, "top_p": 0.9, "num_ctx": NUM_CTX,
                        "num_predict": max(64, min(600, len(german) * 2))},
        }
        if _thinking_model(model):
            body["think"] = False
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
                    if time.perf_counter() - t0 > max_total_s:
                        log.warning("translation took longer than %.0fs - keeping what we have", max_total_s)
                        break
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
        if record and text and self.plausible(german, text):
            with self._lock:
                self.history.append((german, text))
                self._trim_history()
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
    def summarize(self, transcript: str, on_delta: Callable[[str], None] | None = None) -> str:
        """Meeting notes in English from the bilingual transcript (long meetings are chunked)."""
        chunks, cur = [], ""
        for line in transcript.splitlines(keepends=True):
            if len(cur) + len(line) > 12000 and cur:
                chunks.append(cur)
                cur = ""
            cur += line
        if cur:
            chunks.append(cur)
        instr = ("Below is the transcript of a business meeting (German original with English translation). "
                 "Write concise meeting notes in English for the listener, a non-native German speaker:\n"
                 "1. **Summary** - 3-6 bullet points.\n2. **Decisions**.\n3. **Action items / what is expected from me** "
                 "(who, what, deadline).\n4. **Open questions**.\nOnly use what is in the transcript. Keep names, numbers and dates exact.")
        if len(chunks) > 1:
            notes = []
            for i, c in enumerate(chunks, 1):
                notes.append(self._chat_once(
                    f"Part {i} of {len(chunks)} of a meeting transcript. List every important point, decision, number, "
                    f"date and action item from this part as short English bullet points.\n\n{c}", num_ctx=8192))
            body = "Notes from consecutive parts of the meeting:\n\n" + "\n\n".join(notes)
        else:
            body = chunks[0] if chunks else ""
        return self._chat_once(f"{instr}\n\n---\n{body}", num_ctx=8192, on_delta=on_delta)

    def _chat_once(self, prompt: str, num_ctx: int = 8192, on_delta: Callable[[str], None] | None = None) -> str:
        body = {"model": self.model, "messages": [{"role": "user", "content": prompt}], "stream": True,
                "keep_alive": KEEP_ALIVE, "options": {"temperature": 0.2, "num_ctx": num_ctx}}
        if _thinking_model(self.model):
            body["think"] = False
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
