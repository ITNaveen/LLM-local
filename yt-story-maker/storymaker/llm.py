"""Local LLM via Ollama (free, on your Mac). Every caller has a non-LLM fallback."""

import json
import re
import urllib.error
import urllib.request

import numpy as np

from .util import tokens


class LLMError(RuntimeError):
    pass


class OllamaLLM:
    def __init__(self, url, model, embed_models=(), think=False, timeout=600, log=None):
        self.url = url.rstrip("/")
        self.model = model
        self.embed_candidates = list(embed_models)
        self.think = think
        self.timeout = timeout
        self.log = log or (lambda msg: None)
        self._models = None
        self._embed_model = None
        self._think_supported = True

    # ---- plumbing -------------------------------------------------------
    def _post(self, path, payload, timeout=None):
        req = urllib.request.Request(
            self.url + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            raise LLMError(f"Ollama {path} HTTP {e.code}: {body[:300]}") from e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise LLMError(f"Ollama not reachable at {self.url}: {e}") from e

    def models(self):
        if self._models is None:
            try:
                with urllib.request.urlopen(self.url + "/api/tags", timeout=4) as resp:
                    data = json.loads(resp.read().decode())
                self._models = [m["name"] for m in data.get("models", [])]
            except (urllib.error.URLError, OSError, ValueError):
                self._models = []
        return self._models

    @staticmethod
    def _matches(name, wanted):
        """'bge-m3' matches 'bge-m3:latest' or any tag; 'qwen3.5:9b' needs that tag."""
        if ":" in wanted:
            return name == wanted
        return name.split(":")[0] == wanted

    def available(self):
        return any(self._matches(m, self.model) for m in self.models())

    # ---- chat -----------------------------------------------------------
    def chat_json(self, system, user, temperature=0.7, max_tokens=4096):
        """Ask for a JSON object; parse defensively. Raises LLMError on failure."""
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "stream": False,
            "format": "json",
            "options": {"temperature": temperature, "num_predict": max_tokens,
                        "num_ctx": 16384},
        }
        if self._think_supported:
            payload["think"] = bool(self.think)
        try:
            data = self._post("/api/chat", payload)
        except LLMError as e:
            if "think" in str(e).lower() and self._think_supported:
                self._think_supported = False
                payload.pop("think", None)
                data = self._post("/api/chat", payload)
            else:
                raise
        content = (data.get("message") or {}).get("content", "")
        return parse_json_loose(content)

    # ---- embeddings -----------------------------------------------------
    def embed_model(self):
        if self._embed_model is None:
            installed = self.models()
            self._embed_model = next(
                (c for c in self.embed_candidates
                 if any(self._matches(m, c) for m in installed)), "")
        return self._embed_model

    def embed(self, texts):
        model = self.embed_model()
        if not model:
            raise LLMError("no embedding model installed")
        out = []
        for i in range(0, len(texts), 64):
            batch = [t[:2000] or "-" for t in texts[i:i + 64]]
            data = self._post("/api/embed", {"model": model, "input": batch}, timeout=300)
            out.extend(data["embeddings"])
        return np.array(out, dtype=np.float32)


class NoLLM:
    """Used when Ollama is missing: forces the rule-based fallbacks."""

    model = "none"

    def available(self):
        return False

    def models(self):
        return []

    def chat_json(self, *a, **k):
        raise LLMError("no local LLM available")

    def embed_model(self):
        return ""

    def embed(self, texts):
        raise LLMError("no embedding model available")


def parse_json_loose(text):
    text = (text or "").strip()
    text = re.sub(r"^<think>.*?</think>", "", text, flags=re.S).strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    try:
        return json.loads(text)
    except ValueError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except ValueError:
            pass
    raise LLMError(f"LLM did not return valid JSON: {text[:200]}")


class Similarity:
    """Semantic similarity: Ollama embeddings when available, else TF-IDF cosine."""

    def __init__(self, llm, log=None):
        self.llm = llm
        self.log = log or (lambda m: None)
        self.mode = "embeddings" if llm.embed_model() else "keywords"

    def matrix(self, queries, docs):
        """Return len(queries) x len(docs) similarity in [0, 1]."""
        if not queries or not docs:
            return np.zeros((len(queries), len(docs)), dtype=np.float32)
        if self.mode == "embeddings":
            try:
                q = self.llm.embed(queries)
                d = self.llm.embed(docs)
                q /= np.linalg.norm(q, axis=1, keepdims=True) + 1e-9
                d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
                sims = q @ d.T
                # Rescale cosine (typically 0.3-0.9) to a usable 0-1 spread.
                lo, hi = float(sims.min()), float(sims.max())
                return (sims - lo) / (hi - lo + 1e-9)
            except Exception as e:  # noqa: BLE001 - any embed failure -> fallback
                self.log(f"Embeddings failed ({e}); using keyword matching.")
                self.mode = "keywords"
        return tfidf_similarity(queries, docs)


def tfidf_similarity(queries, docs):
    vocab = {}
    def vec_tokens(text):
        return [vocab.setdefault(t, len(vocab)) for t in tokens(text)]
    q_tok = [vec_tokens(q) for q in queries]
    d_tok = [vec_tokens(d) for d in docs]
    n = len(vocab)
    if n == 0:
        return np.zeros((len(queries), len(docs)), dtype=np.float32)
    df = np.zeros(n)
    for toks in d_tok:
        for t in set(toks):
            df[t] += 1
    idf = np.log((1 + len(docs)) / (1 + df)) + 1

    def to_mat(tok_lists):
        m = np.zeros((len(tok_lists), n), dtype=np.float32)
        for i, toks in enumerate(tok_lists):
            for t in toks:
                m[i, t] += 1
        m *= idf
        m /= np.linalg.norm(m, axis=1, keepdims=True) + 1e-9
        return m

    return to_mat(q_tok) @ to_mat(d_tok).T
