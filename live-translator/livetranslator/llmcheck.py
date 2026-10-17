"""Compare translation models on a real meeting - on this Mac.

There is no published German->English benchmark for the small local models, and dialect speech
recognised by Whisper is not like any test set. The honest comparison is your own meeting: the
saved German lines are translated again by each model (with the same conversation context the
live app uses), timed, and written side by side to `model-comparison.md` in the meeting folder.
Read it, then pick the model in Settings. Nothing is changed automatically.
"""
from __future__ import annotations

import statistics
from pathlib import Path

from .translate import OllamaTranslator, check_translation

DEFAULT_MODELS = ["gemma3:4b", "gemma4:e4b", "translategemma:4b"]


def compare(meeting, models: list[str], base_url: str, *, context_lines: int = 8, topic: str = "",
            terms: list[str] | None = None, max_lines: int = 40, progress=print) -> dict:
    """{model: {"ok": bool, "error": str, "english": [...], "first_s": .., "line_s": .., "tok_s": .., "checks": n}}"""
    lines = [ln for ln in meeting.lines if ln.de.strip()][:max_lines]
    out = {}
    before = {m["name"] for m in OllamaTranslator(base_url, models[0]).loaded_models()} if models else set()
    for model in models:
        t = OllamaTranslator(base_url, model, context_lines, topic, terms or [])
        progress(f"{model}: loading…")
        w = t.warmup(first_token_timeout=300.0)
        if not w.ok:
            out[model] = {"ok": False, "error": w.error or "did not load"}
            progress(f"{model}: not usable ({out[model]['error']})")
            continue
        english, firsts, totals, rates, checks = [], [], [], [], 0
        for i, ln in enumerate(lines, 1):
            r = t.translate(ln.de, first_token_timeout=60.0)
            english.append(r.text if r.ok else f"(failed: {r.error})")
            if r.ok:
                firsts.append(r.first_token_s)
                totals.append(r.total_s)
                if r.tok_s:
                    rates.append(r.tok_s)
                checks += bool(check_translation(ln.de, r.text))
            if i % 10 == 0:
                progress(f"{model}: {i}/{len(lines)} lines")
        out[model] = {"ok": bool(firsts), "english": english,
                      "first_s": round(statistics.median(firsts), 2) if firsts else None,
                      "line_s": round(statistics.median(totals), 2) if totals else None,
                      "tok_s": round(statistics.median(rates), 1) if rates else None, "checks": checks,
                      "error": "" if firsts else "no line translated"}
        progress(f"{model}: {describe(model, out[model])}")
        if model not in before and model + ":latest" not in before:
            t.unload(model)          # give the memory back (a model the app is using stays loaded)
    return {"lines": lines, "results": out}


def describe(model: str, r: dict) -> str:
    if not r.get("ok"):
        return f"not usable ({r.get('error')})"
    return (f"first word after {r['first_s']} s, whole line {r['line_s']} s, {r['tok_s']} word-pieces/s, "
            f"{r['checks']} line(s) with a lost negation/number")


def write_markdown(meeting, cmp: dict, path: Path) -> Path:
    models = list(cmp["results"])
    out = [f"# Model comparison: {meeting.title}", "",
           "Each model translated the saved German lines again, with the same context the live app uses.",
           "Pick the one whose English you like best in Settings → Translation.", "", "## Speed on this Mac", "",
           "| Model | First word | Whole line | Word-pieces/s | Lines with a lost negation/number |",
           "|---|---|---|---|---|"]
    for m in models:
        r = cmp["results"][m]
        if r.get("ok"):
            out.append(f"| {m} | {r['first_s']} s | {r['line_s']} s | {r['tok_s']} | {r['checks']} |")
        else:
            out.append(f"| {m} | not usable: {r.get('error')} | | | |")
    out += ["", "## Lines", ""]
    for i, ln in enumerate(cmp["lines"]):
        out.append(f"**{ln.time}**  \nDE: {ln.de}  ")
        for m in models:
            r = cmp["results"][m]
            if r.get("ok"):
                out.append(f"{m}: {r['english'][i]}  ")
        out.append("")
    path.write_text("\n".join(out) + "\n", "utf-8")
    return path
