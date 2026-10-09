"""Meeting transcripts on disk.

    ~/LiveTranslator/Meetings/
        2026-10-08/
            15-15 - Weekly sync with Thomas/
                transcript.md      <- readable: time, German, English
                transcript.jsonl   <- one JSON record per line (crash-safe, append-only)
                meta.json          <- name, start/end time, models used
                summary.md         <- optional meeting notes

Every finished line is appended and flushed to disk immediately, so a crash
or a closed laptop never loses more than the line being spoken.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .config import meetings_dir

_BAD_CHARS = re.compile(r'[\\/:*?"<>|\x00-\x1f]+')
_ID_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{6}(-\d+)?$")


def safe_name(name: str) -> str:
    n = _BAD_CHARS.sub(" ", name or "").strip().strip(".")
    n = re.sub(r"\s+", " ", n)
    return n[:80]


def _fmt_duration(sec: float) -> str:
    sec = int(max(0, sec))
    h, m = divmod(sec // 60, 60)
    return f"{h} h {m:02d} min" if h else f"{m} min"


@dataclass
class Line:
    id: int
    time: str            # wall clock "HH:MM:SS"
    offset: float        # seconds since meeting start
    de: str
    en: str = ""
    ok: bool = True      # translation succeeded
    forced: bool = False
    done: bool = False   # translation finished (or failed)

    def to_dict(self) -> dict:
        return {"id": self.id, "time": self.time, "offset": round(self.offset, 2), "de": self.de,
                "en": self.en, "ok": self.ok, "forced": self.forced}


@dataclass
class Meeting:
    id: str
    folder: Path
    name: str
    started: datetime
    ended: datetime | None = None
    lines: list[Line] = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    @property
    def title(self) -> str:
        return self.name or f"Meeting {self.started:%H:%M}"

    def meta(self) -> dict:
        end = self.ended or datetime.now()
        return {
            "id": self.id,
            "name": self.name,
            "title": self.title,
            "date": f"{self.started:%Y-%m-%d}",
            "started": self.started.isoformat(timespec="seconds"),
            "ended": self.ended.isoformat(timespec="seconds") if self.ended else None,
            "duration_s": int((end - self.started).total_seconds()),
            "line_count": len(self.lines),
            "folder": str(self.folder),
            **self.extra,
        }

    def markdown(self) -> str:
        end = self.ended
        when = f"{self.started:%A, %d %B %Y} · {self.started:%H:%M}"
        if end:
            when += f" – {end:%H:%M} ({_fmt_duration((end - self.started).total_seconds())})"
        out = [f"# {self.title}", "", f"**{when}**", ""]
        for k in ("asr_model", "llm_model"):
            if self.extra.get(k):
                out.append(f"_{'Speech model' if k == 'asr_model' else 'Translation model'}: {self.extra[k]}_  ")
        out += ["", "---", ""]
        for ln in self.lines:
            out.append(f"**{ln.time}**  ")
            out.append(f"DE: {ln.de}  ")
            out.append(f"EN: {ln.en if ln.en else ('(translation unavailable)' if ln.done else '…')}")
            out.append("")
        return "\n".join(out)

    def plain_text(self) -> str:
        out = [self.title, f"{self.started:%Y-%m-%d %H:%M}", ""]
        for ln in self.lines:
            out += [f"[{ln.time}] {ln.de}", f"           {ln.en}", ""]
        return "\n".join(out)


class MeetingStore:
    def __init__(self, root: Path | None = None):
        self.root = root or meetings_dir()
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    # ------------------------------------------------------------- create
    def create(self, name: str = "", now: datetime | None = None, extra: dict | None = None) -> Meeting:
        now = now or datetime.now()
        with self._lock:
            day = self.root / f"{now:%Y-%m-%d}"
            day.mkdir(parents=True, exist_ok=True)
            base = f"{now:%H-%M}"
            folder_name = base + (f" - {safe_name(name)}" if safe_name(name) else "")
            folder = day / folder_name
            n = 2
            while folder.exists():
                folder = day / f"{base} ({n})" if not safe_name(name) else day / f"{base} ({n}) - {safe_name(name)}"
                n += 1
            folder.mkdir()
            mid = f"{now:%Y-%m-%d_%H%M%S}"
            existing = {m["id"] for m in self.list()}
            k = 2
            while mid in existing:
                mid = f"{now:%Y-%m-%d_%H%M%S}-{k}"
                k += 1
            m = Meeting(id=mid, folder=folder, name=safe_name(name), started=now, extra=dict(extra or {}))
            self._write_meta(m)
            self._write_md(m)
            (folder / "transcript.jsonl").touch()
            return m

    # ------------------------------------------------------------- update
    def add_line(self, m: Meeting, line: Line) -> None:
        """German text is saved the moment it is recognised."""
        with self._lock:
            m.lines.append(line)
            self._append_jsonl(m, line.to_dict())
            if len(m.lines) % 20 == 0:
                self._write_meta(m)

    def _folder_is(self, m: Meeting) -> bool:
        """The folder on disk still belongs to this meeting (not deleted / renamed under us)."""
        try:
            return json.loads((m.folder / "meta.json").read_text("utf-8")).get("id") == m.id
        except Exception:  # noqa: BLE001
            return False

    def set_translation(self, m: Meeting, line_id: int, en: str, ok: bool = True) -> Line | None:
        with self._lock:
            if not self._folder_is(m):
                folder = self.find(m.id)        # renamed through another copy: follow it; deleted: drop
                if folder is None:
                    return None
                m.folder = folder
            ln = next((x for x in reversed(m.lines) if x.id == line_id), None)
            if ln is None:
                return None
            first = not ln.done
            self._append_jsonl(m, {"update": line_id, "en": en, "ok": ok})   # saved first: a failed write stays retryable
            ln.en, ln.ok, ln.done = en, ok, True
            if first and all(x.done for x in m.lines if x.id < line_id):
                with open(m.folder / "transcript.md", "a", encoding="utf-8") as f:
                    f.write(f"**{ln.time}**  \nDE: {ln.de}  \nEN: {en if en else '(translation unavailable)'}\n\n")
                    f.flush()
                    os.fsync(f.fileno())
            else:
                self._write_md(m)
            return ln

    def finish(self, m: Meeting, now: datetime | None = None) -> bool:
        """Close the meeting. Returns False if it was empty and got removed."""
        with self._lock:
            m.ended = now or datetime.now()
            if not m.lines and not m.name and not (m.folder / "summary.md").exists():
                shutil.rmtree(m.folder, ignore_errors=True)
                self._cleanup_day(m.folder.parent)
                return False
            self._write_meta(m)
            self._write_md(m)
            return True

    def reopen(self, m: Meeting) -> None:
        with self._lock:
            m.ended = None
            self._write_meta(m)

    def rename(self, m: Meeting, name: str) -> Meeting:
        with self._lock:
            name = safe_name(name)
            m.name = name
            base = m.folder.name.split(" - ", 1)[0]
            target = m.folder.parent / (f"{base} - {name}" if name else base)
            if target != m.folder:
                n = 2
                cand = target
                while cand.exists():
                    cand = target.parent / f"{target.name} ({n})"
                    n += 1
                os.replace(m.folder, cand)
                m.folder = cand
            self._write_meta(m)
            self._write_md(m)
            return m

    def delete(self, m: Meeting) -> None:
        with self._lock:
            shutil.rmtree(m.folder, ignore_errors=True)
            self._cleanup_day(m.folder.parent)

    def save_summary(self, m: Meeting, text: str) -> Path:
        with self._lock:
            p = m.folder / "summary.md"
            p.write_text(f"# Notes: {m.title}\n\n_{m.started:%A, %d %B %Y, %H:%M}_\n\n{text.strip()}\n", "utf-8")
            return p

    # ------------------------------------------------------------- read
    def list(self) -> list[dict]:
        out = []
        for meta in self.root.glob("*/*/meta.json"):
            try:
                d = json.loads(meta.read_text("utf-8"))
                d["folder"] = str(meta.parent)
                d["has_summary"] = (meta.parent / "summary.md").exists()
                out.append(d)
            except Exception:
                continue
        out.sort(key=lambda d: d.get("started", ""), reverse=True)
        return out

    def find(self, meeting_id: str) -> Path | None:
        if not _ID_RE.match(meeting_id or ""):
            return None
        date = meeting_id.split("_", 1)[0]
        for meta in list((self.root / date).glob("*/meta.json")) + list(self.root.glob("*/*/meta.json")):
            try:
                if json.loads(meta.read_text("utf-8")).get("id") == meeting_id:
                    return meta.parent
            except Exception:
                continue
        return None

    def load(self, meeting_id: str) -> Meeting | None:
        folder = self.find(meeting_id)
        if folder is None:
            return None
        meta = json.loads((folder / "meta.json").read_text("utf-8"))
        lines: dict[int, Line] = {}
        order: list[int] = []
        try:
            with open(folder / "transcript.jsonl", encoding="utf-8") as f:
                for raw in f:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        r = json.loads(raw)
                    except json.JSONDecodeError:
                        continue  # torn last line after a crash
                    if "update" in r:
                        ln = lines.get(r["update"])
                        if ln:
                            ln.en, ln.ok, ln.done = r.get("en", ln.en), r.get("ok", True), True
                    else:
                        ln = Line(id=r["id"], time=r["time"], offset=r.get("offset", 0.0), de=r["de"],
                                  en=r.get("en", ""), ok=r.get("ok", True), forced=r.get("forced", False),
                                  done=bool(r.get("en")))
                        if ln.id not in lines:
                            order.append(ln.id)
                        lines[ln.id] = ln
        except FileNotFoundError:
            pass
        known = {"id", "name", "title", "date", "started", "ended", "duration_s", "line_count", "folder", "has_summary"}
        return Meeting(
            id=meta["id"], folder=folder, name=meta.get("name", ""),
            started=datetime.fromisoformat(meta["started"]),
            ended=datetime.fromisoformat(meta["ended"]) if meta.get("ended") else None,
            lines=[lines[i] for i in order],
            extra={k: v for k, v in meta.items() if k not in known},
        )

    def summary(self, m: Meeting) -> str | None:
        p = m.folder / "summary.md"
        return p.read_text("utf-8") if p.exists() else None

    # ------------------------------------------------------------- internals
    def _append_jsonl(self, m: Meeting, rec: dict) -> None:
        with open(m.folder / "transcript.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def _write_meta(self, m: Meeting) -> None:
        tmp = m.folder / "meta.json.tmp"
        tmp.write_text(json.dumps(m.meta(), indent=2, ensure_ascii=False), "utf-8")
        os.replace(tmp, m.folder / "meta.json")

    def _write_md(self, m: Meeting) -> None:
        tmp = m.folder / "transcript.md.tmp"
        tmp.write_text(m.markdown() + ("\n" if m.lines else ""), "utf-8")
        os.replace(tmp, m.folder / "transcript.md")

    @staticmethod
    def _cleanup_day(day: Path) -> None:
        try:
            if day.exists() and not any(day.iterdir()):
                day.rmdir()
        except OSError:
            pass
