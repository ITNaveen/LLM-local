"""
End-to-end test of Living Repos + Local mode, the way Naveen uses it:

  Week 1: drop the work repo.            Week 2: drop the SAME repo again
                                          (2 files added, 1 edited, 1 deleted).

Then ask the questions that decide whether the tool is worth keeping:
  "what's the latest on grafana in dev?"   "latest status of nexus in production?"
  "how did I fix the otel crash?"           "tell me more about it" (follow-up)

Runs fully offline: HOME points to a temp folder (your real DB is never touched)
and a tiny fake Ollama server stands in for the real one.

    python3 -m unittest tests/test_living_repos.py -v
"""
import hashlib
import io
import json
import math
import os
import re
import sys
import tempfile
import threading
import time
import unittest
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# ── Isolate: temp HOME + fake Ollama, BEFORE importing the app ──────────────
TMP_HOME = tempfile.mkdtemp(prefix="localllm-test-")
os.environ["HOME"] = TMP_HOME
FAKE = {"chat_requests": []}


def _vec(text):
    """Deterministic 64-dim bag-of-words vector — enough to test 'meaning' search."""
    v = [0.0] * 64
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 64] += 1.0
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


class FakeOllama(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/tags":
            return self._json({"models": [{"name": "qwen3.5:9b"}, {"name": "nomic-embed-text:latest"}]})
        self._json({"error": "not found"}, 404)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"{}")
        if self.path == "/api/embed":
            return self._json({"embeddings": [_vec(t) for t in body["input"]]})
        if self.path == "/api/chat":
            FAKE["chat_requests"].append(body)
            prompt = body["messages"][-1]["content"]
            m = re.search(r"★ SELECTED: (.+)", prompt)
            answer = f"SELECTED={m.group(1).strip() if m else 'NONE'}"
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()
            for piece in (answer[:10], answer[10:]):
                self.wfile.write((json.dumps({"message": {"content": piece}, "done": False}) + "\n").encode())
            self.wfile.write((json.dumps({"message": {"content": ""}, "done": True,
                                          "prompt_eval_count": 123, "eval_count": 45}) + "\n").encode())
            return
        self._json({"error": "not found"}, 404)


_srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeOllama)
threading.Thread(target=_srv.serve_forever, daemon=True).start()
os.environ["LOCALLLM_OLLAMA_URL"] = f"http://127.0.0.1:{_srv.server_address[1]}"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import app as A          # noqa: E402  (must come after the env setup above)
import repo_kb as R      # noqa: E402


def ms(date_str):
    return int(datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc).timestamp() * 1000)


WEEK1 = {
    "dev/grafana/loki-retention/notes.md":
        ("# Loki retention\nSet retention to 30d in dev.\n", "2026-05-01"),
    "dev/grafana/dashboard-perms/notes.md":
        ("# Dashboard permissions\nTeam folders got viewer role.\n", "2026-06-10"),
    "dev/grafana/otel-collector-crash/notes.md":
        ("# OpenTelemetry collector CrashLoopBackOff\n## Issue\nCollector pod OOMKilled after traces spike.\n"
         "## Root cause\nmemory_limiter processor missing.\n## Fix\nAdded memory_limiter (limit_mib: 400) "
         "and raised the pod limit to 512Mi.\n", "2026-09-20"),
    "prod/grafana/alerting/notes.md":
        ("# Alerting rules in prod\nMoved alerts to unified alerting.\n", "2026-09-25"),
    "prod/nexus/ha-deployment/3.68/values.yaml":
        ("image:\n  tag: 3.68.0\nreplicas: 1\n", "2026-03-01"),
    "prod/nexus/ha-deployment/3.68/notes.md":
        ("# Nexus 3.68 HA\nFirst HA attempt with 1 replica.\n", "2026-03-02"),
    "dev/kafka/strimzi-kafka-a/notes.md":
        ("# Strimzi Kafka A\nKRaft cluster, 3 controllers.\n", "2026-07-01"),
    "dev/kafka/kafka-b/notes.md":
        ("# Kafka B\nSecond kafka for connect tests. Deployment is at step 3 of 5.\n", "2026-08-01"),
    "maint/keycloak/realm.md":
        ("# Keycloak realm\nLDAP federation configured.\n", "2026-04-01"),
    ".git/config": ("[core]\n", "2026-01-01"),        # must be ignored
}


class LivingReposTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = A.app.test_client()

    # ── helpers ──────────────────────────────────────────────────────────────
    def drop(self, name, files, with_sha=True):
        """Do what the browser does: plan → upload the files it asks for → commit."""
        manifest = []
        for path, (text, date) in files.items():
            raw = text.encode()
            entry = {"path": path, "size": len(raw), "mtime": ms(date)}
            if with_sha:
                entry["sha256"] = hashlib.sha256(raw).hexdigest()
            manifest.append(entry)
        plan = self.c.post("/api/repos/plan", json={"name": name, "files": manifest}).get_json()
        self.assertIn("plan_id", plan, plan)
        if plan["upload"]:
            data = {"plan_id": plan["plan_id"],
                    "files": [(io.BytesIO(files[p][0].encode()), p.rsplit("/", 1)[-1]) for p in plan["upload"]],
                    "paths": plan["upload"]}
            r = self.c.post("/api/repos/upload", data=data, content_type="multipart/form-data").get_json()
            self.assertEqual(r["received"], len(plan["upload"]), r)
        r = self.c.post("/api/repos/commit", json={"plan_id": plan["plan_id"]}).get_json()
        self.assertTrue(r.get("ok"), r)
        for _ in range(200):
            st = self.c.get(f"/api/repos/sync/{plan['plan_id']}").get_json()
            if st["status"] in ("committed", "failed"):
                break
            time.sleep(0.05)
        self.assertEqual(st["status"], "committed", st)
        return plan, st["result"]

    def ask(self, q):
        return self.c.get("/api/repos/ask", query_string={"q": q}).get_json()

    def chat(self, chat_id, text):
        resp = self.c.post(f"/api/chats/{chat_id}/messages", json={"message": text})
        self.assertEqual(resp.status_code, 200, resp.get_data(as_text=True)[:300])
        events = [json.loads(l[5:]) for l in resp.get_data(as_text=True).splitlines() if l.startswith("data:")]
        return "".join(e.get("text", "") for e in events if e["type"] == "text"), events

    # ── the actual scenario ─────────────────────────────────────────────────
    def test_weekly_drops_and_latest(self):
        # WEEK 1 — first drop creates the repo
        plan1, res1 = self.drop("local-knowledge", WEEK1)
        self.assertFalse(plan1["repo"]["exists"])
        self.assertEqual(plan1["counts"]["new"], 9)                 # .git ignored
        self.assertEqual(res1["added"], 9)

        r = self.ask("what is the latest on grafana in dev?")
        self.assertEqual(r["selected"]["item"], "dev/grafana/otel-collector-crash", r["understood"])
        r = self.ask("latest status of nexus in production")
        self.assertEqual(r["selected"]["item"], "prod/nexus/ha-deployment")
        self.assertIn("3.68", r["context"])

        # WEEK 2 — same repo again: +2 files (Nexus 3.95), 1 edited, 1 deleted
        week2 = dict(WEEK1)
        week2["prod/nexus/ha-deployment/3.95/values.yaml"] = ("image:\n  tag: 3.95.0\nreplicas: 3\n", "2026-09-28")
        week2["prod/nexus/ha-deployment/3.95/notes.md"] = (
            "# Nexus 3.95 HA\nUpgraded from 3.68. 3 replicas, PostgreSQL backend.\n", "2026-09-28")
        week2["dev/grafana/dashboard-perms/notes.md"] = (
            "# Dashboard permissions\nTeam folders got viewer role.\nUPDATE: SSO groups mapped to editor.\n",
            "2026-10-02")
        del week2["dev/kafka/kafka-b/notes.md"]
        plan2, res2 = self.drop("local-knowledge", week2)
        self.assertTrue(plan2["repo"]["exists"])
        self.assertEqual(plan2["counts"]["new"], 2)
        self.assertEqual(plan2["counts"]["changed"], 1)
        self.assertEqual(plan2["counts"]["missing"], 1)
        self.assertEqual(plan2["counts"]["unchanged"], 7)
        self.assertEqual(sorted(plan2["upload"]), sorted([
            "prod/nexus/ha-deployment/3.95/values.yaml", "prod/nexus/ha-deployment/3.95/notes.md",
            "dev/grafana/dashboard-perms/notes.md"]))          # only 3 files travel, not 11

        # ONE repo, not two
        repos = [x for x in self.c.get("/api/repos").get_json()["repos"] if x["name"] == "local-knowledge"]
        self.assertEqual(len(repos), 1)
        self.assertEqual(repos[0]["files"], 10)
        self.assertEqual(repos[0]["removed"], 1)

        # Nothing deleted: old version archived, removed file still on disk
        mirror = R.REPOS_DIR / "local-knowledge"
        self.assertIn("SSO groups", (mirror / "dev/grafana/dashboard-perms/notes.md").read_text())
        archived = list(R.HISTORY_DIR.rglob("dashboard-perms/notes.md"))
        self.assertEqual(len(archived), 1)
        self.assertNotIn("SSO", archived[0].read_text())
        self.assertTrue((mirror / "dev/kafka/kafka-b/notes.md").exists())

        # The Mac copy keeps the laptop's "Date Modified"
        f395 = mirror / "prod/nexus/ha-deployment/3.95/notes.md"
        self.assertEqual(int(f395.stat().st_mtime * 1000), ms("2026-09-28"))
        os.utime(f395, None)                                   # like a file synced by the old version
        R._lock_existing_files()                               # startup pass restores the laptop date
        self.assertEqual(int(f395.stat().st_mtime * 1000), ms("2026-09-28"))

        # Readable sync log, one entry per drop
        log_txt = (R.HISTORY_DIR / "local-knowledge" / "SYNC-LOG.txt").read_text()
        self.assertIn("Sync #1", log_txt)
        self.assertIn("Sync #2", log_txt)
        self.assertIn("+ prod/nexus/ha-deployment/3.95/notes.md", log_txt)
        self.assertIn("~ dev/grafana/dashboard-perms/notes.md", log_txt)
        self.assertIn("- dev/kafka/kafka-b/notes.md", log_txt)

        # ✓ Verify: every file matches its fingerprint; history + missing are kept
        v = self.c.get("/api/repos/local-knowledge/verify").get_json()
        self.assertEqual(v["checked"], 11)
        self.assertEqual(v["intact"], 11)
        self.assertEqual(v["old_versions_on_disk"], 1)
        self.assertEqual(v["removed_but_kept"], 1)
        # …and it notices a file edited behind the app's back
        f_a = mirror / "dev/kafka/strimzi-kafka-a/notes.md"
        original = f_a.read_text()
        f_a.write_text(original + "tampered\n")
        v = self.c.get("/api/repos/local-knowledge/verify").get_json()
        self.assertEqual(v["changed_outside_app"], ["dev/kafka/strimzi-kafka-a/notes.md"])
        f_a.write_text(original)

        # "latest" now follows the edit / the new version
        r = self.ask("what is the latest on grafana in dev?")
        self.assertEqual(r["selected"]["item"], "dev/grafana/dashboard-perms")
        r = self.ask("what's the latest status of my Nexus in production?")
        self.assertEqual(r["selected"]["item"], "prod/nexus/ha-deployment")
        self.assertIn("3.95.0", r["context"])
        self.assertNotIn("tag: 3.68.0", r["context"])         # the old version is NOT mixed in
        self.assertIn("highest of 2", r["context"])
        r = self.ask("nexus 3.68 in prod")                       # but asking for it explicitly works
        self.assertIn("tag: 3.68.0", r["context"])

        # topic questions, typos, env aliases, prod↔maint fallback
        r = self.ask("how did I fix the opentelemetry crash in dev?")
        self.assertEqual(r["selected"]["item"], "dev/grafana/otel-collector-crash")
        self.assertIn("memory_limiter", r["context"])
        r = self.ask("grafna dev latest work")
        self.assertEqual(r["selected"]["item"], "dev/grafana/dashboard-perms")
        r = self.ask("latest on keycloak in production")
        self.assertEqual(r["selected"]["item"], "maint/keycloak/realm.md")
        self.assertIn("maint", r["context"])
        r = self.ask("latest on grafana in prod")
        self.assertEqual(r["selected"]["item"], "prod/grafana/alerting")
        r = self.ask("latest on jenkins in dev")                 # not in the repo → honest
        self.assertFalse(r["found"])
        self.assertIn("NOTHING in the repo matches", r["context"])
        r = self.ask("latest kafka in dev")                      # kafka-b removed → only A left
        self.assertEqual(r["selected"]["item"], "dev/kafka/strimzi-kafka-a")

        # Browser without fingerprints (plain http on the LAN) still dedupes
        plan3, res3 = self.drop("local-knowledge", week2, with_sha=False)
        self.assertEqual(plan3["counts"]["unchanged"], 10)       # same size + mtime
        self.assertEqual(res3["changed"] + res3["added"], 0)

        self._local_chat()
        self._lock_and_delete()

    def _local_chat(self):
        chat = self.c.post("/api/chats", json={"model": "ollama:qwen3.5:9b"}).get_json()
        self.assertEqual(chat["model"], "ollama:qwen3.5:9b")
        text, events = self.chat(chat["id"], "what is the latest on grafana in dev?")
        self.assertIn("SELECTED=dev / grafana / dashboard-perms", text)
        self.assertIn("dev/grafana/dashboard-perms/notes.md", text)        # sources footer
        self.assertEqual(events[-1]["type"], "done")
        self.assertEqual(events[-1]["cost"], 0)
        req = FAKE["chat_requests"][-1]
        self.assertEqual(req["options"]["num_ctx"], R.DEFAULT_NUM_CTX)
        self.assertIn("LOCAL KNOWLEDGE MODE", req["messages"][0]["content"])

        # Follow-up stays on the same item…
        text, _ = self.chat(chat["id"], "how did I fix it?")
        self.assertIn("SELECTED=dev / grafana / dashboard-perms", text)
        # …a new topic in a short question moves on
        text, _ = self.chat(chat["id"], "latest otel crash?")
        self.assertIn("otel-collector-crash", text)
        # "and in prod?" keeps the product, changes the env
        text, _ = self.chat(chat["id"], "and in prod?")
        self.assertIn("prod / grafana", text)

        with A.get_db() as conn:
            rows = conn.execute("SELECT role, cost_usd, model FROM messages WHERE chat_id=?",
                                (chat["id"],)).fetchall()
            title = conn.execute("SELECT title FROM chats WHERE id=?", (chat["id"],)).fetchone()[0]
        self.assertEqual(len(rows), 8)
        self.assertTrue(all((r["cost_usd"] or 0) == 0 for r in rows))
        self.assertTrue(title.startswith("what is the latest on grafana"))
        # history sent to the model has the footer stripped
        hist = FAKE["chat_requests"][-1]["messages"][1:-1]
        self.assertTrue(hist and all("📁" not in m["content"] for m in hist))

    def _lock_and_delete(self):
        r = self.c.post("/api/repos/local-knowledge/delete", json={"confirm": "local-knowledge"})
        self.assertEqual(r.status_code, 403)                              # locked
        r = self.c.post("/api/repos/local-knowledge/lock", json={"locked": False, "confirm": "wrong"})
        self.assertEqual(r.status_code, 400)
        r = self.c.post("/api/repos/local-knowledge/lock", json={"locked": False, "confirm": "local-knowledge"})
        self.assertEqual(r.status_code, 200)
        r = self.c.post("/api/repos/local-knowledge/delete", json={"confirm": "local-knowledge"}).get_json()
        self.assertTrue(r["ok"])
        self.assertTrue(Path(r["moved_to"]).exists())                     # trash, not erased
        self.assertNotIn("local-knowledge", [x["slug"] for x in self.c.get("/api/repos").get_json()["repos"]])

    def test_finder_lock_is_lifted_only_to_archive(self):
        """Simulate macOS's Locked flag: moving a locked file fails, exactly like
        on the Mac. A sync that replaces a file must unlock → archive → re-lock."""
        from unittest import mock
        locked = set()
        real_move = R.shutil.move

        def strict_move(src, dst, *a, **k):
            if str(Path(src).resolve()) in locked:
                raise PermissionError(f"locked: {src}")
            return real_move(src, dst, *a, **k)

        with mock.patch.object(R, "_fs_lock", lambda p: locked.add(str(Path(p).resolve()))), \
             mock.patch.object(R, "_fs_unlock", lambda p: locked.discard(str(Path(p).resolve()))), \
             mock.patch.object(R.shutil, "move", strict_move):
            files = {"dev/x/notes.md": ("# v1\n", "2026-01-01"), "dev/x/keep.md": ("same\n", "2026-01-01")}
            _, r1 = self.drop("lock-test", files)
            self.assertEqual(r1["errors"], [])
            mirror = R.REPOS_DIR / "lock-test"
            self.assertIn(str((mirror / "dev/x/notes.md").resolve()), locked)
            files["dev/x/notes.md"] = ("# v2\n", "2026-02-01")
            _, r2 = self.drop("lock-test", files)
            self.assertEqual(r2["errors"], [])                                   # no PermissionError
            self.assertEqual(r2["changed"], 1)
            self.assertIn(str((mirror / "dev/x/notes.md").resolve()), locked)    # new version locked
            archived = next(R.HISTORY_DIR.joinpath("lock-test").rglob("notes.md"))
            self.assertIn(str(archived.resolve()), locked)                       # old version locked
            self.assertEqual(archived.read_text(), "# v1\n")

    def test_stale_uploads_are_tidied(self):
        junk = R.STAGING_DIR / ("f" * 32)
        (junk / "a").mkdir(parents=True)
        (junk / "a" / "x.md").write_text("half upload")
        R._cleanup_stale_staging()
        self.assertFalse(junk.exists())

    def test_unsafe_paths_rejected(self):
        for bad in ["../etc/passwd", "a/../../b", "C:/x", ""]:
            with self.assertRaises(ValueError):
                R.clean_relpath(bad)
        self.assertEqual(R.clean_relpath("/abs//path/./f.md"), "abs/path/f.md")   # made relative
        plan = self.c.post("/api/repos/plan", json={"name": "x", "files": [
            {"path": "../../evil.sh", "size": 3, "mtime": 1}]}).get_json()
        self.assertIn("error", plan)

    def test_normal_folder_upload_keeps_tree(self):
        data = {"target": "20261003_120000_photos",
                "files": [(io.BytesIO(b"a"), "a.txt"), (io.BytesIO(b"b"), "b.txt")],
                "paths": ["trip/a.txt", "trip/day2/b.txt"]}
        r = self.c.post("/api/dropbox/upload-folder", data=data, content_type="multipart/form-data").get_json()
        self.assertEqual(r["saved"], 2)
        self.assertTrue((R.DROP_DIR / "20261003_120000_photos/trip/day2/b.txt").exists())

    def test_local_status_and_models(self):
        st = self.c.get("/api/local/status").get_json()
        self.assertTrue(st["running"])
        self.assertEqual(st["selected"], "qwen3.5:9b")
        self.assertTrue(st["embed_ready"])
        ids = [m["id"] for m in self.c.get("/api/models").get_json()]
        self.assertIn("ollama:qwen3.5:9b", ids)
        self.assertNotIn("ollama:nomic-embed-text:latest", ids)          # embedder isn't a chat model
        self.assertIn("claude-sonnet-4-6", ids)                          # cloud models untouched


if __name__ == "__main__":
    unittest.main(verbosity=2)
