"""
Chat folders on disk must read like a book and never lose anything:

  • folder name = date + the chat title exactly as in the UI (+ the id tail)
  • chat.txt = the conversation word for word (#, *, ` kept — they matter in commands)
  • renames / folder moves / delete / restore move ONE folder, never make duplicates
  • a restart never pulls deleted chats out of _Trash_
  • the old layout (Dev_Ops/2026-06-13__Kafka_mTLS__id/ + stray duplicates) is
    reorganised automatically, and nothing inside those folders is lost

    python3 -m unittest tests/test_chat_folders.py -v
"""
import os
import sys
import tempfile
import unittest
import uuid
from pathlib import Path

if "app" not in sys.modules:                       # standalone run: isolate HOME
    os.environ["HOME"] = tempfile.mkdtemp(prefix="localllm-chats-")
    os.environ.setdefault("LOCALLLM_OLLAMA_URL", "http://127.0.0.1:9")   # no Ollama needed
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import app as A  # noqa: E402

CODE = "```bash\n# keep this comment\nls *.yaml | grep `whoami`\n```"


def dirs_of(chat_id):
    return [p for p in A.CHATS_DIR.rglob(f"*__{chat_id[:8]}") if p.is_dir()]


class ChatFoldersTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = A.app.test_client()

    def add_msgs(self, chat_id):
        with A.get_db() as conn:
            conn.execute("INSERT INTO messages (id, chat_id, role, content, tokens_in, tokens_out, model, created_at) "
                         "VALUES (?,?, 'user', ?, 0, 0, 'claude-sonnet-4-6', '2026-06-13T10:00:00')",
                         (str(uuid.uuid4()), chat_id, "How do I list *.yaml files?"))
            conn.execute("INSERT INTO messages (id, chat_id, role, content, tokens_in, tokens_out, model, created_at, cost_usd) "
                         "VALUES (?,?, 'assistant', ?, 10, 20, 'claude-sonnet-4-6', '2026-06-13T10:00:05', 0.0123)",
                         (str(uuid.uuid4()), chat_id, "Run this:\n" + CODE))

    def test_old_layout_is_reorganised_without_losing_files(self):
        cid, fid = str(uuid.uuid4()), str(uuid.uuid4())
        with A.get_db() as conn:
            conn.execute("INSERT INTO folders (id, name, created_at) VALUES (?, 'Dev Ops', '2026-06-01T00:00:00')", (fid,))
            conn.execute("INSERT INTO chats (id, title, model, created_at, updated_at, folder_id) "
                         "VALUES (?, 'Kafka mTLS: broker/cert fix', 'claude-sonnet-4-6', "
                         "'2026-06-13T10:00:00', '2026-06-13T10:00:05', ?)", (cid, fid))
        self.add_msgs(cid)
        s8 = cid[:8]
        old = A.CHATS_DIR / "Dev_Ops" / f"2026-06-13__Kafka_mTLS_brokercert_fix__{s8}"
        old.mkdir(parents=True)
        (old / "chat.txt").write_text("OLD FORMAT")
        (old / "notes.md").write_text("my notes")
        dup1 = A.CHATS_DIR / f"Kafka mTLS brokercert fix__{s8}"          # empty dup from the old rename bug
        dup1.mkdir()
        dup2 = A.CHATS_DIR / f"2026-06-13__Kafka__{s8}"                   # dup holding a clashing file
        dup2.mkdir()
        (dup2 / "notes.md").write_text("other notes")

        A.backfill_all_chat_txt()                                          # = what happens at app start

        found = dirs_of(cid)
        self.assertEqual(len(found), 1, found)
        d = found[0]
        self.assertEqual(d.parent, A.CHATS_DIR / "Dev Ops")
        self.assertEqual(d.name, f"2026-06-13 · Kafka mTLS broker cert fix__{s8}")
        book = (d / "chat.txt").read_text()
        self.assertIn(CODE, book)                                          # word for word
        self.assertIn("Kafka mTLS: broker/cert fix", book)                 # real title inside
        self.assertIn("▶ YOU", book)
        self.assertIn("◀ JARVIS", book)
        self.assertIn("Sonnet 4.6", book)
        self.assertEqual((d / "notes.md").read_text(), "my notes")         # nothing lost…
        clash = [p for p in d.iterdir() if p.name.startswith("notes (from")]
        self.assertEqual(len(clash), 1)
        self.assertEqual(clash[0].read_text(), "other notes")              # …not even the clash
        self.assertFalse((A.CHATS_DIR / "Dev_Ops").exists())               # old empty folder tidied
        index = (A.CHATS_DIR / A.CHAT_INDEX_NAME).read_text()
        self.assertIn("DEV OPS", index)
        self.assertIn("Kafka mTLS broker cert fix", index)

    def test_lifecycle_rename_move_delete_restart_restore(self):
        r = self.c.post("/api/chats", json={"model": "claude-sonnet-4-6"}).get_json()
        cid = r["id"]
        self.assertEqual(len(dirs_of(cid)), 1)
        self.assertEqual(dirs_of(cid)[0].parent, A.CHATS_DIR / "Unfiled")
        self.assertIn("· New Chat__", dirs_of(cid)[0].name)

        self.c.patch(f"/api/chats/{cid}", json={"title": "Grafana: OTel collector fix"})
        found = dirs_of(cid)
        self.assertEqual(len(found), 1, found)                             # the old bug made 2
        self.assertTrue(found[0].name.endswith(f" · Grafana OTel collector fix__{cid[:8]}"))

        fid = self.c.post("/api/folders", json={"name": "Observability"}).get_json()["id"]
        self.c.patch(f"/api/chats/{cid}/folder", json={"folder_id": fid})
        self.assertEqual(dirs_of(cid)[0].parent, A.CHATS_DIR / "Observability")
        self.c.patch(f"/api/folders/{fid}", json={"name": "Observability & Logs"})
        self.assertEqual(dirs_of(cid)[0].parent, A.CHATS_DIR / "Observability & Logs")
        self.assertFalse((A.CHATS_DIR / "Observability").exists())

        self.c.delete(f"/api/chats/{cid}")
        self.assertEqual(dirs_of(cid)[0].parent, A.CHATS_DIR / "Observability & Logs" / "_Trash_")
        A.backfill_all_chat_txt()                                          # restart must not undo it
        self.assertEqual(dirs_of(cid)[0].parent.name, "_Trash_")
        self.assertIn("deleted — in Trash", (dirs_of(cid)[0] / "chat.txt").read_text())

        self.c.post(f"/api/chats/{cid}/restore")
        self.assertEqual(dirs_of(cid)[0].parent, A.CHATS_DIR / "Observability & Logs")
        self.assertFalse((A.CHATS_DIR / "Observability & Logs" / "_Trash_").exists())

        self.c.delete(f"/api/chats/{cid}/hard-delete")                     # "Delete forever"
        kept = list((A.CHATS_DIR / "_Orphan").glob(f"*__{cid[:8]}"))
        self.assertEqual(len(kept), 1)                                     # record kept on disk
        self.assertNotIn("Grafana OTel collector fix", (A.CHATS_DIR / A.CHAT_INDEX_NAME).read_text())

    def test_orphan_cleanup_moves_never_deletes(self):
        keep = self.c.post("/api/chats", json={"model": "claude-sonnet-4-6"}).get_json()["id"]
        stray = A.CHATS_DIR / "Unfiled" / "2026-01-01 · something old__deadbeef"
        stray.mkdir(parents=True)
        (stray / "chat.txt").write_text("old record")
        r = self.c.delete("/api/chats/cleanup/orphans").get_json()
        self.assertIn(stray.name, r["moved_to_orphan"])
        self.assertEqual((A.CHATS_DIR / "_Orphan" / stray.name / "chat.txt").read_text(), "old record")
        self.assertTrue((A.CHATS_DIR / "Unfiled").exists())                # the old code rmtree'd this
        self.assertEqual(len(dirs_of(keep)), 1)

    def test_status_dots_see_chats_inside_folders(self):
        cid = self.c.post("/api/chats", json={"model": "claude-sonnet-4-6"}).get_json()["id"]
        st = self.c.get("/api/chats/status").get_json()
        row = next(x for x in st["chats"] if x["chat_id"] == cid)
        self.assertTrue(row["has_folder"])
        self.assertNotIn("Unfiled", st["orphan_folders"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
