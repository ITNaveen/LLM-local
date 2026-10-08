import json
from datetime import datetime

from livetranslator.storage import Line, MeetingStore, safe_name


def mk(tmp_path):
    return MeetingStore(tmp_path / "Meetings")


def test_folder_layout_by_date_and_time(tmp_path):
    st = mk(tmp_path)
    m = st.create(now=datetime(2026, 10, 8, 15, 15, 7))
    assert m.folder == tmp_path / "Meetings" / "2026-10-08" / "15-15"
    assert (m.folder / "meta.json").exists() and (m.folder / "transcript.md").exists()
    m2 = st.create("Weekly sync", now=datetime(2026, 10, 8, 15, 15, 40))
    assert m2.folder.name == "15-15 - Weekly sync"
    m3 = st.create(now=datetime(2026, 10, 8, 15, 15, 50))
    assert m3.folder.name == "15-15 (2)"
    assert len({m.id, m2.id, m3.id}) == 3
    st.rename(m, "Weekly sync")          # would clash with m2's folder
    assert m.folder.name == "15-15 - Weekly sync (2)"


def test_lines_saved_immediately_and_reloaded(tmp_path):
    st = mk(tmp_path)
    m = st.create(now=datetime(2026, 10, 8, 9, 0))
    st.add_line(m, Line(1, "09:00:05", 5.0, "Guten Morgen."))
    # German is on disk before the translation exists
    recs = [json.loads(x) for x in (m.folder / "transcript.jsonl").read_text().splitlines()]
    assert recs[0]["de"] == "Guten Morgen." and recs[0]["en"] == ""
    st.set_translation(m, 1, "Good morning.")
    st.add_line(m, Line(2, "09:00:09", 9.0, "Wie geht's?"))
    st.set_translation(m, 2, "", ok=False)
    md = (m.folder / "transcript.md").read_text()
    assert "DE: Guten Morgen." in md and "EN: Good morning." in md and "(translation unavailable)" in md
    m2 = st.load(m.id)
    assert [(x.de, x.en, x.ok) for x in m2.lines] == [("Guten Morgen.", "Good morning.", True), ("Wie geht's?", "", False)]
    # retry later
    st.set_translation(m, 2, "How are you?")
    assert st.load(m.id).lines[1].en == "How are you?"
    assert "EN: How are you?" in (m.folder / "transcript.md").read_text()


def test_rename_moves_folder_and_keeps_everything(tmp_path):
    st = mk(tmp_path)
    m = st.create(now=datetime(2026, 10, 8, 15, 15))
    st.add_line(m, Line(1, "15:15:01", 1.0, "Hallo", "Hello", done=True))
    st.rename(m, "Call with Thomas / Budget?")
    assert m.folder.name == "15-15 - Call with Thomas Budget"
    assert st.load(m.id).name == "Call with Thomas Budget"
    assert "# Call with Thomas Budget" in (m.folder / "transcript.md").read_text()
    # writing continues in the renamed folder
    st.add_line(m, Line(2, "15:15:09", 9.0, "Tschüss"))
    assert len(st.load(m.id).lines) == 2
    st.rename(m, "")
    assert m.folder.name == "15-15"


def test_empty_unnamed_meeting_is_removed_on_finish(tmp_path):
    st = mk(tmp_path)
    m = st.create(now=datetime(2026, 10, 8, 10, 0))
    assert st.finish(m) is False
    assert not m.folder.exists() and not m.folder.parent.exists()
    m2 = st.create("Named", now=datetime(2026, 10, 8, 10, 1))
    assert st.finish(m2) is True and m2.folder.exists()


def test_list_and_delete(tmp_path):
    st = mk(tmp_path)
    a = st.create("A", now=datetime(2026, 10, 7, 9, 0))
    b = st.create("B", now=datetime(2026, 10, 8, 9, 0))
    ids = [x["id"] for x in st.list()]
    assert ids == [b.id, a.id]  # newest first
    st.delete(a)
    assert [x["id"] for x in st.list()] == [b.id]


def test_torn_last_line_after_crash_is_ignored(tmp_path):
    st = mk(tmp_path)
    m = st.create(now=datetime(2026, 10, 8, 11, 0))
    st.add_line(m, Line(1, "11:00:01", 1.0, "Eins"))
    with open(m.folder / "transcript.jsonl", "a") as f:
        f.write('{"id": 2, "time": "11:0')  # power cut mid-write
    assert [x.de for x in st.load(m.id).lines] == ["Eins"]


def test_safe_name():
    assert safe_name('a/b:c*?"<>|d') == "a b c d"
    assert len(safe_name("x" * 300)) == 80
