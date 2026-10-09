"""Automatic retries of failed translations and the English preview must never get in the
way of live lines, and must never write into the wrong meeting."""
import json
import time
from datetime import datetime, timedelta

from fake_ollama import FakeOllama

from livetranslator.asr import ScriptedASR
from livetranslator.audio_io import PushSource
from livetranslator.config import SettingsStore
from livetranslator.pipeline import Pipeline
from livetranslator.storage import Line, MeetingStore


def wait_for(cond, timeout=20.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.05)
    return False


def make_pipeline(tmp_path, fo, **settings):
    cfg = SettingsStore(tmp_path / "s.json")
    cfg.update({"ollama_url": fo.url, "llm_model": "gemma3:4b", "llm_fallback": "gemma3:4b",
                "live_preview": False, **settings})
    events = []
    pipe = Pipeline(cfg, MeetingStore(tmp_path / "M"), lambda e: events.append({**e, "_t": time.monotonic()}),
                    asr_factory=lambda s: ScriptedASR())
    assert wait_for(lambda: pipe.asr_state["status"] == "ready" and pipe.llm_state.get("loaded"))
    return pipe, events


def add_line(pipe, sess, de):
    m = sess.meeting
    ln = Line(id=(m.lines[-1].id + 1) if m.lines else 1, time="10:00:00", offset=float(len(m.lines)), de=de)
    pipe.store.add_line(m, ln)
    return ln


def live(pipe, sess, ln):
    pipe._tr_q.put(("line", sess, ln, True, None))


def finals(events, ln_id=None, ok=None):
    return [e for e in events if e["type"] == "tr" and e.get("final")
            and (ln_id is None or e["id"] == ln_id) and (ok is None or e["ok"] == ok)]


def fail_lines(pipe, fo, events, n):
    """n lines of the current meeting whose translation failed (saved without English, retry pending)."""
    sess = pipe.session
    lines = [add_line(pipe, sess, f"Satz Nummer {i}.") for i in range(n)]
    for ln in lines:
        pipe.store.set_translation(sess.meeting, ln.id, "", False)
        pipe._note_retry(sess, ln, False)
    pipe._safe_set_llm(stuck=True)
    return sess, lines


# ----------------------------------------------------------------- backoff
def test_retry_backoff_grows_and_gives_up(tmp_path):
    with FakeOllama(models=["gemma3:4b"]) as fo:
        pipe, events = make_pipeline(tmp_path, fo)
        pipe.start(name="t", source=PushSource(pipe.push_audio))
        sess = pipe.session
        ln = add_line(pipe, sess, "Hallo.")
        gaps = []
        for _ in range(pipe.RETRY_MAX_ATTEMPTS + 1):   # the live try + up to RETRY_MAX_ATTEMPTS retries
            t = time.monotonic()
            pipe._note_retry(sess, ln, False)
            e = pipe._retry_pending.get((sess.meeting.id, ln.id))
            if e is None:
                break
            gaps.append(round(e["next_at"] - t))
        assert gaps[:6] == [10, 20, 40, 80, 160, 300] and max(gaps) == 300
        assert (sess.meeting.id, ln.id) not in pipe._retry_pending     # gave up (↻ still works)
        pipe._note_retry(sess, ln, False)
        pipe._note_retry(sess, ln, True)                                  # success clears it
        assert not pipe._retry_pending
        pipe.stop(wait=True)
        pipe.shutdown()


# ----------------------------------------------------------------- live lines first
def test_many_retries_never_delay_a_live_line(tmp_path):
    with FakeOllama(models=["gemma3:4b"], token_delay=0.1) as fo:   # ~0.4 s per retried line
        pipe, events = make_pipeline(tmp_path, fo)
        pipe.RETRY_FIRST_S = 3600.0          # failed lines wait until we release them
        pipe.start(name="t", source=PushSource(pipe.push_audio))
        sess, failed = fail_lines(pipe, fo, events, 25)
        pipe._safe_set_llm(stuck=False)
        pipe._retry_soon()                   # e.g. Ollama came back: all 25 are due at once
        assert wait_for(lambda: len(finals(events, ok=True)) >= 1, 10)   # retries are running
        ln = add_line(pipe, sess, "Das ist neu.")
        t0 = time.monotonic()
        live(pipe, sess, ln)
        assert wait_for(lambda: finals(events, ln.id, ok=True), 20)
        took = finals(events, ln.id, ok=True)[0]["_t"] - t0
        assert took < 1.5, f"live English waited {took:.1f} s behind retries"
        # ... and the retries still finish afterwards, each line exactly once
        assert wait_for(lambda: len({e["id"] for e in finals(events, ok=True)}) == 26, 60)
        assert len(finals(events, ok=True)) == 26
        pipe.stop(wait=True)
        pipe.shutdown()


def test_a_running_retry_gives_way_to_a_live_line(tmp_path):
    beh = {"gemma3:4b": {"token_s": 0.25}}
    with FakeOllama(models=["gemma3:4b"], behaviour=beh) as fo:
        pipe, events = make_pipeline(tmp_path, fo)
        pipe.RETRY_FIRST_S = 3600.0
        pipe.start(name="t", source=PushSource(pipe.push_audio))
        sess = pipe.session
        long_de = " ".join(["Wir", "besprechen", "heute", "das", "Budget", "für", "das", "nächste", "Jahr"] * 3)
        old = add_line(pipe, sess, long_de)          # failed earlier; its retry takes ~7 s to write
        pipe.store.set_translation(sess.meeting, old.id, "", False)
        pipe._note_retry(sess, old, False)
        pipe._retry_soon()
        assert wait_for(lambda: any(r["body"]["messages"][-1]["content"] == long_de for r in fo.chats()), 5)
        time.sleep(0.8)                              # the retry is writing now
        ln = add_line(pipe, sess, "Ja.")
        t0 = time.monotonic()
        live(pipe, sess, ln)
        assert wait_for(lambda: finals(events, ln.id, ok=True), 15)
        took = finals(events, ln.id, ok=True)[0]["_t"] - t0
        assert took < 2.0, f"the live line waited {took:.1f} s behind a retry"
        # the retry comes back afterwards and completes
        assert wait_for(lambda: finals(events, old.id, ok=True), 30)
        assert pipe.store.load(sess.meeting.id).lines[0].ok
        pipe.stop(wait=True)
        pipe.shutdown()


def test_retry_after_stop_lets_the_model_finish_loading(tmp_path):
    """Ollama abandons a load when the request gives up: a retry must not cut a slow load short."""
    beh = {"gemma3:4b": {"load_s": 3.0}}
    with FakeOllama(models=["gemma3:4b"], behaviour=beh) as fo:
        pipe, events = make_pipeline(tmp_path, fo)
        pipe.RETRY_FIRST_S = 3600.0
        pipe.RETRY_FIRST_TOKEN_S = 0.5
        pipe.start(name="t", source=PushSource(pipe.push_audio))
        sess, lines = fail_lines(pipe, fo, events, 1)
        pipe.stop(wait=True)
        fo.loaded.clear()                            # e.g. unloaded to free memory meanwhile
        pipe._safe_set_llm(stuck=False)
        pipe._retry_soon()
        assert wait_for(lambda: finals(events, lines[0].id, ok=True), 20)
        tries = [r for r in fo.chats() if r["body"]["messages"][-1]["content"] == lines[0].de]
        assert len(tries) == 1                       # one request, which waited for the load
        pipe.shutdown()


def test_stop_is_not_held_up_by_retries(tmp_path):
    with FakeOllama(models=["gemma3:4b"], token_delay=0.1) as fo:
        pipe, events = make_pipeline(tmp_path, fo)
        pipe.RETRY_FIRST_S = 3600.0
        pipe.start(name="t", source=PushSource(pipe.push_audio))
        fail_lines(pipe, fo, events, 20)
        pipe._safe_set_llm(stuck=False)
        pipe._retry_soon()
        t0 = time.monotonic()
        pipe.stop(wait=True)
        assert time.monotonic() - t0 < 5 and pipe.state == "idle"
        pipe.shutdown()


def test_still_failing_ollama_is_probed_not_hammered(tmp_path):
    with FakeOllama(models=["gemma3:4b"]) as fo:
        pipe, events = make_pipeline(tmp_path, fo)
        pipe.RETRY_FIRST_S = 1.0
        pipe.start(name="t", source=PushSource(pipe.push_audio))
        fail_lines(pipe, fo, events, 10)
        fo.fail = True                       # still broken
        n0 = len(fo.chats())
        pipe._retry_soon()
        time.sleep(3.0)
        # one probe roughly every RETRY_FIRST_S (each probe = request + one retry), not 10 lines in a burst
        assert len(fo.chats()) - n0 <= 8, len(fo.chats()) - n0
        fo.fail = False
        pipe.stop(wait=True)
        pipe.shutdown()


# ----------------------------------------------------------------- the right meeting
def test_failed_line_is_saved_without_cut_off_english(tmp_path):
    beh = {"gemma3:4b": {"token_s": 0.6}}
    with FakeOllama(models=["gemma3:4b"], behaviour=beh) as fo:
        pipe, events = make_pipeline(tmp_path, fo)
        orig = pipe.translator.translate
        pipe.translator.translate = lambda *a, **k: orig(*a, **{**k, "max_gen_s": 1.0})
        pipe.RETRY_FIRST_S = 3600.0
        pipe.start(name="t", source=PushSource(pipe.push_audio))
        sess = pipe.session
        ln = add_line(pipe, sess, "Wir müssen das Angebot bis Freitag fertig haben und dem Kunden schicken.")
        live(pipe, sess, ln)
        assert wait_for(lambda: finals(events, ln.id), 30)
        assert finals(events, ln.id)[0]["ok"] is False and finals(events, ln.id)[0]["en"] == ""
        pipe.stop(wait=True)
        m = pipe.store.load(sess.meeting.id)
        assert m.lines[0].en == "" and m.lines[0].ok is False
        md = (m.folder / "transcript.md").read_text()
        assert "[EN]" not in md and "(translation unavailable)" in md
        pipe.shutdown()


def test_renaming_a_just_stopped_meeting_keeps_its_retried_english(tmp_path):
    with FakeOllama(models=["gemma3:4b"]) as fo:
        pipe, events = make_pipeline(tmp_path, fo)
        pipe.RETRY_FIRST_S = 3600.0
        pipe.start(name="t", source=PushSource(pipe.push_audio))
        sess, lines = fail_lines(pipe, fo, events, 3)
        pipe.stop(wait=True)
        mid = sess.meeting.id
        m = pipe.meeting_object(mid)
        assert m is sess.meeting                      # the server edits the app's own object
        pipe.store.rename(m, "Kundentermin")
        pipe._safe_set_llm(stuck=False)
        pipe._retry_soon()
        assert wait_for(lambda: all(x.ok and x.en for x in pipe.store.load(mid).lines), 20)
        disk = pipe.store.load(mid)
        assert disk.folder.name.endswith("Kundentermin") and disk.folder == m.folder
        assert "[EN] Satz Nummer 2." in (disk.folder / "transcript.md").read_text()
        pipe.shutdown()


def test_deleted_meeting_never_writes_into_the_next_one(tmp_path):
    with FakeOllama(models=["gemma3:4b"]) as fo:
        pipe, events = make_pipeline(tmp_path, fo)
        pipe.RETRY_FIRST_S = 3600.0
        pipe.start(name="", source=PushSource(pipe.push_audio))
        old, _ = fail_lines(pipe, fo, events, 3)
        pipe.stop(wait=True)
        pipe.forget_meeting(old.meeting.id)
        pipe.store.delete(pipe.store.load(old.meeting.id))
        assert not pipe._retry_pending
        pipe.start(name="", source=PushSource(pipe.push_audio))   # same minute: may reuse the folder
        new = pipe.session
        ln = add_line(pipe, new, "Neuer Satz.")
        live(pipe, new, ln)
        assert wait_for(lambda: finals(events, ln.id, ok=True) and finals(events, ln.id, ok=True)[-1]["meeting"] == new.meeting.id)
        pipe._retry_soon()
        time.sleep(1.5)
        pipe.stop(wait=True)
        recs = [json.loads(x) for x in (new.meeting.folder / "transcript.jsonl").read_text().splitlines()]
        assert [r.get("en") for r in recs if "update" in r] == ["[EN] Neuer Satz."], recs
        pipe.shutdown()


def test_stale_copy_follows_a_rename_and_drops_after_delete(tmp_path):
    store = MeetingStore(tmp_path / "M")
    t = datetime(2026, 10, 9, 10, 0, 5)
    a = store.create("", now=t)
    store.add_line(a, Line(id=1, time="10:00:00", offset=0.0, de="Eins."))
    copy = store.load(a.id)                       # e.g. ↻ on an old meeting
    store.rename(a, "Neu")
    assert store.set_translation(copy, 1, "One.", True) is not None
    assert store.load(a.id).lines[0].en == "One." and copy.folder == a.folder
    store.delete(a)
    b = store.create("", now=t + timedelta(seconds=30))   # same minute: same folder name as a had
    assert b.folder == copy.folder.parent / "10-00"
    store.add_line(b, Line(id=1, time="10:00:01", offset=0.0, de="Zwei."))
    assert store.set_translation(copy, 1, "Wrong.", True) is None
    assert store.load(b.id).lines[0].en == ""


def test_continue_keeps_retrying_and_translates_each_line_once(tmp_path):
    with FakeOllama(models=["gemma3:4b"]) as fo:
        pipe, events = make_pipeline(tmp_path, fo)
        pipe.RETRY_FIRST_S = 3600.0
        pipe.start(name="t", source=PushSource(pipe.push_audio))
        sess, lines = fail_lines(pipe, fo, events, 3)
        pipe.stop(wait=True)
        mid = sess.meeting.id
        pipe.start(continue_id=mid, source=PushSource(pipe.push_audio))       # back from the coffee break
        new = pipe.session
        assert new.meeting is sess.meeting
        assert all(e["sess"] is new for e in pipe._retry_pending.values()) and len(pipe._retry_pending) == 3
        ln = add_line(pipe, new, "Weiter geht es.")
        live(pipe, new, ln)
        pipe._safe_set_llm(stuck=False)
        pipe._retry_soon()
        assert wait_for(lambda: all(x.ok for x in new.meeting.lines), 20)
        time.sleep(0.5)
        per_line = {}
        for r in fo.chats():
            k = r["body"]["messages"][-1]["content"]
            per_line[k] = per_line.get(k, 0) + 1
        assert per_line["Weiter geht es."] == 1
        assert all(per_line[x.de] == 1 for x in lines)     # retried once, by the continued session only
        md = (new.meeting.folder / "transcript.md").read_text()
        assert "Weiter geht es." in md and "[EN] Satz Nummer 0." in md
        pipe.stop(wait=True)
        pipe.shutdown()


# ----------------------------------------------------------------- the preview gives way
def test_english_preview_gives_way_to_a_finished_line(tmp_path):
    beh = {"gemma3:4b": {"token_s": 0.25}}
    with FakeOllama(models=["gemma3:4b"], behaviour=beh) as fo:
        pipe, events = make_pipeline(tmp_path, fo, live_preview=True)
        pipe.start(name="t", source=PushSource(pipe.push_audio))
        sess = pipe.session
        long_de = " ".join(["Wir", "besprechen", "heute", "das", "Budget", "für", "das", "nächste", "Jahr"] * 4)
        pipe._cur_seg_start = 1.0
        pipe._partial_tr_slot = (sess, 1.0, long_de)      # preview of ~37 words: ~9 s of writing
        assert wait_for(lambda: any(r["body"]["messages"][-1]["content"] == long_de for r in fo.chats()), 5)
        time.sleep(0.5)                                   # the preview is writing now
        ln = add_line(pipe, sess, "Ja.")
        t0 = time.monotonic()
        live(pipe, sess, ln)
        assert wait_for(lambda: finals(events, ln.id, ok=True), 15)
        took = finals(events, ln.id, ok=True)[0]["_t"] - t0
        assert took < 2.0, f"the finished line waited {took:.1f} s for the preview"
        assert not [e for e in events if e["type"] == "partial_en"]
        assert all(d != long_de for d, _ in pipe.translator.history)
        pipe.stop(wait=True)
        pipe.shutdown()
