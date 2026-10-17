import time

from fake_ollama import FakeOllama

from livetranslator.translate import OllamaTranslator, build_system_prompt, clean_translation


def test_clean_translation():
    assert clean_translation("English: We need it by Friday.") == "We need it by Friday."
    assert clean_translation("**Translation:** Hello.") == "Hello."
    assert clean_translation('"We need it by Friday."', "Wir brauchen es bis Freitag.") == "We need it by Friday."
    assert clean_translation("We need it.\n(Note: literally 'we need that')") == "We need it."
    assert clean_translation("<think>hmm</think>Good morning.") == "Good morning."
    assert clean_translation("We will.\nAlternative: We shall.") == "We will."
    assert clean_translation("Ok.") == "Ok."


def test_system_prompt_includes_glossary_and_topic():
    p = build_system_prompt("Quarterly budget review", ["Müller", "SAP S/4HANA"])
    assert "Quarterly budget review" in p and "Müller, SAP S/4HANA" in p
    assert "Output only the English translation" in p


def test_streaming_translation_and_context():
    with FakeOllama(token_delay=0.005) as f:
        t = OllamaTranslator(f.url, "gemma3:12b", context_lines=4)
        deltas = []
        r = t.translate("Bitte sprechen Sie vorher mit Frau Müller aus der Buchhaltung.", deltas.append)
        assert r.ok and r.text == "Please talk to Ms. Müller from accounting beforehand."
        assert len(deltas) > 3 and deltas[-1] == r.text      # streamed word by word
        assert r.first_token_s <= r.total_s
        t.translate("Zweiter Satz.")
        body = f.requests[-1]["body"]
        assert body["keep_alive"] and body["stream"] is True
        msgs = body["messages"]
        assert msgs[0]["role"] == "system"
        # previous line is in the conversation as a user/assistant pair
        assert msgs[1]["content"].startswith("Bitte sprechen") and msgs[2]["role"] == "assistant"
        assert msgs[-1] == {"role": "user", "content": "Zweiter Satz."}


def test_history_is_append_only_between_trims():
    """Prompt-cache friendliness: consecutive requests share the whole previous prompt as prefix."""
    with FakeOllama(token_delay=0) as f:
        t = OllamaTranslator(f.url, "gemma3:12b", context_lines=6)
        prefixes_kept = 0
        for i in range(30):
            t.translate(f"Satz Nummer {i}.")
        reqs = [r["body"]["messages"] for r in f.requests if r["path"] == "/api/chat"]
        for a, b in zip(reqs, reqs[1:]):
            if b[:len(a) - 1] == a[:-1] and b[len(a) - 1] == a[-1]:
                prefixes_kept += 1
        # only the occasional trim breaks the prefix
        assert prefixes_kept >= 26, prefixes_kept      # only every ~10th line breaks the cached prefix
        assert max(len(m) for m in reqs) <= 1 + 2 * (6 + 10) + 1


def test_unknown_model_and_server_down_fail_fast():
    with FakeOllama() as f:
        t = OllamaTranslator(f.url, "nope:1b")
        r = t.translate("Hallo.")
        assert not r.ok and "not found" in r.error
        assert t.health()["running"] and not t.health()["model_ready"]
    t = OllamaTranslator("http://127.0.0.1:9", "gemma3:12b")
    t0 = time.monotonic()
    r = t.translate("Hallo.")
    assert not r.ok and time.monotonic() - t0 < 5
    assert t.health()["running"] is False


def test_thinking_models_get_think_false():
    with FakeOllama(models=("qwen3:14b",)) as f:
        OllamaTranslator(f.url, "qwen3:14b").translate("Hallo.")
        assert f.requests[-1]["body"]["think"] is False


def test_pull_reports_progress():
    with FakeOllama(models=()) as f:
        t = OllamaTranslator(f.url, "gemma3:4b")
        seen = []
        t.pull("gemma3:4b", seen.append)
        assert seen[-1]["status"] == "success"
        assert t.health()["model_ready"]


def test_summary():
    with FakeOllama() as f:
        t = OllamaTranslator(f.url, "gemma3:12b")
        out = t.summarize("[10:00] DE: Hallo\n EN: Hello\n" * 5)
        assert "Summary" in out


def test_garbage_is_not_kept_as_context_and_history_is_size_capped():
    from livetranslator.translate import OllamaTranslator

    assert OllamaTranslator.plausible("Hallo.", "Hello.")
    assert not OllamaTranslator.plausible("Hallo.", "You are a German-to-English interpreter " * 10)
    assert not OllamaTranslator.plausible("Hallo.", "")
    t = OllamaTranslator("http://127.0.0.1:9", "x", context_lines=40)
    t.history = [("Ein langer Satz " * 20, "A long sentence " * 20)] * 30
    t._trim_history()
    assert sum(len(d) + len(e) for d, e in t.history) <= t.MAX_HISTORY_CHARS
    assert len(t.history) >= 1


# ----------------------------------------------------------------- 1.4: accuracy checks, TranslateGemma
def test_lost_negation_is_asked_again_once():
    from fake_ollama import FakeOllama

    from livetranslator.translate import OllamaTranslator

    hint = "Das geht so nicht.\n\n(Check your translation: keep the negation.)"
    with FakeOllama(models=["gemma3:4b"], answers={"Das geht so nicht.": "That works like this.",
                                                    hint: "That does not work like this.",
                                                    "Das Budget beträgt 25.000 Euro.": "The budget is 25,000 euros."}) as fo:
        t = OllamaTranslator(fo.url, "gemma3:4b")
        r = t.translate("Das geht so nicht.")
        assert r.ok and r.text == "That does not work like this."
        assert len(fo.chats()) == 2
        assert t.history == [("Das geht so nicht.", "That does not work like this.")]   # without the hint
        assert all(c["body"]["think"] is False and c["body"]["options"]["temperature"] == 0.0 for c in fo.chats())
        # a correct line is not asked twice; a preview (record=False) is never checked
        t.translate("Das Budget beträgt 25.000 Euro.")
        assert len(fo.chats()) == 3
        t.translate("Das geht so nicht.", record=False)
        assert len(fo.chats()) == 4


def test_a_worse_second_answer_is_not_taken():
    from fake_ollama import FakeOllama

    from livetranslator.translate import OllamaTranslator

    g = "Wir brauchen 3 Leute bis 17. Oktober."
    with FakeOllama(models=["gemma3:4b"], answers={g: "We need people by October 17.",
                                                    g + "\n\n(Check your translation: keep the numbers 3.)": "Hi."}) as fo:
        t = OllamaTranslator(fo.url, "gemma3:4b")
        assert t.translate(g).text == "We need people by October 17."


def test_translategemma_gets_its_own_prompt_and_echo_is_caught():
    from fake_ollama import FakeOllama

    from livetranslator.translate import TG_PREFIX, OllamaTranslator

    with FakeOllama(models=["translategemma:4b"], answers={"Guten Morgen zusammen.": "Good morning, everyone.",
                                                            "Das machen wir morgen.": "Das machen wir morgen."}) as fo:
        t = OllamaTranslator(fo.url, "translategemma:4b", terms=["SAP"])
        r = t.translate("Guten Morgen zusammen.")
        assert r.ok and r.text == "Good morning, everyone."
        msgs = fo.chats()[-1]["body"]["messages"]
        assert [m["role"] for m in msgs] == ["user"] and msgs[0]["content"] == TG_PREFIX + "Guten Morgen zusammen."
        r = t.translate("Das machen wir morgen.")              # echoes the German twice -> a failure, not saved
        assert not r.ok and "repeated the German" in r.error
        assert t.history == [("Guten Morgen zusammen.", "Good morning, everyone.")]
        msgs = fo.chats()[-2]["body"]["messages"]
        assert [m["role"] for m in msgs] == ["user", "assistant", "user"]       # previous line as context
        assert [m["role"] for m in fo.chats()[-1]["body"]["messages"]] == ["user"]   # retry without context


def test_checks_know_german_times_dates_dialect_and_tag_questions():
    from livetranslator.translate import check_translation as c

    fine = [("Treffen wir uns um 15 Uhr.", "Let's meet at 3 pm."), ("um 14 Uhr 30", "at 2:30 pm"),
            ("Bis zum 15.3.", "by March 15"), ("der 3. Punkt", "the third point"), ("20 Optionen", "twenty options"),
            ("Das sind 2,50 Euro.", "That is 2.5 euros."), ("Wir nutzen .NET 8.", "We use .NET 8."),
            ("Net Zero ist das Ziel.", "Net zero is the goal."), ("Die Koalition steht.", "The coalition stands."),
            ("Passt scho, oder net?", "That's fine, right?"), ("Das geht so nicht.", "That doesn’t work like this."),
            ("I hob koa Zeit.", "I have no time."), ("Das passt, oder nicht", "That works, right?"),
            ("Das passt doch, oder nicht, Thomas?", "That works, right, Thomas?"),
            ("Das passt, nicht wahr.", "That works, doesn't it."), ("Passt scho, gell net", "It's fine, right"),
            ("Das passt, oder nicht, Herr Müller?", "That works, right, Mr Müller?"),
            ("Das passt, oder etwa nicht?", "That works, right?"), ("Das passt, gell, ned?", "Fine, right?")]
    for de, en in fine:
        assert c(de, en) == [], (de, en)
    wrong = [("Wir brauchen 20 Leute.", "We need 2 people.", "the numbers 20"),
             ("Das kostet 500 Euro.", "That costs 5 euros.", "the numbers 500"),
             ("Das schaffen wir nicht bis Freitag, oder nicht?", "We'll make it by Friday, right?", "the negation"),
             ("Des passt net.", "That is fine.", "the negation"), ("I hob koa Zeit.", "I have time.", "the negation"),
             ("Treffen wir uns um 15 Uhr.", "Let's meet at 5 pm.", "the numbers 15 Uhr"),
             ("Das ist nicht wahr.", "That is true.", "the negation"),
             ("Ob wir das schaffen oder nicht", "Whether we can do it", "the negation")]
    for de, en, problem in wrong:
        assert problem in c(de, en), (de, en, c(de, en))


def test_echo_guard_accepts_lines_that_are_the_same_in_english():
    from livetranslator.translate import _untranslated

    for same in ["Okay.", "Sorry.", "Super.", "Thomas?", "Status?", "Next Steps.", "Müller, Schäfer, Köhler?"]:
        assert not _untranslated(same, same), same
    assert _untranslated("Das machen wir morgen.", "Das machen wir morgen.")
    assert _untranslated("Bis später alle zusammen.", "Bis später alle zusammen.")
    assert not _untranslated("Wir nutzen die Mobiliar und das Örtliche.", "We use Die Mobiliar and Das Örtliche.")
    assert _untranslated("Das machen wir morgen und dann ist es fertig.", "Das machen wir morgen und dann ist es fertig")
