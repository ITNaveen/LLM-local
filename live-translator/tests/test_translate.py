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
        assert prefixes_kept >= 23, prefixes_kept
        assert max(len(m) for m in reqs) <= 1 + 2 * (6 + 4) + 1


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
