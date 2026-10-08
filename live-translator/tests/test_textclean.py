from livetranslator.asr import build_prompt
from livetranslator.textclean import clean_transcript, collapse_repeats, similarity


def test_known_subtitle_hallucinations_are_dropped():
    for t in ["Untertitel im Auftrag des ZDF, 2021", "Untertitel der Amara.org-Community",
              "Untertitelung im Auftrag des ZDF für funk, 2017", "(Musik)", "[Musik]", "*Musik*", "Musik.",
              "Thank you for watching!", "  ...  ", ""]:
        assert clean_transcript(t) == "", t


def test_real_speech_is_kept():
    s = "Wir müssen das Angebot bis Freitag fertig haben."
    assert clean_transcript(s) == s
    # a confident short "Danke." is real speech
    assert clean_transcript("Danke.", duration_s=0.5, avg_logprob=-0.2, no_speech_prob=0.05) == "Danke."
    assert clean_transcript("Vielen Dank.", duration_s=0.9, avg_logprob=-0.3, no_speech_prob=0.1) == "Vielen Dank."


def test_weak_polite_phrases_are_dropped():
    assert clean_transcript("Vielen Dank fürs Zuschauen!", avg_logprob=-1.1, no_speech_prob=0.6) == ""
    assert clean_transcript("Tschüss.", avg_logprob=-0.4, no_speech_prob=0.7) == ""


def test_fake_sentence_removed_but_real_part_kept():
    t = "Das machen wir morgen. Untertitel im Auftrag des ZDF."
    assert clean_transcript(t) == "Das machen wir morgen."


def test_tags_inside_text_removed():
    assert clean_transcript("Also (Lachen) das ist gut.") == "Also das ist gut."


def test_repetition_loops_collapsed():
    assert collapse_repeats("und dann und dann und dann und dann gehen wir") == "und dann und dann gehen wir"
    assert collapse_repeats("ja ja ja ja ja ja") == "ja ja"
    assert collapse_repeats("das ist das Ziel") == "das ist das Ziel"


def test_prompt_echo_dropped():
    terms = "Müller, SAP-System, Projekt Phoenix"
    assert clean_transcript("Müller, SAP-System, Projekt Phoenix", prompt_terms=terms) == ""


def test_similarity():
    assert similarity("Wir müssen das machen.", "wir müssen das machen") > 0.95
    assert similarity("Guten Morgen", "Bis morgen") < 0.8


def test_build_prompt():
    assert build_prompt([], "") is None
    p = build_prompt(["Müller", "SAP"], "Das Angebot ist fertig.")
    assert p.startswith("Müller, SAP") and "Das Angebot ist fertig" in p
    long_prev = "wort " * 200
    assert len(build_prompt(["A"], long_prev)) <= 360
