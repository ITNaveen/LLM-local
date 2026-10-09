"""Editor v5 ("newsroom"): the narrator only sets the base, lines are grounded in the footage,
the cold open, cutaways, the built-in score and sound effects, Hindi text without libass,
and the expressive voice."""

import asyncio
import subprocess

import numpy as np
import pytest

from storymaker import editor, music, render, research, textimg, timeline, voice
from storymaker.llm import NoLLM
from storymaker.util import ffmpeg, media_duration, probe


# ------------------------------------------------------------------ language and lines
@pytest.mark.parametrize("text,lang", [
    ("ही वाज़ अनकॉन्शियस एट द आर्म हॉस्पिटल फॉर टू डेज़", "en"),        # English, Hindi letters
    ("31 जुलाई को घायल दिल्ली पुलिसकर्मियों के चार परिवारों ने एक प्रेस कॉन्फ्रेंस की।", "hi"),
    ("बिहाइंड एव्री यूनिफार्म इज अ फैमिली। यानी हर वर्दी के पीछे भी एक परिवार होता है।", "hi"),
    ("ये कैसे चलो", "hi"),                                              # too short to judge
])
def test_english_captioned_in_devanagari_is_detected(text, lang):
    assert editor.speech_language(text) == lang


def test_lines_are_polished_for_the_narrator():
    # a picture description the AI invented (it can't see the footage) is removed
    # (the whole line goes: what is left of it usually leans on the invented picture)
    assert editor.polish_line("देख रहे हो ये लाइटें? ये तूफ़ान है।") == ""
    assert editor.polish_line("ये सिर्फ फ्लैश नहीं हैं, ये तूफ़ानी हवा है!") == ""
    # never cut mid-thought ('... मेट्रो के 16')
    line = editor.polish_line("पुलिस की तरफ से साफ़ नो है! पुलिस ने कहा है कि धारा 163 के तहत नई "
                              "दिल्ली ज़िले में इकट्ठा होने पर रोक है। मेट्रो के 16", 30)
    assert line == "पुलिस की तरफ से साफ़ नो है! पुलिस ने कहा है कि धारा 163 के तहत नई दिल्ली ज़िले में इकट्ठा होने पर रोक है।"
    # spoken Hindi, not formal
    assert editor.polish_line("यह अत्यंत आवश्यक है।") == "यह बेहद ज़रूरी है।"
    assert editor.polish_line("पहली बात साफ़ है! " + " ".join(["शब्द"] * 30) + "।", 22) == "पहली बात साफ़ है!"
    assert editor.polish_line(" ".join(["शब्द"] * 40)) == ""            # no sentence fits: dropped


def test_unsupported_claims_are_caught():
    corpus = "delhi police says 2,873 criminals were tracked. 23,000 जवान तैनात हैं। section 163"
    assert editor.unsupported_claim("कुल 2873 अपराधी और 23000 जवान!", corpus) == ""
    assert editor.unsupported_claim("कुल 2900 अपराधी!", corpus) == "number 2900"
    assert editor.unsupported_claim("प्रधानमंत्री ने चोटों को मान्यता दी है।", corpus) == "प्रधानमंत्री"
    assert editor.unsupported_claim("PM मोदी ने कहा", corpus + " prime minister modi said") == ""


def test_fact_check_cuts_lines_the_footage_does_not_support():
    cat = {"P1": {"id": "a#0", "video_id": "a", "text": "पुलिस ने 250 जवान घायल बताए।", "summary": "police",
                  "strength": 4, "start": 0, "end": 12}}
    outline = {"scenes": [
        {"act": "opening", "type": "narration", "text": "250 जवान घायल! सोचिए ज़रा।"},
        {"act": "buildup", "type": "dialogue", "pid": "P1"},
        {"act": "rising", "type": "narration", "text": "गृह मंत्री ने 900 लोगों को पकड़ा।"}]}
    logs = []
    editor.check_facts(NoLLM(), "delhi", "", outline, cat, list(cat.values()), [], logs.append)
    assert [s["type"] for s in outline["scenes"]] == ["narration", "dialogue"]
    assert any("doesn't support" in line for line in logs)


def test_narrator_is_never_heard_twice_in_a_row():
    scenes = [{"type": "narration", "text": "a"}, {"type": "voiceover", "text": "b", "pid": "P1"},
              {"type": "dialogue", "pid": "P2"}, {"type": "voiceover", "text": "c", "pid": "P3"},
              {"type": "voiceover", "text": "d", "pid": "P4"}, {"type": "dialogue", "pid": "P5"},
              {"type": "voiceover", "text": "e", "pid": "P6"}, {"type": "dialogue", "pid": "P7"}]
    out = editor.space_out_narrator(scenes)
    assert [s.get("pid") or s["text"] for s in out] == ["a", "P2", "P4", "P5", "P6", "P7"]
    # a headline right next to the narrator is dropped: one of them sets the base, not both
    out = editor.space_out_narrator([{"type": "text", "text": "x"}, {"type": "narration", "text": "y"},
                                     {"type": "dialogue", "pid": "P1"}, {"type": "text", "text": "z"},
                                     {"type": "dialogue", "pid": "P2"}])
    assert [s.get("pid") or s["text"] for s in out] == ["y", "P1", "z", "P2"]
    assert editor._same_quote("जनता का भरोसा हमारी सबसे बड़ी ताकत है v5k11 v5k17",
                              "जनता का भरोसा हमारी सबसे बड़ी ताकत है v0k14 v0k26 v0k5")
    assert not editor._same_quote("पुलिस ने इजाज़त नहीं दी", "पुलिस पर पत्थर बरसाए गए")


def test_closing_line_asks_the_viewer():
    scenes = [{"act": "climax", "type": "dialogue", "pid": "P1"},
              {"act": "ending", "type": "narration", "text": "ये लड़ाई अभी खत्म नहीं हुई।"},
              {"act": "ending", "type": "dialogue", "pid": "P2"}]
    out = editor.add_closing([dict(s) for s in scenes], "medium")
    assert out[1]["text"].endswith(editor.CTA_LINE) and len(out) == 3
    assert editor.add_closing([dict(s) for s in scenes], "none") == scenes


def test_missing_cutaway_picture_shows_the_speaker_instead():
    story = {"acts": [{"beats": [{"clips": [
        {"video_id": "spk", "start": 10.0, "end": 14.0},
        {"video_id": "brl", "start": 50.0, "end": 53.0, "audio": {"video_id": "spk", "start": 14.0}},
        {"video_id": "spk", "start": 17.0, "end": 20.0}]}]}]}
    out = editor.without_clips(story, {("brl", 50.0)})
    clips = out["acts"][0]["beats"][0]["clips"]
    assert [(c["video_id"], c["start"], c["end"]) for c in clips] == \
        [("spk", 10.0, 14.0), ("spk", 14.0, 17.0), ("spk", 17.0, 20.0)]   # words never jump
    assert "audio" not in clips[1]
    assert story["acts"][0]["beats"][0]["clips"][1]["video_id"] == "brl"   # original untouched


def test_long_slogan_topics_search_their_keywords():
    plan = research.make_queries(NoLLM(), "kal CJP ki sutai hogi aur tote laal bhi honge, ab batting "
                                 "India ki hai", "Delhi police vs CJP protest", lambda m: None)
    assert plan["queries"][1].startswith("delhi police vs cjp")
    assert not any(q.endswith("highlights") and "sutai hogi" in q for q in plan["queries"])


# ------------------------------------------------------------------ timeline
def _beat(kind, clips, **kw):
    return {"kind": kind, "audio": kw.pop("audio", "original"), "clips": clips, **kw}


def test_timeline_cutaway_sound_is_seamless_and_effects_are_placed(settings):
    clips = [{"video_id": "spk", "start": 10.0, "end": 13.517},
             {"video_id": "brl", "start": 50.0, "end": 53.011, "audio": {"video_id": "spk", "start": 13.517}},
             {"video_id": "spk", "start": 16.528, "end": 20.0}]
    story = {"title_hi": "शीर्षक", "acts": [
        {"key": "opening", "mood": "tense", "beats": [
            _beat("teaser", [{"video_id": "t", "start": 1, "end": 4, "fx": "punch", "zoom": True}], sfx="hit"),
            _beat("title", [{"video_id": "b", "start": 1, "end": 4}], audio="text", narration="शीर्षक",
                  overlay_style="title", sfx="boom")]},
        {"key": "rising", "mood": "tense", "beats": [_beat("dialogue", clips, sfx="whoosh")]},
        {"key": "climax", "mood": "epic", "beats": [_beat("dialogue", [{"video_id": "c", "start": 0, "end": 5}])]}]}
    tl = timeline.build(story, {}, {}, {}, settings)
    segs = tl["segments"]
    assert not any(s["type"] == "card" for s in segs)            # title is a beat, no extra card
    teaser, title = segs[0], segs[1]
    assert teaser["fx"] == "punch" and teaser["mode"] == "teaser"
    assert title.get("overlay_style") == "title" and title["overlay"] == "शीर्षक"
    a, b, c = segs[2:5]
    # the speaker's sound runs on sample-exactly across the picture changes
    assert b["audio_src"]["video_id"] == "spk"
    assert b["audio_src"]["src_start"] == pytest.approx(a["src_start"] + a["frames"] / 30, abs=1e-3)
    sound_c = c.get("audio_src", {}).get("src_start", c["src_start"])
    assert sound_c == pytest.approx(b["audio_src"]["src_start"] + b["frames"] / 30, abs=1e-3)
    assert c["src_start"] == 16.528                                # picture times stay put
    fx = {(i, e["kind"]) for i, act in enumerate(tl["acts"]) for e in act["effects"]}
    assert {(0, "hit"), (0, "boom"), (1, "whoosh"), (1, "riser")} <= fx
    riser = next(e for e in tl["acts"][1]["effects"] if e["kind"] == "riser")
    assert riser["t"] == pytest.approx(tl["acts"][1]["end"] - tl["acts"][1]["start"], abs=0.01)


# ------------------------------------------------------------------ sound
@pytest.mark.parametrize("mood", list(music.MOOD_SYNTH))
def test_builtin_score_is_level_and_clean(mood):
    x = music.synth_bed(mood, 12, seed=3)
    rms = float(np.sqrt((x ** 2).mean()))
    assert x.shape == (12 * music.SR, 2) and 0.05 < rms < 0.15
    assert np.abs(x).max() < 0.85 and not np.isnan(x).any()


def test_sound_effects_and_bed_mix(tmp_path):
    for kind in music.SFX_KINDS:
        y = music.sfx(kind)
        assert y.ndim == 2 and 0.3 < np.abs(y).max() <= 1.0
    out = tmp_path / "bed.wav"
    music.render_act_bed(out, 6.0, None, "tense", [(0, 6, 0.0)], [], generate=False,
                         effects=[{"t": 2.0, "kind": "hit", "gain": 0.6},
                                  {"t": 6.0, "kind": "riser", "gain": 0.5}])
    data = music.decode(out)
    assert len(data) == 6 * 48000
    rms = lambda a, b: float(np.sqrt((data[int(a * 48000):int(b * 48000)] ** 2).mean()))
    assert rms(0, 1.5) < 1e-4                                       # music off, nothing else
    assert rms(2.0, 2.3) > 0.05                                     # the hit
    assert rms(5.5, 6.0) > rms(3.0, 3.5) * 2                        # the riser swells to its end


# ------------------------------------------------------------------ pictures
def test_text_pictures(tmp_path):
    assert textimg.available()
    textimg.headline(tmp_path / "h.png", 640, 360, "Delhi Police ने कहा — कोई इजाज़त नहीं!")
    textimg.title(tmp_path / "t.png", 640, 360, "कल दिल्ली में क्या होगा?")
    textimg.subtitle(tmp_path / "s.png", 640, 360, "सोचिए ज़रा!")
    from PIL import Image
    for name in ("h", "t"):
        img = Image.open(tmp_path / f"{name}.png")
        assert img.size == (640, 360) and img.mode == "RGBA" and img.getbbox()
    sub = Image.open(tmp_path / "s.png")
    assert sub.size == (640, textimg.subtitle_band(360)) and sub.getbbox()


def test_subtitle_track_timing(tmp_path):
    subs = [{"start": 1.0, "end": 2.0, "text": "सोचिए ज़रा!"}]
    track = render.subtitle_track(subs, 320, 180, 4.0, tmp_path)
    base = tmp_path / "base.mp4"
    ffmpeg("-f", "lavfi", "-i", "color=c=black:s=320x180:r=30:d=4", "-c:v", "libx264", "-preset",
           "ultrafast", str(base))
    out = tmp_path / "o.mp4"
    ffmpeg("-i", str(base), "-i", str(track), "-filter_complex",
           "[0:v][1:v]overlay=0:H-h:eof_action=pass:format=auto,format=yuv420p", "-c:v", "libx264",
           "-preset", "ultrafast", str(out))

    def ink(t):
        raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(t), "-i", str(out), "-frames:v", "1",
                              "-vf", "format=gray", "-f", "rawvideo", "-"], capture_output=True).stdout
        return np.frombuffer(raw, dtype=np.uint8).max()
    assert ink(0.5) < 40 and ink(1.5) > 200 and ink(3.0) < 40


def _tone_clip(path, seconds=6, freq=440, picture=True):
    args = ["-f", "lavfi", "-i", "testsrc2=s=320x180:r=30:d=%d" % seconds] if picture else []
    ffmpeg(*args, "-f", "lavfi", "-i", f"sine=frequency={freq}:duration={seconds}",
           *(["-c:v", "libx264", "-preset", "ultrafast"] if picture else []), "-shortest", str(path))


def test_cutaway_plays_the_speakers_sound_over_other_footage(tmp_path):
    speaker, broll = tmp_path / "s.mp4", tmp_path / "b.mp4"
    _tone_clip(speaker, freq=440)
    ffmpeg("-f", "lavfi", "-i", "mandelbrot=s=320x180:r=30", "-t", "6", "-c:v", "libx264",
           "-preset", "ultrafast", str(broll))                           # no sound at all
    seg = {"frames": 60, "fade_in": 0, "fade_out": 0, "clip_gain": 1.0}
    render.render_segment(seg, str(broll), 1.0, tmp_path / "v.mp4", tmp_path / "a.wav",
                          {"width": 320, "height": 180, "fps": 30}, {"preset": "ultrafast"},
                          asrc=str(speaker), aoffset=2.0)
    a = music.decode(tmp_path / "a.wav")
    assert len(a) == 2 * 48000 and float(np.sqrt((a ** 2).mean())) > 0.05   # the speaker is heard
    assert int(probe(tmp_path / "v.mp4")["streams"][0]["nb_frames"]) == 60


def test_teaser_shot_flashes_in(tmp_path):
    src = tmp_path / "s.mp4"
    ffmpeg("-f", "lavfi", "-i", "color=c=black:s=320x180:r=30:d=3", "-f", "lavfi", "-i", "sine=duration=3",
           "-c:v", "libx264", "-preset", "ultrafast", "-shortest", str(src))
    seg = {"frames": 45, "fade_in": 0, "fade_out": 0, "clip_gain": 1.0, "fx": "punch", "zoom": True}
    out = tmp_path / "v.mp4"
    render.render_segment(seg, str(src), 0.0, out, tmp_path / "a.wav",
                          {"width": 320, "height": 180, "fps": 30}, {"preset": "ultrafast"})

    def luma(t):
        raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(t), "-i", str(out), "-frames:v", "1",
                              "-vf", "format=gray", "-f", "rawvideo", "-"], capture_output=True).stdout
        return np.frombuffer(raw, dtype=np.uint8).mean()
    assert luma(0.0) > 150 and luma(1.0) < 40                        # white flash, then the shot


def test_shot_availability_needs_picture_and_sound(tmp_path):
    files = {"pic": [{"start": 0, "end": 10, "file": __file__}]}
    seg = {"type": "clip", "video_id": "pic", "src_start": 2.0, "dur": 3.0,
           "audio_src": {"video_id": "spk", "src_start": 5.0}}
    assert not render.shot_available(files, seg)
    files["spk"] = [{"start": 4, "end": 9, "file": __file__}]
    assert render.shot_available(files, seg)
    plan = render.plan_downloads([seg])
    assert {r["video_id"] for r in plan} == {"pic", "spk"}           # the sound is downloaded too


# ------------------------------------------------------------------ voice
def test_each_sentence_gets_its_own_delivery():
    parts = voice.speech_parts("पुलिस ने साफ़ कह दिया! अब सवाल ये है... कल क्या होगा? सब तैयार हैं।", "+10%")
    assert [p[0] for p in parts] == ["पुलिस ने साफ़ कह दिया!", "अब सवाल ये है...", "कल क्या होगा?",
                                     "सब तैयार हैं।"]
    excl, pause, quest, plain = parts
    assert excl[1] == "+18%" and quest[2] == "+7Hz" and pause[4] > plain[4]


def test_expressive_microsoft_voice_joins_sentences(tmp_path, monkeypatch):
    import edge_tts
    calls = []

    class FakeCommunicate:
        def __init__(self, text, voice_name, rate, pitch, volume):
            calls.append((text, rate, pitch, volume))

        async def save(self, path):
            await asyncio.sleep(0)
            ffmpeg("-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono", "-f", "lavfi", "-i",
                   "sine=frequency=220:duration=1", "-filter_complex",
                   "[0]atrim=0:0.4[s];[s][1]concat=n=2:v=0:a=1", "-c:a", "libmp3lame", path)
    monkeypatch.setattr(edge_tts, "Communicate", FakeCommunicate)
    out = tmp_path / "line.wav"
    voice._edge("पहली बात! दूसरी बात?", out, {"edge_voice": "hi-IN-MadhurNeural", "edge_rate": "+10%",
                                              "edge_pitch": "+0Hz"})
    assert [c[0] for c in calls] == ["पहली बात!", "दूसरी बात?"]
    # leading silence of each sentence is trimmed: about 1 s + pause + 1 s + pause
    assert 2.0 < media_duration(out) < 2.8
    assert not list(tmp_path.glob("*.part*"))


def test_emotional_voice_states(tmp_path, monkeypatch):
    monkeypatch.setattr(voice, "PARLER_PY", tmp_path / "nope")
    assert voice.parler_state() == "missing"
    py = tmp_path / "python"
    py.write_text("")
    monkeypatch.setattr(voice, "PARLER_PY", py)
    monkeypatch.setattr(voice, "PARLER_READY", tmp_path / "READY")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    assert voice.parler_state() == "locked"
    assert voice.engine_order({"tts_engine": "parler"})[0] != "parler"   # locked: not tried
    (tmp_path / "READY").touch()
    assert voice.parler_state() == "ready" and voice.engine_order({"tts_engine": "auto"})[0] == "parler"
    s = {"parler_speaker": "Rohit"}
    assert "angry" in voice.parler_description(s, "ये बर्दाश्त नहीं होगा!")
    assert "questioning" in voice.parler_description(s, "कल क्या होगा?")
