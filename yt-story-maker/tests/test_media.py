"""Tests that run real ffmpeg: cut accuracy, framing, music beats, style learning."""

import numpy as np
import pytest

from storymaker import music, render, style
from storymaker.util import ffmpeg, media_duration, probe


def test_plan_downloads_merges_close_ranges():
    segs = [{"type": "clip", "video_id": "a", "src_start": 10, "dur": 3},
            {"type": "clip", "video_id": "a", "src_start": 16, "dur": 2},
            {"type": "clip", "video_id": "a", "src_start": 100, "dur": 2},
            {"type": "card"}]
    plan = render.plan_downloads(segs)
    assert plan == [{"video_id": "a", "start": 9.0, "end": 19.0},
                    {"video_id": "a", "start": 99.0, "end": 103.0}]
    files = {"a": [{**p, "file": f"f{i}"} for i, p in enumerate(plan)]}
    assert render.locate(files, segs[1]) == ("f0", 7.0)
    assert render.locate(files, segs[2]) == ("f1", 1.0)


def _yavg(path, t=0.0, centre=False):
    import subprocess
    crop = "crop=iw/4:ih/2:iw*3/8:ih/4," if centre else ""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(t), "-i", str(path), "-frames:v", "1",
                          "-vf", f"{crop}scale=8:8,format=gray", "-f", "rawvideo", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.uint8).mean()


@pytest.mark.parametrize("w,h", [(640, 360), (360, 640)])
def test_render_segment_cuts_exact_second_and_frames(tmp_path, w, h):
    # Source whose brightness encodes time: luma = 20 * t (+16 offset)
    src = tmp_path / "ramp.mp4"
    ffmpeg("-f", "lavfi", "-i", f"color=c=black:s={w}x{h}:r=30:d=10", "-f", "lavfi",
           "-i", "sine=frequency=440:duration=10", "-vf", "geq=lum='16+20*T':cb=128:cr=128",
           "-c:v", "libx264", "-preset", "ultrafast", "-qp", "0", "-c:a", "aac", "-shortest", str(src))
    tl = {"width": 320, "height": 180, "fps": 30}
    seg = {"frames": 60, "fade_in": 0, "fade_out": 0, "clip_gain": 1.0}
    out_v, out_a = tmp_path / "v.mp4", tmp_path / "a.wav"
    render.render_segment(seg, str(src), 4.0, out_v, out_a, tl, {"preset": "ultrafast"})
    info = probe(out_v)["streams"][0]
    assert (info["width"], info["height"]) == (320, 180)
    assert int(info["nb_frames"]) == 60
    assert media_duration(out_a) == pytest.approx(2.0, abs=0.001)
    expected = 16 + 20 * 4.0
    if w > h:  # 16:9 fills the frame: brightness should match t=4s (eq adds a little contrast)
        assert abs(_yavg(out_v) - expected) < 10
    else:      # vertical: sharp copy in the centre, darker blurred fill at the sides
        assert abs(_yavg(out_v, centre=True) - expected) < 10
        assert _yavg(out_v) < _yavg(out_v, centre=True)


def test_render_segment_without_audio_stream(tmp_path):
    src = tmp_path / "silent.mp4"
    ffmpeg("-f", "lavfi", "-i", "testsrc2=s=640x360:r=25:d=3", "-c:v", "libx264",
           "-preset", "ultrafast", str(src))
    seg = {"frames": 45, "fade_in": 0.3, "fade_out": 0.3, "clip_gain": 0.5}
    render.render_segment(seg, str(src), 0.5, tmp_path / "v.mp4", tmp_path / "a.wav",
                          {"width": 320, "height": 180, "fps": 30}, {"preset": "ultrafast"})
    assert media_duration(tmp_path / "a.wav") == pytest.approx(1.5, abs=0.001)


@pytest.mark.parametrize("bpm", [96, 120, 140])
def test_beat_grid_stays_locked_for_three_minutes(bpm):
    sr, secs, gap = 11025, 180, 60 / bpm
    x = np.zeros(sr * secs, dtype=np.float32)
    clicks = [0.1 + i * gap for i in range(int((secs - 0.2) / gap))]
    for c in clicks:
        s = int(c * sr)
        x[s:s + 200] = np.hanning(200)
    beats = music.beats_from_signal(x, sr)
    assert np.median(np.diff(beats)) == pytest.approx(gap, abs=0.01)
    # every detected beat sits on a real click, including the last minute (no drift)
    late = [b for b in beats if b > 120]
    assert late and all(min(abs(b - c) for c in clicks) < 0.03 for b in late)


def test_music_library_and_mood_fallback(tmp_path):
    (tmp_path / "epic").mkdir()
    (tmp_path / "epic" / "a.mp3").write_bytes(b"x")
    (tmp_path / "calm_piano.wav").write_bytes(b"x")
    lib = music.scan_library(tmp_path)
    assert lib["epic"] and lib["calm"]
    tracks = music.choose_tracks([{"key": "climax", "mood": "triumphant"},
                                  {"key": "ending", "mood": "emotional"},
                                  {"key": "opening", "mood": "dark"}], tmp_path)
    assert tracks["climax"].endswith("a.mp3")     # triumphant -> epic
    assert tracks["ending"].endswith("calm_piano.wav")  # emotional -> calm


def test_envelope_ducks_smoothly():
    env = music.envelope(48000 * 10, [(0, 5, 0.9), (5, 10, 0.1)])
    assert env[48000] == pytest.approx(0.9, abs=1e-3)
    assert env[48000 * 8] == pytest.approx(0.1, abs=1e-3)
    assert np.max(np.abs(np.diff(env))) < 0.01   # no clicks


def test_render_act_bed_exact_length_and_narration(tmp_path):
    voice = tmp_path / "n.wav"
    ffmpeg("-f", "lavfi", "-i", "sine=frequency=300:sample_rate=48000:duration=2", "-ac", "1", str(voice))
    out = tmp_path / "bed.wav"
    music.render_act_bed(out, 7.5, None, "epic", [(0, 3, 0.9), (3, 7.5, 0.2)],
                         [{"file": str(voice), "t": 4.0}])
    data = music.decode(out)
    assert len(data) == int(7.5 * 48000)
    rms = lambda a, b: float(np.sqrt((data[int(a * 48000):int(b * 48000)] ** 2).mean()))
    assert rms(4.2, 5.8) > rms(6.2, 7.0) * 2     # narration clearly on top


def test_style_learning_detects_cuts(tmp_path):
    cuts = [1.0, 2.5, 3.0, 5.0, 6.2]
    path = tmp_path / "ref.mp4"
    style.make_test_video(path, cuts, 8.0)
    found = style.detect_cuts(str(path))
    assert len(found) == len(cuts)
    assert all(abs(a - b) < 0.1 for a, b in zip(found, cuts))
    st = style.analyze_reference(str(path), "unit-test-style",
                                 captions=[{"start": 0, "end": 2, "text": "x"}], title="ref")
    assert st["stats"]["shots"] == 6
    assert st["dialogue_ratio"] == pytest.approx(0.25, abs=0.01)
    assert set(st["acts"]) == set(style.ACT_KEYS)
    assert style.get_style("unit-test-style")["name"] == "unit-test-style"


def test_card_and_thumbnail_render_hindi(tmp_path):
    seg = {"i": 0, "text": "विराट का जवाब", "frames": 30, "fade_in": 0.3, "fade_out": 0.3}
    tl = {"width": 320, "height": 180, "fps": 30}
    render.render_card(seg, None, tmp_path / "c.mp4", tmp_path / "c.wav", tl,
                       {"preset": "ultrafast"}, tmp_path)
    assert int(probe(tmp_path / "c.mp4")["streams"][0]["nb_frames"]) == 30
    png = tmp_path / "f.png"
    ffmpeg("-f", "lavfi", "-i", "testsrc2=s=640x360", "-frames:v", "1", str(png))
    render.make_thumbnail(png, "विराट कोहली का सबसे बड़ा जवाब", tmp_path / "t.jpg", tmp_path)
    s = probe(tmp_path / "t.jpg")["streams"][0]
    assert (s["width"], s["height"]) == (1280, 720)


def _ink_columns(img):
    a = np.asarray(img.convert("L"))
    cols = np.where(a.max(axis=0) > 128)[0]
    return a[:, cols.min():cols.min() + 12] > 128


def test_hindi_matra_is_shaped_correctly():
    """'वि' must start with the ि sign (drawn left of व). With broken (simple) shaping it
    starts with व exactly like 'व' alone. This is what made विराट show as वरिट."""
    from PIL import Image, ImageDraw

    from storymaker import textimg
    assert textimg.available()
    imgs = {}
    for name, text in (("va", "व"), ("vi", "वि")):
        img = Image.new("L", (400, 120), 0)
        textimg.draw_line(ImageDraw.Draw(img), 20, 90, text, 70, 255)
        imgs[name] = _ink_columns(img)
    assert (imgs["va"] != imgs["vi"]).mean() > 0.05


def test_english_words_inside_hindi_use_the_latin_font():
    from storymaker import textimg
    assert textimg.runs("पुलिस vs CJP: 23000 जवान") == [
        ("पुलिस ", False), ("vs", True), (" ", False), ("CJP", True), (": 23000 जवान", False)]
    # every letter has a real glyph (the Hindi font alone draws empty boxes for Latin)
    assert textimg.line_width("CJP", 60) > textimg.line_width("।", 60) * 2


def test_thumbnail_text_keeps_every_word():
    assert render.thumbnail_text("विराट का जवाब") == "विराट का जवाब"
    two = render.thumbnail_text("विराट कोहली का सबसे बड़ा जवाब")
    assert two.replace("\\N", " ") == "विराट कोहली का सबसे बड़ा जवाब" and "\\N" in two


def test_locate_prefers_full_cover_then_start_cover():
    files = {"a": [{"start": 10.0, "end": 20.0, "file": "f1"}]}
    inside = {"video_id": "a", "src_start": 12.0, "dur": 5.0}
    overflow = {"video_id": "a", "src_start": 15.0, "dur": 9.0}      # ends 4 s past the file
    outside = {"video_id": "a", "src_start": 30.0, "dur": 2.0}
    assert render.locate(files, inside) == ("f1", 2.0)
    assert render.locate(files, overflow) == ("f1", 5.0)
    assert render.locate(files, overflow, strict=True) == (None, 0.0)
    assert render.locate(files, outside) == (None, 0.0)
    render.merge_files(files, {"a": [{"start": 14.0, "end": 26.0, "file": "f2"}]})
    assert render.locate(files, overflow, strict=True) == ("f2", 1.0)


def test_download_all_accepts_whole_video_fallback(tmp_path):
    full = tmp_path / "full.mp4"
    ffmpeg("-f", "lavfi", "-i", "testsrc2=s=160x90:r=10:d=2", "-c:v", "libx264", str(full))

    class WholeVideo:
        def download_section(self, video_id, start, end, out_base):
            return {"file": str(full), "start": 0.0, "end": 300.0}
    segs = [{"type": "clip", "video_id": "v", "src_start": 100.0, "dur": 4.0}]
    files, failed = render.download_all(WholeVideo(), segs, tmp_path, lambda m: None)
    assert not failed
    assert render.locate(files, segs[0], strict=True) == (str(full), 100.0)


def test_render_holds_last_frame_when_source_is_short(tmp_path):
    src = tmp_path / "short.mp4"
    ffmpeg("-f", "lavfi", "-i", "testsrc2=s=640x360:r=30:d=2", "-f", "lavfi", "-i",
           "sine=duration=2", "-c:v", "libx264", "-preset", "ultrafast", "-shortest", str(src))
    seg = {"frames": 240, "fade_in": 0, "fade_out": 0, "clip_gain": 1.0}   # 8 s from a 2 s file
    render.render_segment(seg, str(src), 0.5, tmp_path / "v.mp4", tmp_path / "a.wav",
                          {"width": 320, "height": 180, "fps": 30}, {"preset": "ultrafast"})
    assert int(probe(tmp_path / "v.mp4")["streams"][0]["nb_frames"]) == 240


def test_same_footage_from_two_channels_is_detected(tmp_path):
    a, b, c = tmp_path / "a.mp4", tmp_path / "b.mp4", tmp_path / "c.mp4"
    for path, src in ((a, "testsrc2"), (b, "testsrc2"), (c, "testsrc2=s=320x180:r=25:d=6,hflip,vflip,rotate=1.2")):
        lavfi = src if "=" in src else f"{src}=s=320x180:r=25:d=6"
        ffmpeg("-f", "lavfi", "-i", lavfi, "-c:v", "libx264", "-preset", "ultrafast", str(path))
    files = {"x": [{"start": 0, "end": 6, "file": str(a)}],
             "y": [{"start": 0, "end": 6, "file": str(b)}],     # the same clip, re-aired
             "z": [{"start": 0, "end": 6, "file": str(c)}]}
    seg = lambda vid, mode: {"type": "clip", "video_id": vid, "src_start": 1.0, "dur": 2.0,
                             "mode": mode, "act": "climax", "beat": 0}
    segs = [seg("x", "music"), seg("y", "music"), seg("z", "music")]
    assert render.visual_duplicates(segs, files) == {("y", 1.0)}
    # a person speaking is never dropped as a duplicate
    assert render.visual_duplicates([seg("x", "music"), seg("y", "original")], files) == set()


def test_text_overlay_on_moving_footage(tmp_path):
    src = tmp_path / "s.mp4"
    ffmpeg("-f", "lavfi", "-i", "color=c=black:s=640x360:r=30:d=4", "-f", "lavfi", "-i",
           "sine=duration=4", "-vf", "geq=lum='16+50*T':cb=128:cr=128", "-c:v", "libx264",
           "-preset", "ultrafast", "-shortest", str(src))
    seg = {"frames": 90, "fade_in": 0, "fade_out": 0, "clip_gain": 0.3, "overlay": "सच्चाई सामने आई"}
    out = tmp_path / "v.mp4"
    render.render_segment(seg, str(src), 0.0, out, tmp_path / "a.wav",
                          {"width": 320, "height": 180, "fps": 30}, {"preset": "ultrafast"})
    assert int(probe(out)["streams"][0]["nb_frames"]) == 90
    # footage keeps moving under the text (frames differ), unlike the old frozen cards
    assert _yavg(out, 2.5) - _yavg(out, 0.2) > 30


def test_duplicate_check_never_empties_a_narration_line(tmp_path):
    a = tmp_path / "a.mp4"
    ffmpeg("-f", "lavfi", "-i", "testsrc2=s=320x180:r=25:d=6", "-c:v", "libx264", "-preset",
           "ultrafast", str(a))
    files = {"x": [{"start": 0, "end": 6, "file": str(a)}], "y": [{"start": 0, "end": 6, "file": str(a)}]}
    segs = [{"type": "clip", "video_id": "x", "src_start": 1.0, "dur": 2.0, "mode": "music", "act": "a", "beat": 0},
            {"type": "clip", "video_id": "y", "src_start": 1.0, "dur": 2.0, "mode": "narration", "act": "a", "beat": 1}]
    assert render.visual_duplicates(segs, files) == set()      # it's the only picture under the line


def _audio_only(path, seconds=4):
    ffmpeg("-f", "lavfi", "-i", f"sine=frequency=500:duration={seconds}", "-c:a", "libopus", str(path))


def test_audio_only_download_is_rejected(tmp_path):
    from storymaker.util import has_video
    audio = tmp_path / "x.f251.webm"            # what a failed merge leaves behind
    _audio_only(audio)
    assert not has_video(audio)

    class AudioOnly:
        def download_section(self, video_id, start, end, out_base):
            return str(audio)
    segs = [{"type": "clip", "video_id": "v", "src_start": 1.0, "dur": 2.0}]
    files, failed = render.download_all(AudioOnly(), segs, tmp_path, lambda m: None)
    assert files == {} and "no picture" in failed[0]


def test_render_never_crashes_on_a_file_without_picture(tmp_path):
    audio = tmp_path / "a.webm"
    _audio_only(audio)
    seg = {"frames": 60, "fade_in": 0, "fade_out": 0, "clip_gain": 0.75}
    render.render_segment(seg, str(audio), 0.5, tmp_path / "v.mp4", tmp_path / "a.wav",
                          {"width": 320, "height": 180, "fps": 30}, {"preset": "ultrafast"})
    assert int(probe(tmp_path / "v.mp4")["streams"][0]["nb_frames"]) == 60
    assert media_duration(tmp_path / "a.wav") == pytest.approx(2.0, abs=0.001)
