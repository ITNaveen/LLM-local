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


def _ink_columns(png):
    import subprocess
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(png), "-vf", "format=gray",
                          "-f", "rawvideo", "-"], capture_output=True, check=True).stdout
    img = np.frombuffer(raw, dtype=np.uint8).reshape(120, 400)
    cols = np.where(img.max(axis=0) > 128)[0]
    return img[:, cols.min():cols.min() + 12] > 128


def test_hindi_matra_is_shaped_correctly(tmp_path):
    """'वि' must start with the ि sign (drawn left of व). With broken (simple) shaping it
    starts with व exactly like 'व' alone. This is what made विराट show as वरिट."""
    imgs = {}
    for name, text in (("va", "व"), ("vi", "वि")):
        ass = tmp_path / f"{name}.ass"
        ass.write_text(render.ass_header(400, 120, 70, 0) +
                       f"Dialogue: 0,0:00:00.00,0:00:05.00,Default,,0,0,0,,{{\\an7\\pos(20,10)}}{text}\n",
                       encoding="utf-8")
        png = tmp_path / f"{name}.png"
        ffmpeg("-f", "lavfi", "-i", "color=c=black:s=400x120", "-vf", render.ass_filter(ass),
               "-frames:v", "1", str(png))
        imgs[name] = _ink_columns(png)
    assert (imgs["va"] != imgs["vi"]).mean() > 0.05


def test_thumbnail_text_keeps_every_word():
    assert render.thumbnail_text("विराट का जवाब") == "विराट का जवाब"
    two = render.thumbnail_text("विराट कोहली का सबसे बड़ा जवाब")
    assert two.replace("\\N", " ") == "विराट कोहली का सबसे बड़ा जवाब" and "\\N" in two
