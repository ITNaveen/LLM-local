"""Downloads: yt-dlp's temporary files are never used as footage, the whole-video fallback runs
once per video even when several sections need it, and a resumed render never reuses shots
from an earlier cut."""

import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from storymaker import render
from storymaker.source import YouTubeSource, finished_download
from storymaker.util import ffmpeg, write_json


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    path = tmp_path_factory.mktemp("clip") / "clip.mp4"
    ffmpeg("-f", "lavfi", "-i", "testsrc2=s=160x90:r=10:d=3", "-f", "lavfi", "-i", "sine=duration=3",
           "-c:v", "libx264", "-preset", "ultrafast", "-shortest", str(path))
    return path


def test_finished_download_ignores_ytdlp_temporary_files(tmp_path, clip):
    base = tmp_path / "RzI1ODN3atM_full"
    for name in (".f398.mp4", ".temp.mp4", ".f251.webm", ".mp4.part"):
        shutil.copy(clip, f"{base}{name}")
    assert finished_download(base) == []
    shutil.copy(clip, f"{base}.mp4")
    assert finished_download(base) == [f"{base}.mp4"]


def test_whole_video_fallback_runs_once_and_returns_the_merged_file(tmp_path, clip, settings, monkeypatch):
    """The real crash: three sections of one video fell back to the whole video at the same
    time; one picked up 'RzI1ODN3atM_full.f398.mp4', which yt-dlp deleted after merging."""
    import yt_dlp
    vid, guard, calls = "RzI1ODN3atM", threading.Lock(), {"full": 0}

    class FakeYDL:
        def __init__(self, opts):
            self.opts = opts

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def download(self, urls):
            base = self.opts["outtmpl"]["default"].replace(".%(ext)s", "")
            if "download_ranges" in self.opts:
                raise RuntimeError("ERROR: ffmpeg exited with code 8")
            with guard:
                calls["full"] += 1
            # like yt-dlp: the picture-only part lands first, then the merge replaces it
            shutil.copy(clip, base + ".f398.mp4")
            time.sleep(0.4)
            shutil.copy(clip, base + ".temp.mp4")
            time.sleep(0.2)
            os.replace(base + ".temp.mp4", base + ".mp4")
            os.remove(base + ".f398.mp4")

    monkeypatch.setattr(yt_dlp, "YoutubeDL", FakeYDL)
    src = YouTubeSource(settings)
    write_json(src.details_dir / f"{vid}.json", {"duration": 300})
    ranges = [(10, 20), (60, 75), (120, 140)]
    with ThreadPoolExecutor(3) as pool:
        got = list(pool.map(lambda r: src.download_section(vid, r[0], r[1], tmp_path / f"{vid}_{r[0]}"),
                            ranges))
    assert calls["full"] == 1
    for g in got:
        assert isinstance(g, dict) and g["file"] == str(tmp_path / f"{vid}_full.mp4")
        assert os.path.exists(g["file"]) and g["start"] == 0.0 and g["end"] > 2


def test_files_that_disappeared_are_dropped(tmp_path, clip):
    real = tmp_path / "real.mp4"
    shutil.copy(clip, real)
    files = {"a": [{"start": 0, "end": 3, "file": str(real)},
                   {"start": 5, "end": 9, "file": str(tmp_path / "x_full.f398.mp4")}],
             "b": [{"start": 0, "end": 3, "file": str(tmp_path / "gone.mp4")}]}
    assert render.drop_missing(files) == 2
    assert files == {"a": [{"start": 0, "end": 3, "file": str(real)}]}
    seg = {"video_id": "b", "src_start": 1.0, "dur": 1.0}
    assert render.locate(files, seg) == (None, 0.0)


def test_shot_cache_is_tied_to_the_cut():
    tl = {"width": 320, "height": 180, "fps": 30}
    seg = {"i": 7, "type": "clip", "video_id": "a", "src_start": 12.0, "frames": 90}
    assert render.shot_name(seg, tl) == render.shot_name(dict(seg), tl)
    assert render.shot_name(seg, tl).startswith("0007_")
    assert render.shot_name({**seg, "src_start": 40.0}, tl) != render.shot_name(seg, tl)
    assert render.shot_name(seg, {**tl, "width": 1920}) != render.shot_name(seg, tl)
