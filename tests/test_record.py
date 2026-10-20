from datetime import datetime, timedelta

import cv2
import numpy as np

from spotwatch.config import parse_config
from spotwatch.record import record
from spotwatch.sources import Frame, FrameSource, ImageFolderSource


class FakeClock:
    def __init__(self, start):
        self.current = start

    def now(self):
        return self.current

    def sleep(self, seconds):
        self.current += timedelta(seconds=seconds)


def config_for(tmp_path, **camera_extra):
    cam = {"name": "cam", "source": "image_folder", "path": str(tmp_path / "in"), "interval_seconds": 60}
    cam.update(camera_extra)
    return parse_config({"spots": [{"id": "s1"}], "cameras": [cam]})


def make_images(folder, n):
    folder.mkdir()
    for i in range(n):
        cv2.imwrite(str(folder / f"{i:03}.png"), np.full((10, 10, 3), (i * 20) % 256, dtype=np.uint8))


def test_count_stops_after_n_frames(tmp_path):
    make_images(tmp_path / "in", 5)
    config = config_for(tmp_path)
    clock = FakeClock(datetime(2026, 10, 13, 9, 0))
    sources = {"cam": ImageFolderSource(tmp_path / "in")}
    saved = record(config, sources, tmp_path / "out", count=2, clock=clock.now, sleep=clock.sleep)
    assert saved == {"cam": 2}
    files = sorted((tmp_path / "out" / "cam").iterdir())
    assert [f.name for f in files] == ["20261013-090000.jpg", "20261013-090100.jpg"]


def test_hours_limit_and_interval(tmp_path):
    make_images(tmp_path / "in", 50)
    config = config_for(tmp_path)
    clock = FakeClock(datetime(2026, 10, 13, 9, 0))
    sources = {"cam": ImageFolderSource(tmp_path / "in")}
    saved = record(config, sources, tmp_path / "out", every=300, hours=0.5, clock=clock.now, sleep=clock.sleep)
    assert saved == {"cam": 6}  # 9:00, 9:05 ... 9:25


class FlakySource(FrameSource):
    def __init__(self):
        self.reads = 0

    def read(self, now):
        self.reads += 1
        if self.reads % 2 == 0:
            return Frame(image=None, time=now, good=False, reason="stale snapshot")
        return Frame(image=np.zeros((5, 5, 3), dtype=np.uint8), time=now)


def test_bad_frames_are_skipped(tmp_path, capsys):
    (tmp_path / "in").mkdir()
    config = config_for(tmp_path)
    clock = FakeClock(datetime(2026, 10, 13, 9, 0))
    saved = record(config, {"cam": FlakySource()}, tmp_path / "out", count=3, clock=clock.now, sleep=clock.sleep)
    assert saved == {"cam": 3}
    assert "skipped (stale snapshot)" in capsys.readouterr().out


def test_respects_active_hours_unless_all_hours(tmp_path):
    (tmp_path / "in").mkdir()
    config = config_for(tmp_path, active_hours={"start": "07:00", "end": "20:00"})
    night = datetime(2026, 10, 13, 22, 0)

    clock = FakeClock(night)
    saved = record(config, {"cam": FlakySource()}, tmp_path / "a", hours=0.1, clock=clock.now, sleep=clock.sleep)
    assert saved == {"cam": 0}

    clock = FakeClock(night)
    saved = record(
        config, {"cam": FlakySource()}, tmp_path / "b", count=1, all_hours=True, clock=clock.now, sleep=clock.sleep
    )
    assert saved == {"cam": 1}
