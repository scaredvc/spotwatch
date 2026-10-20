from datetime import datetime

import cv2
import numpy as np
import pytest

from spotwatch.config import parse_config
from spotwatch.sources import ImageFolderSource, VideoFileSource, make_source

NOW = datetime(2026, 10, 12, 12, 0)


def write_image(path, value):
    cv2.imwrite(str(path), np.full((20, 30, 3), value, dtype=np.uint8))


def test_image_folder_plays_in_name_order(tmp_path):
    write_image(tmp_path / "002.png", 200)
    write_image(tmp_path / "001.png", 100)
    (tmp_path / "notes.txt").write_text("ignore me")

    source = ImageFolderSource(tmp_path)
    first = source.read(NOW)
    second = source.read(NOW)
    assert first.good and second.good
    assert first.image[0, 0, 0] == 100
    assert second.image[0, 0, 0] == 200
    assert not source.finished

    end = source.read(NOW)
    assert not end.good
    assert source.finished


def test_image_folder_unreadable_file_is_bad_frame(tmp_path):
    (tmp_path / "001.jpg").write_bytes(b"not an image")
    frame = ImageFolderSource(tmp_path).read(NOW)
    assert not frame.good
    assert "could not read" in frame.reason


def test_image_folder_missing():
    with pytest.raises(FileNotFoundError):
        ImageFolderSource("/no/such/folder")


def test_video_file_steps_through(tmp_path):
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (32, 24))
    if not writer.isOpened():
        pytest.skip("no video codec available")
    for i in range(30):  # 3 seconds at 10 fps
        writer.write(np.full((24, 32, 3), i * 8, dtype=np.uint8))
    writer.release()

    source = VideoFileSource(path, step_seconds=1)
    frames = []
    while not source.finished:
        frames.append(source.read(NOW))
    good = [f for f in frames if f.good]
    assert len(good) == 3
    # later frames are brighter, so we really moved forward through the video
    assert good[0].image.mean() < good[1].image.mean() < good[2].image.mean()


def test_make_source_builds_blink_source(tmp_path):
    from spotwatch.blink_source import BlinkSnapshotSource

    config = parse_config(
        {
            "settings": {"blink_session_file": str(tmp_path / "s.json")},
            "spots": [{"id": "s1"}],
            "cameras": [
                {"name": "a", "source": "blink", "blink_camera": "Front", "polygons": {"s1": [[0, 0], [1, 0], [1, 1]]}},
                {"name": "b", "source": "blink", "blink_camera": "Side", "polygons": {"s1": [[0, 0], [1, 0], [1, 1]]}},
            ],
        }
    )
    a = make_source(config.cameras[0], config)
    b = make_source(config.cameras[1], config)
    assert isinstance(a, BlinkSnapshotSource)
    assert a.account is b.account  # one login shared by both cameras


def test_blink_without_session_file_gives_bad_frame_not_crash(tmp_path):
    config = parse_config(
        {
            "settings": {"blink_session_file": str(tmp_path / "missing.json")},
            "spots": [{"id": "s1"}],
            "cameras": [{"name": "a", "source": "blink", "blink_camera": "Front", "polygons": {"s1": [[0, 0], [1, 0], [1, 1]]}}],
        }
    )
    frame = make_source(config.cameras[0], config).read(NOW)
    assert not frame.good
    assert "blink_setup" in frame.reason
