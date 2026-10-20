"""Frame sources: where images come from.

Every source has the same small interface, so the runner doesn't care whether
a frame came from a folder, a video file, a Blink snapshot, or an RTSP stream.
RTSP is not built yet.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass
class Frame:
    image: np.ndarray | None
    time: datetime  # when we asked for it
    good: bool = True
    reason: str = ""
    captured_at: datetime | None = None  # when the camera took it, if the camera says


class FrameSource:
    """Base class. read() is called once per poll and returns one Frame."""

    finished = False  # True when a finite source (folder, video) has run out

    def read(self, now: datetime) -> Frame:
        raise NotImplementedError

    def power_info(self) -> dict | None:
        """Battery or power details, if the camera reports any."""
        return None

    def close(self) -> None:
        pass


def bad_frame(now: datetime, reason: str) -> Frame:
    return Frame(image=None, time=now, good=False, reason=reason)


class ImageFolderSource(FrameSource):
    """Plays a folder of images in name order, one per read."""

    def __init__(self, folder: str | Path):
        self.folder = Path(folder)
        if not self.folder.is_dir():
            raise FileNotFoundError(f"image folder not found: {self.folder}")
        self.files = sorted(p for p in self.folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        self.index = 0

    def read(self, now: datetime) -> Frame:
        if self.index >= len(self.files):
            self.finished = True
            return bad_frame(now, "no more images")
        path = self.files[self.index]
        self.index += 1
        image = cv2.imread(str(path))
        if image is None:
            return bad_frame(now, f"could not read {path.name}")
        return Frame(image=image, time=now)


class VideoFileSource(FrameSource):
    """Plays a video file, jumping ahead step_seconds of video on each read.

    With step_seconds equal to the camera interval, an hour of recorded video
    behaves like an hour of polling.
    """

    def __init__(self, path: str | Path, step_seconds: float = 60):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"video not found: {self.path}")
        self.capture = cv2.VideoCapture(str(self.path))
        if not self.capture.isOpened():
            raise ValueError(f"could not open video: {self.path}")
        self.step_ms = step_seconds * 1000
        self.position_ms = 0.0

    def read(self, now: datetime) -> Frame:
        if self.finished:
            return bad_frame(now, "end of video")
        self.capture.set(cv2.CAP_PROP_POS_MSEC, self.position_ms)
        ok, image = self.capture.read()
        self.position_ms += self.step_ms
        if not ok or image is None:
            self.finished = True
            return bad_frame(now, "end of video")
        return Frame(image=image, time=now)

    def close(self) -> None:
        self.capture.release()


def make_source(camera, config=None) -> FrameSource:
    """Build a FrameSource from a config.Camera."""
    opts = camera.options
    if camera.source == "image_folder":
        return ImageFolderSource(opts["path"])
    if camera.source == "video_file":
        return VideoFileSource(opts["path"], step_seconds=opts.get("step_seconds", camera.interval_seconds))
    if camera.source == "blink":
        # imported here so folder/video runs don't need blinkpy installed
        from spotwatch.blink_source import BlinkSnapshotSource, get_account

        session_file = config.blink_session_file if config else "blink_session.json"
        account = get_account(session_file, refresh_rate=camera.interval_seconds)
        return BlinkSnapshotSource(opts["blink_camera"], account)
    if camera.source == "rtsp":
        raise NotImplementedError("the rtsp source is not built yet; use blink, image_folder or video_file")
    raise ValueError(f"unknown source type: {camera.source}")
