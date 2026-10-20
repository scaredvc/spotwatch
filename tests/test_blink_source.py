import asyncio
import json
import os
import stat
from datetime import datetime, timedelta

import cv2
import numpy as np
import pytest

from spotwatch.blink_source import (
    BlinkAuthError,
    BlinkSnapshotSource,
    backoff_seconds,
    check_fresh,
    image_hash,
    load_session,
    save_session,
    thumbnail_time,
)

NOW = datetime(2026, 10, 13, 9, 0, 0)


@pytest.fixture(autouse=True)
def no_real_waits(monkeypatch):
    """take_snapshot waits a few seconds between refreshes; tests don't need to."""
    import spotwatch.blink_source as bs

    async def no_sleep(seconds):
        pass

    monkeypatch.setattr(bs.asyncio, "sleep", no_sleep)


def jpeg(value: int) -> bytes:
    ok, buf = cv2.imencode(".jpg", np.full((24, 32, 3), value, dtype=np.uint8))
    return buf.tobytes()


# ---------- pure helpers ----------


def test_check_fresh():
    img = jpeg(100)
    assert check_fresh("u?ts=1", "u?ts=2", img, None) == (True, "")
    assert check_fresh("u?ts=1", "u?ts=1", img, None)[0] is False
    assert check_fresh("u?ts=1", "u?ts=2", img, image_hash(img))[0] is False
    assert check_fresh("u?ts=1", "u?ts=2", None, None)[0] is False
    assert check_fresh("u?ts=1", None, img, None)[0] is False


def test_thumbnail_time():
    url = "https://rest-u011.immedia-semi.com/api/v3/media/accounts/1/networks/2/catalina/3/thumbnail/thumbnail.jpg?ts=1760371200&ext="
    assert thumbnail_time(url) == datetime.fromtimestamp(1760371200)
    assert thumbnail_time("https://x/old_style_thumb.jpg") is None
    assert thumbnail_time(None) is None
    assert thumbnail_time("https://x/t.jpg?ts=notanumber") is None


def test_backoff_doubles_and_caps():
    assert [backoff_seconds(n) for n in range(0, 7)] == [0, 60, 120, 240, 480, 900, 900]


# ---------- session file ----------


class FakeAuthForSave:
    login_attributes = {
        "username": "me@example.com",
        "password": "hunter2",
        "refresh_token": "rt",
        "token": "t",
        "hardware_id": "ABC",
    }


class FakeBlinkForSave:
    auth = FakeAuthForSave()


def test_save_session_drops_password_and_is_private(tmp_path):
    path = tmp_path / "blink_session.json"
    save_session(FakeBlinkForSave(), path)
    data = json.loads(path.read_text())
    assert "password" not in data
    assert data["refresh_token"] == "rt"
    if os.name != "nt":  # Windows has no 0o600; chmod there only toggles read-only
        assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    # the original dict inside blinkpy still has it; we only strip the copy
    assert FakeAuthForSave.login_attributes["password"] == "hunter2"


def test_load_session_errors(tmp_path):
    with pytest.raises(BlinkAuthError):
        load_session(tmp_path / "missing.json")
    path = tmp_path / "s.json"
    path.write_text(json.dumps({"username": "me"}))
    with pytest.raises(BlinkAuthError):
        load_session(path)


# ---------- the source, against a fake Blink ----------


class FakeCamera:
    """Behaves like blinkpy 0.25.9's BlinkCamera for what we use."""

    def __init__(self, images):
        self.images = list(images)  # what each new photo looks like
        self.online = True
        self.thumbnail = "https://x/thumb.jpg?ts=1000&ext="
        self._cached = jpeg(0)
        self.pending = None
        self.snaps = 0
        self.attributes = {"battery": "ok", "wifi_strength": 4, "sync_signal_strength": 5, "type": "catalina"}

    async def snap_picture(self):
        self.snaps += 1
        # Real blinkpy re-downloads the OLD thumbnail here.
        self.pending = self.images.pop(0) if self.images else None

    @property
    def image_from_cache(self):
        return self._cached


class FakeBlink:
    def __init__(self, camera, refresh_updates=True, refresh_error=None):
        self.cameras = {"Front Window": camera}
        self.refresh_updates = refresh_updates
        self.refresh_error = refresh_error
        self.ts = 1000

    async def refresh(self, force_cache=False):
        if self.refresh_error:
            raise self.refresh_error
        for cam in self.cameras.values():
            if self.refresh_updates and cam.pending is not None:
                self.ts += 60
                cam.thumbnail = f"https://x/thumb.jpg?ts={self.ts}&ext="
                cam._cached = cam.pending
                cam.pending = None


class FakeAccount:
    def __init__(self, blink=None, login_error=None):
        self.blink = blink
        self.login_error = login_error
        self.loop = asyncio.new_event_loop()
        self.disconnects = 0
        self.saves = 0

    def run(self, coro):
        return self.loop.run_until_complete(coro)

    async def get_blink(self):
        if self.login_error:
            raise self.login_error
        return self.blink

    def save_if_tokens_changed(self):
        self.saves += 1

    def disconnect(self):
        self.disconnects += 1


def test_fresh_snapshot_is_a_good_frame():
    camera = FakeCamera([jpeg(200)])
    source = BlinkSnapshotSource("Front Window", FakeAccount(FakeBlink(camera)))
    frame = source.read(NOW)
    assert frame.good, frame.reason
    assert frame.image.shape == (24, 32, 3)
    assert frame.image.mean() > 150  # the new photo, not the old cached one
    assert frame.captured_at == datetime.fromtimestamp(1060)
    assert source.power_info()["wifi_strength"] == 4


def test_camera_name_lookup_failure_lists_cameras():
    source = BlinkSnapshotSource("Back Yard", FakeAccount(FakeBlink(FakeCamera([jpeg(1)]))))
    frame = source.read(NOW)
    assert not frame.good
    assert "Front Window" in frame.reason


def test_thumbnail_not_updated_is_stale():
    camera = FakeCamera([jpeg(200)])
    source = BlinkSnapshotSource("Front Window", FakeAccount(FakeBlink(camera, refresh_updates=False)))
    frame = source.read(NOW)
    assert not frame.good
    assert "stale" in frame.reason


def test_same_image_twice_is_stale():
    same = jpeg(150)
    camera = FakeCamera([same, same])
    source = BlinkSnapshotSource("Front Window", FakeAccount(FakeBlink(camera)))
    assert source.read(NOW).good
    second = source.read(NOW + timedelta(seconds=60))
    assert not second.good
    assert "same image" in second.reason


def test_offline_camera_is_bad_frame_without_snapping():
    camera = FakeCamera([jpeg(200)])
    camera.online = False
    source = BlinkSnapshotSource("Front Window", FakeAccount(FakeBlink(camera)))
    frame = source.read(NOW)
    assert not frame.good and frame.reason == "camera offline"
    assert camera.snaps == 0


def test_errors_back_off_then_recover():
    camera = FakeCamera([jpeg(200)])
    blink = FakeBlink(camera, refresh_error=ConnectionError("throttled"))
    account = FakeAccount(blink)
    source = BlinkSnapshotSource("Front Window", account)

    first = source.read(NOW)
    assert not first.good and "throttled" in first.reason
    assert account.disconnects == 1

    # two failures in a row -> wait 120s before trying again
    source.read(NOW + timedelta(seconds=60))
    skipped = source.read(NOW + timedelta(seconds=120))
    assert not skipped.good and "backing off" in skipped.reason
    assert camera.snaps == 2  # the skipped read didn't touch Blink

    blink.refresh_error = None
    camera.images = [jpeg(220)]
    ok = source.read(NOW + timedelta(seconds=181))
    assert ok.good, ok.reason
    assert source.failures == 0 and source.retry_at is None


def test_auth_failure_is_bad_frame_not_crash():
    account = FakeAccount(login_error=BlinkAuthError("session expired; run setup"))
    source = BlinkSnapshotSource("Front Window", account)
    frame = source.read(NOW)
    assert not frame.good
    assert "run setup" in frame.reason


def test_runner_keeps_going_when_blink_fails():
    """A failing Blink camera marks its spots unknown and the runner carries on."""
    from spotwatch.config import parse_config
    from spotwatch.detector import ScriptedDetector
    from spotwatch.notifier import ListNotifier
    from spotwatch.runner import Runner
    from spotwatch.storage import Store

    config = parse_config(
        {
            "spots": [{"id": "s1"}],
            "cameras": [
                {
                    "name": "win",
                    "source": "blink",
                    "blink_camera": "Front Window",
                    "polygons": {"s1": [[0, 0], [10, 0], [10, 10]]},
                }
            ],
        }
    )
    blink = FakeBlink(FakeCamera([jpeg(1)]), refresh_error=TimeoutError())
    source = BlinkSnapshotSource("Front Window", FakeAccount(blink))
    runner = Runner(config, {"win": source}, ScriptedDetector([]), ListNotifier(), Store(":memory:"))
    for i in range(5):
        runner.tick(NOW + timedelta(minutes=i))
    assert runner.spot_states()["s1"] == "unknown"


def test_camera_name_match_ignores_spaces_dashes_and_case():
    from spotwatch.blink_source import find_camera

    cameras = {"XT2 - 1234": "a", "XT2 - 5678": "b"}
    assert find_camera(cameras, "XT2 - 1234") == "a"
    assert find_camera(cameras, "XT2-1234") == "a"
    assert find_camera(cameras, "xt2 5678") == "b"
    assert find_camera(cameras, "XT2-9999") is None


def test_ambiguous_loose_match_finds_nothing():
    from spotwatch.blink_source import find_camera

    assert find_camera({"Front-1": "a", "Front 1": "b"}, "front1") is None


def test_source_finds_camera_with_spaced_name():
    camera = FakeCamera([jpeg(200)])
    blink = FakeBlink(camera)
    blink.cameras = {"XT2 - 1234": camera}
    source = BlinkSnapshotSource("XT2-1234", FakeAccount(blink))
    frame = source.read(NOW)
    assert frame.good, frame.reason


class SlowPostBlink(FakeBlink):
    """The new photo only shows up on the Nth refresh after a snap."""

    def __init__(self, camera, show_on_refresh):
        super().__init__(camera)
        self.show_on_refresh = show_on_refresh
        self.refreshes = 0

    async def refresh(self, force_cache=False):
        self.refreshes += 1
        if self.refreshes >= self.show_on_refresh:
            await super().refresh(force_cache)


def test_waits_for_slow_snapshot_to_post():
    camera = FakeCamera([jpeg(210)])
    blink = SlowPostBlink(camera, show_on_refresh=3)
    frame = BlinkSnapshotSource("Front Window", FakeAccount(blink)).read(NOW)
    assert frame.good, frame.reason
    assert blink.refreshes == 3


def test_gives_up_after_four_refreshes():
    camera = FakeCamera([jpeg(210)])
    blink = SlowPostBlink(camera, show_on_refresh=10)
    frame = BlinkSnapshotSource("Front Window", FakeAccount(blink)).read(NOW)
    assert not frame.good and "stale" in frame.reason
    assert blink.refreshes == 4


class OldPhotoAtNewUrlBlink(FakeBlink):
    """The thumbnail URL moves right away, but the first download is still the
    previous photo. The new one shows up on a later refresh."""

    def __init__(self, camera, old_photo_refreshes):
        super().__init__(camera)
        self.old_photo_refreshes = old_photo_refreshes
        self.refreshes = 0

    async def refresh(self, force_cache=False):
        self.refreshes += 1
        cam = self.cameras["Front Window"]
        if cam.pending is not None and self.refreshes <= self.old_photo_refreshes:
            self.ts += 1
            cam.thumbnail = f"https://x/thumb.jpg?ts={self.ts}&ext="  # new URL, old bytes
            return
        await super().refresh(force_cache)


def test_new_url_with_old_photo_waits_for_the_real_one():
    camera = FakeCamera([jpeg(100), jpeg(200)])
    blink = OldPhotoAtNewUrlBlink(camera, old_photo_refreshes=0)
    source = BlinkSnapshotSource("Front Window", FakeAccount(blink))
    assert source.read(NOW).good

    blink.old_photo_refreshes = blink.refreshes + 2  # next poll: two refreshes of the old photo
    frame = source.read(NOW + timedelta(seconds=60))
    assert frame.good, frame.reason
    assert frame.image.mean() > 150


def test_close_disconnects_account():
    account = FakeAccount(FakeBlink(FakeCamera([jpeg(1)])))
    source = BlinkSnapshotSource("Front Window", account)
    source.close()
    assert account.disconnects == 1
