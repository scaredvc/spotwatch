"""Blink cameras, through the unofficial blinkpy library (pinned to 0.25.9).

Blink has no video stream we can read and no official API. What we can do is
ask a camera to take a still photo ("snapshot") and download it.

How one snapshot works, checked against the blinkpy 0.25.9 source:
1. camera.snap_picture() asks the camera for a new photo and waits until
   Blink says the command finished.
2. snap_picture() then downloads from the thumbnail URL it already had. That's
   the OLD photo, so we ignore what it cached.
3. blink.refresh(force_cache=True) pulls the camera's latest info. The new
   thumbnail URL has a new timestamp in it, and blinkpy downloads that image.
4. We only accept the image if the thumbnail URL changed AND the bytes differ
   from the last image we used. Anything else is a stale frame.

blinkpy logs most network errors and returns None instead of raising, so
step 4 is also how we notice that a request quietly failed.
"""

import asyncio
import hashlib
import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

from spotwatch.sources import Frame, FrameSource, bad_frame

log = logging.getLogger("spotwatch.blink")

SETUP_HINT = "run `python -m spotwatch.blink_setup` to log in again"

# Fields from blinkpy's camera.attributes worth keeping in the snapshot log.
POWER_FIELDS = (
    "battery",
    "battery_level",
    "battery_voltage",
    "temperature_c",
    "wifi_strength",
    "sync_signal_strength",
)


class BlinkAuthError(Exception):
    """The saved session no longer works. Needs a human to log in again."""


# ---------- small pure helpers (unit tested) ----------


def image_hash(image_bytes: bytes) -> str:
    return hashlib.sha1(image_bytes).hexdigest()


def check_fresh(old_thumbnail, new_thumbnail, image_bytes, last_hash) -> tuple[bool, str]:
    """Is the image we got after snap + refresh really a new photo?"""
    if not image_bytes:
        return False, "no image from Blink"
    if not new_thumbnail:
        return False, "no thumbnail URL from Blink"
    if new_thumbnail == old_thumbnail:
        return False, "stale snapshot: thumbnail did not change"
    if last_hash is not None and image_hash(image_bytes) == last_hash:
        return False, "stale snapshot: same image as last time"
    return True, ""


def thumbnail_time(url) -> datetime | None:
    """Blink thumbnail URLs end in ?ts=<unix seconds>. Turn that into a time."""
    if not url:
        return None
    try:
        ts = parse_qs(urlparse(url).query).get("ts", [None])[0]
        return datetime.fromtimestamp(int(ts)) if ts else None
    except (ValueError, OverflowError, OSError):
        return None


def power_info_from(camera) -> dict:
    attrs = camera.attributes
    return {key: attrs.get(key) for key in POWER_FIELDS if attrs.get(key) is not None}


def backoff_seconds(failures: int, base: int = 60, cap: int = 900) -> int:
    """1 failure: wait base. Then double each time, up to cap (15 min)."""
    if failures <= 0:
        return 0
    return min(base * 2 ** (failures - 1), cap)


# ---------- session file ----------


def save_session(blink, path) -> None:
    """Save login tokens so we can log in without the password or a PIN.

    The password is left out on purpose. Token refresh only needs the
    refresh token. If that ever stops working, run blink_setup again.
    """
    data = dict(blink.auth.login_attributes)
    data.pop("password", None)
    data.pop("2fa_code", None)
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.chmod(tmp, 0o600)  # only you can read it
    tmp.replace(path)


def load_session(path) -> dict:
    path = Path(path)
    if not path.exists():
        raise BlinkAuthError(f"no Blink session file at {path}; {SETUP_HINT}")
    with open(path) as f:
        data = json.load(f)
    if not data.get("refresh_token"):
        raise BlinkAuthError(f"Blink session file has no refresh token; {SETUP_HINT}")
    return data


# ---------- the shared account ----------


class BlinkAccount:
    """One logged-in Blink account, shared by every Blink camera in the config.

    blinkpy is async; the rest of spotwatch isn't. So the account owns its own
    event loop and the sources call run() to do async work on it.
    """

    def __init__(self, session_file, refresh_rate: int = 60):
        self.session_file = Path(session_file)
        self.refresh_rate = refresh_rate
        self.loop = asyncio.new_event_loop()
        self.blink = None
        self.saved_refresh_token = None

    def run(self, coro):
        return self.loop.run_until_complete(coro)

    async def get_blink(self):
        if self.blink is None:
            self.blink = await self._login()
        return self.blink

    async def _login(self):
        # imported here so the rest of the package works without blinkpy
        from aiohttp import ClientSession
        from blinkpy.auth import Auth, BlinkTwoFARequiredError
        from blinkpy.blinkpy import Blink

        data = load_session(self.session_file)
        session = ClientSession()
        blink = Blink(session=session, refresh_rate=self.refresh_rate)
        blink.auth = Auth(data, no_prompt=True, session=session)
        try:
            ok = await blink.start()
        except BlinkTwoFARequiredError:
            ok = False
        except BaseException:  # network errors etc: don't leak the session
            await session.close()
            raise
        if not ok:
            await session.close()
            raise BlinkAuthError(f"Blink login with saved session failed; {SETUP_HINT}")
        log.info("logged in to Blink, cameras: %s", ", ".join(blink.cameras.keys()))
        self.blink = blink
        self.save_if_tokens_changed()
        return blink

    def save_if_tokens_changed(self) -> None:
        """Blink can rotate the refresh token. Keep the file current."""
        if self.blink is None:
            return
        token = self.blink.auth.refresh_token
        if token and token != self.saved_refresh_token:
            save_session(self.blink, self.session_file)
            self.saved_refresh_token = token

    def disconnect(self) -> None:
        """Drop the session. The next read logs in again from the file."""
        if self.blink is not None:
            try:
                self.run(self.blink.auth.session.close())
            except Exception:
                pass
        self.blink = None


_accounts: dict[str, BlinkAccount] = {}


def get_account(session_file, refresh_rate: int = 60) -> BlinkAccount:
    # blinkpy logs a full network dump at INFO on every login; keep its warnings only
    logging.getLogger("blinkpy").setLevel(logging.WARNING)
    key = str(Path(session_file).resolve())
    if key not in _accounts:
        _accounts[key] = BlinkAccount(session_file, refresh_rate)
    return _accounts[key]


# ---------- one snapshot ----------


def name_key(name: str) -> str:
    """'XT2 - 1234', 'xt2-1234' and 'XT2 1234' all become 'xt21234'."""
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def find_camera(cameras, wanted: str):
    """Exact name first, then ignoring case, spaces and dashes."""
    if wanted in cameras:
        return cameras[wanted]
    matches = [cam for name, cam in cameras.items() if name_key(name) == name_key(wanted)]
    return matches[0] if len(matches) == 1 else None


async def take_snapshot(blink, camera_name: str, last_hash=None, refresh_tries: int = 4, refresh_wait: float = 4.0):
    """Snap, refresh, check freshness.

    Returns (image_bytes or None, reason, captured_at, power_info).
    """
    camera = find_camera(blink.cameras, camera_name)
    if camera is None:
        names = ", ".join(blink.cameras.keys()) or "none"
        return None, f"no Blink camera named {camera_name!r} (cameras: {names})", None, {}

    if not camera.online:
        return None, "camera offline", None, power_info_from(camera)

    old_thumbnail = camera.thumbnail
    await camera.snap_picture()

    # Blink can say the snap finished a few seconds before the new photo shows
    # up on its servers. Two ways that looks:
    #   - the thumbnail URL hasn't changed yet
    #   - the URL changed, but it still serves the previous photo (seen on the
    #     Oct 15 run, about 1 poll in 6)
    # Either way, wait and look again before calling it stale. refresh with
    # force_cache re-downloads the image even when the URL is the same.
    for attempt in range(1, refresh_tries + 1):
        await blink.refresh(force_cache=True)
        posted = camera.thumbnail != old_thumbnail
        same_photo = last_hash is not None and camera.image_from_cache and image_hash(camera.image_from_cache) == last_hash
        if (posted and not same_photo) or attempt == refresh_tries:
            break
        why = "same photo as last time" if posted else "not posted yet"
        log.info("%s: new snapshot %s (try %d), waiting", camera_name, why, attempt)
        await asyncio.sleep(refresh_wait)

    image_bytes = camera.image_from_cache
    fresh, reason = check_fresh(old_thumbnail, camera.thumbnail, image_bytes, last_hash)
    power = power_info_from(camera)
    if not fresh:
        return None, reason, None, power
    return image_bytes, "", thumbnail_time(camera.thumbnail), power


class BlinkSnapshotSource(FrameSource):
    def __init__(self, blink_camera: str, account: BlinkAccount):
        self.blink_camera = blink_camera
        self.account = account
        self.last_hash = None
        self.last_power = None
        self.failures = 0
        self.retry_at: datetime | None = None

    def read(self, now: datetime) -> Frame:
        if self.retry_at is not None and now < self.retry_at:
            return bad_frame(now, f"backing off until {self.retry_at:%H:%M:%S}")

        try:
            frame = self.account.run(self._read(now))
        except BlinkAuthError as error:
            # Needs you to log in again. Back off hard so we don't hammer Blink.
            log.error("%s", error)
            self.account.disconnect()
            frame = bad_frame(now, str(error))
        except Exception as error:  # network, throttling, Blink API changes
            log.warning("Blink read for %s failed: %r", self.blink_camera, error)
            self.account.disconnect()
            frame = bad_frame(now, f"blink error: {type(error).__name__}: {str(error)[:80]}")

        if frame.good:
            self.failures = 0
            self.retry_at = None
        else:
            self.failures += 1
            wait = backoff_seconds(self.failures)
            self.retry_at = now + timedelta(seconds=wait)
            if self.failures > 1:
                log.info("camera %s: %d failures in a row, next try in %ds", self.blink_camera, self.failures, wait)
        return frame

    async def _read(self, now: datetime) -> Frame:
        blink = await self.account.get_blink()
        image_bytes, reason, captured_at, power = await take_snapshot(blink, self.blink_camera, self.last_hash)
        self.last_power = power or self.last_power
        self.account.save_if_tokens_changed()

        if image_bytes is None:
            return bad_frame(now, reason)

        image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return bad_frame(now, "could not decode Blink image")
        self.last_hash = image_hash(image_bytes)
        return Frame(image=image, time=now, captured_at=captured_at)

    def power_info(self) -> dict | None:
        return self.last_power

    def close(self) -> None:
        # Closes the shared Blink connection. Safe to call once per camera:
        # disconnect() does nothing if it's already closed.
        self.account.disconnect()
