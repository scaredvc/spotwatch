"""Polls each camera on its own schedule and drives the spot state machines.

One poll of one camera:
  grab frame -> detect vehicles -> read each of its spots -> store the reading
  -> fuse with the other cameras' latest readings for that spot
  -> feed the state machine -> log changes, notify if it says so
"""

import logging
import time as time_module
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from spotwatch.config import Camera, Config
from spotwatch.detector import Detector
from spotwatch.fusion import fuse
from spotwatch.notifier import Alert, Notifier, highlight_spot
from spotwatch.occupancy import read_spots
from spotwatch.quality import Reference, align_polygons, check_frame, reference_path
from spotwatch.sources import Frame, FrameSource, bad_frame
from spotwatch.state_machine import SpotStateMachine
from spotwatch.states import CameraReading, SpotState
from spotwatch.storage import Store

log = logging.getLogger("spotwatch")


class Runner:
    def __init__(
        self,
        config: Config,
        sources: dict[str, FrameSource],
        detector: Detector,
        notifier: Notifier,
        store: Store,
        clock: Callable[[], datetime] = datetime.now,
        check_quality: bool = True,
        save_dir: str | None = None,
        save_every: int = 300,
    ):
        self.config = config
        self.sources = sources
        self.detector = detector
        self.notifier = notifier
        self.store = store
        self.clock = clock

        self.machines = {
            spot_id: SpotStateMachine(spot_id, config.free_frames, config.occupied_frames)
            for spot_id in config.spots
        }
        # latest reading from each camera for each spot: {spot_id: {camera: CameraReading}}
        self.latest: dict[str, dict[str, CameraReading]] = {spot_id: {} for spot_id in config.spots}
        self.max_age = {
            c.name: timedelta(seconds=c.interval_seconds * config.stale_after_intervals) for c in config.cameras
        }
        self.next_poll: dict[str, datetime | None] = {c.name: None for c in config.cameras}
        self.inactive: dict[str, bool] = {c.name: False for c in config.cameras}
        self.last_frames: dict[str, Frame] = {}
        # spot outlines as placed on each camera's last frame (moved when the camera wobbles)
        self.frame_polygons: dict[str, dict] = {}

        # Optional: keep a frame every save_every seconds per camera, plus the
        # picture of every alert, so a day's run can be checked afterwards.
        self.save_dir = Path(save_dir) if save_dir else None
        self.save_every = timedelta(seconds=save_every)
        self.last_saved: dict[str, datetime] = {}

        # Frame quality: thresholds from config, plus each camera's reference
        # frame (saved by calibrate.py) for the camera-moved check.
        self.check_quality = check_quality
        self.quality = config.quality_settings()
        self.references: dict[str, Reference | None] = {}
        if check_quality:
            for camera in config.cameras:
                path = reference_path(config.reference_dir, camera.name)
                self.references[camera.name] = Reference.load(path, camera.polygons) if path.exists() else None
                if self.references[camera.name] is None:
                    log.info("camera %s: no reference frame at %s, camera-moved check off", camera.name, path)

    # ---- scheduling ----

    def tick(self, now: datetime | None = None) -> None:
        """Poll every camera that is due. Call this often (e.g. every second)."""
        now = now or self.clock()
        touched_spots = set()
        for camera in self.config.cameras:
            due = self.next_poll[camera.name]
            if due is not None and now < due:
                continue
            self.next_poll[camera.name] = now + timedelta(seconds=camera.interval_seconds)
            touched_spots.update(self.poll_camera(camera, now))

        # Update spots only after every due camera has reported. Otherwise, at
        # startup, the first camera would decide a shared spot on its own before
        # the second camera got a say.
        for spot_id in self.config.spots:  # config order, so logs are stable
            if spot_id in touched_spots:
                self.update_spot(spot_id, now)

    def run_forever(self, sleep: Callable[[float], None] = time_module.sleep, stop_when_finished: bool = False):
        log.info("spotwatch running with %d cameras and %d spots", len(self.config.cameras), len(self.config.spots))
        while True:
            self.tick()
            if stop_when_finished and all(s.finished for s in self.sources.values()):
                log.info("all sources finished")
                return
            sleep(1)

    # ---- one camera poll ----

    def poll_camera(self, camera: Camera, now: datetime) -> list[str]:
        """Read one camera and store its readings. Returns the spot ids it covers."""
        if camera.active_hours and not camera.active_hours.contains(now.time()):
            # Outside active hours: don't touch the camera, mark its spots unknown.
            # Only write the readings to the db once, when we first go inactive.
            first_time = not self.inactive[camera.name]
            self.inactive[camera.name] = True
            readings = self.bad_readings(camera, now, "outside active hours")
            self.store_readings(camera, readings, now, log_readings=first_time)
            return list(camera.polygons)
        self.inactive[camera.name] = False

        source = self.sources[camera.name]
        try:
            frame = source.read(now)
        except Exception as error:  # camera down, network error, etc.
            log.warning("camera %s read failed: %s", camera.name, error)
            frame = bad_frame(now, f"read error: {error}")

        polygons = camera.polygons
        if frame.good and self.check_quality:
            result = check_frame(frame.image, self.quality, self.references.get(camera.name))
            polygons = align_polygons(camera.polygons, result.transform)
            if result.notes and result.transform is not None:
                log.info("camera %s: %s", camera.name, "; ".join(result.notes))
            if not result.good:
                log.info("camera %s: bad frame, %s", camera.name, result.reason)
                frame = Frame(image=frame.image, time=frame.time, good=False, reason=result.reason,
                              captured_at=frame.captured_at)

        self.store.log_snapshot(now, camera.name, frame.good, frame.reason, source.power_info())
        self.last_frames[camera.name] = frame
        self.frame_polygons[camera.name] = polygons
        self.maybe_save_frame(camera.name, frame, now)

        if not frame.good:
            readings = self.bad_readings(camera, now, frame.reason)
        else:
            detections = self.detector.detect(frame.image)
            values = read_spots(
                detections,
                polygons,
                min_confidence=self.config.min_confidence,
                occlusion_overlap=self.config.occlusion_overlap,
                tolerance_px=self.config.ground_tolerance_px,
                weak_confidence=self.config.weak_confidence,
                use_masks=self.config.occupancy_mode == "masks",
            )
            readings = [
                CameraReading(camera.name, spot_id, value, good=True, time=now)
                for spot_id, value in values.items()
            ]

        self.store_readings(camera, readings, now)
        return list(camera.polygons)

    def bad_readings(self, camera: Camera, now: datetime, reason: str) -> list[CameraReading]:
        return [
            CameraReading(camera.name, spot_id, SpotState.UNKNOWN, good=False, time=now, reason=reason)
            for spot_id in camera.polygons
        ]

    def store_readings(self, camera: Camera, readings: list[CameraReading], now: datetime, log_readings: bool = True):
        for reading in readings:
            if log_readings:
                self.store.log_reading(now, camera.name, reading.spot_id, reading.value.value, reading.good, reading.reason)
            self.latest[reading.spot_id][camera.name] = reading

    def update_spot(self, spot_id: str, now: datetime) -> None:
        fused = fuse(list(self.latest[spot_id].values()), now, self.max_age)
        update = self.machines[spot_id].update(fused)

        if update.changed:
            self.store.log_state_change(now, spot_id, update.old_state.value, update.new_state.value)
            log.info("spot %s: %s -> %s", spot_id, update.old_state.value, update.new_state.value)

        if update.notify or (self.config.notify_taken and update.notify_taken):
            spot = self.config.spots[spot_id]
            state = SpotState.OCCUPIED if update.notify_taken else SpotState.FREE
            alert = Alert(spot_id=spot_id, spot_name=spot.name, time=now,
                          image=self.alert_picture(spot_id, state), state=state)
            try:
                message = self.notifier.notify(alert)
            except Exception as error:  # a broken notifier must never stop the monitor
                log.error("notifier failed for spot %s: %r", spot_id, error)
                message = f"notify failed: {type(error).__name__}"
            self.store.log_notification(now, spot_id, message)
            if self.save_dir is not None and alert.image is not None:
                self.save_image(self.save_dir / "alerts" / f"{now:%Y%m%d-%H%M%S}-spot{spot_id}.jpg", alert.image)

    def maybe_save_frame(self, camera_name: str, frame: Frame, now: datetime) -> None:
        if self.save_dir is None or not isinstance(frame.image, np.ndarray):
            return
        last = self.last_saved.get(camera_name)
        if last is not None and now - last < self.save_every:
            return
        self.last_saved[camera_name] = now
        # bad frames are kept too (marked in the name): useful for tuning the quality check
        tag = "" if frame.good else "-bad"
        self.save_image(self.save_dir / camera_name / f"{now:%Y%m%d-%H%M%S}{tag}.jpg", frame.image)

    def save_image(self, path: Path, image) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), image)
        except Exception as error:  # a full disk shouldn't stop the monitor
            log.warning("couldn't save %s: %r", path, error)

    def alert_picture(self, spot_id: str, state: SpotState = SpotState.FREE):
        """Latest frame from a camera supporting the alert's state."""
        for camera_name, reading in self.latest[spot_id].items():
            if not reading.good or reading.value != state:
                continue
            frame = self.last_frames.get(camera_name)
            camera = next(c for c in self.config.cameras if c.name == camera_name)
            if frame is not None and frame.good and isinstance(frame.image, np.ndarray):
                polygons = self.frame_polygons.get(camera_name, camera.polygons)
                return highlight_spot(frame.image, polygons[spot_id], state)
        return None

    # ---- for the API later ----

    def spot_states(self) -> dict[str, str]:
        return {spot_id: m.state.value for spot_id, m in self.machines.items()}
