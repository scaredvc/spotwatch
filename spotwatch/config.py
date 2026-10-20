"""Load and check config.yaml."""

from dataclasses import dataclass, field
from datetime import time
from pathlib import Path

import yaml

SOURCE_TYPES = {"image_folder", "video_file", "blink", "rtsp"}


@dataclass
class ActiveHours:
    start: time
    end: time

    def contains(self, t: time) -> bool:
        if self.start <= self.end:
            return self.start <= t < self.end
        # window that crosses midnight, e.g. 22:00 to 06:00
        return t >= self.start or t < self.end


@dataclass
class Spot:
    id: str
    name: str


@dataclass
class Camera:
    name: str
    source: str
    interval_seconds: int = 60
    active_hours: ActiveHours | None = None
    polygons: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    # source-specific settings (path, env var name, blink camera name, ...)
    options: dict = field(default_factory=dict)


@dataclass
class Config:
    spots: dict[str, Spot]
    cameras: list[Camera]
    free_frames: int = 1
    occupied_frames: int = 2
    stale_after_intervals: float = 3.0
    occlusion_overlap: float = 0.4
    ground_tolerance_px: float = 10.0
    min_confidence: float = 0.35
    weak_confidence: float = 0.15
    model: str = "yolo11n.pt"
    occupancy_mode: str = "boxes"  # boxes | masks (requires a segmentation model)
    notify_taken: bool = False  # opt in to confirmed free -> occupied notifications
    db_path: str = "spotwatch.db"
    blink_session_file: str = "blink_session.json"
    # frame quality (see spotwatch/quality.py)
    min_brightness: float = 40.0
    min_sharpness: float = 30.0
    max_glare: float = 0.12
    max_shift_px: float = 10.0
    max_align_shift_px: float = 0.0  # >0: align spot outlines to shifts up to this size
    reference_dir: str = "calibration"
    # alerts: "console" or "discord" (webhook URL from this env var or .env)
    notifier: str = "console"
    discord_webhook_env: str = "DISCORD_WEBHOOK_URL"

    def quality_settings(self):
        from spotwatch.quality import QualitySettings

        return QualitySettings(
            min_brightness=self.min_brightness,
            min_sharpness=self.min_sharpness,
            max_glare=self.max_glare,
            max_shift_px=self.max_shift_px,
            max_align_shift_px=self.max_align_shift_px,
        )

    def cameras_for_spot(self, spot_id: str) -> list[Camera]:
        return [c for c in self.cameras if spot_id in c.polygons]


class ConfigError(ValueError):
    pass


def _parse_time(value) -> time:
    if isinstance(value, int):  # yaml reads 07:00 as an int (sexagesimal) sometimes
        return time(value // 60, value % 60)
    hours, minutes = str(value).split(":")
    return time(int(hours), int(minutes))


def parse_config(data: dict) -> Config:
    spots = {}
    for s in data.get("spots") or []:
        spot_id = str(s["id"])
        if spot_id in spots:
            raise ConfigError(f"duplicate spot id: {spot_id}")
        spots[spot_id] = Spot(id=spot_id, name=s.get("name", spot_id))

    cameras = []
    names = set()
    known_keys = {"name", "source", "interval_seconds", "active_hours", "polygons"}
    for c in data.get("cameras") or []:
        name = c["name"]
        if name in names:
            raise ConfigError(f"duplicate camera name: {name}")
        names.add(name)

        source = c["source"]
        if source not in SOURCE_TYPES:
            raise ConfigError(f"camera {name}: unknown source type {source!r}")

        hours = None
        if c.get("active_hours"):
            hours = ActiveHours(
                start=_parse_time(c["active_hours"]["start"]),
                end=_parse_time(c["active_hours"]["end"]),
            )

        polygons = {}
        for spot_id, points in (c.get("polygons") or {}).items():
            spot_id = str(spot_id)
            if spot_id not in spots:
                raise ConfigError(f"camera {name}: polygon for unknown spot {spot_id!r}")
            if len(points) < 3:
                raise ConfigError(f"camera {name}: polygon for {spot_id} needs at least 3 points")
            polygons[spot_id] = [(int(x), int(y)) for x, y in points]

        interval = int(c.get("interval_seconds", 60))
        if interval < 1:
            raise ConfigError(f"camera {name}: interval_seconds must be at least 1")

        if source == "blink" and not c.get("blink_camera"):
            raise ConfigError(
                f"camera {name}: blink cameras need blink_camera (the camera's name in the Blink app)"
            )
        if source in ("image_folder", "video_file") and not c.get("path"):
            raise ConfigError(f"camera {name}: {source} needs a path")

        cameras.append(
            Camera(
                name=name,
                source=source,
                interval_seconds=interval,
                active_hours=hours,
                polygons=polygons,
                options={k: v for k, v in c.items() if k not in known_keys},
            )
        )

    settings = data.get("settings") or {}
    config = Config(spots=spots, cameras=cameras)
    for key in (
        "free_frames",
        "occupied_frames",
        "stale_after_intervals",
        "occlusion_overlap",
        "ground_tolerance_px",
        "min_confidence",
        "weak_confidence",
        "model",
        "occupancy_mode",
        "db_path",
        "blink_session_file",
        "min_brightness",
        "min_sharpness",
        "max_glare",
        "max_shift_px",
        "max_align_shift_px",
        "reference_dir",
        "notifier",
        "discord_webhook_env",
    ):
        if key in settings:
            setattr(config, key, type(getattr(config, key))(settings[key]))

    if config.free_frames < 1 or config.occupied_frames < 1:
        raise ConfigError("free_frames and occupied_frames must be at least 1")
    if config.occupancy_mode not in {"boxes", "masks"}:
        raise ConfigError("occupancy_mode must be boxes or masks")
    if "notify_taken" in settings:
        if not isinstance(settings["notify_taken"], bool):
            raise ConfigError("notify_taken must be a boolean")
        config.notify_taken = settings["notify_taken"]
    return config


def load_config(path: str | Path) -> Config:
    with open(path) as f:
        return parse_config(yaml.safe_load(f) or {})
