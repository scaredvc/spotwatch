"""Save frames from each camera to a folder, to build a test set from your street.

    python -m spotwatch.record --count 1             # one frame per camera, then stop
    python -m spotwatch.record --hours 3             # every camera's interval, for 3 hours
    python -m spotwatch.record --every 120 --hours 2 --camera window-left

Frames go to recordings/<camera>/<YYYYMMDD-HHMMSS>.jpg. Bad or stale frames
are skipped and the reason is printed. Respects each camera's active hours
unless you pass --all-hours (useful for recording dusk and night).
"""

import argparse
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

import cv2

from spotwatch.config import Config, load_config
from spotwatch.sources import FrameSource, make_source

log = logging.getLogger("spotwatch.record")


def record(
    config: Config,
    sources: dict[str, FrameSource],
    out_dir: Path,
    every: int | None = None,
    hours: float | None = None,
    count: int | None = None,
    all_hours: bool = False,
    clock: Callable[[], datetime] = datetime.now,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, int]:
    """Returns how many frames were saved per camera."""
    cameras = [c for c in config.cameras if c.name in sources]
    start = clock()
    end = start + timedelta(hours=hours) if hours else None
    next_due = {c.name: start for c in cameras}
    saved = {c.name: 0 for c in cameras}

    def done() -> bool:
        if end is not None and clock() >= end:
            return True
        if count is not None and all(saved[c.name] >= count for c in cameras):
            return True
        return all(sources[c.name].finished for c in cameras)

    while not done():
        now = clock()
        for camera in cameras:
            if count is not None and saved[camera.name] >= count:
                continue
            if now < next_due[camera.name]:
                continue
            next_due[camera.name] = now + timedelta(seconds=every or camera.interval_seconds)

            if not all_hours and camera.active_hours and not camera.active_hours.contains(now.time()):
                continue

            frame = sources[camera.name].read(now)
            if not frame.good:
                print(f"{now:%H:%M:%S} {camera.name}: skipped ({frame.reason})", flush=True)
                continue

            folder = out_dir / camera.name
            folder.mkdir(parents=True, exist_ok=True)
            taken = frame.captured_at or now
            path = folder / f"{taken:%Y%m%d-%H%M%S}.jpg"
            cv2.imwrite(str(path), frame.image)
            saved[camera.name] += 1
            print(f"{now:%H:%M:%S} {camera.name}: saved {path}", flush=True)
        sleep(1)
    return saved


def main():
    parser = argparse.ArgumentParser(description="Save camera frames to a folder.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--out", default="recordings")
    parser.add_argument("--camera", action="append", help="camera name from config (repeatable); default all")
    parser.add_argument("--every", type=int, help="seconds between frames (default: each camera's interval)")
    parser.add_argument("--hours", type=float, help="stop after this many hours")
    parser.add_argument("--count", type=int, help="stop after this many frames per camera")
    parser.add_argument("--all-hours", action="store_true", help="ignore active hours")
    args = parser.parse_args()

    if args.hours is None and args.count is None:
        parser.error("give --hours or --count so it knows when to stop")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(args.config)
    wanted = set(args.camera) if args.camera else {c.name for c in config.cameras}
    unknown = wanted - {c.name for c in config.cameras}
    if unknown:
        parser.error(f"unknown camera(s): {', '.join(sorted(unknown))}")

    sources = {c.name: make_source(c, config) for c in config.cameras if c.name in wanted}
    try:
        saved = record(config, sources, Path(args.out), args.every, args.hours, args.count, args.all_hours)
    except KeyboardInterrupt:
        print("stopped")
        return
    finally:
        for source in sources.values():
            source.close()
    print("done:", ", ".join(f"{name} {n}" for name, n in saved.items()))


if __name__ == "__main__":
    main()
