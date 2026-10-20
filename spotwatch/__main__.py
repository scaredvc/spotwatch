"""Run spotwatch: python -m spotwatch [--config config.yaml]"""

import argparse
import logging
import time
from datetime import datetime, timedelta

from spotwatch.config import load_config
from spotwatch.detector import YoloDetector
from spotwatch.notifier import make_notifier
from spotwatch.runner import Runner
from spotwatch.sources import make_source
from spotwatch.storage import Store


class FakeClock:
    """Time that only moves when sleep() is called. Lets a folder of images
    play through at full speed while the runner thinks minutes are passing."""

    def __init__(self, start: datetime):
        self.current = start

    def now(self) -> datetime:
        return self.current

    def sleep(self, seconds: float) -> None:
        self.current += timedelta(seconds=seconds)


def main():
    parser = argparse.ArgumentParser(description="Watch parking spots and say when one is free.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--until-done",
        action="store_true",
        help="stop when every folder/video source has run out (for dev runs)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="don't wait between polls; pretend time passes (for folder/video sources)",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        metavar="SECONDS",
        help="also save a frame per camera this often (and every alert picture) to --save-dir",
    )
    parser.add_argument("--save-dir", default="runs", help="where --save-every puts frames (default: runs)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(args.config)
    # keep weak boxes too: read_spots uses them to avoid calling a covered spot free
    detector = YoloDetector(config.model, min(config.weak_confidence, config.min_confidence),
                            use_masks=config.occupancy_mode == "masks")
    sources = {camera.name: make_source(camera, config) for camera in config.cameras}

    clock, sleep = datetime.now, time.sleep
    if args.fast:
        fake = FakeClock(datetime.now())
        clock, sleep = fake.now, fake.sleep

    save_dir = None
    if args.save_every:
        save_dir = args.save_dir  # filenames carry the date and time
        logging.info("saving a frame every %ds and every alert picture to %s", args.save_every, save_dir)
    runner = Runner(
        config,
        sources,
        detector,
        make_notifier(config),
        Store(config.db_path),
        clock=clock,
        save_dir=save_dir,
        save_every=args.save_every or 300,
    )
    try:
        runner.run_forever(sleep=sleep, stop_when_finished=args.until_done)
    except KeyboardInterrupt:
        pass
    finally:
        for source in sources.values():
            source.close()


if __name__ == "__main__":
    main()
