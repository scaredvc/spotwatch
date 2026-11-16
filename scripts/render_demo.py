"""Render the README demo: run a video clip through SpotWatch and draw what it decided.

The script doesn't make any decisions itself. Each poll goes through the same Runner the live monitor
uses: quality gate (quality.py), YOLO11-seg (detector.py), spot reading (occupancy.py), fusion and
the per-spot state machine, then the notifier. After each poll this script reads the runner's state
and draws it: the spot outlines colored by this frame's reading, the state machine's state for each
spot, and the alert picture the notifier received. The notifier is the in-memory ListNotifier the
tests use, so alerts are recorded instead of printed or posted.

    uv run python scripts/render_demo.py --video path/to/the-downloaded-clip.mp4

Needs ultralytics (downloads the model on first run) and ffmpeg on PATH for the GIF.
See docs/demo.md for where the clip comes from.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spotwatch.check import COLORS, draw_readings  # noqa: E402
from spotwatch.config import parse_config  # noqa: E402
from spotwatch.detector import Detector, YoloDetector  # noqa: E402
from spotwatch.notifier import ListNotifier  # noqa: E402
from spotwatch.quality import reference_path  # noqa: E402
from spotwatch.runner import Runner  # noqa: E402
from spotwatch.sources import VideoFileSource  # noqa: E402
from spotwatch.states import SpotState  # noqa: E402
from spotwatch.storage import Store  # noqa: E402

FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE, MUTED, PANEL = (240, 240, 240), (165, 165, 165), (22, 17, 13)


class RecordingDetector(Detector):
    """Passes calls to the real detector and keeps the last result for drawing."""

    def __init__(self, inner: Detector):
        self.inner = inner
        self.last = []

    def detect(self, image):
        self.last = self.inner.detect(image)
        return self.last


def load_demo_config(path: Path, video: Path) -> tuple[dict, tuple[int, int, int, int] | None]:
    """Read the demo config, point it at the video and scale outlines to the video's frame size.

    Returns the config and the display crop (x1, y1, x2, y2) in the video's pixels, if one is set.
    The crop only affects the picture; SpotWatch always sees the full frame."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    capture = cv2.VideoCapture(str(video))
    width, height = capture.get(cv2.CAP_PROP_FRAME_WIDTH), capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    capture.release()
    if not width:
        raise SystemExit(f"couldn't open {video}")
    crop = None
    for camera in data["cameras"]:
        camera["path"] = str(video)
        drawn_w, drawn_h = camera.pop("frame_size")
        sx, sy = width / drawn_w, height / drawn_h
        camera["polygons"] = {
            spot: [[round(x * sx), round(y * sy)] for x, y in points] for spot, points in camera["polygons"].items()
        }
        if "display_crop" in camera:
            x1, y1, x2, y2 = camera.pop("display_crop")
            crop = (round(x1 * sx), round(y1 * sy), round(x2 * sx), round(y2 * sy))
    return data, crop


def text(image, s, org, scale=0.55, color=WHITE, thickness=1):
    cv2.putText(image, s, org, FONT, scale, color, thickness, cv2.LINE_AA)


def text_width(s, scale, thickness=1) -> int:
    return cv2.getTextSize(s, FONT, scale, thickness)[0][0]


def panel(width, clip_s, frame, spots, readings, machines, config, events) -> np.ndarray:
    """Status bar under the picture: frame quality, then each spot's reading and state.

    Columns are placed from measured text widths so nothing overlaps at any GIF width."""
    big, small = 0.55, 0.45
    pad, gap = 12, 18
    names = {s: f"{s}  {spot.name}" for s, spot in spots.items()}
    col_reading = pad + max(text_width(n, big) for n in names.values()) + gap
    col_state = col_reading + max(text_width(w, big) for w in ("this frame", "occupied")) + gap
    col_note = col_state + max(text_width(w, big, 2) for w in ("spot state", "OCCUPIED")) + gap
    if col_note + text_width("1/2 to confirm", small) > width - pad:
        raise SystemExit(f"--width {width} is too narrow for the status bar")

    out = np.full((34 + 26 * (len(spots) + 1) + 52, width, 3), PANEL, np.uint8)
    # Only the short reason ("too dark"); the full one, with its numbers, is in the report.
    quality = "frame OK" if frame.good else "frame rejected: " + frame.reason.split(" (")[0]
    text(out, f"clip {clip_s:5.2f}s", (pad, 24), big)
    text(out, quality, (pad + text_width("clip 00.00s", big) + gap, 24), big,
         WHITE if frame.good else (90, 170, 255))
    y = 52
    text(out, "spot", (pad, y), small, MUTED)
    text(out, "this frame", (col_reading, y), small, MUTED)
    text(out, "spot state", (col_state, y), small, MUTED)
    for spot_id in spots:
        y += 26
        text(out, names[spot_id], (pad, y), big)
        reading, machine = readings[spot_id], machines[spot_id]
        text(out, reading.value, (col_reading, y), big, COLORS[reading.value])
        text(out, machine.state.value.upper(), (col_state, y), big, COLORS[machine.state.value], 2)
        # A reading that hasn't changed the state yet: show how far it is toward confirming.
        if reading != machine.state and reading != SpotState.UNKNOWN:
            if reading == SpotState.OCCUPIED:
                streak, needed = machine.occupied_streak, config.occupied_frames
            else:
                streak, needed = machine.free_streak, config.free_frames
            text(out, f"{streak}/{needed} to confirm", (col_note, y), small, MUTED)
    last = events[-1] if events else "no alerts yet"
    text(out, last, (pad, out.shape[0] - 30), big, (120, 230, 120) if events else MUTED)
    text(out, "outline color = this frame's reading, dot = a car's ground point", (pad, out.shape[0] - 10),
         small, MUTED)
    return out


def banner(image, message) -> np.ndarray:
    """Green bar over the alert photo. The photo is cropped to the GIF's view like every other frame."""
    out = image.copy()
    note = "alert photo, cropped"
    cv2.rectangle(out, (0, 0), (out.shape[1], 36), (30, 120, 30), -1)
    text(out, message, (12, 25), 0.6, WHITE, 2)
    text(out, note, (out.shape[1] - text_width(note, 0.45) - 12, 24), 0.45, WHITE)
    return out


def fit(image, width, crop=None):
    if crop is not None:
        x1, y1, x2, y2 = crop
        image = image[y1:y2, x1:x2]
    h, w = image.shape[:2]
    return cv2.resize(image, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)


def montage(samples: list[tuple[float, object, np.ndarray]], count: int = 12, columns: int = 3,
            tile_width: int = 400) -> np.ndarray:
    """Evenly spaced polls in a grid, each labelled with the quality gate's verdict."""
    picks = [samples[round(i * (len(samples) - 1) / (count - 1))] for i in range(count)]
    tiles = []
    for clip_s, frame, picture in picks:
        tile = fit(picture, tile_width)
        label = "frame OK" if frame.good else frame.reason.split(" (")[0]
        cv2.rectangle(tile, (0, 0), (tile_width, 24), PANEL, -1)
        text(tile, f"{clip_s:5.2f}s  {label}", (6, 17), 0.5, WHITE if frame.good else (90, 170, 255))
        tiles.append(tile)
    rows = [np.hstack(tiles[i:i + columns]) for i in range(0, len(tiles), columns)]
    return np.vstack(rows)


def write_gif(frames: list[np.ndarray], path: Path, fps: int, colors: int) -> None:
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg not found on PATH; it's needed to write the GIF")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for i, frame in enumerate(frames):
            cv2.imwrite(str(Path(tmp) / f"{i:05d}.png"), frame)
        pattern = str(Path(tmp) / "%05d.png")
        palette = str(Path(tmp) / "palette.png")
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(fps), "-i", pattern,
                        "-vf", f"palettegen=max_colors={colors}:stats_mode=diff", palette], check=True)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(fps), "-i", pattern, "-i", palette,
                        "-lavfi", "paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle", "-loop", "0",
                        str(path)], check=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def environment() -> dict:
    """Library versions and device, since detections can shift slightly between them."""
    import torch
    import ultralytics

    return {"python": sys.version.split()[0], "opencv": cv2.__version__, "numpy": np.__version__,
            "ultralytics": ultralytics.__version__, "torch": torch.__version__,
            "device": "cuda" if torch.cuda.is_available() else "cpu"}


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", type=Path, required=True, help="the Pexels clip (see docs/demo.md)")
    parser.add_argument("--config", type=Path, default=root / "scripts" / "demo-config.yaml")
    parser.add_argument("--start", type=float, default=17.0, help="clip second to start at")
    parser.add_argument("--end", type=float, default=32.0, help="clip second to stop at")
    parser.add_argument("--step", type=float, default=0.25, help="clip seconds between polls")
    parser.add_argument("--reference-at", type=float, default=22.0,
                        help="clip second used as the calibration reference frame (camera-moved check)")
    parser.add_argument("--width", type=int, default=640, help="GIF width in pixels")
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--colors", type=int, default=128, help="GIF palette size (fewer = smaller file)")
    parser.add_argument("--hold", type=float, default=2.0, help="seconds to show each alert picture")
    parser.add_argument("--out", type=Path, default=root / "docs" / "images" / "demo.gif")
    parser.add_argument("--alert-out", type=Path, help="also save the full-size picture of the free alert here")
    parser.add_argument("--report", type=Path, default=root / "docs" / "demo-readings.json")
    parser.add_argument("--montage-out", type=Path, default=root / "docs" / "images" / "demo-montage.jpg",
                        help="grid of evenly spaced polls: detections, ground points and quality verdicts")
    args = parser.parse_args()

    data, crop = load_demo_config(args.config, args.video)
    with tempfile.TemporaryDirectory() as reference_dir:
        data.setdefault("settings", {})["reference_dir"] = reference_dir
        config = parse_config(data)
        camera = config.cameras[0]

        # Calibration step: keep one daylight frame as the reference, as `calibrate` does.
        ref_source = VideoFileSource(camera.options["path"])
        ref_source.position_ms = args.reference_at * 1000
        reference = ref_source.read(datetime.now())
        ref_source.close()
        cv2.imwrite(str(reference_path(reference_dir, camera.name)), reference.image)

        source = VideoFileSource(camera.options["path"], step_seconds=args.step)
        source.position_ms = args.start * 1000
        detector = RecordingDetector(YoloDetector(config.model, min(config.weak_confidence, config.min_confidence),
                                                  use_masks=config.occupancy_mode == "masks"))
        notifier = ListNotifier()
        runner = Runner(config, {camera.name: source}, detector, notifier, Store(":memory:"))

        start = datetime(2020, 9, 10, 6, 0)  # simulated wall clock: one poll per camera interval
        frames, report, events, samples = [], [], [], []
        poll = 0
        while True:
            clip_s = args.start + poll * args.step
            if clip_s > args.end + 1e-6:
                break
            detector.last = []
            alerts_before = len(notifier.alerts)
            runner.tick(start + timedelta(seconds=camera.interval_seconds * poll))
            poll += 1
            frame = runner.last_frames[camera.name]
            if source.finished or frame.image is None:
                break

            readings = {s: runner.latest[s][camera.name].value for s in camera.polygons}
            machines = {s: runner.machines[s] for s in camera.polygons}
            states = {s: m.state for s, m in machines.items()}
            polygons = runner.frame_polygons[camera.name]
            new_alerts = notifier.alerts[alerts_before:]
            for alert in new_alerts:
                word = "taken" if alert.state == SpotState.OCCUPIED else "free"
                events.append(f"{clip_s:5.2f}s  alert sent: spot {alert.spot_id} is {word}")
            report.append({
                "clip_seconds": round(clip_s, 2),
                "frame_good": frame.good,
                "frame_reason": frame.reason,
                "vehicles": [{"label": d.label, "confidence": round(d.confidence, 3),
                              "ground_point": [round(v) for v in d.ground_point()]} for d in detector.last],
                "readings": {s: r.value for s, r in readings.items()},
                "states": {s: st.value for s, st in states.items()},
                "alerts": [{"spot": a.spot_id, "state": a.state.value} for a in new_alerts],
            })

            picture = draw_readings(frame.image, polygons, readings, detector.last)
            bar = panel(args.width, clip_s, frame, {s: config.spots[s] for s in camera.polygons},
                        readings, machines, config, events)
            frames.append(np.vstack([fit(picture, args.width, crop), bar]))
            samples.append((clip_s, frame, fit(picture, picture.shape[1], crop)))

            for alert in new_alerts:
                if alert.image is None:
                    continue
                if alert.state == SpotState.FREE and args.alert_out:
                    args.alert_out.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(args.alert_out), alert.image, [cv2.IMWRITE_JPEG_QUALITY, 85])
                word = "taken" if alert.state == SpotState.OCCUPIED else "free"
                picture = fit(alert.image, args.width, crop)
                shown = np.vstack([banner(picture, f"Alert sent: spot {alert.spot_id} is {word}"), bar])
                frames.extend([shown] * round(args.hold * args.fps))

        source.close()

    if not frames:
        raise SystemExit("no frames rendered; check --start/--end against the clip length")
    frames.extend([frames[-1]] * args.fps)  # pause on the last frame before looping
    write_gif(frames, args.out, args.fps, args.colors)
    args.montage_out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.montage_out), montage(samples), [cv2.IMWRITE_JPEG_QUALITY, 82])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({
        "clip": args.video.name, "config": args.config.relative_to(root).as_posix() if args.config.is_relative_to(root)
        else str(args.config), "model": config.model, "occupancy_mode": config.occupancy_mode,
        "clip_sha256": sha256(args.video), "environment": environment(),
        "start": args.start, "end": args.end, "step": args.step, "reference_at": args.reference_at,
        "polls": report,
    }, indent=1), encoding="utf-8", newline="\n")
    print(f"{len(report)} polls, {len(notifier.alerts)} alerts")
    if not any(a.state == SpotState.FREE for a in notifier.alerts):
        print("warning: no free alert in this range" + (f"; {args.alert_out} was not written" if args.alert_out else ""))
    for event in events:
        print(" ", event)
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB), {args.montage_out} and {args.report}")


if __name__ == "__main__":
    main()
