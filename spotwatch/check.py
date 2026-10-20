"""Check one frame: is it good enough, has the camera moved, and what does each spot read?

    uv run python -m spotwatch.check --camera front-camera --image recordings/front-camera/20261014-172754.jpg
    uv run python -m spotwatch.check --camera front-camera                # grab a live frame
    uv run python -m spotwatch.check --camera front-camera --image ... --detect

--detect also runs YOLO (needs `ultralytics` installed) and saves a picture
next to the frame, <name>-check.jpg, with each spot colored:
green = free, red = occupied, gray = unknown. Yellow dots are where each
detected car touches the ground.
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from spotwatch.config import load_config
from spotwatch.quality import Reference, align_polygons, check_frame, reference_path

COLORS = {"occupied": (0, 0, 255), "free": (0, 200, 0), "unknown": (160, 160, 160)}


def draw_readings(image, polygons, readings, detections):
    out = image.copy()
    overlay = image.copy()
    for spot_id, pts in polygons.items():
        arr = np.array(pts, np.int32)
        color = COLORS[readings[spot_id].value]
        cv2.fillPoly(overlay, [arr], color)
        cv2.polylines(out, [arr], True, color, 2)
    out = cv2.addWeighted(overlay, 0.35, out, 0.65, 0)
    for det in detections:
        cv2.rectangle(out, (int(det.x1), int(det.y1)), (int(det.x2), int(det.y2)), (255, 255, 255), 1)
        gx, gy = det.ground_point()
        cv2.circle(out, (int(gx), int(gy)), 6, (0, 255, 255), -1)
    for spot_id, pts in polygons.items():
        cx, cy = np.array(pts).mean(axis=0).astype(int)
        text = f"{spot_id}: {readings[spot_id].value}"
        cv2.putText(out, text, (int(cx) - 45, int(cy) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(out, text, (int(cx) - 45, int(cy) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return out


def main():
    parser = argparse.ArgumentParser(description="Check a frame's quality and spot readings.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--camera", required=True)
    parser.add_argument("--image", help="saved frame to check (default: grab a live one)")
    parser.add_argument("--detect", action="store_true", help="also run YOLO and save an annotated picture")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    config = load_config(args.config)
    camera = next((c for c in config.cameras if c.name == args.camera), None)
    if camera is None:
        parser.error(f"no camera named {args.camera!r} in {args.config}")

    if args.image:
        image = cv2.imread(args.image)
        if image is None:
            parser.error(f"couldn't read {args.image}")
        out_path = Path(args.image).with_name(Path(args.image).stem + "-check.jpg")
    else:
        from spotwatch.sources import make_source

        source = make_source(camera, config)
        print("Grabbing a live frame...")
        frame = source.read(datetime.now())
        source.close()
        if not frame.good:
            raise SystemExit(f"couldn't get a frame: {frame.reason}")
        image = frame.image
        out_path = Path(f"check-{camera.name}-{datetime.now():%Y%m%d-%H%M%S}.jpg")

    ref_file = reference_path(config.reference_dir, camera.name)
    reference = Reference.load(ref_file, camera.polygons) if ref_file.exists() else None
    result = check_frame(image, config.quality_settings(), reference)

    q = config.quality_settings()
    print(f"\n{camera.name}: {'GOOD' if result.good else 'BAD'} frame" + (f" ({result.reason})" if result.reason else ""))
    print(f"  brightness   {result.brightness:6.1f}   (bad below {q.min_brightness:.0f})")
    print(f"  sharpness    {result.sharpness:6.1f}   (bad below {q.min_sharpness:.0f})")
    print(f"  glare        {result.glare:6.1%}   (bad above {q.max_glare:.0%})")
    if result.shift_px is not None:
        limit = max(q.max_shift_px, q.max_align_shift_px)
        print(f"  camera shift {result.shift_px:6.1f}px (bad above {limit:.0f}px, vs {ref_file})")
    elif result.good:
        pass
    for note in result.notes:
        print(f"  note: {note}")

    if not args.detect:
        return
    if not camera.polygons:
        print("\nNo spots drawn for this camera yet; run calibrate first.")
        return

    from spotwatch.detector import YoloDetector  # needs ultralytics
    from spotwatch.occupancy import filter_vehicles, read_spots

    polygons = align_polygons(camera.polygons, result.transform)

    detections = YoloDetector(config.model, min(config.weak_confidence, config.min_confidence),
                              use_masks=config.occupancy_mode == "masks").detect(image)
    readings = read_spots(
        detections, polygons, config.min_confidence, config.occlusion_overlap, config.ground_tolerance_px,
        weak_confidence=config.weak_confidence,
        use_masks=config.occupancy_mode == "masks",
    )
    print(f"\nSpots (cars found: {len(filter_vehicles(detections, config.min_confidence))}):")
    for spot_id in camera.polygons:
        note = "" if result.good else "   <- live system would say unknown (bad frame)"
        print(f"  {spot_id}: {readings[spot_id].value}{note}")
    cv2.imwrite(str(out_path), draw_readings(image, polygons, readings, detections))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
