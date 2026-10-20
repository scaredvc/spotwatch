"""Evaluate saved photos without camera access, database writes or notifications.

    python -m spotwatch.replay --config config.segmentation.yaml \
        --camera front-camera --images runs/front-camera --pattern '20261016*.jpg'
"""

import argparse
import hashlib
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from time import perf_counter

import cv2

from spotwatch.config import Camera, Config, load_config
from spotwatch.detector import Detector, YoloDetector
from spotwatch.occupancy import read_spots
from spotwatch.quality import Reference, align_polygons, check_frame, reference_path


def replay(config: Config, camera: Camera, paths: list[Path], detector: Detector) -> list[dict]:
    """Return per-frame readings; sparse saved photos cannot reproduce live timing."""
    ref_path = reference_path(config.reference_dir, camera.name)
    reference = Reference.load(ref_path, camera.polygons) if ref_path.exists() else None
    rows = []
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Could not read image: {path}")
        quality = check_frame(image, config.quality_settings(), reference)
        readings = {spot: "unknown" for spot in camera.polygons}
        detections = []
        elapsed = None
        if quality.good:
            start = perf_counter()
            detections = detector.detect(image)
            elapsed = (perf_counter() - start) * 1000
            readings = {
                spot: state.value for spot, state in read_spots(
                    detections, align_polygons(camera.polygons, quality.transform), config.min_confidence,
                    config.occlusion_overlap, config.ground_tolerance_px,
                    weak_confidence=config.weak_confidence,
                    use_masks=config.occupancy_mode == "masks",
                ).items()
            }
        rows.append({"file": str(path), "good": quality.good, "quality": asdict(quality),
                     "readings": readings, "predict_ms": elapsed,
                     "detections": [asdict(d) for d in detections]})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--camera", required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--pattern", action="append", help="filename glob; repeatable (default *.jpg)")
    parser.add_argument("--out", type=Path, help="JSON report (default: timestamped file in runs/replays)")
    args = parser.parse_args()
    config = load_config(args.config)
    camera = next((c for c in config.cameras if c.name == args.camera), None)
    if camera is None:
        parser.error(f"Unknown camera: {args.camera}")
    if not args.images.is_dir():
        parser.error(f"Image directory does not exist: {args.images}")
    paths = sorted({p for pattern in args.pattern or ["*.jpg"] for p in args.images.glob(pattern) if p.is_file()})
    if not paths:
        parser.error("No matching photos")
    output = args.out or Path("runs/replays") / f"{datetime.now():%Y%m%d-%H%M%S-%f}-{config.occupancy_mode}.json"
    if output.suffix.lower() != ".json" or output.resolve() in {p.resolve() for p in paths} | {Path(args.config).resolve()}:
        parser.error("Output must be a separate .json report")
    detector = YoloDetector(config.model, min(config.weak_confidence, config.min_confidence),
                            use_masks=config.occupancy_mode == "masks")
    rows = replay(config, camera, paths, detector)
    report = {"model": config.model, "occupancy_mode": config.occupancy_mode,
              "config_sha256": hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
              "camera": camera.name, "polygons": camera.polygons,
              "reference": str(reference_path(config.reference_dir, camera.name)), "rows": rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for row in rows:
        values = " ".join(f"{s}:{v}" for s, v in row["readings"].items())
        print(f"{Path(row['file']).name}: {values}" + (" [quality rejected]" if not row["good"] else ""))
    print(f"Saved {len(rows)} frame results to {output}; {sum(not r['good'] for r in rows)} rejected by quality checks.")


if __name__ == "__main__":
    main()
