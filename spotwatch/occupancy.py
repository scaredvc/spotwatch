"""Decide occupied / free / unknown for each spot from vehicle detections.

Occupied test: the vehicle's ground-contact point (bottom-center of its box)
is inside the spot polygon. We don't use plain box overlap, because from a
second-floor window a car's box spills up over the spots behind it.

Occlusion: if no vehicle stands in the spot but some vehicle box covers a big
enough share of the polygon, we can't see the spot, so it reads unknown.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from spotwatch.states import SpotState

VEHICLE_CLASSES = {"car", "truck", "motorcycle", "bus"}


@dataclass
class Detection:
    label: str
    confidence: float
    # box corners in pixels
    x1: float
    y1: float
    x2: float
    y2: float
    mask_polygon: list[tuple[float, float]] | None = None
    edge_clipped: bool = False

    def ground_point(self) -> tuple[float, float]:
        """Where the tires meet the road.

        With a segmentation outline: its lowest point, the tire nearest the
        camera. Without one: middle of the box's bottom edge. For a car parked
        at an angle the box bottom is set by one tire, so its middle lands on
        empty road below the car, outside a spot the car is clearly in.
        """
        if self.mask_polygon and len(self.mask_polygon) >= 3:
            pts = np.asarray(self.mask_polygon, dtype=np.float64)
            if np.isfinite(pts).all():
                x, y = pts[pts[:, 1].argmax()]
                return (float(x), float(y))
        return ((self.x1 + self.x2) / 2, self.y2)


def point_in_polygon(point: tuple[float, float], polygon: list[tuple[int, int]]) -> bool:
    contour = np.array(polygon, dtype=np.float32).reshape(-1, 1, 2)
    # >= 0 means inside or on the edge
    return cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), False) >= 0


def box_covers_fraction(det: Detection, polygon: list[tuple[int, int]]) -> float:
    """Share of the polygon's area that falls inside the detection box (0 to 1)."""
    pts = np.array(polygon, dtype=np.int32)
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    width = int(x_max - x_min) + 1
    height = int(y_max - y_min) + 1

    # draw the polygon on a small mask that just fits it
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [pts - [x_min, y_min]], 1)
    polygon_area = int(mask.sum())
    if polygon_area == 0:
        return 0.0

    # clip the box into the mask's coordinates
    bx1 = int(max(det.x1 - x_min, 0))
    by1 = int(max(det.y1 - y_min, 0))
    bx2 = int(min(det.x2 - x_min, width))
    by2 = int(min(det.y2 - y_min, height))
    if bx2 <= bx1 or by2 <= by1:
        return 0.0

    covered = int(mask[by1:by2, bx1:bx2].sum())
    return covered / polygon_area


def read_spot(
    detections: list[Detection],
    polygon: list[tuple[int, int]],
    occlusion_overlap: float = 0.4,
) -> SpotState:
    for det in detections:
        if point_in_polygon(det.ground_point(), polygon):
            return SpotState.OCCUPIED

    for det in detections:
        if box_covers_fraction(det, polygon) >= occlusion_overlap:
            return SpotState.UNKNOWN

    return SpotState.FREE


def filter_vehicles(detections: list[Detection], min_confidence: float) -> list[Detection]:
    return [d for d in detections if d.label in VEHICLE_CLASSES and d.confidence >= min_confidence]


def distance_to_polygon(point: tuple[float, float], polygon: list[tuple[int, int]]) -> float:
    """0 if the point is inside, else how many pixels it is outside the edge."""
    contour = np.array(polygon, dtype=np.float32).reshape(-1, 1, 2)
    signed = cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), True)
    return max(0.0, -signed)


def assign_vehicles(
    vehicles: list[Detection],
    polygons: dict[str, list[tuple[int, int]]],
    tolerance_px: float = 10.0,
) -> dict[str, list[Detection]]:
    """Which spot is each vehicle parked in? At most one spot per vehicle.

    A vehicle whose ground point is inside a polygon goes to that spot. If it's
    inside none, it goes to the nearest polygon within tolerance_px. That covers
    near-side cars whose ground point lands right on the curb line, a few
    pixels below a hand-drawn outline.
    """
    parked: dict[str, list[Detection]] = {spot_id: [] for spot_id in polygons}
    for det in vehicles:
        point = det.ground_point()
        distances = {spot_id: distance_to_polygon(point, poly) for spot_id, poly in polygons.items()}
        if not distances:
            continue
        nearest = min(distances, key=distances.get)
        if distances[nearest] <= tolerance_px:
            parked[nearest].append(det)
    return parked


def read_spots(
    detections: list[Detection],
    polygons: dict[str, list[tuple[int, int]]],
    min_confidence: float = 0.35,
    occlusion_overlap: float = 0.4,
    tolerance_px: float = 10.0,
    weak_confidence: float = 0.15,
    use_masks: bool = False,
) -> dict[str, SpotState]:
    vehicles = filter_vehicles(detections, min_confidence)
    # A truncated box's bottom-center is not a reliable ground contact point.
    assignable = [d for d in vehicles if not (use_masks and d.edge_clipped)]
    parked = assign_vehicles(assignable, polygons, tolerance_px)
    uncertain = assign_vehicles(
        [d for d in vehicles if use_masks and d.edge_clipped], polygons, tolerance_px,
    )
    # Boxes too weak to count as a car, but too strong to ignore. At dusk a
    # parked car's score can dip below min_confidence for a frame. That should
    # not read as "free", so a weak box sitting on a spot makes it unknown.
    weak = [d for d in filter_vehicles(detections, weak_confidence) if d.confidence < min_confidence]
    result = {}
    for spot_id, poly in polygons.items():
        if parked[spot_id]:
            result[spot_id] = SpotState.OCCUPIED
        elif uncertain[spot_id]:
            # A small visible fragment can have low overlap but still be in
            # this spot. Truncation invalidates occupied, not evidence of a car.
            result[spot_id] = SpotState.UNKNOWN
        elif any(occlusion_fraction(det, poly, use_masks) >= occlusion_overlap for det in vehicles):
            result[spot_id] = SpotState.UNKNOWN
        elif any(weak_box_on_spot(det, poly, occlusion_overlap) for det in weak):
            result[spot_id] = SpotState.UNKNOWN
        else:
            result[spot_id] = SpotState.FREE
    return result


def occlusion_fraction(det: Detection, polygon: list[tuple[int, int]], use_masks: bool) -> float:
    """Use precise silhouettes only when the vehicle is strong and complete.

    Invalid/missing masks and clipped vehicles retain the conservative box check.
    The caller keeps weak detections on the separate weak-box path.
    """
    if not use_masks or det.edge_clipped or det.mask_polygon is None:
        return box_covers_fraction(det, polygon)
    contour = np.asarray(det.mask_polygon, dtype=np.float32)
    if (contour.ndim != 2 or contour.shape[1] != 2 or len(contour) < 3
            or not np.isfinite(contour).all() or abs(cv2.contourArea(contour)) < 1):
        return box_covers_fraction(det, polygon)
    pts = np.asarray(polygon, dtype=np.int32)
    origin = pts.min(axis=0)
    width, height = pts.max(axis=0) - origin + 1
    spot_mask = np.zeros((int(height), int(width)), np.uint8)
    vehicle_mask = np.zeros_like(spot_mask)
    cv2.fillPoly(spot_mask, [pts - origin], 1)
    cv2.fillPoly(vehicle_mask, [np.rint(contour - origin).astype(np.int32)], 1)
    area = int(spot_mask.sum())
    return float((spot_mask & vehicle_mask).sum() / area) if area else 0.0


def weak_box_on_spot(det: Detection, polygon: list[tuple[int, int]], occlusion_overlap: float) -> bool:
    return point_in_polygon(det.ground_point(), polygon) or box_covers_fraction(det, polygon) >= occlusion_overlap
