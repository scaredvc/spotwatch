"""Is this frame trustworthy enough to read parking spots from?

Checks, cheapest first:
- too dark      mean brightness below min_brightness (night behind glass)
- too blurry    sharpness (variance of the Laplacian) below min_sharpness
- glare         share of blown-out pixels above max_glare
- camera moved  the fixed background (buildings, poles) has shifted versus the
                reference frame saved at calibration, by more than max_shift_px.
                With max_align_shift_px set, shifts up to that size are not
                rejected: the measured transform is returned so the caller can
                move the spot outlines to match the frame (align_polygons).

A frame that fails any check is a bad frame: that camera's spots read
"unknown", which never triggers an alert. Better a missed minute than a false
"spot is free".

The default thresholds come from real frames of one Blink XT2 behind a window;
tune them for your camera.
"""

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

BLOWN_LEVEL = 250  # gray value counted as blown out


@dataclass
class QualitySettings:
    min_brightness: float = 40.0
    min_sharpness: float = 30.0
    max_glare: float = 0.12  # share of the frame, 0 to 1
    max_shift_px: float = 10.0
    max_align_shift_px: float = 0.0  # 0 = never align, reject above max_shift_px


@dataclass
class QualityResult:
    good: bool
    reason: str = ""
    brightness: float = 0.0
    sharpness: float = 0.0
    glare: float = 0.0
    shift_px: float | None = None  # None = no reference, or couldn't measure
    # reference -> frame transform (2x3) when the spot outlines should be moved to match
    transform: list[list[float]] | None = None
    notes: list[str] = field(default_factory=list)


def gray(image: np.ndarray) -> np.ndarray:
    return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# ---------- camera moved ----------


class Reference:
    """The frame the spot polygons were drawn on, with its features precomputed.

    Given the spot polygons, movement is measured only on the street strip
    around them, and reported at the polygon corners. Measured over the whole
    frame, features near the top made the fit tilt: on Oct 17 it reported
    10-22px while the street around the spots moved under 5px.
    """

    MIN_INLIERS = 20
    # Features come from 100px above the highest spot (house fronts and the far
    # curb: fixed, textured) down to the lowest spot. Going further down only
    # adds pavement and parked cars, which crowd out the useful features.
    STRIP_ABOVE = 100

    def __init__(self, image: np.ndarray, polygons: dict | None = None):
        self.gray = gray(image)
        self.mask = None
        self.points = None
        if polygons:
            self.points = np.float32([pt for poly in polygons.values() for pt in poly])
            top = max(0, int(self.points[:, 1].min()) - self.STRIP_ABOVE)
            bottom = min(self.gray.shape[0], int(self.points[:, 1].max()))
            self.mask = np.zeros_like(self.gray)
            self.mask[top:bottom, :] = 255
        self.orb = cv2.ORB_create(4000 if polygons else 2000)  # the strip needs a denser sample
        self.keypoints, self.descriptors = self.orb.detectAndCompute(self.gray, self.mask)

    @classmethod
    def load(cls, path, polygons: dict | None = None) -> "Reference | None":
        image = cv2.imread(str(path))
        return cls(image, polygons) if image is not None else None

    def shift(self, image: np.ndarray) -> tuple[float | None, int]:
        """How far the view moved, in pixels (worst corner). Returns (shift, inliers)."""
        shift, inliers, _ = self.measure(image)
        return shift, inliers

    def measure(self, image: np.ndarray) -> tuple[float | None, int, np.ndarray | None]:
        """Like shift(), plus the fitted reference -> frame transform (2x3).

        Matches ORB features between the reference and this frame, then fits a
        rotation + shift + scale with RANSAC. RANSAC throws out matches that
        don't agree, like the ones on cars that came or went.
        """
        g = gray(image)
        if g.shape != self.gray.shape or self.descriptors is None:
            return None, 0, None
        keypoints, descriptors = self.orb.detectAndCompute(g, self.mask)
        if descriptors is None or len(keypoints) < self.MIN_INLIERS:
            return None, 0, None

        matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(self.descriptors, descriptors)
        matches = sorted(matches, key=lambda m: m.distance)[:300]
        if len(matches) < self.MIN_INLIERS:
            return None, len(matches), None

        ref_pts = np.float32([self.keypoints[m.queryIdx].pt for m in matches])
        new_pts = np.float32([keypoints[m.trainIdx].pt for m in matches])
        transform, inliers = cv2.estimateAffinePartial2D(ref_pts, new_pts, method=cv2.RANSAC, ransacReprojThreshold=4)
        n_inliers = int(inliers.sum()) if inliers is not None else 0
        if transform is None or n_inliers < self.MIN_INLIERS:
            return None, n_inliers, None

        h, w = g.shape
        if self.points is not None:
            corners = self.points
        else:
            corners = np.float32([[0, 0], [w, 0], [0, h], [w, h], [w / 2, h / 2]])
        moved = corners @ transform[:, :2].T + transform[:, 2] - corners
        return float(np.linalg.norm(moved, axis=1).max()), n_inliers, transform


def align_polygons(polygons: dict, transform) -> dict:
    """Move spot outlines drawn on the reference frame onto a shifted frame."""
    if transform is None:
        return polygons
    t = np.asarray(transform, dtype=np.float64)
    aligned = {}
    for spot_id, polygon in polygons.items():
        pts = np.asarray(polygon, dtype=np.float64) @ t[:, :2].T + t[:, 2]
        aligned[spot_id] = [(int(round(x)), int(round(y))) for x, y in pts]
    return aligned


def reference_path(reference_dir, camera_name: str) -> Path:
    return Path(reference_dir) / f"{camera_name}.jpg"


# ---------- the check ----------


def check_frame(image: np.ndarray, settings: QualitySettings, reference: Reference | None = None) -> QualityResult:
    g = gray(image)
    result = QualityResult(good=True)
    result.brightness = float(g.mean())
    result.sharpness = float(cv2.Laplacian(g, cv2.CV_64F).var())
    result.glare = float(np.mean(g >= BLOWN_LEVEL))

    if result.brightness < settings.min_brightness:
        return _bad(result, f"too dark (brightness {result.brightness:.0f} < {settings.min_brightness:.0f})")
    if result.sharpness < settings.min_sharpness:
        return _bad(result, f"too blurry or hazy (sharpness {result.sharpness:.0f} < {settings.min_sharpness:.0f})")
    if result.glare > settings.max_glare:
        return _bad(result, f"glare ({result.glare:.0%} blown out > {settings.max_glare:.0%})")

    if reference is None:
        result.notes.append("no reference frame, camera position not checked")
    else:
        shift, inliers, transform = reference.measure(image)
        result.shift_px = shift
        if shift is None:
            # Can't tell (too few matching features). Don't fail the frame for it.
            result.notes.append(f"couldn't measure camera position ({inliers} feature matches)")
        elif settings.max_align_shift_px > 0 and shift <= settings.max_align_shift_px:
            # The camera wobbles a few pixels between snapshots; follow it instead
            # of throwing the frame away.
            result.transform = transform.tolist()
            if shift > settings.max_shift_px:
                result.notes.append(f"camera shifted ~{shift:.0f}px; spot outlines aligned to the frame")
        elif shift > settings.max_shift_px:
            return _bad(result, f"camera moved ~{shift:.0f}px since calibration; rerun calibrate")
    return result


def _bad(result: QualityResult, reason: str) -> QualityResult:
    result.good = False
    result.reason = reason
    return result
