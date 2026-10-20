import sys
from types import SimpleNamespace

import numpy as np
import pytest

from spotwatch.detector import YoloDetector


def install_model(monkeypatch, task="segment", masks=True):
    box = SimpleNamespace(xyxy=np.array([[50, 20, 199, 90]]),
                          cls=np.array([2]), conf=np.array([0.8]))
    result = SimpleNamespace(boxes=[box], names={2: "car"},
                             masks=SimpleNamespace(xy=[np.array([[50, 20], [199, 20], [199, 90]])]) if masks else None)
    model = SimpleNamespace(task=task, names={2: "car"}, predict=lambda *a, **k: [result])
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=lambda path: model))


def test_converts_mask_in_image_coordinates_and_flags_clipped_box(monkeypatch):
    install_model(monkeypatch)
    detector = YoloDetector("fake.pt", use_masks=True)
    detection = detector.detect(np.zeros((100, 200, 3), np.uint8))[0]
    assert detection.mask_polygon == [(50.0, 20.0), (199.0, 20.0), (199.0, 90.0)]
    assert detection.edge_clipped


def test_mask_mode_requires_segmentation_model(monkeypatch):
    install_model(monkeypatch, task="detect")
    with pytest.raises(ValueError, match="segmentation"):
        YoloDetector("fake.pt", use_masks=True)


def test_missing_masks_preserves_vehicle_box(monkeypatch):
    install_model(monkeypatch, masks=False)
    detection = YoloDetector("fake.pt", use_masks=True).detect(np.zeros((100, 200, 3), np.uint8))[0]
    assert detection.mask_polygon is None
    assert detection.x1 == 50
