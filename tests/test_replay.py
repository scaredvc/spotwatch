from pathlib import Path

import cv2
import numpy as np
import pytest

from spotwatch.config import Camera, Config, Spot
from spotwatch.detector import ScriptedDetector
from spotwatch.replay import replay


def config():
    camera = Camera("test", "image_folder", polygons={"1": [(10, 10), (90, 10), (90, 90), (10, 90)]})
    return Config(spots={"1": Spot("1", "1")}, cameras=[camera],
                  min_brightness=0, min_sharpness=0, max_glare=1)


def test_replay_uses_quality_gate_and_never_infers_rejected_frame(tmp_path):
    path = tmp_path / "dark.jpg"
    cv2.imwrite(str(path), np.zeros((100, 100, 3), np.uint8))
    cfg = config()
    cfg.min_brightness = 40
    detector = ScriptedDetector([[]])
    rows = replay(cfg, cfg.cameras[0], [path], detector)
    assert rows[0]["readings"] == {"1": "unknown"}
    assert not rows[0]["good"]
    assert detector.calls == 0


def test_replay_empty_valid_frame_reports_free(tmp_path):
    path = tmp_path / "good.jpg"
    cv2.imwrite(str(path), np.full((100, 100, 3), 100, np.uint8))
    cfg = config()
    assert replay(cfg, cfg.cameras[0], [path], ScriptedDetector([[]]))[0]["readings"] == {"1": "free"}


def test_replay_unreadable_file_fails_instead_of_reporting_free(tmp_path):
    cfg = config()
    with pytest.raises(ValueError, match="read"):
        replay(cfg, cfg.cameras[0], [tmp_path / "missing.jpg"], ScriptedDetector([[]]))
