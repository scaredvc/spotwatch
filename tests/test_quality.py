import cv2
import numpy as np
import pytest

from spotwatch.quality import QualitySettings, Reference, align_polygons, check_frame

SETTINGS = QualitySettings()


def street(seed=1, w=640, h=360):
    """A fake scene with lots of fixed texture (like buildings and windows)."""
    rng = np.random.default_rng(seed)
    image = np.full((h, w, 3), 110, dtype=np.uint8)
    for _ in range(250):
        x, y = int(rng.integers(0, w - 40)), int(rng.integers(0, h - 40))
        cw, ch = int(rng.integers(8, 40)), int(rng.integers(8, 40))
        color = tuple(int(c) for c in rng.integers(20, 230, size=3))
        cv2.rectangle(image, (x, y), (x + cw, y + ch), color, -1)
    return image


def shifted(image, dx, dy):
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(image, m, (image.shape[1], image.shape[0]), borderMode=cv2.BORDER_REPLICATE)


def test_normal_frame_is_good():
    result = check_frame(street(), SETTINGS)
    assert result.good, result.reason
    assert "no reference" in result.notes[0]


def test_too_dark():
    dark = (street() * 0.2).astype(np.uint8)
    result = check_frame(dark, SETTINGS)
    assert not result.good and "too dark" in result.reason


def test_too_blurry():
    blurry = cv2.GaussianBlur(street(), (31, 31), 12)
    result = check_frame(blurry, SETTINGS)
    assert not result.good and "blurry" in result.reason


def test_glare():
    image = street()
    cv2.circle(image, (150, 100), 140, (255, 255, 255), -1)  # sun blob
    result = check_frame(image, SETTINGS)
    assert not result.good and "glare" in result.reason
    assert result.glare > SETTINGS.max_glare


def test_same_position_passes_camera_check():
    image = street()
    result = check_frame(image, SETTINGS, Reference(image))
    assert result.good, result.reason
    assert result.shift_px == pytest.approx(0, abs=1)


def test_small_jitter_is_tolerated():
    image = street()
    result = check_frame(shifted(image, 3, 2), SETTINGS, Reference(image))
    assert result.good, result.reason
    assert result.shift_px == pytest.approx(3.6, abs=1.5)


def test_camera_moved_is_bad_frame():
    image = street()
    result = check_frame(shifted(image, 25, 0), SETTINGS, Reference(image))
    assert not result.good
    assert "camera moved" in result.reason
    assert result.shift_px == pytest.approx(25, abs=2)


def test_moved_ignores_a_car_driving_through():
    image = street()
    later = image.copy()
    cv2.rectangle(later, (300, 250), (420, 320), (40, 40, 40), -1)  # a car appeared
    result = check_frame(later, SETTINGS, Reference(image))
    assert result.good, result.reason


ALIGN = QualitySettings(max_align_shift_px=25)


def test_moderate_shift_is_aligned_instead_of_rejected():
    image = street()
    result = check_frame(shifted(image, 15, -6), ALIGN, Reference(image))
    assert result.good, result.reason
    assert result.transform is not None
    assert "aligned" in result.notes[0]
    spot = {"1": [(100, 100), (200, 100), (200, 150), (100, 150)]}
    moved = align_polygons(spot, result.transform)["1"]
    for (x0, y0), (x1, y1) in zip(spot["1"], moved):
        assert x1 - x0 == pytest.approx(15, abs=2)
        assert y1 - y0 == pytest.approx(-6, abs=2)


def test_large_shift_is_still_rejected_with_alignment_on():
    image = street()
    result = check_frame(shifted(image, 40, 0), ALIGN, Reference(image))
    assert not result.good
    assert "camera moved" in result.reason
    assert result.transform is None


def test_alignment_off_by_default_keeps_outlines_unchanged():
    image = street()
    result = check_frame(shifted(image, 3, 2), SETTINGS, Reference(image))
    assert result.good and result.transform is None
    spot = {"1": [(1, 2), (3, 4), (5, 6)]}
    assert align_polygons(spot, result.transform) is spot



def test_movement_outside_the_spot_strip_is_ignored():
    image = street()
    later = image.copy()
    later[:120] = shifted(image, 20, 0)[:120]  # only the top of the view changed
    spots = {"1": [(100, 250), (300, 250), (300, 290), (100, 290)]}
    shift, _ = Reference(image, spots).shift(later)
    assert shift == pytest.approx(0, abs=2)


def test_strip_measurement_follows_a_real_move():
    image = street()
    spots = {"1": [(100, 250), (300, 250), (300, 290), (100, 290)]}
    shift, _ = Reference(image, spots).shift(shifted(image, 12, 0))
    assert shift == pytest.approx(12, abs=2)

def test_cannot_measure_is_not_a_failure():
    image = street()
    flat = np.full_like(image, 120)
    flat[::7, :] = 160  # some sharpness but no matchable features
    result = Reference(image).shift(flat)
    assert result[0] is None


def test_different_size_frame_cannot_be_compared():
    ref = Reference(street())
    assert ref.shift(street(w=320, h=180)) == (None, 0)


def test_runner_marks_spots_unknown_on_a_dark_frame():
    from datetime import datetime

    from spotwatch.config import parse_config
    from spotwatch.detector import ScriptedDetector
    from spotwatch.notifier import ListNotifier
    from spotwatch.runner import Runner
    from spotwatch.sources import Frame, FrameSource
    from spotwatch.storage import Store

    class OneFrame(FrameSource):
        def __init__(self, image):
            self.image = image

        def read(self, now):
            return Frame(image=self.image, time=now)

    config = parse_config(
        {
            "settings": {"reference_dir": "/nonexistent"},
            "spots": [{"id": "s1"}],
            "cameras": [
                {"name": "cam", "source": "image_folder", "path": "x", "polygons": {"s1": [[0, 0], [50, 0], [50, 50]]}}
            ],
        }
    )
    dark = (street() * 0.2).astype(np.uint8)
    store = Store(":memory:")
    notifier = ListNotifier()
    runner = Runner(config, {"cam": OneFrame(dark)}, ScriptedDetector([[]]), notifier, store)
    runner.tick(datetime(2026, 10, 14, 12, 0))
    assert runner.spot_states() == {"s1": "unknown"}  # an empty dark frame must not alert "free"
    assert notifier.sent == []
    row = store.conn.execute("SELECT good, reason FROM snapshots").fetchone()
    assert row["good"] == 0 and "too dark" in row["reason"]
