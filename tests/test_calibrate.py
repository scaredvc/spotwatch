import shutil
from pathlib import Path

import numpy as np
import pytest

from spotwatch.calibrate import Session, draw, dump_doc, load_doc, remove_polygon, save_doc, set_polygon
from spotwatch.config import load_config

REPO = Path(__file__).resolve().parent.parent
SQUARE = [(100, 400), (300, 400), (300, 480), (100, 480)]


@pytest.fixture
def config_path(tmp_path):
    path = tmp_path / "config.yaml"
    shutil.copy(REPO / "tests" / "fixtures" / "blank_config.yaml", path)
    return path


def test_set_polygon_on_empty_placeholder_and_keep_comments(config_path):
    doc = load_doc(config_path)
    set_polygon(doc, "side-camera", "s1", SQUARE)
    save_doc(doc, config_path)

    text = config_path.read_text()
    assert "# exact name in the Blink app" in text
    assert "# N: free readings in a row" in text
    assert "s1: [[100, 400], [300, 400], [300, 480], [100, 480]]" in text
    # the other camera is untouched
    assert text.count("polygons: {}") == 1

    config = load_config(config_path)
    assert config.cameras[0].polygons["s1"] == SQUARE
    assert config.cameras[1].polygons == {}


def test_replacing_a_polygon(config_path):
    doc = load_doc(config_path)
    set_polygon(doc, "side-camera", "s1", SQUARE)
    set_polygon(doc, "side-camera", "s1", [(0, 0), (10, 0), (10, 10)])
    save_doc(doc, config_path)
    assert load_config(config_path).cameras[0].polygons["s1"] == [(0, 0), (10, 0), (10, 10)]


def test_new_spot_is_created(config_path):
    doc = load_doc(config_path)
    set_polygon(doc, "side-camera", "s8", SQUARE, new_spot_name="By the driveway")
    save_doc(doc, config_path)
    config = load_config(config_path)
    assert config.spots["s8"].name == "By the driveway"
    assert "s8" in config.cameras[0].polygons


def test_same_spot_on_two_cameras(config_path):
    doc = load_doc(config_path)
    set_polygon(doc, "side-camera", "s4", SQUARE)
    set_polygon(doc, "front-camera", "s4", [(50, 420), (220, 420), (230, 480)])
    save_doc(doc, config_path)
    config = load_config(config_path)
    assert [c.name for c in config.cameras_for_spot("s4")] == ["side-camera", "front-camera"]


def test_remove_polygon(config_path):
    doc = load_doc(config_path)
    set_polygon(doc, "side-camera", "s1", SQUARE)
    assert remove_polygon(doc, "side-camera", "s1")
    assert not remove_polygon(doc, "side-camera", "s1")
    save_doc(doc, config_path)
    assert load_config(config_path).cameras[0].polygons == {}


def test_too_few_points_rejected(config_path):
    doc = load_doc(config_path)
    with pytest.raises(ValueError):
        set_polygon(doc, "side-camera", "s1", [(0, 0), (1, 1)])


def test_unknown_camera_rejected(config_path):
    doc = load_doc(config_path)
    with pytest.raises(KeyError):
        set_polygon(doc, "nope", "s1", SQUARE)


def test_session_click_undo_finish(config_path):
    session = Session(load_doc(config_path), "side-camera")
    for x, y in SQUARE + [(999, 999)]:
        session.click(x, y)
    session.undo()  # drop the misclick
    session.finish("s2")
    assert session.current == []
    assert session.unsaved
    assert session.polygons() == {"s2": SQUARE}


def test_draw_does_not_change_the_input_image():
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    out = draw(image, {"s1": SQUARE}, [(10, 10), (50, 50)], "note")
    assert image.sum() == 0
    assert out.shape == image.shape
    assert out.sum() > 0


def test_dump_is_stable_without_changes(config_path):
    # Loading and saving without edits shouldn't reshuffle the file.
    original = config_path.read_text()
    assert dump_doc(load_doc(config_path)) == original


def test_save_reference_writes_the_frame(config_path, tmp_path, monkeypatch):
    from spotwatch.calibrate import save_reference

    monkeypatch.chdir(tmp_path)
    image = np.full((72, 128, 3), 90, dtype=np.uint8)
    path = save_reference(load_doc(config_path), "front-camera", image)
    assert path.as_posix() == "calibration/front-camera.jpg"
    assert (tmp_path / path).exists()
