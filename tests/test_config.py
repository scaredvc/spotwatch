from datetime import time
from pathlib import Path

import pytest

from spotwatch.config import ActiveHours, ConfigError, load_config, parse_config

REPO = Path(__file__).resolve().parent.parent


def base():
    return {
        "spots": [{"id": "s1", "name": "Front"}],
        "cameras": [
            {
                "name": "cam",
                "source": "image_folder",
                "path": "x",
                "polygons": {"s1": [[0, 0], [10, 0], [10, 10]]},
            }
        ],
    }


def test_example_config_loads():
    config = load_config(REPO / "tests" / "fixtures" / "sample_config.yaml")
    assert set(config.spots) == {"s1", "s2", "s3"}
    assert [c.name for c in config.cameras] == ["window-left", "window-right"]
    # s2 is seen by both cameras
    assert [c.name for c in config.cameras_for_spot("s2")] == ["window-left", "window-right"]
    assert config.cameras[0].options["path"] == "samples/window-left"


def test_defaults():
    config = parse_config(base())
    assert config.free_frames == 1
    assert config.cameras[0].interval_seconds == 60
    assert config.cameras[0].active_hours is None
    assert config.occupancy_mode == "boxes"


def test_mask_mode_and_invalid_mode():
    data = base()
    data["settings"] = {"occupancy_mode": "masks", "model": "yolo11n-seg.pt"}
    assert parse_config(data).occupancy_mode == "masks"
    data["settings"]["occupancy_mode"] = "mask"
    with pytest.raises(ConfigError, match="occupancy_mode"):
        parse_config(data)


def test_polygon_for_unknown_spot_rejected():
    data = base()
    data["cameras"][0]["polygons"] = {"nope": [[0, 0], [1, 0], [1, 1]]}
    with pytest.raises(ConfigError):
        parse_config(data)


def test_unknown_source_rejected():
    data = base()
    data["cameras"][0]["source"] = "webcam"
    with pytest.raises(ConfigError):
        parse_config(data)


def test_settings_override():
    data = base()
    data["settings"] = {"free_frames": 2, "occlusion_overlap": 0.5}
    config = parse_config(data)
    assert config.free_frames == 2
    assert config.occlusion_overlap == 0.5


def test_taken_notifications_require_boolean():
    data = base()
    data["settings"] = {"notify_taken": True}
    assert parse_config(data).notify_taken
    data["settings"]["notify_taken"] = "false"
    with pytest.raises(ConfigError, match="boolean"):
        parse_config(data)


def test_active_hours():
    day = ActiveHours(time(7), time(20))
    assert day.contains(time(12))
    assert not day.contains(time(21))
    assert not day.contains(time(20))

    overnight = ActiveHours(time(22), time(6))
    assert overnight.contains(time(23))
    assert overnight.contains(time(2))
    assert not overnight.contains(time(12))


def test_unquoted_yaml_time():
    import yaml

    data = base()
    data["cameras"][0]["active_hours"] = yaml.safe_load("{start: 07:00, end: 20:30}")
    hours = parse_config(data).cameras[0].active_hours
    assert hours.start == time(7, 0)
    assert hours.end == time(20, 30)


def test_blink_camera_needs_blink_camera_name():
    data = base()
    data["cameras"][0]["source"] = "blink"
    with pytest.raises(ConfigError):
        parse_config(data)
    data["cameras"][0]["blink_camera"] = "Front Window"
    assert parse_config(data).cameras[0].options["blink_camera"] == "Front Window"


def test_example_config_loads_and_uses_only_real_settings():
    import yaml

    path = REPO / "config.example.yaml"
    config = load_config(path)
    assert set(config.cameras[0].polygons) == set(config.spots)
    # parse_config ignores unknown setting names, so a typo in the example would go unnoticed
    for key in yaml.safe_load(path.read_text())["settings"]:
        assert hasattr(config, key), f"config.example.yaml has an unknown setting: {key}"
