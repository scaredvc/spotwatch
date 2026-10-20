import pytest

from spotwatch.occupancy import Detection, read_spots
from spotwatch.states import SpotState

NEAR = [(100, 400), (300, 400), (300, 480), (100, 480)]
FAR = [(100, 330), (300, 330), (300, 390), (100, 390)]
BODY = [(110, 400), (290, 400), (290, 470), (110, 470)]


def vehicle(mask=BODY, confidence=0.9, clipped=False):
    return Detection("car", confidence, 110, 300, 290, 470,
                     mask_polygon=mask, edge_clipped=clipped)


def test_mask_recovers_visible_far_spot_without_changing_occupied_near_spot():
    readings = read_spots([vehicle()], {"near": NEAR, "far": FAR}, use_masks=True)
    assert readings == {"near": SpotState.OCCUPIED, "far": SpotState.FREE}
    assert read_spots([vehicle()], {"far": FAR})["far"] == SpotState.UNKNOWN


def test_actual_mask_obstruction_remains_unknown():
    mask = [(110, 300), (290, 300), (290, 470), (110, 470)]
    assert read_spots([vehicle(mask)], {"far": FAR}, use_masks=True)["far"] == SpotState.UNKNOWN


@pytest.mark.parametrize("mask", [None, [], [(1, 1)], [(1, 1), (2, 2), (3, 3)],
                                 [(float('nan'), 1), (2, 2), (3, 1)]])
def test_missing_or_invalid_mask_keeps_box_protection(mask):
    assert read_spots([vehicle(mask)], {"far": FAR}, use_masks=True)["far"] == SpotState.UNKNOWN


def test_clipped_vehicle_cannot_establish_occupied_or_remove_box_protection():
    readings = read_spots([vehicle(clipped=True)], {"near": NEAR, "far": FAR}, use_masks=True)
    assert readings == {"near": SpotState.UNKNOWN, "far": SpotState.UNKNOWN}


@pytest.mark.parametrize("bottom", [60, 105])
def test_small_clipped_vehicle_in_or_near_spot_is_unknown(bottom):
    # Even tiny box coverage is evidence against free if the truncated ground
    # point is in the spot (or within the normal assignment tolerance).
    spot = [(0, 0), (100, 0), (100, 100), (0, 100)]
    clipped = Detection("car", 0.9, 0, bottom - 20, 20, bottom, edge_clipped=True)
    assert read_spots([clipped], {"s": spot}, use_masks=True)["s"] == SpotState.UNKNOWN


def test_weak_vehicle_retains_box_protection_even_with_a_small_mask():
    assert read_spots([vehicle(confidence=0.2)], {"far": FAR}, use_masks=True)["far"] == SpotState.UNKNOWN


def test_empty_detection_list_still_reports_free():
    assert read_spots([], {"far": FAR}, use_masks=True)["far"] == SpotState.FREE


def test_saved_camera_regressions():
    import json
    from pathlib import Path
    from spotwatch.config import load_config

    fixture = json.loads((Path(__file__).parent / "fixtures/segmentation_cases.json").read_text())
    candidate = load_config(Path(__file__).resolve().parent.parent / "config.segmentation.yaml")
    for case in fixture["cases"]:
        readings = read_spots([Detection(**d) for d in case["detections"]],
                              candidate.cameras[0].polygons, use_masks=True)
        for spot, expected in case["expected"].items():
            assert readings[spot].value == expected, (case["frame"], spot, readings)
