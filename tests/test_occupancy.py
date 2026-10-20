import pytest

from spotwatch.occupancy import Detection, box_covers_fraction, point_in_polygon, read_spot, read_spots
from spotwatch.states import SpotState

# Two spots along the curb. "near" is lower in the frame (closer to the window),
# "far" is just above it, the way spots look from a second-floor window.
NEAR = [(100, 400), (300, 400), (300, 480), (100, 480)]
FAR = [(100, 330), (300, 330), (300, 390), (100, 390)]


def car(x1, y1, x2, y2, label="car", conf=0.9):
    return Detection(label, conf, x1, y1, x2, y2)


def test_point_in_polygon():
    assert point_in_polygon((200, 440), NEAR)
    assert point_in_polygon((100, 400), NEAR)  # on the edge counts
    assert not point_in_polygon((50, 440), NEAR)


def test_empty_spot_is_free():
    assert read_spot([], NEAR) == SpotState.FREE


def test_car_standing_in_spot_is_occupied():
    assert read_spot([car(120, 380, 280, 460)], NEAR) == SpotState.OCCUPIED


def test_near_car_box_spilling_over_far_spot_does_not_make_it_occupied():
    # Car parked in NEAR. Its box reaches up over FAR, but its tires are in NEAR.
    near_car = car(110, 300, 290, 470)
    assert read_spot([near_car], NEAR) == SpotState.OCCUPIED
    # FAR is mostly hidden behind it, so we don't know, rather than "occupied"
    assert read_spot([near_car], FAR) == SpotState.UNKNOWN


def test_small_overlap_does_not_hide_spot():
    # a car in NEAR whose box only nicks the bottom of FAR
    assert read_spot([car(110, 385, 290, 470)], FAR) == SpotState.FREE


def test_double_parked_truck_makes_spot_unknown():
    # Truck in the travel lane: box covers NEAR, but its tires are below it.
    truck = car(80, 300, 320, 520, label="truck")
    assert read_spot([truck], NEAR) == SpotState.UNKNOWN


def test_occupied_wins_over_occlusion():
    truck = car(80, 300, 320, 520, label="truck")
    parked = car(120, 380, 280, 460)
    assert read_spot([truck, parked], NEAR) == SpotState.OCCUPIED


def test_box_covers_fraction():
    assert box_covers_fraction(car(0, 0, 1000, 1000), NEAR) == pytest.approx(1.0)
    assert box_covers_fraction(car(0, 0, 50, 50), NEAR) == 0.0
    assert box_covers_fraction(car(100, 400, 200, 480), NEAR) == pytest.approx(0.5, abs=0.02)


def test_read_spots_filters_non_vehicles_and_low_confidence():
    dets = [
        car(120, 380, 280, 460, label="person"),
        car(120, 310, 280, 380, conf=0.1),
    ]
    result = read_spots(dets, {"near": NEAR, "far": FAR}, min_confidence=0.35)
    assert result == {"near": SpotState.FREE, "far": SpotState.FREE}


def test_weak_box_on_a_spot_makes_it_unknown_not_free():
    # at dusk a parked car's score dipped to 0.3: not a car, but not proof of empty either
    dets = [car(120, 380, 280, 460, conf=0.3)]  # tires in NEAR
    result = read_spots(dets, {"near": NEAR, "far": FAR}, min_confidence=0.35)
    assert result == {"near": SpotState.UNKNOWN, "far": SpotState.FREE}


def test_read_spots_multiple_spots():
    dets = [car(120, 300, 280, 380)]  # tires in FAR, box doesn't touch NEAR
    result = read_spots(dets, {"near": NEAR, "far": FAR})
    assert result == {"near": SpotState.FREE, "far": SpotState.OCCUPIED}


def test_ground_point_just_outside_counts_within_tolerance():
    # tires land 4 px below the drawn outline (near-side car on the curb line)
    near_car = car(120, 380, 280, 484)
    assert read_spots([near_car], {"near": NEAR}, tolerance_px=10) == {"near": SpotState.OCCUPIED}
    assert read_spots([near_car], {"near": NEAR}, tolerance_px=0)["near"] != SpotState.OCCUPIED


def test_far_outside_tolerance_is_not_occupied():
    far_below = car(120, 380, 280, 520)  # 40 px below
    assert read_spots([far_below], {"near": NEAR}, tolerance_px=10)["near"] != SpotState.OCCUPIED


def test_one_car_never_occupies_two_spots():
    left = [(100, 400), (200, 400), (200, 480), (100, 480)]
    right = [(200, 400), (300, 400), (300, 480), (200, 480)]
    # ground point at x=203, inside "right", within 10 px of "left"
    straddler = car(150, 380, 256, 450)
    result = read_spots([straddler], {"left": left, "right": right}, tolerance_px=10)
    assert result["right"] == SpotState.OCCUPIED
    assert result["left"] != SpotState.OCCUPIED


def test_distance_to_polygon():
    from spotwatch.occupancy import distance_to_polygon

    assert distance_to_polygon((200, 440), NEAR) == 0
    assert distance_to_polygon((200, 490), NEAR) == pytest.approx(10)


def test_ground_point_is_lowest_mask_point_for_an_angled_car():
    # box bottom is set by the front tire on the right; the car's middle is higher
    mask = [(100, 400), (500, 420), (520, 480), (480, 500), (120, 460)]
    det = Detection("car", 0.9, 100, 400, 520, 500, mask_polygon=mask)
    assert det.ground_point() == (480.0, 500.0)


def test_ground_point_falls_back_to_box_without_a_usable_mask():
    det = Detection("car", 0.9, 100, 400, 500, 500, mask_polygon=[(1, 2), (3, float("nan")), (5, 6)])
    assert det.ground_point() == (300.0, 500.0)
    assert Detection("car", 0.9, 100, 400, 500, 500).ground_point() == (300.0, 500.0)
