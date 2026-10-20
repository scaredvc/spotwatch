from datetime import datetime, timedelta

from spotwatch.fusion import fuse
from spotwatch.state_machine import SpotStateMachine
from spotwatch.states import CameraReading, SpotState

O = SpotState.OCCUPIED
F = SpotState.FREE
U = SpotState.UNKNOWN

NOW = datetime(2026, 10, 12, 12, 0, 0)
AGE = {"a": timedelta(minutes=3), "b": timedelta(minutes=3)}


def reading(camera, value, good=True, age_s=0):
    return CameraReading(camera, "s1", value, good, NOW - timedelta(seconds=age_s))


def test_no_readings_is_unknown():
    assert fuse([], NOW, AGE) == U


def test_single_camera_passes_through():
    assert fuse([reading("a", F)], NOW, AGE) == F
    assert fuse([reading("a", O)], NOW, AGE) == O


def test_two_cameras_disagree_occupied_wins():
    assert fuse([reading("a", F), reading("b", O)], NOW, AGE) == O


def test_two_cameras_both_free():
    assert fuse([reading("a", F), reading("b", F)], NOW, AGE) == F


def test_bad_frame_is_ignored():
    assert fuse([reading("a", F), reading("b", O, good=False)], NOW, AGE) == F


def test_stale_reading_is_ignored():
    assert fuse([reading("a", F), reading("b", O, age_s=600)], NOW, AGE) == F


def test_all_bad_is_unknown():
    assert fuse([reading("a", F, good=False), reading("b", F, age_s=600)], NOW, AGE) == U


def test_blocked_view_is_ignored_when_other_camera_sees_free():
    # truck blocks spot from camera a, camera b sees it empty
    assert fuse([reading("a", U), reading("b", F)], NOW, AGE) == F


def test_blocked_view_and_occupied_view_is_occupied():
    assert fuse([reading("a", U), reading("b", O)], NOW, AGE) == O


def test_blocked_on_every_camera_is_unknown():
    assert fuse([reading("a", U), reading("b", U)], NOW, AGE) == U


def test_blocked_on_one_camera_and_other_camera_down_is_unknown():
    assert fuse([reading("a", U), reading("b", F, good=False)], NOW, AGE) == U


def test_two_cameras_disagreeing_through_state_machine():
    m = SpotStateMachine("s1", occupied_frames=1)
    alerts = 0
    # a says free the whole time, b sees a car for a while, then it leaves
    for b_value in [O, O, O, F, F]:
        fused = fuse([reading("a", F), reading("b", b_value)], NOW, AGE)
        alerts += m.update(fused).notify
    assert alerts == 1
    assert m.state == F
