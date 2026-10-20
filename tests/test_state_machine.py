from spotwatch.state_machine import SpotStateMachine
from spotwatch.states import SpotState

O = SpotState.OCCUPIED
F = SpotState.FREE
U = SpotState.UNKNOWN


def run(machine, readings):
    """Feed readings in, return the list of reading indexes that notified."""
    alerts = []
    for i, r in enumerate(readings):
        if machine.update(r).notify:
            alerts.append(i)
    return alerts


def test_startup_is_unknown():
    m = SpotStateMachine("s1")
    assert m.state == U
    assert not m.alerted


def test_free_on_first_frame_alerts():
    m = SpotStateMachine("s1")
    update = m.update(F)
    assert update.notify
    assert update.changed
    assert m.state == F


def test_occupied_then_free_alerts_once():
    m = SpotStateMachine("s1", occupied_frames=2)
    assert run(m, [O, O, O, F, F, F]) == [3]


def test_no_repeat_alert_after_unknown_blip():
    m = SpotStateMachine("s1")
    # free, truck passes, free again
    assert run(m, [F, U, F, F]) == [0]


def test_no_repeat_alert_after_a_night_of_unknown():
    m = SpotStateMachine("s1")
    night = [U] * 600
    assert run(m, [F] + night + [F]) == [0]


def test_re_alert_only_after_confirmed_occupied():
    m = SpotStateMachine("s1", occupied_frames=2)
    # one occupied reading isn't confirmed, so no second alert
    assert run(m, [F, O, F]) == [0]
    # two in a row confirms it, so the next free alerts
    assert run(m, [O, O, F]) == [2]


def test_unknown_between_occupied_readings_still_confirms():
    m = SpotStateMachine("s1", occupied_frames=2)
    assert run(m, [F, O, U, O, F]) == [0, 4]


def test_flapping_does_not_spam():
    m = SpotStateMachine("s1", occupied_frames=2)
    assert run(m, [F, O, F, O, F, O, F]) == [0]


def test_bad_frames_do_not_alert_or_reset():
    m = SpotStateMachine("s1")
    assert run(m, [O, O, U, U, U]) == []
    assert m.state == U
    assert run(m, [F]) == [0]


def test_spot_never_free():
    m = SpotStateMachine("s1")
    assert run(m, [O, U, O, O, U, O] * 50) == []


def test_free_frames_greater_than_one():
    m = SpotStateMachine("s1", free_frames=2)
    update = m.update(F)
    assert not update.notify
    assert m.state == U  # not confirmed yet, keeps old state
    assert m.update(F).notify
    assert m.state == F


def test_occupied_breaks_free_streak():
    m = SpotStateMachine("s1", free_frames=3)
    assert run(m, [F, F, O, F, F, F]) == [5]


def test_unknown_pauses_free_streak():
    m = SpotStateMachine("s1", free_frames=2)
    assert run(m, [F, U, F]) == [2]


def test_alert_from_unknown_and_from_occupied_states():
    m = SpotStateMachine("s1", occupied_frames=1)
    assert run(m, [U, F]) == [1]
    assert run(m, [O, F]) == [1]


def test_state_change_reported():
    m = SpotStateMachine("s1", occupied_frames=1)
    u = m.update(O)
    assert (u.old_state, u.new_state, u.changed) == (U, O, True)
    u = m.update(O)
    assert not u.changed


def test_bad_config_rejected():
    import pytest

    with pytest.raises(ValueError):
        SpotStateMachine("s1", free_frames=0)
def test_taken_once_after_confirmed_occupancy_even_across_unknown():
    from spotwatch.state_machine import SpotStateMachine
    from spotwatch.states import SpotState as S
    m = SpotStateMachine("1")
    assert not m.update(S.OCCUPIED).notify_taken
    assert not m.update(S.OCCUPIED).notify_taken  # no startup occupied alert
    assert m.update(S.FREE).notify
    assert not m.update(S.OCCUPIED).notify_taken
    assert not m.update(S.UNKNOWN).notify_taken
    assert m.update(S.OCCUPIED).notify_taken
    assert not m.update(S.OCCUPIED).notify_taken
    assert m.update(S.FREE).notify
