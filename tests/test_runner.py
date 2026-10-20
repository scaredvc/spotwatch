from datetime import datetime, timedelta


from spotwatch.config import parse_config
from spotwatch.detector import Detector
from spotwatch.notifier import ListNotifier
from spotwatch.occupancy import Detection
from spotwatch.runner import Runner
from spotwatch.sources import Frame, FrameSource
from spotwatch.storage import Store

START = datetime(2026, 10, 12, 9, 0, 0)

SPOT = [[100, 400], [300, 400], [300, 480], [100, 480]]
PARKED = Detection("car", 0.9, 120, 380, 280, 460)  # tires inside SPOT
TRUCK = Detection("truck", 0.9, 80, 300, 320, 520)  # covers SPOT, tires outside


class ScriptSource(FrameSource):
    """Each read returns the next item: a list of detections, "bad", or "error"."""

    def __init__(self, script):
        self.script = list(script)
        self.reads = 0

    def read(self, now):
        item = self.script[self.reads] if self.reads < len(self.script) else "bad"
        self.reads += 1
        if item == "error":
            raise ConnectionError("camera offline")
        if item == "bad":
            return Frame(image=None, time=now, good=False, reason="bad frame")
        return Frame(image=FakeImage(item), time=now)


class FakeImage:
    """Stands in for a real image. Carries the detections the fake detector returns."""

    def __init__(self, detections):
        self.detections = detections


class PassThroughDetector(Detector):
    def detect(self, image):
        return image.detections


def make_config(cameras, **settings):
    return parse_config(
        {
            "settings": settings,
            "spots": [{"id": "s1", "name": "Front"}, {"id": "s2", "name": "Side"}],
            "cameras": cameras,
        }
    )


def camera(name, spots=("s1",), interval=60, hours=None):
    cam = {"name": name, "source": "image_folder", "path": "unused", "interval_seconds": interval}
    cam["polygons"] = {s: SPOT for s in spots}
    if hours:
        cam["active_hours"] = {"start": hours[0], "end": hours[1]}
    return cam


def build(config, scripts):
    sources = {name: ScriptSource(script) for name, script in scripts.items()}
    notifier = ListNotifier()
    store = Store(":memory:")
    runner = Runner(config, sources, PassThroughDetector(), notifier, store, check_quality=False)
    return runner, sources, notifier, store


def run_minutes(runner, minutes, start=START):
    for i in range(minutes):
        runner.tick(start + timedelta(minutes=i))


EMPTY = []  # good frame, no vehicles


def test_taken_alert_follows_free_after_two_confirmations():
    cfg = make_config([camera("a")], notify_taken=True)
    runner, _, notifier, store = build(cfg, {"a": [EMPTY, [PARKED], "bad", [PARKED], [PARKED]]})
    run_minutes(runner, 5)
    assert [a.state.value for a in notifier.alerts] == ["free", "occupied"]
    assert "Spot taken" in store.notifications()[1]["message"]
    store.close()


def test_runner_uses_opt_in_mask_occlusion():
    # Box hides SPOT, but the actual vehicle silhouette is below it.
    truck = Detection("truck", 0.9, 80, 300, 320, 520,
                      mask_polygon=[(80, 490), (320, 490), (320, 520), (80, 520)])
    cfg = make_config([camera("a")], occupancy_mode="masks")
    runner, _, notifier, store = build(cfg, {"a": [[truck]]})
    runner.tick(START)
    assert runner.spot_states()["s1"] == "free"
    assert len(notifier.sent) == 1
    store.close()


def test_single_camera_spot_frees_up_once():
    config = make_config([camera("a")], occupied_frames=2)
    runner, _, notifier, store = build(config, {"a": [[PARKED], [PARKED], EMPTY, EMPTY, [TRUCK], EMPTY]})
    run_minutes(runner, 6)

    assert [spot for spot, _ in notifier.sent] == ["s1"]
    assert notifier.sent[0][1] == START + timedelta(minutes=2)
    assert len(store.notifications("s1")) == 1
    states = [c["new_state"] for c in store.state_changes("s1")]
    assert states == ["occupied", "free", "unknown", "free"]


def test_ticks_respect_interval():
    config = make_config([camera("a", interval=60)])
    runner, sources, _, _ = build(config, {"a": [EMPTY] * 10})
    for s in range(0, 180, 10):  # tick every 10 seconds for 3 minutes
        runner.tick(START + timedelta(seconds=s))
    assert sources["a"].reads == 3


def test_camera_error_marks_spot_unknown_and_keeps_running():
    config = make_config([camera("a")], occupied_frames=1)
    runner, _, notifier, store = build(config, {"a": [[PARKED], "error", "bad", EMPTY]})
    run_minutes(runner, 4)
    assert len(notifier.sent) == 1
    states = [c["new_state"] for c in store.state_changes("s1")]
    assert states == ["occupied", "unknown", "free"]
    stats = store.snapshot_stats()
    assert stats == [{"camera": "a", "day": "2026-10-12", "total": 4, "good": 2}]


def test_outside_active_hours_does_not_poll_and_does_not_realert():
    config = make_config([camera("a", hours=("07:00", "20:00"))])
    runner, sources, notifier, store = build(config, {"a": [EMPTY, EMPTY, EMPTY]})

    evening = datetime(2026, 10, 12, 19, 59)
    runner.tick(evening)  # polls, spot is free, alert
    assert runner.spot_states()["s1"] == "free"

    for i in range(1, 60 * 11):  # through the night, one tick per minute
        runner.tick(evening + timedelta(minutes=i))
    assert sources["a"].reads == 1
    assert runner.spot_states()["s1"] == "unknown"

    runner.tick(datetime(2026, 10, 13, 7, 0))  # morning: still free
    assert sources["a"].reads == 2
    assert runner.spot_states()["s1"] == "free"
    assert len(notifier.sent) == 1  # no second alert after the night

    # the night only wrote one "outside active hours" reading, not hundreds
    rows = store.conn.execute("SELECT COUNT(*) FROM readings WHERE reason = 'outside active hours'").fetchone()
    assert rows[0] == 1


def test_two_cameras_on_one_spot():
    # a sees s1 free all along; b sees a car in s1 for 3 minutes
    config = make_config([camera("a"), camera("b")], occupied_frames=1)
    runner, _, notifier, _ = build(
        config,
        {"a": [EMPTY] * 6, "b": [[PARKED], [PARKED], [PARKED], EMPTY, EMPTY, EMPTY]},
    )
    run_minutes(runner, 3)
    assert notifier.sent == []
    run_minutes(runner, 3, start=START + timedelta(minutes=3))
    assert len(notifier.sent) == 1
    assert notifier.sent[0][1] == START + timedelta(minutes=3)


def test_down_camera_is_ignored_for_shared_spot():
    config = make_config([camera("a"), camera("b")], occupied_frames=1)
    runner, _, notifier, _ = build(
        config,
        {"a": [[PARKED], EMPTY, EMPTY], "b": [[PARKED], "error", "error"]},
    )
    run_minutes(runner, 3)
    # b is down, a alone says free, so we alert
    assert len(notifier.sent) == 1
    assert notifier.sent[0][1] == START + timedelta(minutes=1)


def test_spots_without_polygon_on_camera_untouched():
    config = make_config([camera("a", spots=("s1",))])
    runner, _, _, _ = build(config, {"a": [EMPTY]})
    runner.tick(START)
    assert runner.spot_states() == {"s1": "free", "s2": "unknown"}


def test_shared_spot_blocked_on_one_camera_free_on_other():
    # a truck blocks s1 from camera a; camera b sees it empty
    config = make_config([camera("a"), camera("b")], occupied_frames=1)
    runner, _, notifier, _ = build(config, {"a": [[PARKED], [TRUCK]], "b": [[PARKED], EMPTY]})
    run_minutes(runner, 2)
    assert runner.spot_states()["s1"] == "free"
    assert len(notifier.sent) == 1



def test_save_every_keeps_frames_and_alert_pictures(tmp_path):
    import numpy as np

    from spotwatch.detector import ScriptedDetector
    from spotwatch.notifier import ListNotifier

    class Still(FrameSource):
        def read(self, now):
            return Frame(image=np.full((40, 60, 3), 100, dtype=np.uint8), time=now)

    config = make_config([camera("a")])
    runner = Runner(
        config, {"a": Still()}, ScriptedDetector([]), ListNotifier(), Store(":memory:"),
        check_quality=False, save_dir=str(tmp_path), save_every=300,
    )
    for minute in range(11):  # 10 minutes of polling, once a minute
        runner.tick(START + timedelta(minutes=minute))
    frames = sorted(p.name for p in (tmp_path / "a").iterdir())
    assert frames == ["20261012-090000.jpg", "20261012-090500.jpg", "20261012-091000.jpg"]
    alerts = list((tmp_path / "alerts").iterdir())
    assert [p.name for p in alerts] == ["20261012-090000-spots1.jpg"]
