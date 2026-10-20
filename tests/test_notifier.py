import json
import logging
from datetime import datetime

import numpy as np
import pytest

from spotwatch.config import parse_config
from spotwatch.notifier import (
    Alert,
    ConsoleNotifier,
    DiscordNotifier,
    highlight_spot,
    load_dotenv,
    make_notifier,
)

URL = "https://discord.com/api/webhooks/123/SECRET-TOKEN"
WHEN = datetime(2026, 10, 14, 18, 5)


class FakeResponse:
    def __init__(self, status_code, body=None):
        self.status_code = status_code
        self.body = body or {}

    def json(self):
        return self.body


class FakePost:
    """Records calls; returns the queued responses in order."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, url, **kwargs):
        self.calls.append((url, kwargs))
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def test_console_prints(capsys):
    msg = ConsoleNotifier().notify(Alert("1", "Front", WHEN))
    assert msg == "Spot free: Front (18:05)"
    assert "Spot free: Front" in capsys.readouterr().out


def test_taken_discord_message():
    from spotwatch.states import SpotState
    post = FakePost(FakeResponse(204))
    message = DiscordNotifier(URL, post=post).notify(
        Alert("3", "3", WHEN, state=SpotState.OCCUPIED))
    assert message == "Spot taken: 3 (18:05)"
    assert "Spot taken: 3" in post.calls[0][1]["json"]["content"]


def test_discord_text_only():
    post = FakePost(FakeResponse(204))
    msg = DiscordNotifier(URL, also_print=False, post=post).notify(Alert("1", "Front", WHEN))
    assert msg == "Spot free: Front (18:05)"
    url, kwargs = post.calls[0]
    assert url == URL
    assert "Spot free: Front (18:05)" in kwargs["json"]["content"]


def test_discord_with_picture_sends_multipart():
    post = FakePost(FakeResponse(200))
    image = np.zeros((72, 128, 3), dtype=np.uint8)
    DiscordNotifier(URL, also_print=False, post=post).notify(Alert("1", "Front", WHEN, image=image))
    _, kwargs = post.calls[0]
    name, data, mime = kwargs["files"]["files[0]"]
    assert name == "spot.jpg" and mime == "image/jpeg" and data[:2] == b"\xff\xd8"  # JPEG header
    assert "Spot free: Front" in json.loads(kwargs["data"]["payload_json"])["content"]


def test_discord_retries_once_after_rate_limit(monkeypatch):
    import spotwatch.notifier as n

    monkeypatch.setattr(n.time_module, "sleep", lambda s: None)
    post = FakePost(FakeResponse(429, {"retry_after": 0.5}), FakeResponse(204))
    msg = DiscordNotifier(URL, also_print=False, post=post).notify(Alert("1", "Front", WHEN))
    assert len(post.calls) == 2
    assert "failed" not in msg


def test_discord_failure_is_reported_not_raised(caplog):
    post = FakePost(FakeResponse(500))
    with caplog.at_level(logging.WARNING):
        msg = DiscordNotifier(URL, also_print=False, post=post).notify(Alert("1", "Front", WHEN))
    assert msg.endswith("[discord send failed]")
    assert "SECRET-TOKEN" not in caplog.text


def test_discord_network_error_is_reported_not_raised(caplog):
    post = FakePost(ConnectionError(f"could not reach {URL}"))
    with caplog.at_level(logging.WARNING):
        msg = DiscordNotifier(URL, also_print=False, post=post).notify(Alert("1", "Front", WHEN))
    assert "failed" in msg
    assert "SECRET-TOKEN" not in caplog.text  # only the error type is logged, never the URL


def test_discord_rejects_non_https():
    with pytest.raises(ValueError):
        DiscordNotifier("http://example.com/hook", post=FakePost())


def config_with(notifier):
    return parse_config({"settings": {"notifier": notifier}, "spots": [{"id": "1"}], "cameras": []})


def test_make_notifier_console():
    assert isinstance(make_notifier(config_with("console")), ConsoleNotifier)


def test_make_notifier_discord_needs_url(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # no .env here
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
    with pytest.raises(SystemExit) as exc:
        make_notifier(config_with("discord"))
    assert "DISCORD_WEBHOOK_URL" in str(exc.value)


def test_make_notifier_discord_from_dotenv(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
    (tmp_path / ".env").write_text(f'# my secrets\nDISCORD_WEBHOOK_URL="{URL}"\n')
    notifier = make_notifier(config_with("discord"))
    assert isinstance(notifier, DiscordNotifier)
    assert notifier.webhook_url == URL


def test_dotenv_does_not_override_real_env(monkeypatch, tmp_path):
    monkeypatch.setenv("SOME_KEY", "from-env")
    (tmp_path / ".env").write_text("SOME_KEY=from-file\n")
    load_dotenv(str(tmp_path / ".env"))
    import os

    assert os.environ["SOME_KEY"] == "from-env"


def test_highlight_spot_marks_the_polygon():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    out = highlight_spot(image, [(10, 10), (60, 10), (60, 60), (10, 60)])
    assert out[35, 35, 1] > 50  # green inside
    assert out[90, 90].sum() == 0  # untouched outside
    assert image.sum() == 0  # original not modified


def test_taken_picture_is_red():
    from spotwatch.states import SpotState
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    out = highlight_spot(image, [(10, 10), (60, 10), (60, 60), (10, 60)], SpotState.OCCUPIED)
    assert out[35, 35, 2] > 50
    assert out[35, 35, 1] == 0
    assert image.sum() == 0


def test_runner_attaches_picture_and_survives_a_broken_notifier():
    from spotwatch.detector import ScriptedDetector
    from spotwatch.notifier import ListNotifier, Notifier
    from spotwatch.runner import Runner
    from spotwatch.sources import Frame, FrameSource
    from spotwatch.storage import Store

    class Still(FrameSource):
        def read(self, now):
            return Frame(image=np.full((60, 80, 3), 120, dtype=np.uint8), time=now)

    config = parse_config(
        {
            "spots": [{"id": "1", "name": "Front"}, {"id": "2"}],
            "cameras": [
                {
                    "name": "cam",
                    "source": "image_folder",
                    "path": "x",
                    "polygons": {"1": [[5, 5], [40, 5], [40, 40]], "2": [[45, 5], [75, 5], [75, 40]]},
                }
            ],
        }
    )
    notifier = ListNotifier()
    runner = Runner(config, {"cam": Still()}, ScriptedDetector([[]]), notifier, Store(":memory:"), check_quality=False)
    runner.tick(WHEN)
    assert {a.spot_id for a in notifier.alerts} == {"1", "2"}
    assert all(a.image is not None and a.image.shape == (60, 80, 3) for a in notifier.alerts)

    class Broken(Notifier):
        def notify(self, alert):
            raise RuntimeError("boom")

    store = Store(":memory:")
    runner = Runner(config, {"cam": Still()}, ScriptedDetector([[]]), Broken(), store, check_quality=False)
    runner.tick(WHEN)  # must not raise
    rows = store.conn.execute("SELECT message FROM notifications").fetchall()
    assert len(rows) == 2 and all("notify failed" in r["message"] for r in rows)
