"""Notifiers send free and confirmed-taken parking alerts.

- ConsoleNotifier: prints it.
- DiscordNotifier: posts it to a Discord channel through a webhook, with a
  picture of the spot attached (green for free, red for taken).

Test your Discord setup without waiting for a real free spot:

    uv run python -m spotwatch.notifier --test
"""

import argparse
import json
import logging
import os
import time as time_module
from dataclasses import dataclass
from datetime import datetime

import cv2
import numpy as np

from spotwatch.states import SpotState

log = logging.getLogger("spotwatch.notify")


@dataclass
class Alert:
    spot_id: str
    spot_name: str
    time: datetime
    image: np.ndarray | None = None  # frame with the relevant spot outlined, if available
    state: SpotState = SpotState.FREE


class Notifier:
    def notify(self, alert: Alert) -> str:
        """Send the alert and return the message text (it goes in the log)."""
        raise NotImplementedError


def free_message(spot_name: str, time: datetime) -> str:
    return f"Spot free: {spot_name} ({time:%H:%M})"


def alert_message(alert: Alert) -> str:
    if alert.state == SpotState.OCCUPIED:
        return f"Spot taken: {alert.spot_name} ({alert.time:%H:%M})"
    return free_message(alert.spot_name, alert.time)


def highlight_spot(image: np.ndarray, polygon, state: SpotState = SpotState.FREE) -> np.ndarray:
    """Highlight free in green and occupied in red."""
    out = image.copy()
    overlay = image.copy()
    pts = np.array(polygon, dtype=np.int32)
    color = (0, 0, 200) if state == SpotState.OCCUPIED else (0, 200, 0)
    cv2.fillPoly(overlay, [pts], color)
    out = cv2.addWeighted(overlay, 0.4, out, 0.6, 0)
    cv2.polylines(out, [pts], True, color, 3)
    return out


class ConsoleNotifier(Notifier):
    def notify(self, alert: Alert) -> str:
        message = alert_message(alert)
        print(f"[spotwatch] {message}", flush=True)
        return message


class ListNotifier(Notifier):
    """Keeps alerts in a list. Used by tests."""

    def __init__(self):
        self.sent: list[tuple[str, datetime]] = []
        self.alerts: list[Alert] = []

    def notify(self, alert: Alert) -> str:
        self.sent.append((alert.spot_id, alert.time))
        self.alerts.append(alert)
        return alert_message(alert)


class DiscordNotifier(Notifier):
    """Posts to a Discord channel webhook.

    The webhook URL is a secret (anyone with it can post to your channel), so
    it comes from an environment variable or the gitignored .env file, and is
    never logged.

    If Discord is down or slow, the alert is logged as failed and spotwatch
    keeps running. It retries once after a rate limit (HTTP 429).
    """

    def __init__(self, webhook_url: str, also_print: bool = True, timeout: float = 10, post=None):
        if not webhook_url.startswith("https://"):
            raise ValueError("Discord webhook URL should start with https://")
        self.webhook_url = webhook_url
        self.also_print = also_print
        self.timeout = timeout
        if post is None:
            import requests

            post = requests.post
        self._post = post  # swappable in tests

    def notify(self, alert: Alert) -> str:
        message = alert_message(alert)
        if self.also_print:
            print(f"[spotwatch] {message}", flush=True)
        ok = self.send(message, alert.image)
        return message if ok else f"{message} [discord send failed]"

    def send(self, message: str, image: np.ndarray | None = None) -> bool:
        payload = {"content": f"🅿️ **{message}**"}
        for attempt in range(2):
            try:
                if image is not None:
                    ok_jpg, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    files = {"files[0]": ("spot.jpg", buf.tobytes(), "image/jpeg")} if ok_jpg else None
                    response = self._post(
                        self.webhook_url,
                        data={"payload_json": json.dumps(payload)},
                        files=files,
                        timeout=self.timeout,
                    )
                else:
                    response = self._post(self.webhook_url, json=payload, timeout=self.timeout)
            except Exception as error:  # network down, DNS, timeout
                log.warning("Discord send failed: %s", type(error).__name__)
                return False

            if response.status_code in (200, 204):
                return True
            if response.status_code == 429 and attempt == 0:
                wait = _retry_after(response)
                log.warning("Discord rate limited, retrying in %.1fs", wait)
                time_module.sleep(wait)
                continue
            log.warning("Discord send failed: HTTP %s", response.status_code)
            return False
        return False


def _retry_after(response) -> float:
    try:
        return min(float(response.json().get("retry_after", 1)), 10.0)
    except Exception:
        return 1.0


# ---------- building the notifier from config ----------


def load_dotenv(path: str = ".env") -> None:
    """Read KEY=VALUE lines from .env into the environment (if not already set)."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def make_notifier(config) -> Notifier:
    kind = config.notifier
    if kind == "console":
        return ConsoleNotifier()
    if kind == "discord":
        load_dotenv()
        url = os.environ.get(config.discord_webhook_env, "").strip()
        if not url:
            raise SystemExit(
                f"notifier is 'discord' but {config.discord_webhook_env} isn't set. "
                f"Put {config.discord_webhook_env}=<your webhook url> in a .env file in the repo folder."
            )
        return DiscordNotifier(url)
    raise SystemExit(f"unknown notifier {kind!r} (use 'console' or 'discord')")


def main():
    parser = argparse.ArgumentParser(description="Send a test alert with the configured notifier.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--test", action="store_true", help="send a test message")
    args = parser.parse_args()
    if not args.test:
        parser.error("use --test to send a test message")

    from spotwatch.config import load_config

    logging.basicConfig(level=logging.WARNING)
    config = load_config(args.config)
    notifier = make_notifier(config)
    result = notifier.notify(Alert(spot_id="test", spot_name="TEST (spotwatch setup check)", time=datetime.now()))
    print("sent:", result)


if __name__ == "__main__":
    main()
