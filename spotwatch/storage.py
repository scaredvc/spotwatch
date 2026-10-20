"""SQLite log of readings, state changes, notifications and snapshots."""

import json
import sqlite3
from datetime import datetime

SCHEMA = """
CREATE TABLE IF NOT EXISTS readings (
    id INTEGER PRIMARY KEY,
    time TEXT NOT NULL,
    camera TEXT NOT NULL,
    spot_id TEXT NOT NULL,
    value TEXT NOT NULL,
    good INTEGER NOT NULL,
    reason TEXT
);
CREATE TABLE IF NOT EXISTS state_changes (
    id INTEGER PRIMARY KEY,
    time TEXT NOT NULL,
    spot_id TEXT NOT NULL,
    old_state TEXT NOT NULL,
    new_state TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS notifications (
    id INTEGER PRIMARY KEY,
    time TEXT NOT NULL,
    spot_id TEXT NOT NULL,
    message TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS snapshots (
    id INTEGER PRIMARY KEY,
    time TEXT NOT NULL,
    camera TEXT NOT NULL,
    good INTEGER NOT NULL,
    reason TEXT,
    power_info TEXT
);
CREATE INDEX IF NOT EXISTS idx_readings_spot ON readings (spot_id, time);
CREATE INDEX IF NOT EXISTS idx_changes_spot ON state_changes (spot_id, time);
CREATE INDEX IF NOT EXISTS idx_snapshots_camera ON snapshots (camera, time);
"""


class Store:
    def __init__(self, path: str = "spotwatch.db"):
        # check_same_thread=False so another thread (e.g. a future web view) can read it too
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    def log_reading(self, time: datetime, camera: str, spot_id: str, value: str, good: bool, reason: str = ""):
        self.conn.execute(
            "INSERT INTO readings (time, camera, spot_id, value, good, reason) VALUES (?, ?, ?, ?, ?, ?)",
            (time.isoformat(), camera, spot_id, value, int(good), reason),
        )
        self.conn.commit()

    def log_state_change(self, time: datetime, spot_id: str, old_state: str, new_state: str):
        self.conn.execute(
            "INSERT INTO state_changes (time, spot_id, old_state, new_state) VALUES (?, ?, ?, ?)",
            (time.isoformat(), spot_id, old_state, new_state),
        )
        self.conn.commit()

    def log_notification(self, time: datetime, spot_id: str, message: str):
        self.conn.execute(
            "INSERT INTO notifications (time, spot_id, message) VALUES (?, ?, ?)",
            (time.isoformat(), spot_id, message),
        )
        self.conn.commit()

    def log_snapshot(self, time: datetime, camera: str, good: bool, reason: str = "", power_info: dict | None = None):
        self.conn.execute(
            "INSERT INTO snapshots (time, camera, good, reason, power_info) VALUES (?, ?, ?, ?, ?)",
            (time.isoformat(), camera, int(good), reason, json.dumps(power_info) if power_info else None),
        )
        self.conn.commit()

    def snapshot_stats(self) -> list[dict]:
        """Snapshots per camera per day, with how many were good."""
        rows = self.conn.execute(
            """
            SELECT camera, substr(time, 1, 10) AS day, COUNT(*) AS total, SUM(good) AS good
            FROM snapshots GROUP BY camera, day ORDER BY day, camera
            """
        ).fetchall()
        return [dict(r) for r in rows]

    def notifications(self, spot_id: str | None = None) -> list[dict]:
        if spot_id is None:
            rows = self.conn.execute("SELECT * FROM notifications ORDER BY id").fetchall()
        else:
            rows = self.conn.execute("SELECT * FROM notifications WHERE spot_id = ? ORDER BY id", (spot_id,)).fetchall()
        return [dict(r) for r in rows]

    def state_changes(self, spot_id: str) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM state_changes WHERE spot_id = ? ORDER BY id", (spot_id,)).fetchall()
        return [dict(r) for r in rows]

    def close(self):
        self.conn.close()
