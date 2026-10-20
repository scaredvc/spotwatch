"""Combine readings from several cameras that see the same spot.

Rules:
- Ignore cameras whose latest frame is bad or stale.
- Also ignore cameras whose view of the spot is blocked (UNKNOWN on a good
  frame). Blocked means "this camera can't tell", not "the spot isn't free".
- Of the rest, an OCCUPIED reading from any camera wins.
- Otherwise the spot is FREE if at least one camera can see it.
- If no camera can see it, the spot is UNKNOWN.

Example: camera A sees spots 1-4, camera B sees 4-7. A truck blocks spot 4
from A, B sees it empty. Spot 4 is FREE.
"""

from datetime import datetime, timedelta

from spotwatch.states import CameraReading, SpotState


def fuse(readings: list[CameraReading], now: datetime, max_age: dict[str, timedelta]) -> SpotState:
    """max_age maps camera name to how old its reading may be before it's stale."""
    usable = []
    for r in readings:
        if not r.good or r.value == SpotState.UNKNOWN:
            continue
        limit = max_age.get(r.camera)
        if limit is not None and now - r.time > limit:
            continue
        usable.append(r)

    if not usable:
        return SpotState.UNKNOWN

    if any(r.value == SpotState.OCCUPIED for r in usable):
        return SpotState.OCCUPIED
    return SpotState.FREE
