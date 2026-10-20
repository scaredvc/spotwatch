"""Shared types for spot readings and states."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class SpotState(str, Enum):
    OCCUPIED = "occupied"
    FREE = "free"
    UNKNOWN = "unknown"


@dataclass
class CameraReading:
    """What one camera said about one spot on one frame.

    good=False means the frame itself can't be trusted (bad, stale, camera down,
    outside active hours). A good frame can still read UNKNOWN when the spot is
    occluded.
    """

    camera: str
    spot_id: str
    value: SpotState
    good: bool
    time: datetime
    reason: str = ""
