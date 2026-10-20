"""Per-spot state machine. Pure Python, no camera or model code.

Rules (from the kickoff spec):
- A spot becomes FREE after N free readings in a row (free_frames, default 1).
- When it becomes FREE we notify once, whatever came before (occupied,
  unknown, or startup).
- After a notification we don't notify again for that spot until it has been
  confirmed OCCUPIED (occupied_frames occupied readings in a row).
- UNKNOWN readings (truck passing, bad frame, night) set the shown state to
  UNKNOWN but don't reset anything. They pause the streak counters.
- Confirming occupied after a free alert emits notify_taken once. Startup
  occupied does not emit it. The runner controls whether to deliver this alert.
"""

from dataclasses import dataclass

from spotwatch.states import SpotState


@dataclass
class Update:
    """What happened after feeding one reading in."""

    old_state: SpotState
    new_state: SpotState
    notify: bool = False
    notify_taken: bool = False

    @property
    def changed(self) -> bool:
        return self.old_state != self.new_state


class SpotStateMachine:
    def __init__(self, spot_id: str, free_frames: int = 1, occupied_frames: int = 2):
        if free_frames < 1 or occupied_frames < 1:
            raise ValueError("free_frames and occupied_frames must be at least 1")
        self.spot_id = spot_id
        self.free_frames = free_frames
        self.occupied_frames = occupied_frames

        self.state = SpotState.UNKNOWN  # startup
        self.free_streak = 0
        self.occupied_streak = 0
        # True once we've sent a "free" alert that hasn't been cleared by a
        # confirmed occupied state yet.
        self.alerted = False

    def update(self, reading: SpotState) -> Update:
        old = self.state
        notify = False
        notify_taken = False

        if reading == SpotState.OCCUPIED:
            self.occupied_streak += 1
            self.free_streak = 0
            if self.occupied_streak >= self.occupied_frames:
                self.state = SpotState.OCCUPIED
                notify_taken = self.alerted
                self.alerted = False  # re-arm: next free gets an alert

        elif reading == SpotState.FREE:
            self.free_streak += 1
            self.occupied_streak = 0
            if self.free_streak >= self.free_frames:
                self.state = SpotState.FREE
                if not self.alerted:
                    notify = True
                    self.alerted = True

        else:  # UNKNOWN
            self.state = SpotState.UNKNOWN

        return Update(old_state=old, new_state=self.state, notify=notify, notify_taken=notify_taken)
