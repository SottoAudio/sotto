"""Arming latch: nothing goes back to clean on its own.

State rules (the same rules the real app will use):
  * Start disarmed. HEARTBEAT is on only while armed.
  * Any system fault (probe missing, injected fault, device error) disarms everything and
    latches: clearing the fault does not re-arm.
  * A talker trip puts that talker on dry and latches it until that talker is re-armed.
  * An arm request (everyone or one talker) is accepted only when every check has been green
    for `arm_hold_s` without a break. Requests can come from the console, the HTTP API
    (Companion) or the Riedel panel (via RRCS); the source is logged.
"""
from dataclasses import dataclass, field


@dataclass
class Event:
    t: float
    kind: str
    detail: str = ""
    source: str = ""


@dataclass
class ArmState:
    n_talkers: int
    arm_hold_s: float = 5.0
    armed: bool = False
    talker_armed: list = field(default_factory=list)
    green_since: float = None
    reason: str = "started disarmed"
    events: list = field(default_factory=list)

    def __post_init__(self):
        if not self.talker_armed:
            self.talker_armed = [False] * self.n_talkers

    # ------------------------------------------------------------ health
    def update_health(self, t, healthy, why=""):
        """Call every block. `healthy` is False while any system check is failing."""
        if healthy:
            if self.green_since is None:
                self.green_since = t
        else:
            self.green_since = None
            if self.armed:
                self._disarm(t, why or "system fault", source="auto")

    def green_for(self, t):
        return 0.0 if self.green_since is None else t - self.green_since

    def can_arm(self, t):
        return self.green_for(t) >= self.arm_hold_s

    # ------------------------------------------------------------ requests
    def request_arm(self, t, source, talker=None):
        """Returns (accepted, message)."""
        if not self.can_arm(t):
            need = self.arm_hold_s - self.green_for(t)
            msg = ("checks not green" if self.green_since is None
                   else f"checks green for {self.green_for(t):.1f} s, need {self.arm_hold_s:.0f} s ({need:.1f} s to go)")
            self.events.append(Event(t, "arm_refused", msg, source))
            return False, msg
        if talker is None:
            self.armed = True
            self.talker_armed = [True] * self.n_talkers
            self.reason = ""
            self.events.append(Event(t, "armed", "everyone", source))
            return True, "armed"
        if not 0 <= talker < self.n_talkers:
            return False, "no such talker"
        if not self.armed:
            msg = "arm everyone first: the Riedel is holding all talkers on dry"
            self.events.append(Event(t, "arm_refused", f"talker {talker + 1}: {msg}", source))
            return False, msg
        self.talker_armed[talker] = True
        self.events.append(Event(t, "talker_armed", f"talker {talker + 1}", source))
        return True, f"talker {talker + 1} armed"

    def request_disarm(self, t, source, why="disarmed by engineer"):
        self._disarm(t, why, source)

    def trip_talker(self, t, talker, why, source="auto"):
        if self.talker_armed[talker]:
            self.talker_armed[talker] = False
            self.events.append(Event(t, "talker_tripped", f"talker {talker + 1}: {why}", source))

    def _disarm(self, t, why, source):
        was = self.armed
        self.armed = False
        self.talker_armed = [False] * self.n_talkers
        self.reason = why
        if was:
            self.events.append(Event(t, "disarmed", why, source))

    # ------------------------------------------------------------ outputs
    @property
    def heartbeat_on(self):
        return self.armed

    def drain_events(self):
        ev, self.events = self.events, []
        return ev
