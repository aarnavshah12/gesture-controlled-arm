"""Gesture debounce + the mode state machine. No hardware here: actions go through an interface.

Debounce: a gesture fires only after DEBOUNCE_N consecutive ACCEPTED predictions of the same class
above CONFIDENCE. Any rejected prediction (no detection, the model's `null` class, an unknown class,
or below threshold) resets the count. Once fired, the same gesture held continuously fires nothing
more: the class must change (or drop out) before it can fire again. Every accepted and rejected
prediction is logged with its class, confidence and debounce count.

Modes: MIRROR (default) -> FROZEN (fist) -> ROUTINE (HOME / PICK / FLOURISH in progress).
FREEZE overrides everything, including a routine in progress. RELEASE (open-palm) is the only exit
from FROZEN. Events during a routine are logged and ignored, except FREEZE.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from . import config, runlog

MIRROR, FROZEN, ROUTINE = "MIRROR", "FROZEN", "ROUTINE"
MODES = (MIRROR, FROZEN, ROUTINE)
ROUTINE_EVENTS = ("HOME", "PICK", "FLOURISH", "GRAB")   # RELEASE becomes the PLACE routine while holding


@dataclass(frozen=True)
class Prediction:
    """Top-1 gesture detection for one inference frame (pixels in the displayed frame)."""
    cls: str
    conf: float
    x1: int
    y1: int
    x2: int
    y2: int
    raw_cls: str | None = None   # the model's label when `cls` was corrected by the landmark veto

    @property
    def box(self) -> tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2

    @property
    def corrected(self) -> bool:
        return self.raw_cls is not None and self.raw_cls != self.cls


def reconcile(pred: Prediction | None, hand, log=None) -> Prediction | None:
    """Apply config.LANDMARK_VETO: relabel a model class the landmarks contradict. One direction only."""
    if pred is None or hand is None or pred.cls not in config.LANDMARK_VETO:
        return pred
    from .perception import finger_states, two_fingers_up   # local import: gestures stays hardware-free

    states = finger_states(hand.norm)
    if pred.cls == "peace" and two_fingers_up(states):
        return pred
    if pred.cls == "peace":
        new_cls = config.LANDMARK_VETO["peace"]
        (log or runlog.get_logger()).info(
            "landmark veto: model %s %.2f but fingers %s -> %s", pred.cls, pred.conf,
            "".join(k[0].upper() if v else k[0] for k, v in states.items()), new_cls)
        return Prediction(new_cls, pred.conf, pred.x1, pred.y1, pred.x2, pred.y2, raw_cls=pred.cls)
    return pred


@dataclass(frozen=True)
class Event:
    name: str      # FREEZE / RELEASE / GRIP / HOME / PICK / FLOURISH
    gesture: str   # model class that produced it
    conf: float
    t: float


class Debouncer:
    def __init__(self, n: int | None = None, threshold: float | None = None, log=None):
        self.n = config.DEBOUNCE_N if n is None else int(n)
        self.threshold = config.CONFIDENCE if threshold is None else float(threshold)
        self.log = log or runlog.get_logger()
        self.cls: str | None = None
        self.count = 0
        self.fired = False
        self.last_conf = 0.0
        self.last_reason = "start"

    def reset(self, reason: str) -> None:
        if self.cls is not None or self.count:
            self.log.info("debounce reset (%s) after %s x%d", reason, self.cls, self.count)
        self.cls, self.count, self.fired = None, 0, False
        self.last_reason = reason

    def update(self, pred: Prediction | None, t: float | None = None) -> Event | None:
        """Feed one inference result (or None for no detection). Returns an Event at most once per hold."""
        t = time.time() if t is None else t
        if pred is None:
            self.log.info("gesture rejected: no detection (count was %d)", self.count)
            self.reset("no detection")
            return None
        self.last_conf = pred.conf
        if pred.cls == config.NULL_CLASS:
            self.log.info("gesture rejected: %s %.2f (null class, count was %d)", pred.cls, pred.conf, self.count)
            self.reset("null")
            return None
        if pred.cls in config.NO_EVENT_CLASSES:
            self.log.info("gesture seen: %s %.2f (steering pose, no event; count was %d)", pred.cls, pred.conf, self.count)
            self.reset("steering pose")
            return None
        if pred.cls not in config.GESTURE_EVENTS:
            self.log.warning("gesture rejected: unknown class %r %.2f", pred.cls, pred.conf)
            self.reset("unknown class")
            return None
        if pred.conf < self.threshold:
            self.log.info("gesture rejected: %s %.2f < %.2f (count was %d)", pred.cls, pred.conf,
                          self.threshold, self.count)
            self.reset("below threshold")
            return None
        if pred.cls != self.cls:
            if self.cls is not None:
                self.log.info("debounce switch %s x%d -> %s", self.cls, self.count, pred.cls)
            self.cls, self.count, self.fired = pred.cls, 0, False
        self.count += 1
        self.log.info("gesture accepted: %s %.2f count=%d/%d%s", pred.cls, pred.conf, self.count, self.n,
                      " (already fired)" if self.fired else "")
        if self.count >= self.n and not self.fired:
            self.fired = True
            return Event(config.GESTURE_EVENTS[pred.cls], pred.cls, pred.conf, t)
        return None

    @property
    def progress(self) -> float:
        """0..1 charge for the on-screen arc (1.0 once fired)."""
        if self.fired:
            return 1.0
        return min(1.0, self.count / float(self.n)) if self.n > 0 else 1.0


class Actions:
    """What the state machine can ask the rest of the system to do. Override in the real app / tests.

    start_routine(name, done) must call done(ok) exactly once when the routine ends (or aborts).
    """

    def freeze(self) -> None: ...
    def release(self) -> None: ...
    def grip(self) -> None: ...
    def is_gripping(self) -> bool: return False
    def can_grab(self) -> tuple[bool, str]: return True, ""   # steering established and inside the box?
    def resume_mirror(self, recenter: bool) -> None: ...
    def pause_mirror(self) -> None: ...
    def start_routine(self, name: str, done: Callable[[bool], None]) -> None: ...
    def abort_routine(self) -> None: ...


@dataclass
class Transition:
    t: float
    frm: str
    to: str
    why: str


class StateMachine:
    def __init__(self, actions: Actions, log=None):
        self.actions = actions
        self.log = log or runlog.get_logger()
        self.mode = MIRROR
        self.routine: str | None = None
        self._routine_seq = 0
        self.history: list[Transition] = []
        self.last_event: Event | None = None
        self.last_refusal = ""          # why the last event was ignored (for the toast)

    def _set_mode(self, to: str, why: str) -> None:
        if to not in MODES:
            raise ValueError(to)
        frm = self.mode
        self.mode = to
        self.history.append(Transition(time.time(), frm, to, why))
        self.log.info("mode %s -> %s (%s)", frm, to, why)

    # -- events -------------------------------------------------------------------
    def on_event(self, ev: Event) -> bool:
        """Handle one debounced event. Returns True if it was acted on, False if ignored in this mode."""
        self.last_event = ev
        self.log.info("EVENT %s from %s %.2f in mode %s", ev.name, ev.gesture, ev.conf, self.mode)
        if ev.name == "FREEZE":
            return self._freeze(ev)
        if self.mode == FROZEN:
            if ev.name == "RELEASE":
                self.actions.release()
                self.routine = None
                self._set_mode(MIRROR, "open-palm released the freeze")
                self.actions.resume_mirror(recenter=True)
                return True
            return self._ignore(ev, "frozen")
        if self.mode == ROUTINE:
            return self._ignore(ev, f"routine {self.routine}")
        # MIRROR
        if ev.name == "GRAB":
            if self.actions.is_gripping():
                return self._ignore(ev, "already holding")
            ok, why = self.actions.can_grab()
            if not ok:
                return self._ignore(ev, why)
        if ev.name == "FLOURISH" and self.actions.is_gripping():
            return self._ignore(ev, "holding a block")
        if ev.name == "GRIP":
            self.actions.grip()                 # plain "pump on" in place (not mapped by default)
        elif ev.name == "RELEASE":
            if self.actions.is_gripping():
                ok, why = self.actions.can_grab()
                if not ok:
                    # never drop a held object from an unplanned spot; fist -> open-palm is the deliberate drop
                    return self._ignore(ev, why)
                self._start_routine("PLACE")        # holding something: set it down, don't drop it from height
            else:
                self.actions.release()
        elif ev.name in ROUTINE_EVENTS:
            self._start_routine(ev.name)
        else:
            self.log.warning("event %s has no handler", ev.name)
            return False
        return True

    def _ignore(self, ev: Event, why: str) -> bool:
        self.last_refusal = why
        self.log.info("event %s ignored: %s", ev.name, why)
        return False

    def _freeze(self, ev: Event) -> bool:
        if self.mode == FROZEN:
            self.log.info("already FROZEN: re-asserting the halt")
            self.actions.freeze()          # a repeated fist re-sends the stop; harmless, never wrong
            return False
        was_routine = self.routine if self.mode == ROUTINE else None
        self._set_mode(FROZEN, f"fist{' during ' + was_routine if was_routine else ''}")
        if was_routine:
            self.actions.abort_routine()   # flag first: the routine thread must not send another move
        self.actions.freeze()              # then halt (and inhibit) the arm
        self.actions.pause_mirror()
        return True

    def _start_routine(self, name: str) -> None:
        self._routine_seq += 1
        seq = self._routine_seq
        self.routine = name
        self._set_mode(ROUTINE, f"{name} started")
        self.actions.pause_mirror()

        def done(ok: bool) -> None:
            self.on_routine_done(name, ok, seq)

        self.actions.start_routine(name, done)

    def on_routine_done(self, name: str, ok: bool, seq: int | None = None) -> None:
        if seq is not None and seq != self._routine_seq:
            self.log.info("routine %s (stale #%d) finished ok=%s; ignored", name, seq, ok)
            return
        self.log.info("routine %s finished ok=%s in mode %s", name, ok, self.mode)
        if self.mode == FROZEN:
            self.routine = None
            return          # a freeze interrupted it; open-palm decides what happens next
        if self.mode == ROUTINE:
            self.routine = None
            self._set_mode(MIRROR, f"{name} {'done' if ok else 'failed'}")
            # a failed routine leaves the arm somewhere unplanned: re-centre before following again
            self.actions.resume_mirror(recenter=config.RESUME_RECENTER.get(name, True) or not ok)

    @property
    def banner(self) -> str:
        return self.mode
