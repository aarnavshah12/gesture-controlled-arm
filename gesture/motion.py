"""Mirroring math and the 10 Hz motion controller. No serial code here: the controller drives an
`arm` object with stream_to(x, y, z) / halt() / read_xyz() / dry_run / commanded.

Safety envelope, enforced here before anything reaches the driver (which checks again):
- every target clamped to the mirror box (already intersected with the rig's reach limits) and pulled
  back when it would need more than EXTENSION_MAX of the arm's full stretch (the firmware silently
  refuses such targets, and a streamed command has no read-back to notice);
- the commanded point moves toward the target at most VELOCITY_CAP_MM_S (per tick);
- the loop runs at CONTROL_HZ independent of camera FPS;
- no hand for NO_HAND_HOLD_S = hold; FREEZE halts and disables the loop until resumed.
"""

from __future__ import annotations

import math
import threading
import time

from . import config, runlog

Box = tuple[tuple[float, float], tuple[float, float], tuple[float, float]]

# Link lengths from Hiwonder's ESPMax library (same numbers as the block picker's arm.py).
L0, L1, L2, L3, L4 = 84.4, 8.14, 128.4, 138.0, 16.8


def extension_ratio(x: float, y: float, z: float) -> float:
    d = math.hypot(x, y) - L1 - L4
    h = z - L0
    return math.hypot(d, h) / (L2 + L3)


def intersect_box(a: Box, b: Box) -> Box:
    out = []
    for (alo, ahi), (blo, bhi) in zip(a, b):
        lo, hi = max(alo, blo), min(ahi, bhi)
        if lo > hi:
            raise ValueError(f"empty intersection: [{alo}, {ahi}] and [{blo}, {bhi}]")
        out.append((float(lo), float(hi)))
    return tuple(out)  # type: ignore[return-value]


def clamp_box(p, box: Box) -> tuple[float, float, float]:
    return tuple(min(max(float(v), lo), hi) for v, (lo, hi) in zip(p, box))  # type: ignore[return-value]


def clamp_target(p, box: Box, origin, ext_max: float | None = None) -> tuple[float, float, float]:
    """Clamp to the box, then pull (x, y) toward the origin until the extension is acceptable."""
    ext_max = config.EXTENSION_MAX if ext_max is None else ext_max
    x, y, z = clamp_box(p, box)
    ox, oy = float(origin[0]), float(origin[1])
    for _ in range(40):
        if extension_ratio(x, y, z) <= ext_max:
            break
        x, y = ox + (x - ox) * 0.9, oy + (y - oy) * 0.9
    return clamp_box((x, y, z), box)


def step_toward(cur, target, max_step: float) -> tuple[float, float, float]:
    dx, dy, dz = (target[i] - cur[i] for i in range(3))
    dist = math.sqrt(dx * dx + dy * dy + dz * dz)
    if dist <= max_step or dist == 0.0:
        return tuple(float(v) for v in target)  # type: ignore[return-value]
    f = max_step / dist
    return (cur[0] + dx * f, cur[1] + dy * f, cur[2] + dz * f)


def step_toward_polar(cur, target, max_step: float) -> tuple[float, float, float]:
    """Like step_toward, but interpolating radius / angle about the arm's base (and z linearly).

    Used when the straight line from `cur` to `target` would pass through the base keep-out zone
    (e.g. coming back from HOME): the radius never drops below min(r_cur, r_target), so the path
    swings around the base instead of through it. The Cartesian step is still <= max_step.
    """
    r0, a0 = math.hypot(cur[0], cur[1]), math.atan2(cur[1], cur[0])
    r1, a1 = math.hypot(target[0], target[1]), math.atan2(target[1], target[0])
    da = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi
    arc = abs(da) * max(r0, r1)
    total = math.sqrt(arc * arc + (r1 - r0) ** 2 + (target[2] - cur[2]) ** 2)
    if total <= max_step or total == 0.0:
        return tuple(float(v) for v in target)  # type: ignore[return-value]
    f = max_step / total
    r, a, z = r0 + f * (r1 - r0), a0 + f * da, cur[2] + f * (target[2] - cur[2])
    return (r * math.cos(a), r * math.sin(a), z)


def hand_to_target(wrist_norm, ref_norm, origin, box: Box) -> tuple[float, float, float]:
    """Relative mirroring: arm target = origin + gain * (wrist - reference), clamped."""
    dx = (wrist_norm[0] - ref_norm[0]) * config.MIRROR_GAIN_X_MM * config.MIRROR_X_SIGN
    dz = (wrist_norm[1] - ref_norm[1]) * config.MIRROR_GAIN_Z_MM * config.MIRROR_Z_SIGN
    return clamp_target((origin[0] + dx, origin[1], origin[2] + dz), box, origin)


class MotionController(threading.Thread):
    """Fixed-rate control loop. Feed it wrist positions; it streams capped, clamped targets."""

    def __init__(self, arm, box: Box, origin=None, hz: float | None = None, log=None, check=None):
        super().__init__(name="gesture-motion", daemon=True)
        self.arm = arm
        self.box = box
        # check(x, y, z) -> bool: the driver's envelope (reach limits, table Z, base keep-out radius).
        # A step is only streamed if it passes; the driver checks again before sending.
        self.check = check or (lambda x, y, z: clamp_box((x, y, z), box) == (x, y, z))
        self.origin = tuple(float(v) for v in (origin or config.MIRROR_ORIGIN_XYZ_MM))
        self.hz = config.CONTROL_HZ if hz is None else float(hz)
        self.period = 1.0 / self.hz
        self.max_step = config.VELOCITY_CAP_MM_S / self.hz
        self.log = log or runlog.get_logger()
        self._lock = threading.Lock()
        self._stop_evt = threading.Event()
        # state (guarded by _lock)
        self.enabled = False            # mirroring active (MIRROR mode)
        self.frozen = False
        self.target = self.origin
        self.commanded = None           # last point streamed (None until the first command)
        self.hand_ref = None
        self.recenter_required = True
        self._recenter_since = None
        self.last_hand_t = 0.0
        self.holding = False
        self.actual = None
        self.actual_t = 0.0
        self.epoch = 0                  # bumped by freeze()/pause(): a resume queued before is stale
        # live: never stream from a guessed position (sync_to_arm() must succeed first); dry-run may
        # start from the origin, there is nothing to hit
        self.position_known = bool(getattr(arm, "dry_run", False))
        self._warned_unknown = False
        # stats
        self.ticks = 0
        self.late_ticks = 0
        self.commands = 0
        self.refused = 0
        self.detours = 0
        self.blocked = 0
        self.errors = 0

    # -- inputs -------------------------------------------------------------------
    def update_hand(self, wrist_norm, t: float) -> None:
        """Called every camera frame with the (smoothed) wrist in normalised coords, or None."""
        with self._lock:
            if wrist_norm is None:
                self._recenter_since = None     # the centre-hold restarts when the hand comes back
                return
            self.last_hand_t = t
            if self.holding:
                self.holding = False
                self.log.info("mirror: hand back, resuming from hold")
            if self.recenter_required:
                d = math.hypot(wrist_norm[0] - 0.5, wrist_norm[1] - 0.5)
                if d <= config.RECENTER_RADIUS:
                    if self._recenter_since is None:
                        self._recenter_since = t
                    elif t - self._recenter_since >= config.RECENTER_HOLD_S:
                        self.hand_ref = (float(wrist_norm[0]), float(wrist_norm[1]))
                        self.recenter_required = False
                        self._recenter_since = None
                        self.log.info("mirror: hand re-centred at (%.2f, %.2f); reference set", *self.hand_ref)
                else:
                    self._recenter_since = None
                return
            if self.hand_ref is None:
                self.hand_ref = (float(wrist_norm[0]), float(wrist_norm[1]))
                self.log.info("mirror: reference set at (%.2f, %.2f)", *self.hand_ref)
            self.target = hand_to_target(wrist_norm, self.hand_ref, self.origin, self.box)

    # -- mode changes ---------------------------------------------------------------
    def sync_to_arm(self):
        """Adopt the arm's real position as the commanded point (after a routine / at start).

        Returns the position, or None when it is unknown - in which case a live controller refuses to
        stream (streaming from a guessed point could traverse the whole workspace in one command).
        """
        pos = None
        dry = getattr(self.arm, "dry_run", False)
        if not dry:
            for _ in range(3):
                try:
                    pos = self.arm.read_xyz()
                except Exception as e:  # noqa: BLE001
                    self.log.warning("mirror: read_xyz failed during sync: %r", e)
                if pos is not None:
                    break
        if pos is None:
            pos = getattr(self.arm, "commanded", None)
        with self._lock:
            if pos is not None:
                self.commanded = tuple(float(v) for v in pos)
                self.actual, self.actual_t = self.commanded, time.time()
                self.position_known = True
            else:
                self.position_known = bool(dry)   # dry-run may start from the origin; live may not
            self.target = self.commanded if self.commanded is not None else self.origin
        if pos is None and not dry:
            self.log.error("mirror: arm position UNKNOWN (no read-back); mirroring stays disabled until a "
                           "sync succeeds (thumbs-up HOME re-syncs)")
        else:
            self.log.info("mirror: synced to arm position %s", self.commanded)
        return pos

    def resume(self, recenter: bool = True, epoch: int | None = None) -> bool:
        """Enable mirroring. A resume decided before a later freeze()/pause() (epoch mismatch) is ignored."""
        with self._lock:
            if epoch is not None and epoch != self.epoch:
                self.log.warning("mirror: stale resume ignored (epoch %d != %d)", epoch, self.epoch)
                return False
            if not self.position_known:
                self.log.error("mirror: resume refused: arm position unknown")
                return False
            self.enabled = True
            self.frozen = False
            self.holding = False
            self.recenter_required = recenter
            self._recenter_since = None
            if recenter:
                self.hand_ref = None
            self.target = self.commanded if self.commanded is not None else self.origin
        release = getattr(self.arm, "release_halt", None)
        if release is not None:
            release()
        self.log.info("mirror: resumed (recenter=%s)", recenter)
        return True

    def pause(self) -> None:
        with self._lock:
            self.enabled = False
            self.epoch += 1
        self.log.info("mirror: paused")

    def freeze(self) -> None:
        with self._lock:
            self.enabled = False
            self.frozen = True
            self.epoch += 1
        self.log.info("mirror: FREEZE -> halting arm")
        try:
            self.arm.halt()
        except Exception as e:  # noqa: BLE001 - the freeze itself must never raise into the UI loop
            self.errors += 1
            self.log.error("mirror: halt failed: %r", e)
        with self._lock:
            if self.commanded is not None:
                self.target = self.commanded

    def stop(self) -> None:
        self._stop_evt.set()
        if self.is_alive():
            self.join(timeout=2.0)

    # -- loop -------------------------------------------------------------------------
    def snapshot(self) -> dict:
        with self._lock:
            return {
                "enabled": self.enabled, "frozen": self.frozen, "holding": self.holding,
                "recenter": self.recenter_required, "target": self.target, "commanded": self.commanded,
                "actual": self.actual, "ticks": self.ticks, "late": self.late_ticks,
                "commands": self.commands, "refused": self.refused, "detours": self.detours,
                "blocked": self.blocked, "epoch": self.epoch, "position_known": self.position_known,
            }

    def run(self) -> None:
        next_t = time.monotonic()
        last_readback = 0.0
        while not self._stop_evt.is_set():
            now = time.monotonic()
            if now < next_t:
                time.sleep(min(self.period, next_t - now))
                continue
            if now - next_t > self.period:
                self.late_ticks += 1
                next_t = now            # never replay missed ticks back-to-back (that would burst commands)
            next_t += self.period
            self.ticks += 1
            self._tick(time.time())
            if (not getattr(self.arm, "dry_run", False) and time.time() - last_readback >= config.READBACK_EVERY_S):
                with self._lock:
                    want = self.enabled or self.frozen
                if want:
                    last_readback = time.time()
                    try:
                        pos = self.arm.read_xyz()
                    except Exception as e:  # noqa: BLE001
                        pos = None
                        self.log.warning("mirror: read_xyz failed: %r", e)
                    if pos is not None:
                        with self._lock:
                            self.actual, self.actual_t = tuple(float(v) for v in pos), time.time()

    def _tick(self, t: float) -> None:
        with self._lock:
            if not self.enabled or self.frozen or self.recenter_required:
                return
            if t - self.last_hand_t > config.NO_HAND_HOLD_S:
                if not self.holding:
                    self.holding = True
                    self.log.info("mirror: no hand for %.1fs -> holding position", config.NO_HAND_HOLD_S)
                return
            if not self.position_known:
                if not self._warned_unknown:
                    self._warned_unknown = True
                    self.log.error("mirror: not streaming: arm position unknown")
                return
            prev = self.commanded
            cur = prev if prev is not None else self.origin
            nxt = step_toward(cur, self.target, self.max_step)
            if prev is not None and max(abs(nxt[i] - cur[i]) for i in range(3)) < 0.5:
                return
            if not self.check(*nxt):
                # straight step leaves the envelope (typically the base keep-out on the way back from
                # HOME): swing around the base instead; if that is unsafe too, hold this tick and log.
                alt = step_toward_polar(cur, self.target, self.max_step)
                if self.check(*alt):
                    self.detours += 1
                    nxt = alt
                else:
                    self.blocked += 1
                    if self.blocked in (1, 10, 100) or self.blocked % 1000 == 0:
                        self.log.warning("mirror: no safe step from %s toward %s (straight %s, arc %s); holding",
                                         tuple(round(v) for v in cur), tuple(round(v) for v in self.target),
                                         tuple(round(v) for v in nxt), tuple(round(v) for v in alt))
                    return
            self.commanded = nxt
        try:
            self.arm.stream_to(*nxt)
            self.commands += 1
        except Exception as e:  # noqa: BLE001 - refused / serial error: log, keep the loop alive
            self.refused += 1
            self.log.error("mirror: stream_to%s refused: %r", tuple(round(v, 1) for v in nxt), e)
            with self._lock:
                self.commanded = prev   # nothing was accepted: assume nothing
