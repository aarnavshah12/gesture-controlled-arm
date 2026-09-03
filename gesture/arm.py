"""The only gesture-side code that talks to the arm: a thin layer over the block picker's driver.

The block picker's arm.py (imported as-is through gesture.bp) is the hardware-validated Hiwonder
MaxArm serial driver: frames `AA 55 | func | len | data | chk`, 9600 8N1, FUNC_SET_XYZ 0x03,
FUNC_SET_SUCTIONNOZZLE 0x07, FUNC_READ_XYZ 0x13, no completion ack, reach-limit / table-Z checks
before any byte is sent, "Workspace clear?" once per session. GestureArm adds:

- stream_to(x, y, z): the 10 Hz mirroring command - validated with the driver's check_target, sent,
  NOT waited on (the driver's move_to blocks for the move duration and verifies by read-back, which
  is right for routines and wrong for a control loop);
- halt(): stop where you are (re-command the read-back position with a short duration) and INHIBIT
  every further motion frame until release_halt() - so a routine thread or a control tick that was
  between waits when the fist fired cannot send one more move after the stop;
- grip() / release(): this arm has a suction nozzle, so GRIP = pump on, RELEASE = vent + valve close;
- a re-entrant lock so the control thread, the routine thread and the UI thread never interleave
  bytes on the serial port.
"""

from __future__ import annotations

import threading

from . import bp, config, runlog
from .motion import Box, clamp_box, intersect_box

_bp = bp.load()
_Base = _bp.arm.Arm
UnsafeTarget = _bp.arm.UnsafeTarget
MoveRefused = _bp.arm.MoveRefused
ArmError = _bp.arm.ArmError


def reach_box() -> Box:
    c = _bp.config
    return (tuple(c.require("REACH_X_MM")), tuple(c.require("REACH_Y_MM")), tuple(c.require("REACH_Z_MM")))


def mirror_box() -> Box:
    """config.MIRROR_* intersected with the rig's reach limits and floored at table Z."""
    c = _bp.config
    box = intersect_box((config.MIRROR_X_MM, config.MIRROR_Y_MM, config.MIRROR_Z_MM), reach_box())
    zlo = max(box[2][0], float(c.require("TABLE_Z_MM")))
    return (box[0], box[1], (zlo, box[2][1]))


class GestureArm(_Base):
    def __init__(self, dry_run: bool = False, port: str | None = None):
        super().__init__(port=port, dry_run=dry_run)
        self.lock = threading.RLock()
        self.commanded: tuple[float, float, float] | None = None
        self.gripping = False          # pump on (kept current by the suction()/vent() overrides below)
        self.inhibited = False         # set by halt(): motion frames are refused until release_halt()
        self._halting = False
        self.glog = runlog.get_logger()

    # -- motion inhibit (the dead-man switch, enforced where the bytes leave) -------------
    def _check_motion_allowed(self) -> None:
        if self.tick is not None:
            self.tick()                # routine thread: unwinds with RoutineAborted after a fist
        if self.inhibited:
            raise ArmError("motion inhibited: arm is halted (FREEZE); open-palm resumes")

    def release_halt(self) -> None:
        with self.lock:
            if self.inhibited:
                self.glog.info("arm: motion re-enabled")
            self.inhibited = False

    # -- connection ------------------------------------------------------------------------
    def connect(self):
        """The driver's connect(), with the macOS "stuck USB-serial driver" failure explained.

        termios EINVAL on open means the OS driver for the arm's USB-serial chip refuses every settings
        change (seen 2026-09-03 after an unclean disconnect). Nothing in software fixes it: the cable must
        be unplugged and plugged back in (or the Mac rebooted).
        """
        import termios

        try:
            return super().connect()
        except termios.error as e:
            raise ArmError(
                f"cannot configure the arm's serial port ({e}): the macOS USB-serial driver is stuck. "
                f"Unplug the arm's USB cable from the Mac, plug it back in, and run again. (If the pump is "
                f"running, switch the arm off and on: it boots with the pump off.)") from e

    def ensure_pump_off(self) -> None:
        """Session start: a previous process that died mid-grab leaves the pump running; vent once."""
        if self.dry_run:
            return
        self.glog.info("arm: session start -> pump off, vent, valve close")
        self.suction(False)

    # -- serial access serialised across threads ------------------------------------
    def _send(self, frame: bytes) -> None:
        with self.lock:
            # Atomic with halt(): once the stop frame is out, no move frame can follow it.
            if (self.inhibited and not self._halting
                    and frame[2] in (_bp.arm.FUNC_SET_XYZ, _bp.arm.FUNC_SET_ANGLE)):
                raise ArmError("motion inhibited: arm is halted (FREEZE); open-palm resumes")
            super()._send(frame)

    def _query(self, func: int, reply_len: int):
        with self.lock:
            return super()._query(func, reply_len)

    # -- blocking moves (routines) record the commanded point --------------------------
    def move_to(self, x, y, z, ms=None):
        self._check_motion_allowed()          # abort / inhibit observed BEFORE the frame is built or sent
        pos = super().move_to(x, y, z, ms)   # raises UnsafeTarget / MoveRefused: commanded stays as it was
        self.commanded = (float(x), float(y), float(z))
        if self.dry_run:
            # the driver returns immediately in dry-run; take the commanded time so routines (and their
            # abort path, pumped through tick()) behave as they will on the rig
            self.wait((_bp.config.MOVE_MS if ms is None else int(ms)) / 1000.0)
        return pos

    # -- streaming (mirroring) ---------------------------------------------------------
    def stream_to(self, x: float, y: float, z: float, ms: int | None = None) -> None:
        """Validate (driver rules) and send one positional command without waiting."""
        ms = config.STREAM_MOVE_MS if ms is None else int(ms)
        self._check_motion_allowed()
        _bp.arm.check_target(x, y, z)  # UnsafeTarget: nothing is sent
        frame = _bp.arm.set_xyz_frame(x, y, z, ms)
        if self.dry_run:
            self.commanded = (float(x), float(y), float(z))
            self.glog.info("[dry-run] arm: stream_to(%.1f, %.1f, %.1f) %d ms", x, y, z, ms)
            return
        if not self.cleared:
            raise ArmError("workspace not confirmed clear; streaming refused")
        self.glog.info("arm: stream_to(%.1f, %.1f, %.1f) %d ms frame=%s", x, y, z, ms, frame.hex())
        self._send(frame)               # re-checks the inhibit under the lock
        self.commanded = (float(x), float(y), float(z))

    def halt(self) -> None:
        """Stop where you are: inhibit motion, read the real position, re-command it with a short duration.

        Live: if the position cannot be read (3 tries) NO stop target is sent - a blind target could be
        hundreds of mm away and a 300 ms move there is worse than letting the in-flight (finite) move end.
        Motion stays inhibited either way until release_halt().
        """
        with self.lock:
            self.inhibited = True
            pos = None
            if self.dry_run:
                pos = self.commanded
            else:
                for _ in range(3):
                    pos = self.read_xyz()
                    if pos is not None:
                        break
            if pos is None:
                if self.dry_run:
                    self.glog.info("[dry-run] arm: HALT (nothing commanded yet); motion inhibited")
                else:
                    self.glog.error("arm: HALT: position unknown after 3 reads; NOT sending a stop target "
                                    "(blind target could jerk the arm); motion stays inhibited, the in-flight "
                                    "move ends on its own")
                return
            tol = _bp.arm.POSITION_TOLERANCE_MM
            if (self.commanded is not None and max(abs(pos[i] - self.commanded[i]) for i in range(3)) <= tol):
                # at rest at the last verified target (e.g. the cup pressed on a block after a GRAB descent):
                # re-send that target, never a noisy read-back that could push the cup further down
                x, y, z = self.commanded
                self.glog.info("arm: HALT at rest (read-back %s within %.0f mm of commanded) -> re-command "
                               "(%.0f, %.0f, %.0f) %d ms; motion inhibited", pos, tol, x, y, z, config.HALT_MOVE_MS)
            else:
                # read-back is ~8 mm noisy; clamp into the reach box so the driver's check never refuses a stop
                x, y, z = clamp_box(pos, reach_box())
                z = max(z, float(_bp.config.require("TABLE_Z_MM")))
                self.glog.info("arm: HALT at read-back %s -> command (%.0f, %.0f, %.0f) %d ms; motion inhibited",
                               pos, x, y, z, config.HALT_MOVE_MS)
            if self.dry_run:
                self.commanded = (x, y, z)
                return
            if not self.cleared:
                return
            self._halting = True
            try:
                self._send(_bp.arm.set_xyz_frame(x, y, z, config.HALT_MOVE_MS))
            finally:
                self._halting = False
            self.commanded = (x, y, z)

    # -- suction state tracking (the block picker's PickLoop calls suction()/vent() directly) ------
    def suction(self, on: bool) -> None:
        if not on:
            self.gripping = False       # the pump-off frame is the first byte out; the vent only lets go
        super().suction(on)
        self.gripping = bool(on)

    def vent(self) -> None:
        super().vent()
        self.gripping = False

    # -- gripper = suction ----------------------------------------------------------------
    def grip(self) -> None:
        self.glog.info("arm: GRIP (suction on)")
        self.suction(True)          # each frame is locked individually; never hold the lock across a wait

    def release(self) -> None:
        if self.gripping:
            self.glog.warning("arm: RELEASE while holding an object at %s - it drops here", self.commanded)
        self.glog.info("arm: RELEASE (vent, valve close)")
        self.suction(False)         # vent for the block picker's VENT_S, then valve close


def make_arm(dry_run: bool, port: str | None = None) -> GestureArm:
    log = runlog.get_logger()
    log.info("rig values from block picker: %s", bp.rig_values())
    return GestureArm(dry_run=dry_run, port=port)
