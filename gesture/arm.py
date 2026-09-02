"""The only gesture-side code that talks to the arm: a thin layer over the block picker's driver.

The block picker's arm.py (imported as-is through gesture.bp) is the hardware-validated Hiwonder
MaxArm serial driver: frames `AA 55 | func | len | data | chk`, 9600 8N1, FUNC_SET_XYZ 0x03,
FUNC_SET_SUCTIONNOZZLE 0x07, FUNC_READ_XYZ 0x13, no completion ack, reach-limit / table-Z checks
before any byte is sent, "Workspace clear?" once per session. GestureArm adds:

- stream_to(x, y, z): the 10 Hz mirroring command - validated with the driver's check_target, sent,
  NOT waited on (the driver's move_to blocks for the move duration and verifies by read-back, which
  is right for routines and wrong for a control loop);
- halt(): stop where you are (re-command the read-back position with a short duration);
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
        self.gripping = False
        self.glog = runlog.get_logger()

    # -- serial access serialised across threads ------------------------------------
    def _send(self, frame: bytes) -> None:
        with self.lock:
            super()._send(frame)

    def _query(self, func: int, reply_len: int):
        with self.lock:
            return super()._query(func, reply_len)

    # -- blocking moves (routines) record the commanded point --------------------------
    def move_to(self, x, y, z, ms=None):
        self.commanded = (float(x), float(y), float(z))
        return super().move_to(x, y, z, ms)

    # -- streaming (mirroring) ---------------------------------------------------------
    def stream_to(self, x: float, y: float, z: float, ms: int | None = None) -> None:
        """Validate (driver rules) and send one positional command without waiting."""
        ms = config.STREAM_MOVE_MS if ms is None else int(ms)
        _bp.arm.check_target(x, y, z)  # UnsafeTarget: nothing is sent
        frame = _bp.arm.set_xyz_frame(x, y, z, ms)
        if self.dry_run:
            self.commanded = (float(x), float(y), float(z))
            self.glog.info("[dry-run] arm: stream_to(%.1f, %.1f, %.1f) %d ms", x, y, z, ms)
            return
        if not self.cleared:
            raise ArmError("workspace not confirmed clear; streaming refused")
        self.glog.info("arm: stream_to(%.1f, %.1f, %.1f) %d ms frame=%s", x, y, z, ms, frame.hex())
        self._send(frame)
        self.commanded = (float(x), float(y), float(z))

    def halt(self) -> None:
        """Stop where you are: read the real position and re-command it with a short duration."""
        with self.lock:
            pos = None
            if not self.dry_run:
                pos = self.read_xyz()
                if pos is None:
                    pos = self.read_xyz()
            if pos is None:
                pos = self.commanded
            if pos is None:
                self.glog.warning("arm: halt requested but position unknown and nothing commanded yet")
                return
            # read-back is ~8 mm noisy; clamp into the reach box so the driver's check never refuses a stop
            x, y, z = clamp_box(pos, reach_box())
            z = max(z, float(_bp.config.require("TABLE_Z_MM")))
            self.glog.info("arm: HALT at read-back %s -> command (%.0f, %.0f, %.0f) %d ms", pos, x, y, z,
                           config.HALT_MOVE_MS)
            if self.dry_run:
                self.commanded = (x, y, z)
                return
            if not self.cleared:
                return
            self._send(_bp.arm.set_xyz_frame(x, y, z, config.HALT_MOVE_MS))
            self.commanded = (x, y, z)

    # -- gripper = suction ----------------------------------------------------------------
    def grip(self) -> None:
        self.glog.info("arm: GRIP (suction on)")
        self.suction(True)          # each frame is locked individually; never hold the lock across a wait
        self.gripping = True

    def release(self) -> None:
        self.glog.info("arm: RELEASE (vent, valve close)")
        self.suction(False)         # vent for the block picker's VENT_S, then valve close
        self.gripping = False


def make_arm(dry_run: bool, port: str | None = None) -> GestureArm:
    log = runlog.get_logger()
    log.info("rig values from block picker: %s", bp.rig_values())
    return GestureArm(dry_run=dry_run, port=port)
