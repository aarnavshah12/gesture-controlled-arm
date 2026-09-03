"""HOME / PICK / FLOURISH, each run in its own thread so the overlay keeps drawing.

Abort: the state machine sets the abort flag on FREEZE. The driver pumps `arm.tick` every ~50 ms
while it waits for a move, and between steps the routines check the flag themselves, so a fist
unwinds the routine within ~50 ms (the arm itself was halted by the controller's freeze()).

PICK reuses the block picker AS-IS: its Detector (its model), its calibration.npy (homography),
its overhead camera + UVC settings and its PickLoop, run for one cycle. Nothing is re-implemented.
"""

from __future__ import annotations

import threading
import time
from typing import Callable

from . import bp, config, runlog
from .motion import clamp_box

try:  # the driver's exception types, when the block picker is present
    _bp_arm = bp.load().arm
    MoveRefused, UnsafeTarget = _bp_arm.MoveRefused, _bp_arm.UnsafeTarget
except Exception:  # noqa: BLE001 - dry-run without the block picker
    class MoveRefused(Exception):  # type: ignore[no-redef]
        pass

    class UnsafeTarget(Exception):  # type: ignore[no-redef]
        pass

DEFAULT_HEIGHTS = (84.0, 40.0, 20.0)   # pick z, hover, place lift: the block picker's values, for dry-run without it


class RoutineAborted(Exception):
    pass


class Routines:
    NAMES = ("HOME", "PICK", "FLOURISH", "GRAB", "PLACE")

    def __init__(self, arm, dry_run: bool, log=None, mac_camera_index: int | None = None, box=None):
        self.arm = arm
        self.dry_run = dry_run
        self.mac_camera_index = mac_camera_index   # PICK must never open the camera the main loop is using
        self.box = box                             # the steering box: GRAB/PLACE columns are clamped into its x/y
        self.log = log or runlog.get_logger()
        self._abort = threading.Event()
        self._thread: threading.Thread | None = None
        self.current: str | None = None
        self.started_at = 0.0
        self.status = ""
        self._pick_detector = None
        arm.tick = self._tick  # pumped by the driver during every wait()

    # -- control ------------------------------------------------------------------
    def _tick(self) -> None:
        # Only the routine thread unwinds on abort: the driver also pumps tick() during a plain
        # grip()/release() vent from other threads, which must complete normally.
        if self._abort.is_set() and threading.current_thread() is self._thread:
            raise RoutineAborted()

    def _check(self, where: str) -> None:
        if self._abort.is_set():
            raise RoutineAborted(where)

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, name: str, done: Callable[[bool], None]) -> None:
        if name not in self.NAMES:
            self.log.error("unknown routine %s", name)
            done(False)
            return
        if self.running:
            self._thread.join(timeout=0.05)         # a routine that just reported done() may still be exiting
        if self.running:
            # never block the UI thread on an aborted routine that is still unwinding (e.g. inside a model
            # load with no tick): refuse now, the operator repeats the gesture once it has stopped
            self.log.warning("routine %s requested while %s is still %s; ignored - repeat the gesture",
                             name, self.current, "stopping" if self._abort.is_set() else "running")
            done(False)
            return
        self._abort.clear()
        self.current = name
        self.started_at = time.time()
        self.status = "starting"
        self._thread = threading.Thread(target=self._run, args=(name, done), name=f"routine-{name}", daemon=True)
        self._thread.start()

    def abort(self) -> None:
        if self.running:
            self.log.info("routine %s: ABORT requested", self.current)
        self._abort.set()

    def abort_and_join(self, timeout: float = 3.0) -> bool:
        """Abort and wait for the routine thread to unwind; False if it is still alive after `timeout`."""
        self.abort()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        return not self.running

    def _run(self, name: str, done: Callable[[bool], None]) -> None:
        ok = False
        t0 = time.time()
        self.log.info("routine %s: start (dry_run=%s)", name, self.dry_run)
        try:
            {"HOME": self._home, "PICK": self._pick, "FLOURISH": self._flourish,
             "GRAB": self._grab, "PLACE": self._place}[name]()
            ok = True
            self.log.info("routine %s: done in %.1fs", name, time.time() - t0)
        except RoutineAborted as e:
            self.log.warning("routine %s: aborted by fist after %.1fs (%s)", name, time.time() - t0, e or "in wait")
        except SystemExit as e:  # the block picker's pre-flight checks exit instead of raising
            self.log.error("routine %s: block picker refused to run: %s", name, e)
        except Exception as e:  # noqa: BLE001 - a failed routine must never kill the app
            self.log.error("routine %s: failed: %r", name, e, exc_info=True)
        finally:
            self.status = ""
            done(ok)
            self.current = None

    # -- routines -------------------------------------------------------------------
    def _heights(self) -> tuple[float, float, float]:
        """(pick z, hover offset, place lift) from config, falling back to the block picker's values."""
        pick_z, hover, lift = config.GRAB_Z_MM, config.GRAB_HOVER_MM, config.PLACE_LIFT_MM
        if pick_z is None or hover is None or lift is None:
            try:
                c = bp.load().config
            except bp.BlockPickerMissing:
                if not self.dry_run:
                    raise
                self.log.warning("block picker absent: GRAB/PLACE heights default to %s (dry-run only)", DEFAULT_HEIGHTS)
                d = DEFAULT_HEIGHTS
                return (float(pick_z if pick_z is not None else d[0]), float(hover if hover is not None else d[1]),
                        float(lift if lift is not None else d[2]))
            if pick_z is None:
                pick_z = c.require("TABLE_Z_MM") + c.BLOCK_HEIGHT_MM - c.CUP_PRESS_MM
            if hover is None:
                hover = c.HOVER_OFFSET_MM
            if lift is None:
                lift = c.RELEASE_LIFT_MM
        return float(pick_z), float(hover), float(lift)

    def _where(self) -> tuple[float, float, float]:
        """The arm's current position: read-back when live, else the last commanded point."""
        pos = None
        if not self.dry_run:
            self.arm.wait(config.STREAM_MOVE_MS / 1000.0 + 0.1)   # the last streamed move may still be running
            pos = self.arm.read_xyz()
            if pos is None:
                pos = self.arm.read_xyz()
        if pos is None:
            pos = getattr(self.arm, "commanded", None)
        if pos is None:
            raise RuntimeError("arm position unknown: cannot grab/place here")
        x, y, z = (float(v) for v in pos)
        # read-back is ~8 mm noisy: keep the column inside the steering box (the calibrated pick area) so a
        # reading just past the edge does not turn into a refused move or a descent outside the area
        if self.box is not None:
            (xlo, xhi), (ylo, yhi), _ = self.box
            x, y = min(max(x, xlo), xhi), min(max(y, ylo), yhi)
        return x, y, z

    def _descend(self, x: float, y: float, hover_z: float, low_z: float, what: str) -> None:
        """Hover, then slowly down. A refused descent (something in the way) retreats to hover first."""
        self.arm.move_to(x, y, hover_z)
        try:
            self.arm.move_to(x, y, low_z, config.GRAB_DESCENT_MS)
        except MoveRefused:
            self.log.error("routine %s: descent to z=%.0f refused (obstacle?); retreating to hover", what, low_z)
            try:
                self.arm.move_to(x, y, hover_z)
            except Exception as e:  # noqa: BLE001
                self.log.error("routine %s: retreat refused too: %r", what, e)
            raise

    def _grab(self) -> None:
        """Descend at the current (x, y), suction on, come back up to where we were."""
        x, y, z0 = self._where()
        pick_z, hover, _ = self._heights()
        z_top = max(z0, pick_z + hover)
        self.log.info("routine GRAB: at (%.0f, %.0f) from z=%.0f: hover %.0f, pick %.0f, back to %.0f",
                      x, y, z0, pick_z + hover, pick_z, z_top)
        self._check("before grab")
        self.status = "descending"
        self._descend(x, y, pick_z + hover, pick_z, "GRAB")
        self.status = "suction on"
        self.arm.suction(True)
        self.arm.wait(config.GRAB_SUCTION_PAUSE_S)
        self.status = "lifting"
        self.arm.move_to(x, y, pick_z + hover)
        self.arm.move_to(x, y, z_top)

    def _place(self) -> None:
        """Descend at the current (x, y), release just above the pick height, come back up."""
        x, y, z0 = self._where()
        pick_z, hover, lift = self._heights()
        z_top = max(z0, pick_z + hover)
        self.log.info("routine PLACE: at (%.0f, %.0f) from z=%.0f: hover %.0f, release at %.0f, back to %.0f",
                      x, y, z0, pick_z + hover, pick_z + lift, z_top)
        self._check("before place")
        self.status = "descending"
        self._descend(x, y, pick_z + hover, pick_z + lift, "PLACE")
        self.status = "releasing"
        # Block-picker sequence (pick.py): pump off + valve OPEN, wait, lift clear, THEN close the valve.
        # Closing it while the bellows cup is still on the block re-grabs it like a sucker (seen 2026-09-03).
        self.arm.vent()
        self.arm.wait(config.PLACE_VENT_S)
        self.status = "lifting"
        self.arm.move_to(x, y, pick_z + hover)
        self.arm.valve_close()
        self.arm.move_to(x, y, z_top)

    def _home(self) -> None:
        self.status = "going home"
        self._check("before home")
        self.arm.home()

    def _flourish(self) -> None:
        ox, oy, oz = config.MIRROR_ORIGIN_XYZ_MM
        self.status = "wave"
        self._check("before flourish")
        self.arm.move_to(ox, oy, oz, 800)
        for i, (dx, dy, dz, ms) in enumerate(config.FLOURISH_STEPS):
            self._check(f"flourish step {i}")
            self.status = f"wave {i + 1}/{len(config.FLOURISH_STEPS)}"
            self.arm.move_to(ox + dx, oy + dy, oz + dz, ms)

    def _pick(self) -> None:
        problems = bp.check()
        if problems:
            raise bp.BlockPickerMissing("; ".join(problems))
        b = bp.load()
        self.status = "loading block picker"
        cal = b.mapping.load_calibration(bp.calibration_path())
        H = cal["H"]
        b.pick.check_fixed_targets()   # SystemExit on a bad config (caught in _run)
        self._check("before pick detector")
        if self._pick_detector is None:
            t0 = time.time()
            self._pick_detector = b.detect.Detector()
            self.log.info("routine PICK: block-picker model %s loaded in %.1fs classes=%s",
                          self._pick_detector.model_id, time.time() - t0, self._pick_detector.class_names)
        det = self._pick_detector
        camera = b.config.WEBCAM_INDEX if config.PICK_CAMERA_INDEX is None else config.PICK_CAMERA_INDEX
        if self.mac_camera_index is not None and camera == self.mac_camera_index:
            raise RuntimeError(f"overhead camera index {camera} is the camera the gesture loop is using; is the "
                               f"overhead webcam plugged in? (block-picker WEBCAM_INDEX / gesture PICK_CAMERA_INDEX)")
        self.status = "opening overhead camera"
        cap = b.detect.open_camera(camera)
        try:
            b.mapping.check_frame_size(cal, b.detect.grab(cap))
            self._check("before pick loop")
            self.status = "pick loop"
            loop = b.pick.PickLoop(det, self.arm, lambda: b.detect.grab(cap), H, dry_run=self.dry_run,
                                   once=True, max_cycles=config.PICK_MAX_CYCLES, show=None)
            picks = loop.run()
            self.log.info("routine PICK: %d pick(s), %d ignored spots", picks, len(loop.ignored))
        finally:
            cap.release()
