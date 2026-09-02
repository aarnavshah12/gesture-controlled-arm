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


class RoutineAborted(Exception):
    pass


class Routines:
    NAMES = ("HOME", "PICK", "FLOURISH")

    def __init__(self, arm, dry_run: bool, log=None, mac_camera_index: int | None = None):
        self.arm = arm
        self.dry_run = dry_run
        self.mac_camera_index = mac_camera_index   # PICK must never open the camera the main loop is using
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
            self.log.warning("routine %s requested while %s is running; ignored", name, self.current)
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

    def _run(self, name: str, done: Callable[[bool], None]) -> None:
        ok = False
        t0 = time.time()
        self.log.info("routine %s: start (dry_run=%s)", name, self.dry_run)
        try:
            {"HOME": self._home, "PICK": self._pick, "FLOURISH": self._flourish}[name]()
            ok = True
            self.log.info("routine %s: done in %.1fs", name, time.time() - t0)
        except RoutineAborted as e:
            self.log.warning("routine %s: aborted by fist after %.1fs (%s)", name, time.time() - t0, e or "in wait")
        except SystemExit as e:  # the block picker's pre-flight checks exit instead of raising
            self.log.error("routine %s: block picker refused to run: %s", name, e)
        except Exception as e:  # noqa: BLE001 - a failed routine must never kill the app
            self.log.error("routine %s: failed: %r", name, e, exc_info=True)
        finally:
            self.current = None
            self.status = ""
            done(ok)

    # -- routines -------------------------------------------------------------------
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
