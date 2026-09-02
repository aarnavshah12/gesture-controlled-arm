#!/usr/bin/env python
"""Gesture Arm: webcam -> MediaPipe landmarks + Roboflow gesture model -> state machine -> MaxArm.

    python gesture_arm.py --dry-run          full pipeline + full overlay, planned targets drawn, NO motion
    python gesture_arm.py                    live: asks "Workspace clear?" once, then mirrors + gestures
    python gesture_arm.py --clean            no status strip (for filming); everything else stays
    python gesture_arm.py --windowed         normal window instead of full screen
    python gesture_arm.py --headless --frames 60 --save-overlay   no window; saves one overlay JPEG (tests)

Keys: q / Esc quit, c toggle the status strip, f toggle full screen, r re-centre the hand reference.
Gestures: fist FREEZE, open-palm RELEASE/resume, pinch GRIP, thumbs-up HOME, point PICK, peace FLOURISH.
"""

from __future__ import annotations

import argparse
import collections
import queue
import sys
import threading
import time

import cv2

from gesture import bp, config, runlog, viz
from gesture.camera import Camera
from gesture.gestures import Actions, Debouncer, StateMachine, FROZEN, MIRROR, ROUTINE
from gesture.motion import MotionController
from gesture.perception import DetectorWorker, GestureDetector, HandTracker

TOAST = {
    "FREEZE": ("FROZEN", config.GESTURE_COLOURS["fist"]),
    "RELEASE": ("RELEASE", config.GESTURE_COLOURS["open-palm"]),
    "GRIP": ("GRIP", config.GESTURE_COLOURS["pinch"]),
    "HOME": ("GOING HOME", config.GESTURE_COLOURS["thumbs-up"]),
    "PICK": ("PICK ROUTINE", config.GESTURE_COLOURS["point"]),
    "FLOURISH": ("FLOURISH", config.GESTURE_COLOURS["peace"]),
}


class NullArm:
    """Dry-run stand-in used ONLY when the block-picker project is not on this machine."""
    dry_run = True
    cleared = True
    tick = None

    def __init__(self, log):
        self.log = log
        self.commanded = None

    def connect(self):
        self.log.warning("arm: block picker not found -> NullArm (dry-run only, nothing is sent anywhere)")
        return self

    def stream_to(self, x, y, z, ms=None):
        self.commanded = (float(x), float(y), float(z))
        self.log.info("[dry-run] arm: stream_to(%.1f, %.1f, %.1f)", x, y, z)

    def move_to(self, x, y, z, ms=None):
        self.commanded = (float(x), float(y), float(z))
        self.log.info("[dry-run] arm: move_to(%.1f, %.1f, %.1f) %s ms", x, y, z, ms)
        if self.tick:
            for _ in range(int((ms or 1000) / 50)):
                self.tick()
                time.sleep(0.05)

    def home(self, ms=1500):
        self.log.info("[dry-run] arm: home")
        self.move_to(*(config.MIRROR_ORIGIN_XYZ_MM), ms)

    def halt(self):
        self.log.info("[dry-run] arm: HALT at %s", self.commanded)

    def grip(self):
        self.log.info("[dry-run] arm: GRIP")

    def release(self):
        self.log.info("[dry-run] arm: RELEASE")

    def read_xyz(self):
        return None

    def close(self):
        pass


class ArmOps(threading.Thread):
    """Single worker that runs blocking gripper ops (vent takes ~1 s) off the UI thread, in order."""

    def __init__(self, log):
        super().__init__(name="gesture-armops", daemon=True)
        self.q: queue.Queue = queue.Queue()
        self.log = log
        self.start()

    def submit(self, fn, label: str) -> None:
        self.q.put((fn, label))

    def run(self) -> None:
        while True:
            fn, label = self.q.get()
            if fn is None:
                return
            try:
                fn()
            except Exception as e:  # noqa: BLE001
                self.log.error("arm op %s failed: %r", label, e)

    def stop(self) -> None:
        self.q.put((None, "stop"))


class AppActions(Actions):
    def __init__(self, arm, controller: MotionController, routines, toasts: viz.Toasts, ops: ArmOps, log):
        self.arm, self.controller, self.routines, self.toasts, self.ops, self.log = arm, controller, routines, toasts, ops, log

    def freeze(self) -> None:
        self.controller.freeze()          # immediate: halts the arm on the UI thread

    def release(self) -> None:
        self.ops.submit(self.arm.release, "release")

    def grip(self) -> None:
        self.ops.submit(self.arm.grip, "grip")

    def resume_mirror(self, recenter: bool) -> None:
        def _resume():
            self.controller.sync_to_arm()
            self.controller.resume(recenter)
        self.ops.submit(_resume, "resume-mirror")   # after any queued release, in order

    def pause_mirror(self) -> None:
        self.controller.pause()

    def start_routine(self, name: str, done) -> None:
        self.routines.start(name, done)

    def abort_routine(self) -> None:
        self.routines.abort()


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="full overlay + planned targets, no motion, no confirmation")
    ap.add_argument("--clean", action="store_true", help="strip the status strip for filming")
    ap.add_argument("--windowed", action="store_true", help="normal window instead of full screen")
    ap.add_argument("--headless", action="store_true", help="no window at all (tests)")
    ap.add_argument("--frames", type=int, default=0, help="stop after N frames (0 = until q)")
    ap.add_argument("--save-overlay", action="store_true", help="save the last overlay frame to logs/")
    ap.add_argument("--camera", type=int, default=None, help="Mac camera index (default: auto, built-in)")
    ap.add_argument("--port", default=None, help="arm serial port (default: block-picker config)")
    ap.add_argument("--conf", type=float, default=None, help="gesture confidence threshold (default config)")
    ap.add_argument("--debounce", type=int, default=None, help="consecutive accepted predictions (default config)")
    ap.add_argument("--every", type=int, default=None, help="run the gesture model every Nth frame")
    ap.add_argument("--record", metavar="FILE.mp4", help="record the overlay to a video file")
    ap.add_argument("--quiet", action="store_true", help="log to file only")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    log = runlog.start_run("gesture-dry" if args.dry_run else "gesture", quiet=args.quiet)
    log.info("args=%s", vars(args))
    for p in bp.check():
        log.warning("block picker: %s", p)

    cam = Camera(args.camera).open()
    tracker = HandTracker()
    log.info("hand landmarker loaded in %.2fs (%s)", tracker.load_seconds, tracker.model_path)
    detector = GestureDetector()
    worker = DetectorWorker(detector, args.every).start()

    # Arm: the block picker's driver, or a stand-in when its project is absent (dry-run only).
    box = (config.MIRROR_X_MM, config.MIRROR_Y_MM, config.MIRROR_Z_MM)
    check = None
    try:
        from gesture.arm import make_arm, mirror_box, UnsafeTarget, _bp
        arm = make_arm(dry_run=args.dry_run, port=args.port)
        box = mirror_box()

        def check(x, y, z):
            try:
                _bp.arm.check_target(x, y, z)
                return True
            except UnsafeTarget:
                return False
    except bp.BlockPickerMissing as e:
        if not args.dry_run:
            raise SystemExit(f"cannot run live without the block-picker project: {e}")
        log.warning("%s", e)
        arm = NullArm(log)
    arm.connect()
    if not args.dry_run:
        arm.confirm_workspace_clear()   # once per session, before anything can move
    log.info("mirror box x%s y%s z%s origin=%s gains x=%.0f z=%.0f mm/frame cap=%.0f mm/s @ %.0f Hz",
             *box, config.MIRROR_ORIGIN_XYZ_MM, config.MIRROR_GAIN_X_MM, config.MIRROR_GAIN_Z_MM,
             config.VELOCITY_CAP_MM_S, config.CONTROL_HZ)

    from gesture.routines import Routines
    controller = MotionController(arm, box, check=check)
    controller.sync_to_arm()
    controller.start()
    routines = Routines(arm, dry_run=args.dry_run)
    toasts = viz.Toasts()
    ops = ArmOps(log)
    actions = AppActions(arm, controller, routines, toasts, ops, log)
    sm = StateMachine(actions)
    deb = Debouncer(args.debounce, args.conf)
    controller.resume(recenter=True)

    clean = args.clean
    fullscreen = not args.windowed
    show = not args.headless
    if show:
        cv2.namedWindow(config.WINDOW_NAME, cv2.WINDOW_NORMAL)
        if fullscreen:
            cv2.setWindowProperty(config.WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    writer = None
    trail: collections.deque = collections.deque(maxlen=config.WRIST_TRAIL_LEN)
    ema = None
    fps, t_prev = 0.0, time.time()
    last_res = None
    img = None
    i = 0
    bad_reads = 0
    try:
        while True:
            frame = cam.read()
            t = time.time()
            if frame is None:
                bad_reads += 1
                log.error("camera read failed (%d)", bad_reads)
                if bad_reads >= 30:
                    raise RuntimeError("camera stopped delivering frames")
                time.sleep(0.05)
                continue
            bad_reads = 0
            i += 1
            dt = t - t_prev
            t_prev = t
            fps = (1.0 / dt) if fps == 0.0 else 0.9 * fps + 0.1 / max(dt, 1e-6)

            # continuous stream: landmarks every frame -> smoothed wrist -> controller
            hand = tracker.process(frame, t)
            if hand is not None:
                w = hand.wrist_norm
                a = config.WRIST_SMOOTHING
                ema = w if ema is None else (ema[0] + a * (w[0] - ema[0]), ema[1] + a * (w[1] - ema[1]))
                trail.append((ema[0] * frame.shape[1], ema[1] * frame.shape[0]))
                controller.update_hand(ema, t)
            else:
                if trail:
                    trail.popleft()
                controller.update_hand(None, t)

            # command channel: gesture model every Nth frame in the worker -> debounce -> events
            worker.submit(frame, i, t)
            res = worker.poll()
            if res is not None:
                last_res = res
                ev = deb.update(res.top, t)
                if ev is not None:
                    text, colour = TOAST[ev.name]
                    if ev.name == "RELEASE" and sm.mode == FROZEN:
                        text = "RESUME"
                    toasts.add(text, colour, t)
                    sm.on_event(ev)

            # overlay
            pred = last_res.top if (last_res is not None and t - last_res.t < 1.0) else None
            progress = deb.progress if (pred is not None and pred.cls == deb.cls) else 0.0
            snap = controller.snapshot()
            if sm.mode == FROZEN:
                sub = "open palm to resume"
            elif sm.mode == ROUTINE:
                sub = f"{routines.current or ''}  {routines.status}".strip()
            elif snap["recenter"]:
                sub = "centre your hand to start mirroring"
            elif snap["holding"]:
                sub = "no hand - holding"
            else:
                sub = ""
            arm_xyz = snap["actual"] if snap["actual"] is not None else snap["commanded"]
            status = dict(fps=fps, infer_ms=(last_res.ms if last_res else 0.0), hand_ms=tracker.last_ms,
                          arm_xyz=arm_xyz, conf=(pred.conf if pred else None), gesture=(pred.cls if pred else None),
                          extra=f"debounce {deb.count}/{deb.n}", dry_run=args.dry_run)
            target_map = None
            if not clean or args.dry_run:
                target_map = dict(box=box, target=snap["target"], commanded=snap["commanded"], actual=snap["actual"],
                                  label="planned target (dry run)" if args.dry_run else "arm target")
            img = viz.render(frame, hand=hand, connections=tracker.connections, pred=pred, progress=progress,
                             trail=list(trail), mode=sm.mode, mode_sub=sub, toasts=toasts, status=status,
                             clean=clean, target_map=target_map, t=t)
            if args.record:
                if writer is None:
                    hh, ww = img.shape[:2]
                    writer = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"), 20, (ww, hh))
                    log.info("recording overlay to %s (%dx%d)", args.record, ww, hh)
                writer.write(img)
            if show:
                cv2.imshow(config.WINDOW_NAME, img)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("c"):
                    clean = not clean
                if key == ord("f"):
                    fullscreen = not fullscreen
                    cv2.setWindowProperty(config.WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                                          cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
                if key == ord("r"):
                    controller.resume(recenter=True)
            if args.frames and i >= args.frames:
                break
    except KeyboardInterrupt:
        log.info("interrupted")
    finally:
        log.info("shutting down: frames=%d fps=%.1f detector submitted=%d skipped=%d errors=%d controller=%s",
                 i, fps, worker.submitted, worker.skipped_busy, worker.errors, controller.snapshot())
        routines.abort()
        controller.stop()
        worker.stop()
        ops.stop()
        tracker.close()
        if args.save_overlay and img is not None:
            log.info("saved overlay %s", runlog.save_frame(img, "overlay"))
        try:
            if not args.dry_run and getattr(arm, "cleared", False):
                arm.halt()
                arm.release()
                arm.home()
        except BaseException as e:  # noqa: BLE001 - always release the port
            log.error("cleanup: %r", e)
        finally:
            arm.close()
            cam.release()
            if writer is not None:
                writer.release()
                log.info("saved recording %s", args.record)
            if show:
                cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
