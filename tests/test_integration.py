"""Debouncer + StateMachine + MotionController + Routines + ArmOps wired as in gesture_arm.py, driven
by a scripted sequence of detections against a fake arm. No camera, no model, no serial."""
import threading
import time
import unittest

from _common import silent_logger

import gesture_arm
from gesture import config, viz
from gesture.gestures import Debouncer, Prediction, StateMachine, FROZEN, MIRROR, ROUTINE
from gesture.motion import MotionController
from gesture.routines import Routines

BOX = ((-115.0, 145.0), (-250.0, -130.0), (100.0, 200.0))


class FakeArm:
    dry_run = True
    cleared = True

    def __init__(self):
        self.log = []
        self.tick = None
        self.commanded = None
        self.lock = threading.Lock()

    def _rec(self, *a):
        with self.lock:
            self.log.append(a)

    def stream_to(self, x, y, z, ms=None):
        self.commanded = (x, y, z)
        self._rec("stream", round(x), round(y), round(z))

    gripping = False

    def suction(self, on):
        self._rec("suction", on)
        self.gripping = bool(on)

    def wait(self, seconds):
        end = time.time() + min(seconds, 0.03)
        while time.time() < end:
            if self.tick:
                self.tick()
            time.sleep(0.005)

    def move_to(self, x, y, z, ms=None):
        self.commanded = (x, y, z)
        self._rec("move", round(x), round(y), round(z))
        end = time.time() + 0.06
        while time.time() < end:
            if self.tick:
                self.tick()
            time.sleep(0.005)
        return (x, y, z)

    def home(self, ms=1500):
        self._rec("home")
        return self.move_to(-210, -56, 190, ms)

    def halt(self):
        self.inhibited = True
        self._rec("halt", self.commanded)

    def release_halt(self):
        self.inhibited = False

    def grip(self):
        self._rec("grip")

    def release(self):
        self._rec("release")
        self.gripping = False

    def read_xyz(self):
        return None

    def kinds(self):
        with self.lock:
            return [e[0] for e in self.log]


def P(cls):
    return Prediction(cls, 0.95, 100, 100, 400, 400)


class IntegrationTest(unittest.TestCase):
    def setUp(self):
        self.log = silent_logger()
        self.arm = FakeArm()
        self.ctl = MotionController(self.arm, BOX, config.MIRROR_ORIGIN_XYZ_MM, hz=50.0, log=self.log)
        self.ctl.start()
        self.routines = Routines(self.arm, dry_run=True, log=self.log)
        self.toasts = viz.Toasts()
        self.ops = gesture_arm.ArmOps(self.log)
        self.actions = gesture_arm.AppActions(self.arm, self.ctl, self.routines, self.toasts, self.ops, self.log, box=BOX)
        self.actions.carry_floor = 160.0
        self.sm = StateMachine(self.actions, log=self.log)
        self.actions.sm = self.sm
        self.deb = Debouncer(n=3, threshold=0.7, log=self.log)
        self.ctl.resume(recenter=True)
        self.t = 0.0

    def tearDown(self):
        self.routines.abort()
        self.ctl.stop()
        self.ops.stop()

    def hold(self, cls, n=3):
        """Feed n detections of cls, dispatching events like the main loop."""
        for _ in range(n):
            self.t += 0.25
            ev = self.deb.update(P(cls), self.t)
            if ev:
                self.sm.on_event(ev)
            self.actions.drain()
        self.deb.update(None, self.t)      # hand goes away between gestures (re-arms the debounce)

    def centre_hand(self):
        now = time.time()
        self.ctl.update_hand((0.5, 0.5), now)
        self.ctl.update_hand((0.5, 0.5), now + config.RECENTER_HOLD_S + 0.05)

    def wait_until(self, pred, timeout=3.0):
        end = time.time() + timeout
        while time.time() < end:
            self.actions.drain()
            if pred():
                return True
            time.sleep(0.01)
        return False

    def test_full_demo_sequence(self):
        a, sm, ctl = self.arm, self.sm, self.ctl
        # 1. centre + mirror: streamed commands stay inside the box
        self.centre_hand()
        self.assertFalse(ctl.recenter_required)
        ctl.update_hand((0.7, 0.4), time.time())
        self.assertTrue(self.wait_until(lambda: "stream" in a.kinds()))
        for e in a.log:
            if e[0] == "stream":
                self.assertTrue(BOX[0][0] <= e[1] <= BOX[0][1] and BOX[2][0] <= e[3] <= BOX[2][1], e)
        # 2. pinch = GRAB routine at the current spot, then open-palm = PLACE (holding), both resume
        #    mirroring without a re-centre
        config.GRAB_Z_MM, config.GRAB_HOVER_MM, config.PLACE_LIFT_MM = 84.0, 40.0, 20.0
        self.hold("pinch")
        self.assertEqual(sm.mode, ROUTINE)
        self.assertTrue(self.wait_until(lambda: sm.mode == MIRROR))
        self.assertIn(("suction", True), [e[:2] for e in a.log])
        self.assertTrue(a.gripping)
        self.assertFalse(ctl.recenter_required)
        self.assertTrue(self.wait_until(lambda: ctl.box[2][0] == 160.0))     # carry floor while holding
        self.assertTrue(self.wait_until(lambda: a.commanded is not None and a.commanded[2] >= 160.0))  # rose first
        self.hold("open-palm")
        self.assertEqual(sm.mode, ROUTINE)
        self.assertEqual(sm.routine, "PLACE")
        self.assertTrue(self.wait_until(lambda: sm.mode == MIRROR))
        self.assertEqual(a.kinds().count("release"), 1)
        self.assertFalse(a.gripping)
        self.assertFalse(ctl.recenter_required)
        self.assertTrue(self.wait_until(lambda: ctl.box[2][0] == BOX[2][0]))  # floor back to normal
        # open-palm with nothing held is a plain release
        self.hold("open-palm")
        self.assertTrue(self.wait_until(lambda: a.kinds().count("release") == 2))
        self.assertEqual(sm.mode, MIRROR)
        # 3. home routine -> back to MIRROR with re-centre required
        self.hold("thumbs-up")
        self.assertEqual(sm.mode, ROUTINE)
        self.assertTrue(self.wait_until(lambda: sm.mode == MIRROR and ctl.recenter_required))
        self.assertIn("home", a.kinds())
        # while re-centre is required, hand motion streams nothing
        n = a.kinds().count("stream")
        ctl.update_hand((0.9, 0.9), time.time())
        time.sleep(0.1)
        self.assertEqual(a.kinds().count("stream"), n)
        # 4. flourish, fist mid-way -> halt + abort, frozen; pinch ignored; open-palm resumes
        self.centre_hand()
        self.hold("peace")
        self.assertEqual(sm.mode, ROUTINE)
        self.assertTrue(self.wait_until(lambda: a.kinds().count("move") >= 1))
        moves_before = a.kinds().count("move")
        self.hold("fist")
        self.assertEqual(sm.mode, FROZEN)
        self.assertIn("halt", a.kinds())
        self.assertTrue(self.wait_until(lambda: not self.routines.running))
        self.assertLessEqual(a.kinds().count("move"), moves_before + 1)
        self.actions.drain()
        self.assertEqual(sm.mode, FROZEN)           # aborted routine's done() must not unfreeze
        self.hold("pinch")
        self.assertEqual(sm.mode, FROZEN)              # ignored while frozen: no GRAB routine
        self.hold("open-palm")
        self.assertTrue(self.wait_until(lambda: sm.mode == MIRROR and ctl.enabled))
        self.assertEqual(a.kinds().count("release"), 3)
        # the resume is queued AFTER the release in the ops queue
        idx_rel = max(i for i, e in enumerate(a.log) if e[0] == "release")
        self.assertTrue(ctl.recenter_required)
        self.assertGreaterEqual(len(a.log), idx_rel + 1)

    def test_fist_right_after_open_palm_stays_frozen(self):
        """The queued resume must not undo a freeze that happened while the release was venting."""
        self.centre_hand()
        self.hold("fist")
        self.assertEqual(self.sm.mode, FROZEN)
        # slow the release so the resume is still queued when the next fist lands
        orig = self.arm.release
        self.arm.release = lambda: (time.sleep(0.4), orig())
        self.hold("open-palm")
        self.assertEqual(self.sm.mode, MIRROR)
        self.hold("fist")                           # within the vent window
        self.assertEqual(self.sm.mode, FROZEN)
        time.sleep(0.8)                             # let the queued resume run (and be ignored)
        self.actions.drain()
        self.assertEqual(self.sm.mode, FROZEN)
        self.assertTrue(self.ctl.frozen)
        self.assertFalse(self.ctl.enabled)

    def test_fist_during_resume_rise_keeps_the_halt(self):
        """A1: the queued resume's vertical rise must not clear a halt from a fist that landed during its sync."""
        a, sm, ctl, actions = self.arm, self.sm, self.ctl, self.actions
        # arm halted below the box floor (fist during PICK's descent)
        a.commanded = (0.0, -175.0, 84.0)
        ctl.sync_to_arm()
        sm.on_event(__import__("gesture.gestures", fromlist=["Event"]).Event("FREEZE", "fist", 0.9, 0.0))
        self.assertEqual(sm.mode, FROZEN)
        self.assertTrue(a.inhibited)
        # capture the resume closure instead of running it on the ops thread
        submitted = []
        actions.ops.submit = lambda fn, label: submitted.append((fn, label))
        sm.on_event(__import__("gesture.gestures", fromlist=["Event"]).Event("RELEASE", "open-palm", 0.9, 0.0))
        self.assertEqual(sm.mode, MIRROR)
        fn = [f for f, l in submitted if l == "resume-mirror"][0]
        # a second fist lands before the queued resume gets to run
        sm.on_event(__import__("gesture.gestures", fromlist=["Event"]).Event("FREEZE", "fist", 0.9, 0.0))
        self.assertEqual(sm.mode, FROZEN)
        moves_before = a.kinds().count("move")
        fn()                                        # the stale resume runs now
        self.assertEqual(a.kinds().count("move"), moves_before)   # no rise was sent
        self.assertTrue(a.inhibited)                # halt still in force
        self.assertFalse(ctl.enabled)

    def test_flicker_moves_nothing(self):
        self.centre_hand()
        for cls in ("fist", "fist", "open-palm", "fist", "pinch", "fist", "null", "fist", "fist"):
            self.t += 0.25
            ev = self.deb.update(P(cls), self.t)
            self.assertIsNone(ev)
        time.sleep(0.05)
        self.assertEqual(self.sm.mode, MIRROR)
        self.assertNotIn("halt", self.arm.kinds())
        self.assertNotIn("grip", self.arm.kinds())


if __name__ == "__main__":
    unittest.main()
