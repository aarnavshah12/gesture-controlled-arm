import math
import threading
import unittest

from _common import silent_logger

from gesture import config
from gesture.motion import (MotionController, clamp_box, clamp_target, extension_ratio, hand_to_target,
                            intersect_box, step_toward)

BOX = ((-115.0, 145.0), (-250.0, -130.0), (100.0, 200.0))
ORIGIN = (0.0, -175.0, 150.0)


class FakeArm:
    dry_run = True

    def __init__(self, refuse=False):
        self.streams = []
        self.halts = 0
        self.commanded = None
        self.refuse = refuse

    def stream_to(self, x, y, z, ms=None):
        if self.refuse:
            raise ValueError("refused")
        self.streams.append((x, y, z))
        self.commanded = (x, y, z)

    def halt(self):
        self.halts += 1

    def read_xyz(self):
        return None


class MathTest(unittest.TestCase):
    def test_clamp_box(self):
        self.assertEqual(clamp_box((999, -999, 0), BOX), (145.0, -250.0, 100.0))
        self.assertEqual(clamp_box((0, -175, 150), BOX), (0.0, -175.0, 150.0))

    def test_intersect_box(self):
        self.assertEqual(intersect_box(BOX, ((-269, 272), (-256, 24), (47, 210))), BOX)
        self.assertEqual(intersect_box(BOX, ((0, 50), (-300, -200), (0, 120)))[0], (0.0, 50.0))
        with self.assertRaises(ValueError):
            intersect_box(BOX, ((200, 300), (-250, -130), (100, 200)))

    def test_extension_guard_pulls_back_far_corners(self):
        for corner in ((145, -250, 200), (-115, -250, 200), (145, -250, 100)):
            x, y, z = clamp_target(corner, BOX, ORIGIN)
            self.assertLessEqual(extension_ratio(x, y, z), config.EXTENSION_MAX + 1e-9, corner)
            self.assertEqual(clamp_box((x, y, z), BOX), (x, y, z))   # still inside the box
        self.assertEqual(clamp_target(ORIGIN, BOX, ORIGIN), ORIGIN)

    def test_step_toward_caps_velocity(self):
        p = step_toward((0, 0, 0), (100, 0, 0), 15.0)
        self.assertAlmostEqual(p[0], 15.0)
        p = step_toward((0, 0, 0), (3, 4, 0), 15.0)
        self.assertEqual(p, (3.0, 4.0, 0.0))
        p = step_toward((0, 0, 0), (30, 40, 0), 10.0)
        self.assertAlmostEqual(math.hypot(*p[:2]), 10.0)

    def test_hand_to_target_signs_and_gain(self):
        ref = (0.5, 0.5)
        t = hand_to_target((0.6, 0.5), ref, ORIGIN, BOX)
        self.assertAlmostEqual(t[0], ORIGIN[0] + 0.1 * config.MIRROR_GAIN_X_MM * config.MIRROR_X_SIGN, places=6)
        self.assertEqual(t[1], ORIGIN[1])
        t = hand_to_target((0.5, 0.4), ref, ORIGIN, BOX)   # hand up on screen (y smaller) -> arm up
        self.assertAlmostEqual(t[2], ORIGIN[2] + 0.1 * config.MIRROR_GAIN_Z_MM * -config.MIRROR_Z_SIGN, places=6)
        self.assertGreater(t[2], ORIGIN[2])
        t = hand_to_target((1.0, 0.0), ref, ORIGIN, BOX)   # far corner stays inside the box
        self.assertEqual(clamp_box(t, BOX), t)


class ControllerTest(unittest.TestCase):
    def make(self, arm=None):
        arm = arm or FakeArm()
        c = MotionController(arm, BOX, ORIGIN, hz=10.0, log=silent_logger())
        return arm, c

    def test_recenter_gate_then_mirror(self):
        arm, c = self.make()
        c.resume(recenter=True)
        c.update_hand((0.9, 0.9), 0.0)          # off-centre: no reference yet
        self.assertTrue(c.recenter_required)
        c.update_hand((0.5, 0.5), 1.0)
        c.update_hand((0.5, 0.5), 1.0 + config.RECENTER_HOLD_S + 0.01)
        self.assertFalse(c.recenter_required)
        self.assertEqual(c.hand_ref, (0.5, 0.5))
        c.update_hand((0.6, 0.5), 2.0)
        c._tick(2.0)
        self.assertEqual(len(arm.streams), 1)
        x, y, z = arm.streams[0]
        self.assertAlmostEqual(math.dist((x, y, z), ORIGIN), c.max_step)   # first step is velocity-capped
        self.assertGreater(x, ORIGIN[0])

    def test_velocity_cap_every_tick(self):
        arm, c = self.make()
        c.resume(recenter=False)
        c.update_hand((0.5, 0.5), 0.0)
        c.update_hand((1.0, 0.5), 0.1)          # big jump requested
        prev = ORIGIN
        for k in range(5):
            c._tick(0.1 + k * 0.1)
            cur = arm.streams[-1]
            self.assertLessEqual(math.dist(prev, cur), c.max_step + 1e-6)
            prev = cur
        self.assertEqual(len(arm.streams), 5)

    def test_no_hand_holds(self):
        arm, c = self.make()
        c.resume(recenter=False)
        c.update_hand((0.5, 0.5), 0.0)
        c.update_hand((0.7, 0.5), 0.5)
        c._tick(0.6)
        n = len(arm.streams)
        c._tick(0.5 + config.NO_HAND_HOLD_S + 0.1)   # no hand for > 1 s
        c._tick(0.5 + config.NO_HAND_HOLD_S + 0.2)
        self.assertEqual(len(arm.streams), n)
        self.assertTrue(c.holding)
        c.update_hand((0.7, 0.5), 3.0)
        self.assertFalse(c.holding)
        c._tick(3.0)
        self.assertEqual(len(arm.streams), n + 1)

    def test_freeze_halts_and_stops_streaming(self):
        arm, c = self.make()
        c.resume(recenter=False)
        c.update_hand((0.5, 0.5), 0.0)
        c.update_hand((0.9, 0.5), 0.1)
        c._tick(0.1)
        c.freeze()
        self.assertEqual(arm.halts, 1)
        n = len(arm.streams)
        c.update_hand((0.1, 0.5), 0.2)
        for k in range(5):
            c._tick(0.2 + 0.1 * k)
        self.assertEqual(len(arm.streams), n)
        c.resume(recenter=True)
        self.assertFalse(c.frozen)

    def test_pause_stops_streaming(self):
        arm, c = self.make()
        c.resume(recenter=False)
        c.update_hand((0.5, 0.5), 0.0)
        c.pause()
        c.update_hand((0.9, 0.5), 0.1)
        c._tick(0.1)
        self.assertEqual(arm.streams, [])

    def test_refused_command_keeps_commanded_point(self):
        arm, c = self.make(FakeArm(refuse=True))
        c.resume(recenter=False)
        c.update_hand((0.5, 0.5), 0.0)
        c.update_hand((0.9, 0.5), 0.1)
        c._tick(0.1)
        self.assertEqual(c.refused, 1)
        self.assertIsNone(c.commanded)     # nothing was accepted, so nothing is assumed
        c._tick(0.2)
        self.assertEqual(c.refused, 2)

    def test_stale_resume_is_ignored_after_freeze(self):
        arm, c = self.make()
        c.resume(recenter=False)
        epoch = c.epoch
        c.freeze()
        self.assertFalse(c.resume(recenter=True, epoch=epoch))
        self.assertTrue(c.frozen)
        self.assertFalse(c.enabled)
        self.assertTrue(c.resume(recenter=True, epoch=c.epoch))
        self.assertFalse(c.frozen)

    def test_pause_also_invalidates_queued_resume(self):
        arm, c = self.make()
        epoch = c.epoch
        c.pause()
        self.assertFalse(c.resume(recenter=True, epoch=epoch))
        self.assertFalse(c.enabled)

    def test_unknown_position_never_streams_live(self):
        class LiveArm(FakeArm):
            dry_run = False
        arm, c = self.make(LiveArm())
        self.assertIsNone(c.sync_to_arm())
        self.assertFalse(c.position_known)
        self.assertFalse(c.resume(recenter=False))
        c.enabled = True                      # even if something enabled it
        c.update_hand((0.5, 0.5), 0.0)
        c.update_hand((0.9, 0.5), 0.1)
        c._tick(0.1)
        self.assertEqual(arm.streams, [])
        arm.read_xyz = lambda: (0, -175, 150)
        self.assertEqual(c.sync_to_arm(), (0, -175, 150))
        self.assertTrue(c.resume(recenter=False))

    def test_live_sync_never_adopts_commanded(self):
        class LiveArm(FakeArm):
            dry_run = False
        arm = LiveArm()
        arm.commanded = (0.0, -175.0, 150.0)       # start of an in-flight move, not where the arm is
        _, c = self.make(arm)
        self.assertIsNone(c.sync_to_arm())
        self.assertFalse(c.position_known)
        self.assertFalse(c.resume(recenter=False))

    def test_release_halt_only_if_epoch_current(self):
        class InhibitArm(FakeArm):
            def __init__(self):
                super().__init__()
                self.inhibited = False
                self.lock = threading.RLock()

            def halt(self):
                super().halt()
                self.inhibited = True

            def release_halt(self):
                self.inhibited = False
        arm = InhibitArm()
        _, c = self.make(arm)
        c.resume(recenter=False)
        epoch = c.epoch
        c.freeze()
        self.assertTrue(arm.inhibited)
        self.assertFalse(c.release_halt_if_current(epoch))   # stale: the halt stays in force
        self.assertTrue(arm.inhibited)
        self.assertTrue(c.release_halt_if_current(c.epoch))
        self.assertFalse(arm.inhibited)

    def test_hand_loss_resets_recenter_timer(self):
        arm, c = self.make()
        c.resume(recenter=True)
        c.update_hand((0.5, 0.5), 0.0)
        c.update_hand(None, 0.1)
        c.update_hand((0.5, 0.5), 0.2)
        self.assertTrue(c.recenter_required)  # the hold restarted at 0.2, not 0.0
        c.update_hand((0.5, 0.5), 0.2 + config.RECENTER_HOLD_S + 0.01)
        self.assertFalse(c.recenter_required)

    def test_targets_always_inside_box(self):
        arm, c = self.make()
        c.resume(recenter=False)
        c.update_hand((0.5, 0.5), 0.0)
        for w in ((0.0, 0.0), (1.0, 1.0), (1.0, 0.0), (0.0, 1.0)):
            c.update_hand(w, 1.0)
            self.assertEqual(clamp_box(c.target, BOX), c.target)
            self.assertLessEqual(extension_ratio(*c.target), config.EXTENSION_MAX + 1e-9)


if __name__ == "__main__":
    unittest.main()


class DetourTest(unittest.TestCase):
    """Coming back from the block picker's HOME, the straight line to a lateral target crosses the base
    keep-out radius; the controller must swing around it, never stall, never step inside."""
    HOME = (-210.0, -56.0, 190.0)
    MIN_R = 120.0

    def check(self, x, y, z):
        return (math.hypot(x, y) >= self.MIN_R and -269 <= x <= 272 and -256 <= y <= 24 and 47 <= z <= 210)

    def drive(self, target, max_ticks=400):
        arm = FakeArm()
        c = MotionController(arm, BOX, ORIGIN, hz=10.0, log=silent_logger(), check=self.check)
        arm.commanded = self.HOME
        c.sync_to_arm()
        c.resume(recenter=False)
        c.update_hand((0.5, 0.5), 0.0)
        for k in range(max_ticks):
            c.last_hand_t = 0.1 * k              # hand stays "present"; the target is pinned below
            c.target = target
            c._tick(0.1 * k)
            if arm.streams and math.dist(arm.streams[-1], target) < 0.6:   # controller deadband is 0.5 mm
                break
        return arm, c

    def test_lateral_target_from_home_swings_around_base(self):
        target = (145.0, -130.0, 150.0)
        arm, c = self.drive(target)
        self.assertTrue(arm.streams and math.dist(arm.streams[-1], target) < 0.6, "never arrived")
        self.assertGreater(c.detours, 0)
        self.assertEqual(c.blocked, 0)
        prev = self.HOME
        for p in arm.streams:
            self.assertGreaterEqual(math.hypot(p[0], p[1]), self.MIN_R - 1e-6, p)
            self.assertLessEqual(math.dist(prev, p), c.max_step + 1e-6)
            prev = p

    def test_straight_path_used_when_safe(self):
        arm, c = self.drive(ORIGIN)
        self.assertEqual(c.detours, 0)
        self.assertEqual(c.blocked, 0)
        self.assertTrue(math.dist(arm.streams[-1], ORIGIN) < 0.6)

    def test_polar_step_respects_cap(self):
        from gesture.motion import step_toward_polar
        cur, tgt = (-210.0, -56.0, 190.0), (145.0, -130.0, 150.0)
        p = step_toward_polar(cur, tgt, 15.0)
        self.assertLessEqual(math.dist(cur, p), 15.0 + 1e-6)
        self.assertGreaterEqual(math.hypot(p[0], p[1]), min(math.hypot(*cur[:2]), math.hypot(*tgt[:2])) - 1e-6)
        self.assertEqual(step_toward_polar(cur, cur, 15.0), cur)
