import threading
import time
import unittest

from _common import silent_logger

from gesture import config
from gesture.routines import RoutineAborted, Routines


class FakeArm:
    """Mimics the driver's blocking move_to + tick pumping during waits."""
    dry_run = True

    def __init__(self, step_s=0.02):
        self.moves = []
        self.homes = 0
        self.tick = None
        self.step_s = step_s
        self.commanded = None
        self.events = []
        self.gripping = False

    def suction(self, on):
        self.events.append(("suction", on))
        self.gripping = bool(on)

    def release(self):
        self.events.append(("release",))
        self.gripping = False

    def vent(self):
        self.events.append(("vent",))
        self.gripping = False

    def valve_close(self):
        self.events.append(("valve_close",))

    def wait(self, seconds):
        end = time.time() + min(seconds, 0.02)
        while time.time() < end:
            if self.tick:
                self.tick()
            time.sleep(0.005)

    def read_xyz(self):
        return None

    def move_to(self, x, y, z, ms=None):
        self.moves.append((x, y, z, ms))
        self.events.append(("move", x, y, z))
        self.commanded = (x, y, z)
        end = time.time() + self.step_s
        while time.time() < end:          # like Arm.wait(): pump tick every ~50 ms
            if self.tick:
                self.tick()
            time.sleep(0.005)
        return (x, y, z)

    def home(self, ms=1500):
        self.homes += 1
        return self.move_to(-210, -56, 190, ms)


class RoutinesTest(unittest.TestCase):
    def run_routine(self, r, name, timeout=5.0):
        done = threading.Event()
        result = {}

        def cb(ok):
            result["ok"] = ok
            done.set()

        r.start(name, cb)
        self.assertTrue(done.wait(timeout), "routine did not finish")
        return result["ok"]

    def test_flourish_runs_all_steps(self):
        arm = FakeArm()
        r = Routines(arm, dry_run=True, log=silent_logger())
        self.assertTrue(self.run_routine(r, "FLOURISH"))
        self.assertEqual(len(arm.moves), 1 + len(config.FLOURISH_STEPS))
        ox, oy, oz = config.MIRROR_ORIGIN_XYZ_MM
        self.assertEqual(arm.moves[0][:3], (ox, oy, oz))
        self.assertIsNone(r.current)

    def _fixed_heights(self):
        config.GRAB_Z_MM, config.GRAB_HOVER_MM, config.PLACE_LIFT_MM = 84.0, 40.0, 20.0

    def test_grab_descends_sucks_and_returns(self):
        self._fixed_heights()
        arm = FakeArm()
        arm.commanded = (30.0, -180.0, 150.0)
        r = Routines(arm, dry_run=True, log=silent_logger())
        self.assertTrue(self.run_routine(r, "GRAB"))
        self.assertEqual([m[:3] for m in arm.moves], [(30.0, -180.0, 124.0), (30.0, -180.0, 84.0),
                                                      (30.0, -180.0, 124.0), (30.0, -180.0, 150.0)])
        self.assertEqual(arm.moves[1][3], config.GRAB_DESCENT_MS)
        kinds = [e[0] if e[0] != "move" else "move" for e in arm.events]
        self.assertEqual(arm.events[2], ("suction", True))      # after the descent, before the lift
        self.assertTrue(arm.gripping)

    def test_grab_column_clamped_into_box(self):
        self._fixed_heights()
        arm = FakeArm()
        arm.commanded = (150.0, -257.0, 150.0)        # noisy read past the box edge
        r = Routines(arm, dry_run=True, log=silent_logger(), box=((-115.0, 145.0), (-250.0, -130.0), (115.0, 200.0)))
        self.assertTrue(self.run_routine(r, "GRAB"))
        self.assertEqual(arm.moves[0][:2], (145.0, -250.0))

    def test_refused_descent_retreats_to_hover(self):
        from gesture.routines import MoveRefused
        self._fixed_heights()

        class Blocked(FakeArm):
            def move_to(self, x, y, z, ms=None):
                if z == 84.0:
                    raise MoveRefused("stack in the way")
                return super().move_to(x, y, z, ms)
        arm = Blocked()
        arm.commanded = (0.0, -175.0, 150.0)
        r = Routines(arm, dry_run=True, log=silent_logger())
        self.assertFalse(self.run_routine(r, "GRAB"))
        self.assertEqual([m[2] for m in arm.moves], [124.0, 124.0])   # hover, (refused), retreat to hover
        self.assertNotIn(("suction", True), arm.events)

    def test_heights_default_in_dry_run_without_block_picker(self):
        from gesture import bp, routines
        config.GRAB_Z_MM = config.GRAB_HOVER_MM = config.PLACE_LIFT_MM = None
        orig = bp.load
        bp.load = lambda: (_ for _ in ()).throw(bp.BlockPickerMissing("absent"))
        try:
            r = Routines(FakeArm(), dry_run=True, log=silent_logger())
            self.assertEqual(r._heights(), routines.DEFAULT_HEIGHTS)
            r = Routines(FakeArm(), dry_run=False, log=silent_logger())
            with self.assertRaises(bp.BlockPickerMissing):
                r._heights()
        finally:
            bp.load = orig
            self._fixed_heights()

    def test_grab_from_below_hover_returns_to_hover_not_lower(self):
        self._fixed_heights()
        arm = FakeArm()
        arm.commanded = (0.0, -175.0, 100.0)
        r = Routines(arm, dry_run=True, log=silent_logger())
        self.assertTrue(self.run_routine(r, "GRAB"))
        self.assertEqual(arm.moves[-1][:3], (0.0, -175.0, 124.0))

    def test_place_releases_above_pick_height(self):
        self._fixed_heights()
        arm = FakeArm()
        arm.commanded = (-50.0, -200.0, 160.0)
        arm.gripping = True
        r = Routines(arm, dry_run=True, log=silent_logger())
        self.assertTrue(self.run_routine(r, "PLACE"))
        self.assertEqual([m[2] for m in arm.moves], [124.0, 104.0, 124.0, 160.0])
        kinds = [e[0] for e in arm.events]
        # vent at 104, lift to hover with the valve still open, close it, then up
        self.assertEqual(kinds, ["move", "move", "vent", "move", "valve_close", "move"])
        self.assertFalse(arm.gripping)

    def test_grab_aborts_mid_descent(self):
        self._fixed_heights()
        arm = FakeArm(step_s=0.3)
        arm.commanded = (0.0, -175.0, 150.0)
        r = Routines(arm, dry_run=True, log=silent_logger())
        done = threading.Event(); out = {}
        r.start("GRAB", lambda ok: (out.__setitem__("ok", ok), done.set()))
        time.sleep(0.1)
        r.abort()
        self.assertTrue(done.wait(2.0))
        self.assertFalse(out["ok"])
        self.assertNotIn(("suction", True), arm.events)          # never switched the pump on

    def test_grab_without_position_fails_cleanly(self):
        self._fixed_heights()
        arm = FakeArm()
        r = Routines(arm, dry_run=True, log=silent_logger())
        self.assertFalse(self.run_routine(r, "GRAB"))
        self.assertEqual(arm.moves, [])

    def test_home_calls_driver_home(self):
        arm = FakeArm()
        r = Routines(arm, dry_run=True, log=silent_logger())
        self.assertTrue(self.run_routine(r, "HOME"))
        self.assertEqual(arm.homes, 1)

    def test_abort_unwinds_inside_a_move(self):
        arm = FakeArm(step_s=0.3)
        r = Routines(arm, dry_run=True, log=silent_logger())
        done = threading.Event()
        result = {}
        r.start("FLOURISH", lambda ok: (result.__setitem__("ok", ok), done.set()))
        time.sleep(0.1)                      # inside the first 300 ms move
        t0 = time.time()
        r.abort()
        self.assertTrue(done.wait(2.0))
        self.assertLess(time.time() - t0, 0.2)   # unwound at the next tick, not at the end of the move
        self.assertFalse(result["ok"])
        self.assertLessEqual(len(arm.moves), 2)
        self.assertFalse(r.running)

    def test_tick_only_raises_in_routine_thread(self):
        arm = FakeArm()
        r = Routines(arm, dry_run=True, log=silent_logger())
        r._abort.set()
        arm.tick()                            # main thread: a plain release() vent must not unwind
        self.assertTrue(True)

    def test_second_routine_refused_while_running(self):
        arm = FakeArm(step_s=0.2)
        r = Routines(arm, dry_run=True, log=silent_logger())
        first = threading.Event()
        r.start("FLOURISH", lambda ok: first.set())
        time.sleep(0.05)
        got = []
        r.start("HOME", lambda ok: got.append(ok))
        self.assertEqual(got, [False])
        r.abort()
        self.assertTrue(first.wait(2.0))

    def test_abort_and_join(self):
        arm = FakeArm(step_s=0.3)
        r = Routines(arm, dry_run=True, log=silent_logger())
        r.start("FLOURISH", lambda ok: None)
        time.sleep(0.05)
        self.assertTrue(r.abort_and_join(2.0))
        self.assertFalse(r.running)

    def test_request_during_unwind_refused_without_blocking(self):
        arm = FakeArm(step_s=1.0)                   # a long move: the abort unwinds only at its tick
        r = Routines(arm, dry_run=True, log=silent_logger())

        class Stuck(FakeArm):
            def move_to(self, x, y, z, ms=None):    # like a model load: no tick for a while
                time.sleep(0.6)
                return super().move_to(x, y, z, ms)
        r.arm = Stuck(step_s=0.05)
        r.arm.tick = r._tick
        r.start("FLOURISH", lambda ok: None)
        time.sleep(0.05)
        r.abort()
        got = []
        t0 = time.time()
        r.start("HOME", lambda ok: got.append(ok))
        self.assertLess(time.time() - t0, 0.3)      # never blocks the UI thread
        self.assertEqual(got, [False])
        self.assertTrue(r.abort_and_join(3.0))

    def test_unknown_routine(self):
        r = Routines(FakeArm(), dry_run=True, log=silent_logger())
        got = []
        r.start("DANCE", lambda ok: got.append(ok))
        self.assertEqual(got, [False])

    def test_abort_flag_cleared_for_next_routine(self):
        arm = FakeArm(step_s=0.2)
        r = Routines(arm, dry_run=True, log=silent_logger())
        done = threading.Event()
        r.start("FLOURISH", lambda ok: done.set())
        time.sleep(0.05)
        r.abort()
        done.wait(2.0)
        self.assertTrue(self.run_routine(r, "HOME"))


if __name__ == "__main__":
    unittest.main()
