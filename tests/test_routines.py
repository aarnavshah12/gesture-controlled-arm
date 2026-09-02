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

    def move_to(self, x, y, z, ms=None):
        self.moves.append((x, y, z, ms))
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
