import unittest

from _common import silent_logger

from gesture.gestures import Actions, Event, StateMachine, FROZEN, MIRROR, ROUTINE


class FakeActions(Actions):
    def __init__(self):
        self.calls = []
        self.pending = []   # (name, done)

    def freeze(self):
        self.calls.append("freeze")

    def release(self):
        self.calls.append("release")

    def grip(self):
        self.calls.append("grip")

    def resume_mirror(self, recenter):
        self.calls.append(f"resume({recenter})")

    def pause_mirror(self):
        self.calls.append("pause")

    def start_routine(self, name, done):
        self.calls.append(f"start:{name}")
        self.pending.append((name, done))

    def abort_routine(self):
        self.calls.append("abort")


def ev(name, gesture="x"):
    return Event(name, gesture, 0.9, 0.0)


class StateMachineTest(unittest.TestCase):
    def setUp(self):
        self.a = FakeActions()
        self.sm = StateMachine(self.a, log=silent_logger())

    def test_grip_release_in_mirror(self):
        self.assertTrue(self.sm.on_event(ev("GRIP")))
        self.assertTrue(self.sm.on_event(ev("RELEASE")))
        self.assertEqual(self.a.calls, ["grip", "release"])
        self.assertEqual(self.sm.mode, MIRROR)

    def test_ignored_events_report_false(self):
        self.assertTrue(self.sm.on_event(ev("FREEZE")))
        self.assertFalse(self.sm.on_event(ev("FREEZE")))      # already frozen: no mode change ...
        self.assertEqual(self.a.calls.count("freeze"), 2)     # ... but the halt is re-asserted
        self.assertFalse(self.sm.on_event(ev("GRIP")))
        self.assertTrue(self.sm.on_event(ev("RELEASE")))
        self.assertTrue(self.sm.on_event(ev("HOME")))
        self.assertFalse(self.sm.on_event(ev("PICK")))        # routine in progress

    def test_freeze_blocks_everything_but_open_palm(self):
        self.sm.on_event(ev("FREEZE", "fist"))
        self.assertEqual(self.sm.mode, FROZEN)
        self.assertEqual(self.a.calls, ["freeze", "pause"])
        for name in ("GRIP", "HOME", "PICK", "FLOURISH", "FREEZE"):
            self.sm.on_event(ev(name))
        self.assertEqual(self.sm.mode, FROZEN)
        self.assertEqual(self.a.calls, ["freeze", "pause", "freeze"])   # only the repeated fist re-asserted the halt
        self.sm.on_event(ev("RELEASE", "open-palm"))
        self.assertEqual(self.sm.mode, MIRROR)
        self.assertEqual(self.a.calls[-2:], ["release", "resume(True)"])

    def test_routine_lifecycle(self):
        self.sm.on_event(ev("HOME"))
        self.assertEqual(self.sm.mode, ROUTINE)
        self.assertEqual(self.sm.routine, "HOME")
        self.assertEqual(self.a.calls, ["pause", "start:HOME"])
        # conflicting gestures mid-routine are ignored
        self.sm.on_event(ev("GRIP"))
        self.sm.on_event(ev("PICK"))
        self.assertEqual(self.a.calls, ["pause", "start:HOME"])
        name, done = self.a.pending.pop()
        done(True)
        self.assertEqual(self.sm.mode, MIRROR)
        self.assertIsNone(self.sm.routine)
        self.assertEqual(self.a.calls[-1], "resume(True)")

    def test_failed_routine_returns_to_mirror(self):
        self.sm.on_event(ev("PICK"))
        _, done = self.a.pending.pop()
        done(False)
        self.assertEqual(self.sm.mode, MIRROR)

    def test_fist_aborts_routine_and_stays_frozen(self):
        self.sm.on_event(ev("FLOURISH"))
        self.sm.on_event(ev("FREEZE", "fist"))
        self.assertEqual(self.sm.mode, FROZEN)
        self.assertEqual(self.a.calls, ["pause", "start:FLOURISH", "abort", "freeze", "pause"])
        _, done = self.a.pending.pop()
        done(False)                      # the aborted routine reports back
        self.assertEqual(self.sm.mode, FROZEN)   # still frozen: only open-palm resumes
        self.sm.on_event(ev("RELEASE", "open-palm"))
        self.assertEqual(self.sm.mode, MIRROR)
        self.assertIsNone(self.sm.routine)

    def test_stale_done_is_ignored(self):
        self.sm.on_event(ev("HOME"))
        _, done1 = self.a.pending.pop()
        self.sm.on_event(ev("FREEZE"))
        self.sm.on_event(ev("RELEASE"))
        self.sm.on_event(ev("PEACE")) if False else None
        self.sm.on_event(ev("FLOURISH"))
        self.assertEqual(self.sm.mode, ROUTINE)
        done1(True)                      # late callback from the first routine
        self.assertEqual(self.sm.mode, ROUTINE)
        self.assertEqual(self.sm.routine, "FLOURISH")

    def test_transitions_are_recorded(self):
        self.sm.on_event(ev("FREEZE"))
        self.sm.on_event(ev("RELEASE"))
        self.assertEqual([(t.frm, t.to) for t in self.sm.history], [(MIRROR, FROZEN), (FROZEN, MIRROR)])


if __name__ == "__main__":
    unittest.main()
