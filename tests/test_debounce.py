import unittest

from _common import silent_logger

from gesture import config
from gesture.gestures import Debouncer, Prediction


def P(cls, conf=0.9):
    return Prediction(cls, conf, 10, 10, 100, 100)


class DebounceTest(unittest.TestCase):
    def setUp(self):
        self.d = Debouncer(n=5, threshold=0.7, log=silent_logger())

    def feed(self, seq):
        events = []
        for p in seq:
            ev = self.d.update(p, t=0.0)
            if ev is not None:
                events.append(ev)
        return events

    def test_fires_once_after_n_consecutive(self):
        evs = self.feed([P("pinch")] * 4)
        self.assertEqual(evs, [])
        self.assertAlmostEqual(self.d.progress, 0.8)
        evs = self.feed([P("pinch")])
        self.assertEqual([e.name for e in evs], ["GRAB"])
        self.assertEqual(evs[0].gesture, "pinch")
        self.assertEqual(self.d.progress, 1.0)
        # holding the same gesture never re-fires
        self.assertEqual(self.feed([P("pinch")] * 20), [])

    def test_flicker_fires_nothing(self):
        seq = [P("fist"), P("fist"), P("open-palm"), P("fist"), P("fist"), P("fist"), P("null"), P("fist"), P("fist")]
        self.assertEqual(self.feed(seq), [])

    def test_below_threshold_resets(self):
        self.feed([P("fist")] * 4)
        self.assertEqual(self.d.count, 4)
        self.assertEqual(self.feed([P("fist", 0.69)]), [])
        self.assertEqual(self.d.count, 0)
        self.assertEqual(self.feed([P("fist")] * 4), [])
        self.assertEqual([e.name for e in self.feed([P("fist")])], ["FREEZE"])

    def test_no_detection_and_null_reset(self):
        self.feed([P("peace")] * 4)
        self.feed([None])
        self.assertEqual(self.d.count, 0)
        self.feed([P("peace")] * 4)
        self.feed([P("null", 0.99)])
        self.assertEqual(self.d.count, 0)
        self.assertEqual(self.d.progress, 0.0)

    def test_unknown_class_rejected(self):
        self.assertEqual(self.feed([P("Robotic-Gestures")] * 10), [])

    def test_refires_after_switching_away(self):
        self.assertEqual([e.name for e in self.feed([P("thumbs-up")] * 5)], ["HOME"])
        self.assertEqual([e.name for e in self.feed([P("peace")] * 5)], ["FLOURISH"])
        self.assertEqual([e.name for e in self.feed([P("thumbs-up")] * 5)], ["HOME"])

    def test_point_is_the_steering_pose_and_never_fires(self):
        self.assertEqual(self.feed([P("point")] * 20), [])
        self.assertEqual(self.d.count, 0)
        self.feed([P("pinch")] * 4)
        self.assertEqual(self.feed([P("point")]), [])       # a steering frame resets the charge
        self.assertEqual(self.d.count, 0)

    def test_every_class_maps_to_its_event(self):
        for cls, name in config.GESTURE_EVENTS.items():
            d = Debouncer(n=3, threshold=0.7, log=silent_logger())
            evs = [d.update(P(cls), 0.0) for _ in range(3)]
            self.assertEqual([e.name for e in evs if e], [name], cls)

    def test_config_classes_are_dashboard_strings(self):
        self.assertEqual(set(config.CLASSES), {"fist", "open-palm", "pinch", "point", "peace", "thumbs-up"})
        self.assertEqual(set(config.GESTURE_EVENTS) | set(config.NO_EVENT_CLASSES), set(config.CLASSES))
        self.assertEqual(config.GESTURE_EVENTS["pinch"], "GRAB")
        self.assertIn("point", config.NO_EVENT_CLASSES)


if __name__ == "__main__":
    unittest.main()
