import unittest

import numpy as np

from _common import silent_logger

from gesture import config
from gesture.gestures import Prediction, reconcile
from gesture.perception import Hand, finger_states, two_fingers_up


def make_hand(extended):
    """Synthetic landmarks: wrist at the bottom, palm above it, fingers up (extended) or curled back."""
    norm = np.zeros((21, 2), dtype=np.float32)
    norm[0] = (0.5, 0.80)                         # wrist
    bases = {"index": (0.44, 0.60, 5, 6, 7, 8), "middle": (0.49, 0.58, 9, 10, 11, 12),
             "ring": (0.54, 0.60, 13, 14, 15, 16), "pinky": (0.59, 0.63, 17, 18, 19, 20)}
    for name, (x, y, mcp, pip, dip, tip) in bases.items():
        norm[mcp] = (x, y)
        norm[pip] = (x, y - 0.06)
        if name in extended:
            norm[dip] = (x, y - 0.11)
            norm[tip] = (x, y - 0.16)
        else:                                      # curled: tip comes back toward the palm
            norm[dip] = (x, y - 0.04)
            norm[tip] = (x, y + 0.02)
    norm[1:5] = [(0.42, 0.74), (0.37, 0.68), (0.34, 0.62), (0.32, 0.57)]   # thumb, irrelevant
    return Hand(pts=norm * [1280, 720], norm=norm, handedness="Right", score=0.9, t=0.0)


def P(cls, conf=0.85):
    return Prediction(cls, conf, 100, 100, 400, 400)


class VetoTest(unittest.TestCase):
    def test_finger_states(self):
        self.assertEqual(finger_states(make_hand({"index"}).norm),
                         {"index": True, "middle": False, "ring": False, "pinky": False})
        self.assertEqual(finger_states(make_hand({"index", "middle"}).norm),
                         {"index": True, "middle": True, "ring": False, "pinky": False})
        self.assertTrue(two_fingers_up(finger_states(make_hand({"index", "middle"}).norm)))
        self.assertFalse(two_fingers_up(finger_states(make_hand({"index"}).norm)))
        self.assertFalse(two_fingers_up(finger_states(make_hand({"index", "middle", "ring", "pinky"}).norm)))

    def test_peace_with_one_finger_becomes_point(self):
        out = reconcile(P("peace"), make_hand({"index"}), silent_logger())
        self.assertEqual(out.cls, "point")
        self.assertEqual(out.raw_cls, "peace")
        self.assertTrue(out.corrected)
        self.assertEqual(out.conf, 0.85)
        self.assertEqual(out.box, (100, 100, 400, 400))

    def test_real_peace_stands(self):
        out = reconcile(P("peace"), make_hand({"index", "middle"}), silent_logger())
        self.assertEqual(out.cls, "peace")
        self.assertFalse(out.corrected)

    def test_no_landmarks_no_veto(self):
        self.assertEqual(reconcile(P("peace"), None, silent_logger()).cls, "peace")
        self.assertIsNone(reconcile(None, make_hand({"index"}), silent_logger()))

    def test_only_peace_is_ever_touched(self):
        for cls in ("point", "fist", "open-palm", "pinch", "thumbs-up", "null"):
            for hand in (make_hand({"index"}), make_hand({"index", "middle"}), make_hand(set())):
                out = reconcile(P(cls), hand, silent_logger())
                self.assertIs(out.cls, cls)
                self.assertFalse(out.corrected)

    def test_veto_never_creates_an_event(self):
        # a point that the landmarks say is two fingers stays point (no upgrade to peace)
        out = reconcile(P("point"), make_hand({"index", "middle"}), silent_logger())
        self.assertEqual(out.cls, "point")
        self.assertEqual(config.LANDMARK_VETO, {"peace": "point"})


if __name__ == "__main__":
    unittest.main()
