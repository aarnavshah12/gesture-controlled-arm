import unittest

import numpy as np

from _common import silent_logger  # noqa: F401

from gesture import viz
from gesture.gestures import Prediction
from gesture.perception import HAND_CONNECTIONS, Hand


def fake_hand(w=1280, h=720):
    norm = np.array([[0.5 + 0.02 * i, 0.5 + 0.015 * (i % 5)] for i in range(21)], dtype=np.float32)
    return Hand(pts=norm * [w, h], norm=norm, handedness="Right", score=0.9, t=0.0)


class VizTest(unittest.TestCase):
    def render(self, **kw):
        frame = np.zeros((720, 1280, 3), np.uint8)
        hand = fake_hand()
        pred = Prediction("pinch", 0.87, 500, 250, 900, 650)
        toasts = viz.Toasts(1.0)
        toasts.add("GRIP", (0, 140, 255), t=0.0)
        status = dict(fps=29.9, infer_ms=180.0, hand_ms=9.5, arm_xyz=(0, -175, 150), conf=0.87, gesture="pinch",
                      extra="debounce 3/5", dry_run=True)
        tmap = dict(box=((-115, 145), (-250, -130), (100, 200)), target=(20, -175, 160), commanded=(10, -175, 155),
                    actual=(8, -176, 154))
        args = dict(hand=hand, connections=HAND_CONNECTIONS, pred=pred, progress=0.6, trail=[(600, 400), (620, 410), (650, 420)],
                    mode="MIRROR", mode_sub="centre your hand", toasts=toasts, status=status, clean=False, target_map=tmap, t=0.5)
        args.update(kw)
        return frame, hand, pred, viz.render(frame, **args)

    def test_box_and_skeleton_both_visible(self):
        frame, hand, pred, img = self.render()
        self.assertEqual(img.shape, frame.shape)
        self.assertFalse(np.array_equal(img, frame))
        # box edge pixel painted in the gesture colour
        x1, y1, x2, y2 = pred.box
        self.assertTrue(img[(y1 + y2) // 2, x1].any())
        # every landmark has a dot (non-black) at its location
        for x, y in hand.pts:
            self.assertTrue(img[int(y), int(x)].any(), (x, y))
        # status strip drawn at the bottom, banner at the top
        self.assertTrue(img[715, :].any())
        self.assertTrue(img[30, :].any())

    def test_clean_strips_status_only(self):
        _, _, _, full = self.render(clean=False, target_map=None)
        _, _, _, clean = self.render(clean=True, target_map=None)
        self.assertFalse(clean[700:720, :].any())      # bottom strip gone
        self.assertTrue(clean[20:80, :].any())          # banner still there
        self.assertFalse(np.array_equal(full, clean))

    def test_modes_and_no_hand(self):
        for mode in ("MIRROR", "FROZEN", "ROUTINE"):
            _, _, _, img = self.render(mode=mode, hand=None, pred=None, trail=[])
            self.assertTrue(img[20:100, :].any(), mode)

    def test_status_strip_fits_the_frame(self):
        parts = ["DRY RUN", "29.9 fps", "infer 200.0 ms", "hand 10.0 ms", "arm (-115, -175, 150)",
                 "open-palm conf 0.87", "debounce 3/5", "brightness UNCONFIRMED"]
        for w in (1280, 1920, 960):
            sc, kept = viz.status_layout(w, parts)
            self.assertLessEqual(viz._text_w(viz.STATUS_GAP.join(kept), sc, 1)[0], w - 28)
            self.assertTrue(any("conf" in k for k in kept), kept)
        img = np.zeros((720, 1280, 3), np.uint8)
        viz.draw_status(img, fps=29.9, infer_ms=200.0, hand_ms=10.0, arm_xyz=(-115, -175, 150), conf=0.87,
                        gesture="open-palm", extra="debounce 3/5", dry_run=True, warn="brightness UNCONFIRMED")
        self.assertTrue(img[688:720, 14:1266].any())        # strip drawn, inside the frame

    def test_toasts_expire(self):
        t = viz.Toasts(1.0)
        t.add("X", t=0.0)
        img = np.zeros((200, 400, 3), np.uint8)
        t.draw(img, t=0.5)
        self.assertEqual(len(t.items), 1)
        t.draw(img, t=1.5)
        self.assertEqual(len(t.items), 0)

    def test_progress_arc_extremes(self):
        for p in (0.0, 0.5, 1.0, 1.7, -1.0):
            img = np.zeros((300, 300, 3), np.uint8)
            viz.draw_bbox(img, (50, 50, 250, 250), "fist 0.99", (0, 0, 255), p)
            self.assertTrue(img.any())


if __name__ == "__main__":
    unittest.main()
