"""All drawing. The demo is judged on this overlay, so every element in the plan's visual spec is here:

1. RF-DETR hand bounding box: thick, corner-accented, gesture-coloured, label + confidence pill.
2. All 21 MediaPipe landmarks with skeleton connections, larger dots on the 5 fingertips + wrist,
   drawn on top of the box so both layers are always visible together.
3. Wrist trail: fading polyline of the last N smoothed wrist positions.
4. Mode banner top centre (FROZEN = solid red fill).
5. Event toasts (~1 s) and the debounce charging arc on the box.
6. Status strip (FPS, inference ms, arm x y z, gesture confidence); `clean=True` removes it.
7. Dry-run: the planned target drawn on a small map of the mirror box.
"""

from __future__ import annotations

import math
import time

import cv2
import numpy as np

from . import config
from .perception import FINGERTIPS, WRIST

FONT = cv2.FONT_HERSHEY_DUPLEX
WHITE = (250, 250, 249)
INK = (39, 24, 17)
GREY = (140, 140, 140)
SKELETON = (255, 255, 255)
SKELETON_DOT = (80, 220, 255)     # warm yellow joints
TIP_DOT = (0, 255, 200)           # green-yellow fingertips + wrist
TRAIL = (255, 200, 80)            # light blue trail
DRY = (11, 158, 245)              # amber


def _text(img, s, org, scale, colour, thick=1):
    cv2.putText(img, s, org, FONT, scale, colour, thick, cv2.LINE_AA)


def _text_w(s, scale, thick=1):
    return cv2.getTextSize(s, FONT, scale, thick)[0]


def translucent(img, x1, y1, x2, y2, colour=INK, alpha=0.7):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    roi = img[y1:y2, x1:x2]
    if roi.size:
        cv2.addWeighted(np.full_like(roi, colour, dtype=np.uint8), alpha, roi, 1 - alpha, 0, roi)


def gesture_colour(cls: str | None):
    return config.GESTURE_COLOURS.get(cls or "", GREY)


# ---------------------------------------------------------------- 1. bounding box + 5. charging arc

def draw_bbox(img, box, label: str, colour, progress: float = 0.0, thick: int = 4, accent: int = 28):
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(img, (x1, y1), (x2, y2), colour, thick, cv2.LINE_AA)
    a = min(accent, max(8, (x2 - x1) // 4), max(8, (y2 - y1) // 4))
    for (cx, cy, dx, dy) in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(img, (cx, cy), (cx + dx * a, cy), colour, thick + 3, cv2.LINE_AA)
        cv2.line(img, (cx, cy), (cx, cy + dy * a), colour, thick + 3, cv2.LINE_AA)
    # label pill above the box
    scale, tk = 0.8, 2
    tw, th = _text_w(label, scale, tk)
    px1, py1 = x1, y1 - th - 18
    if py1 < 0:
        py1 = y1 + 4
    px2, py2 = px1 + tw + 20, py1 + th + 14
    cv2.rectangle(img, (px1, py1), (px2, py2), colour, -1, cv2.LINE_AA)
    _text(img, label, (px1 + 10, py2 - 7), scale, WHITE, tk)
    # charging arc at the top-right corner: grey ring, coloured sweep, full ring = fired
    r = 20
    cx, cy = x2 - r - 6, y1 + r + 6
    cv2.circle(img, (cx, cy), r, INK, -1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), r, GREY, 3, cv2.LINE_AA)
    sweep = int(round(360 * max(0.0, min(1.0, progress))))
    if sweep > 0:
        cv2.ellipse(img, (cx, cy), (r, r), -90, 0, sweep, colour, 5, cv2.LINE_AA)
    if progress >= 1.0:
        cv2.circle(img, (cx, cy), r - 7, colour, -1, cv2.LINE_AA)


# ---------------------------------------------------------------- 2. skeleton

def draw_skeleton(img, pts: np.ndarray, connections, colour=SKELETON):
    """All 21 landmarks: bones, a dot on every joint, larger dots on the fingertips and wrist."""
    p = [(int(round(x)), int(round(y))) for x, y in pts]
    for a, b in connections:
        cv2.line(img, p[a], p[b], INK, 5, cv2.LINE_AA)
        cv2.line(img, p[a], p[b], colour, 2, cv2.LINE_AA)
    for i, pt in enumerate(p):
        if i in FINGERTIPS or i == WRIST:
            cv2.circle(img, pt, 9, INK, -1, cv2.LINE_AA)
            cv2.circle(img, pt, 7, TIP_DOT, -1, cv2.LINE_AA)
        else:
            cv2.circle(img, pt, 6, INK, -1, cv2.LINE_AA)
            cv2.circle(img, pt, 4, SKELETON_DOT, -1, cv2.LINE_AA)


# ---------------------------------------------------------------- 3. wrist trail

def draw_trail(img, trail, colour=TRAIL):
    """Fading polyline: older segments thinner and darker."""
    n = len(trail)
    if n < 2:
        return
    for i in range(1, n):
        f = i / (n - 1)
        c = tuple(int(v * (0.25 + 0.75 * f)) for v in colour)
        t = max(1, int(round(1 + 5 * f)))
        cv2.line(img, (int(trail[i - 1][0]), int(trail[i - 1][1])), (int(trail[i][0]), int(trail[i][1])), c, t, cv2.LINE_AA)
    x, y = trail[-1]
    cv2.circle(img, (int(x), int(y)), 6, colour, 2, cv2.LINE_AA)


# ---------------------------------------------------------------- 4. mode banner

def draw_banner(img, mode: str, sub: str = ""):
    h, w = img.shape[:2]
    colour = config.MODE_COLOURS.get(mode, GREY)
    scale, tk = 1.9, 4
    tw, th = _text_w(mode, scale, tk)
    x1, y1 = (w - tw) // 2 - 36, 16
    x2, y2 = x1 + tw + 72, y1 + th + 34
    if mode == "FROZEN":
        cv2.rectangle(img, (x1, y1), (x2, y2), colour, -1)
        cv2.rectangle(img, (x1, y1), (x2, y2), WHITE, 4)
        _text(img, mode, (x1 + 36, y2 - 18), scale, WHITE, tk)
    else:
        translucent(img, x1, y1, x2, y2, alpha=0.75)
        cv2.rectangle(img, (x1, y1), (x2, y2), colour, 4)
        _text(img, mode, (x1 + 36, y2 - 18), scale, colour, tk)
    if sub:
        sw, sh = _text_w(sub, 0.7, 1)
        translucent(img, (w - sw) // 2 - 10, y2 + 6, (w + sw) // 2 + 10, y2 + sh + 18, alpha=0.6)
        _text(img, sub, ((w - sw) // 2, y2 + sh + 12), 0.7, WHITE, 1)


# ---------------------------------------------------------------- 5. toasts

class Toasts:
    def __init__(self, seconds: float | None = None):
        self.seconds = config.TOAST_S if seconds is None else seconds
        self.items: list[tuple[str, tuple, float]] = []

    def add(self, text: str, colour=WHITE, t: float | None = None) -> None:
        t = time.time() if t is None else t
        self.items.append((text, colour, t + self.seconds))

    def draw(self, img, t: float | None = None) -> None:
        t = time.time() if t is None else t
        self.items = [it for it in self.items if it[2] > t]
        h, w = img.shape[:2]
        y = int(h * 0.30)
        for text, colour, t_end in self.items[-3:]:
            left = (t_end - t) / self.seconds
            scale, tk = 1.5, 3
            tw, th = _text_w(text, scale, tk)
            x1, y1 = (w - tw) // 2 - 28, y - th - 16
            x2, y2 = x1 + tw + 56, y + 16
            translucent(img, x1, y1, x2, y2, alpha=0.55 + 0.3 * left)
            cv2.rectangle(img, (x1, y1), (x2, y2), colour, 3)
            _text(img, text, (x1 + 28, y), scale, colour, tk)
            y += th + 44


# ---------------------------------------------------------------- 6. status strip

def draw_status(img, *, fps: float, infer_ms: float, hand_ms: float, arm_xyz, conf: float | None,
                gesture: str | None, extra: str = "", dry_run: bool = False):
    h, w = img.shape[:2]
    bar = 44
    translucent(img, 0, h - bar, w, h, alpha=0.75)
    xyz = "arm (%s)" % (", ".join("%.0f" % v for v in arm_xyz) if arm_xyz else "  -  ,  -  ,  -  ")
    conf_s = "conf %.2f" % conf if conf is not None else "conf  -  "
    parts = [f"{fps:4.1f} fps", f"infer {infer_ms:5.1f} ms", f"hand {hand_ms:4.1f} ms", xyz,
             f"{(gesture or '-'):>10s} {conf_s}"]
    if dry_run:
        parts.insert(0, "DRY RUN")
    if extra:
        parts.append(extra)
    _text(img, "   |   ".join(parts), (14, h - 14), 0.62, DRY if dry_run else WHITE, 1)


# ---------------------------------------------------------------- 7. planned target map (dry-run / mirroring)

def draw_target_map(img, box, target, commanded, actual=None, label="planned target"):
    """Top-down (x, y) + side (z) view of the mirror box in the bottom-right corner."""
    h, w = img.shape[:2]
    (xlo, xhi), (ylo, yhi), (zlo, zhi) = box
    mw, mh = 220, 150
    x0, y0 = w - mw - 56, h - mh - 60
    translucent(img, x0 - 8, y0 - 26, x0 + mw + 44, y0 + mh + 8, alpha=0.7)
    _text(img, label, (x0, y0 - 8), 0.55, WHITE, 1)
    cv2.rectangle(img, (x0, y0), (x0 + mw, y0 + mh), GREY, 1)
    zx = x0 + mw + 22
    cv2.line(img, (zx, y0), (zx, y0 + mh), GREY, 2)

    def to_px(p):
        fx = (p[0] - xlo) / max(1e-6, xhi - xlo)
        fy = (p[1] - ylo) / max(1e-6, yhi - ylo)       # front of the arm (-y) at the bottom of the map
        fz = (p[2] - zlo) / max(1e-6, zhi - zlo)
        return (int(x0 + fx * mw), int(y0 + (1 - fy) * mh)), int(y0 + (1 - fz) * mh)

    for p, colour, r in ((actual, WHITE, 5), (commanded, DRY, 6), (target, config.MODE_COLOURS["MIRROR"], 8)):
        if p is None:
            continue
        (px, py), pz = to_px(p)
        px, py = min(max(px, x0), x0 + mw), min(max(py, y0), y0 + mh)
        pz = min(max(pz, y0), y0 + mh)
        cv2.circle(img, (px, py), r, colour, 2 if p is target else -1, cv2.LINE_AA)
        cv2.circle(img, (zx, pz), r - 1, colour, 2 if p is target else -1, cv2.LINE_AA)


# ---------------------------------------------------------------- composite

def render(frame, *, hand=None, connections=(), pred=None, progress: float = 0.0, trail=(),
           mode: str = "MIRROR", mode_sub: str = "", toasts: Toasts | None = None, status: dict | None = None,
           clean: bool = False, target_map: dict | None = None, t: float | None = None):
    """Draw everything onto a copy of `frame` and return it. Order: trail, box, skeleton, banner, toasts, strip."""
    img = frame.copy()
    if trail:
        draw_trail(img, trail)
    if pred is not None:
        colour = gesture_colour(pred.cls)
        draw_bbox(img, pred.box, f"{pred.cls} {pred.conf:.2f}", colour, progress)
    if hand is not None:
        draw_skeleton(img, hand.pts, connections)
    draw_banner(img, mode, mode_sub)
    if toasts is not None:
        toasts.draw(img, t)
    if target_map:
        draw_target_map(img, **target_map)
    if not clean and status:
        draw_status(img, **status)
    return img
