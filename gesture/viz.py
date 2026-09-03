"""All drawing. The demo is judged on this overlay; every element of the plan's visual spec is here, drawn
calmly: thin lines, one accent per element, dark translucent pills, nothing shouting except FROZEN.

1. RF-DETR hand box: thin stroke with corner accents in the gesture colour, small label + confidence pill,
   and the debounce charging arc in the top-right corner.
2. All 21 MediaPipe landmarks with skeleton connections, larger dots on fingertips + wrist, a ring on
   the landmark that steers the arm, always drawn on top of the box.
3. Fading trail of the last steering-point positions.
4. Mode pill top centre; FROZEN is a solid red banner that cannot be missed.
5. Event toasts (~1 s), one at a time; the centre-zone ring that fills while you hold your finger in it.
6. Status strip (FPS, inference ms, arm x y z, confidence); `clean=True` removes it.
7. Bottom-right map of the arm's box with target / commanded / actual.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from . import config
from .perception import FINGERTIPS, WRIST

FONT = cv2.FONT_HERSHEY_DUPLEX
WHITE = (245, 245, 243)
INK = (28, 24, 22)
GREY = (150, 150, 150)
MUTED = (105, 105, 105)
SKELETON = (235, 235, 235)
SKELETON_DOT = (80, 220, 255)     # joints
TIP_DOT = (0, 235, 190)           # fingertips + wrist
TRAIL = (255, 200, 80)            # steering trail + ring
DRY = (11, 158, 245)              # amber
OK = (129, 185, 16)               # green


def _text(img, s, org, scale, colour, thick=1):
    cv2.putText(img, s, org, FONT, scale, colour, thick, cv2.LINE_AA)


def _text_w(s, scale, thick=1):
    return cv2.getTextSize(s, FONT, scale, thick)[0]


def translucent(img, x1, y1, x2, y2, colour=INK, alpha=0.62):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    roi = img[y1:y2, x1:x2]
    if roi.size:
        cv2.addWeighted(np.full_like(roi, colour, dtype=np.uint8), alpha, roi, 1 - alpha, 0, roi)


def pill(img, x1, y1, x2, y2, fill=INK, alpha=0.62, border=None, border_thick=2):
    """Translucent rounded box: a rounded-rect mask blended once (no seams at the corners)."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return
    r = max(4, min(14, (y2 - y1) // 2, (x2 - x1) // 2))
    roi = img[y1:y2, x1:x2]
    mask = np.zeros(roi.shape[:2], np.uint8)
    rw, rh = x2 - x1, y2 - y1
    cv2.rectangle(mask, (r, 0), (rw - r - 1, rh - 1), 255, -1)
    cv2.rectangle(mask, (0, r), (rw - 1, rh - r - 1), 255, -1)
    for cx, cy in ((r, r), (rw - r - 1, r), (r, rh - r - 1), (rw - r - 1, rh - r - 1)):
        cv2.circle(mask, (cx, cy), r, 255, -1, cv2.LINE_AA)
    a = (mask.astype(np.float32) / 255.0 * alpha)[..., None]
    roi[:] = (roi * (1 - a) + np.array(fill, np.float32) * a).astype(np.uint8)
    if border is not None:
        cv2.rectangle(img, (x1, y1), (x2, y2), border, border_thick, cv2.LINE_AA)


def gesture_colour(cls: str | None):
    return config.GESTURE_COLOURS.get(cls or "", GREY)


# ---------------------------------------------------------------- 1. bounding box + charging arc

def draw_bbox(img, box, label: str, colour, progress: float = 0.0, thick: int = 2, accent: int = 22):
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(img, (x1, y1), (x2, y2), colour, thick, cv2.LINE_AA)
    a = min(accent, max(8, (x2 - x1) // 4), max(8, (y2 - y1) // 4))
    for (cx, cy, dx, dy) in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(img, (cx, cy), (cx + dx * a, cy), colour, thick + 2, cv2.LINE_AA)
        cv2.line(img, (cx, cy), (cx, cy + dy * a), colour, thick + 2, cv2.LINE_AA)
    # label pill above the box
    scale, tk = 0.6, 1
    tw, th = _text_w(label, scale, tk)
    px1, py1 = x1, y1 - th - 14
    if py1 < 0:
        py1 = y1 + 4
    px2, py2 = px1 + tw + 16, py1 + th + 10
    cv2.rectangle(img, (px1, py1), (px2, py2), colour, -1, cv2.LINE_AA)
    _text(img, label, (px1 + 8, py2 - 5), scale, WHITE, tk)
    # charging arc, top-right corner: thin grey ring, coloured sweep, solid dot once fired
    r = 13
    cx, cy = x2 - r - 8, y1 + r + 8
    cv2.circle(img, (cx, cy), r, INK, -1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), r, MUTED, 2, cv2.LINE_AA)
    sweep = int(round(360 * max(0.0, min(1.0, progress))))
    if sweep > 0:
        cv2.ellipse(img, (cx, cy), (r, r), -90, 0, sweep, colour, 3, cv2.LINE_AA)
    if progress >= 1.0:
        cv2.circle(img, (cx, cy), r - 5, colour, -1, cv2.LINE_AA)


# ---------------------------------------------------------------- 2. skeleton

def draw_skeleton(img, pts: np.ndarray, connections, colour=SKELETON, track: int | None = None):
    """All 21 landmarks: bones, a dot on every joint, larger dots on fingertips + wrist, a ring on `track`."""
    p = [(int(round(x)), int(round(y))) for x, y in pts]
    for a, b in connections:
        cv2.line(img, p[a], p[b], INK, 4, cv2.LINE_AA)
        cv2.line(img, p[a], p[b], colour, 2, cv2.LINE_AA)
    for i, pt in enumerate(p):
        if i in FINGERTIPS or i == WRIST:
            cv2.circle(img, pt, 6, INK, -1, cv2.LINE_AA)
            cv2.circle(img, pt, 5, TIP_DOT, -1, cv2.LINE_AA)
        else:
            cv2.circle(img, pt, 4, INK, -1, cv2.LINE_AA)
            cv2.circle(img, pt, 3, SKELETON_DOT, -1, cv2.LINE_AA)
    if track is not None and 0 <= track < len(p):
        cv2.circle(img, p[track], 13, INK, 4, cv2.LINE_AA)
        cv2.circle(img, p[track], 13, TRAIL, 2, cv2.LINE_AA)


# ---------------------------------------------------------------- 3. trail

def draw_trail(img, trail, colour=TRAIL):
    n = len(trail)
    if n < 2:
        return
    for i in range(1, n):
        f = i / (n - 1)
        c = tuple(int(v * (0.3 + 0.7 * f)) for v in colour)
        t = max(1, int(round(1 + 2 * f)))
        cv2.line(img, (int(trail[i - 1][0]), int(trail[i - 1][1])), (int(trail[i][0]), int(trail[i][1])), c, t, cv2.LINE_AA)


# ---------------------------------------------------------------- 4. mode pill

def draw_banner(img, mode: str, sub: str = ""):
    h, w = img.shape[:2]
    colour = config.MODE_COLOURS.get(mode, GREY)
    if mode == "FROZEN":
        scale, tk = 1.4, 3
        tw, th = _text_w(mode, scale, tk)
        x1, y1 = (w - tw) // 2 - 40, 14
        x2, y2 = x1 + tw + 80, y1 + th + 28
        cv2.rectangle(img, (x1, y1), (x2, y2), colour, -1)
        cv2.rectangle(img, (x1, y1), (x2, y2), WHITE, 3)
        _text(img, mode, (x1 + 40, y2 - 14), scale, WHITE, tk)
    else:
        scale, tk = 0.95, 2
        tw, th = _text_w(mode, scale, tk)
        x1, y1 = (w - tw) // 2 - 22, 14
        x2, y2 = x1 + tw + 44, y1 + th + 20
        pill(img, x1, y1, x2, y2, alpha=0.6)
        cv2.line(img, (x1 + 10, y2 - 3), (x2 - 10, y2 - 3), colour, 2, cv2.LINE_AA)
        _text(img, mode, (x1 + 22, y2 - 11), scale, colour, tk)
    if sub:
        sw, sh = _text_w(sub, 0.58, 1)
        pill(img, (w - sw) // 2 - 12, y2 + 8, (w + sw) // 2 + 12, y2 + sh + 22, alpha=0.5)
        _text(img, sub, ((w - sw) // 2, y2 + sh + 15), 0.58, WHITE, 1)


# ---------------------------------------------------------------- 5. toasts + centre zone

class Toasts:
    def __init__(self, seconds: float | None = None):
        self.seconds = config.TOAST_S if seconds is None else seconds
        self.items: list[tuple[str, tuple, float]] = []

    def add(self, text: str, colour=WHITE, t: float | None = None) -> None:
        t = time.time() if t is None else t
        self.items.append((text, colour, t + self.seconds))

    def draw(self, img, t: float | None = None) -> None:
        """Only the newest toast is shown; it fades over its last third."""
        t = time.time() if t is None else t
        self.items = [it for it in self.items if it[2] > t]
        if not self.items:
            return
        h, w = img.shape[:2]
        text, colour, t_end = self.items[-1]
        left = (t_end - t) / self.seconds
        alpha = 0.65 * min(1.0, left * 3)
        scale, tk = 1.05, 2
        tw, th = _text_w(text, scale, tk)
        y = int(h * 0.26)
        x1, y1 = (w - tw) // 2 - 22, y - th - 12
        x2, y2 = x1 + tw + 44, y + 12
        pill(img, x1, y1, x2, y2, alpha=alpha, border=colour if left > 0.33 else None, border_thick=2)
        _text(img, text, (x1 + 22, y), scale, colour, tk)


def draw_center_zone(img, progress: float, in_zone: bool, hand_seen: bool, label: str = "point here to start"):
    """The re-centre target: a ring at the frame centre that fills while the fingertip is held inside it."""
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    r = int(config.RECENTER_RADIUS * h)
    colour = OK if in_zone else (GREY if hand_seen else MUTED)
    cv2.circle(img, (cx, cy), r, INK, 4, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), r, colour, 2, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), 4, colour, -1, cv2.LINE_AA)
    sweep = int(round(360 * max(0.0, min(1.0, progress))))
    if sweep > 0:
        cv2.ellipse(img, (cx, cy), (r, r), -90, 0, sweep, OK, 6, cv2.LINE_AA)
    if not hand_seen:
        label = "show your hand"
    tw, th = _text_w(label, 0.62, 1)
    pill(img, cx - tw // 2 - 12, cy + r + 12, cx + tw // 2 + 12, cy + r + th + 26, alpha=0.5)
    _text(img, label, (cx - tw // 2, cy + r + th + 19), 0.62, WHITE, 1)


# ---------------------------------------------------------------- 6. status strip

STATUS_GAP = "    "


def status_layout(width: int, parts: list[str], scale: float = 0.5, min_scale: float = 0.38,
                  gap: str = STATUS_GAP) -> tuple[float, list[str]]:
    """Pick a text scale (and, failing that, drop trailing parts) so the strip fits; keep the conf field."""
    parts = list(parts)
    while True:
        sc = scale
        while sc >= min_scale:
            if _text_w(gap.join(parts), sc, 1)[0] <= width - 28:
                return sc, parts
            sc = round(sc - 0.04, 2)
        if len(parts) <= 2:
            return min_scale, parts
        drop = -1 if "conf" not in parts[-1] else -2
        del parts[drop]


def draw_status(img, *, fps: float, infer_ms: float, hand_ms: float, arm_xyz, conf: float | None,
                gesture: str | None, extra: str = "", dry_run: bool = False, warn: str = ""):
    h, w = img.shape[:2]
    bar = 32
    translucent(img, 0, h - bar, w, h, alpha=0.6)
    xyz = "arm %s" % (" ".join("%4.0f" % v for v in arm_xyz) if arm_xyz else "  -    -    -")
    conf_s = "%.2f" % conf if conf is not None else " - "
    parts = [f"{fps:4.1f} fps", f"infer {infer_ms:4.0f} ms", f"hand {hand_ms:3.0f} ms", xyz,
             f"{(gesture or '-')} conf {conf_s}"]
    if dry_run:
        parts.insert(0, "DRY RUN")
    if extra:
        parts.append(extra)
    if warn:
        parts.append(warn)
    gap = STATUS_GAP
    sc, parts = status_layout(w, parts, gap=gap)
    _text(img, gap.join(parts), (14, h - 11), sc, DRY if dry_run else GREY, 1)
    if warn and warn in parts:
        tw, _ = _text_w(gap.join(parts), sc, 1)
        ww, _ = _text_w(warn, sc, 1)
        _text(img, warn, (14 + tw - ww, h - 11), sc, config.GESTURE_COLOURS["fist"], 1)


# ---------------------------------------------------------------- 7. target map (bottom right)

def draw_target_map(img, box, target, commanded, actual=None, label="arm target"):
    """Top-down (x, y) + side (z) view of the arm's box in the bottom-right corner."""
    h, w = img.shape[:2]
    (xlo, xhi), (ylo, yhi), (zlo, zhi) = box
    mw, mh = 200, 140
    x0, y0 = w - mw - 56, h - mh - 56
    pill(img, x0 - 10, y0 - 26, x0 + mw + 44, y0 + mh + 10, alpha=0.6)
    _text(img, label, (x0, y0 - 9), 0.5, GREY, 1)
    cv2.rectangle(img, (x0, y0), (x0 + mw, y0 + mh), MUTED, 1)
    zx = x0 + mw + 22
    cv2.line(img, (zx, y0), (zx, y0 + mh), MUTED, 2)

    def to_px(p):
        fx = (p[0] - xlo) / max(1e-6, xhi - xlo)
        fy = (p[1] - ylo) / max(1e-6, yhi - ylo)       # front of the arm (-y) at the bottom of the map
        fz = (p[2] - zlo) / max(1e-6, zhi - zlo)
        return (int(x0 + fx * mw), int(y0 + (1 - fy) * mh)), int(y0 + (1 - fz) * mh)

    for p, colour, r in ((actual, WHITE, 4), (commanded, DRY, 5), (target, OK, 7)):
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
           clean: bool = False, target_map: dict | None = None, center: dict | None = None, t: float | None = None):
    """Draw everything onto a copy of `frame`. Order: trail, box, skeleton, centre zone, pill, toasts, map, strip."""
    img = frame.copy()
    if trail:
        draw_trail(img, trail)
    if pred is not None:
        colour = gesture_colour(pred.cls)
        label = f"{pred.cls} {pred.conf:.2f}" + (" (lm)" if getattr(pred, "corrected", False) else "")
        draw_bbox(img, pred.box, label, colour, progress)
    if hand is not None:
        draw_skeleton(img, hand.pts, connections, track=config.TRACK_LANDMARK)
    if center:
        draw_center_zone(img, center.get("progress", 0.0), center.get("in_zone", False), hand is not None)
    draw_banner(img, mode, mode_sub)
    if toasts is not None:
        toasts.draw(img, t)
    if target_map:
        draw_target_map(img, **target_map)
    if not clean and status:
        draw_status(img, **status)
    return img
