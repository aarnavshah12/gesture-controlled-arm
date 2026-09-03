"""Dashboard demo video: the screen-recorded take + a live UI rebuilt from the session log.

    python demo_ui.py "demo draft.mov" logs/20260903-125002.log --offset 12.64 --out Demo/demo-ui.mp4

Layout (1920x1080): the take fills the top-left (1440x810); a right rail shows the mode, the gesture the model
sees right now with its confidence and debounce charge, the six gesture chips (the active one lit, the fired
one flashing), the arm's commanded target and the counters; the bottom strip is an event timeline with a
playhead, the arm's path in x/z from every streamed command, and a ticker of the last log lines.

--offset  seconds between the log's first line and the first frame of the clip (log_time = clip_time + offset).
Find it from any mode change: the pill turns purple in the clip at 5.23 s, the log says GRAB at 17.87 s after
its first line -> 12.64. Everything drawn is exact to the log's timestamps, so it stays in sync for the whole take.
Also writes <out>.title.png, a title card for the editor.
"""

from __future__ import annotations

import argparse
import re
import time

import cv2
import numpy as np

from gesture import config
from gesture.viz import FONT, GREY, INK, MUTED, TRAIL, WHITE, _text, _text_w, pill

W, H = 1920, 1080
VID_W, VID_H = 1440, 810
RAIL_X = VID_W
STRIP_Y = VID_H
ACCENT = TRAIL
OK = config.MODE_COLOURS["MIRROR"]
FLASH_S = 1.2
GESTURES = ["point", "pinch", "open-palm", "fist", "thumbs-up", "peace"]
EVENT_NAME = {"GRAB": "GRAB", "RELEASE": "PLACE", "FLOURISH": "HANDSHAKE", "HOME": "HOME", "FREEZE": "FREEZE", "PICK": "PICK"}
EVENT_GESTURE = {"GRAB": "pinch", "RELEASE": "open-palm", "FLOURISH": "peace", "HOME": "thumbs-up", "FREEZE": "fist"}


# ---------------------------------------------------------------- log

def parse_log(path: str):
    """(t, kind, data) events, t in seconds after the log's first line."""
    P = {
        "event": re.compile(r"EVENT (\S+) from (\S+) ([\d.]+) in mode (\S+)"),
        "mode": re.compile(r"mode (\S+) -> (\S+) \((.*)\)"),
        "routine": re.compile(r"routine (\S+): (start|done in ([\d.]+)s|failed|aborted)"),
        "plan": re.compile(r"routine (GRAB|PLACE): at \(([-\d]+), ([-\d]+)\) from z=([-\d]+)"),
        "stream": re.compile(r"stream_to\(([-\d.]+), ([-\d.]+), ([-\d.]+)\)"),
        "move": re.compile(r"move_to\(([-\d.]+), ([-\d.]+), ([-\d.]+)\) (\d+) ms"),
        "accepted": re.compile(r"gesture accepted: (\S+) ([\d.]+) count=(\d+)/(\d+)"),
        "seen": re.compile(r"gesture seen: (\S+) ([\d.]+)"),
        "rejected": re.compile(r"gesture rejected: (\S+) ([\d.]+)"),
        "nodet": re.compile(r"gesture rejected: no detection"),
        "veto": re.compile(r"landmark veto: model (\S+) ([\d.]+)"),
        "suction": re.compile(r"suction (ON|OFF)"),
        "halt": re.compile(r"arm: HALT"),
        "recentred": re.compile(r"hand re-centred"),
        "resumed": re.compile(r"mirror: resumed"),
        "handshake": re.compile(r"handshake at"),
    }

    def ts(line):
        h, m, s = line[:12].split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)

    ev, t0 = [], None
    for line in open(path):
        if not re.match(r"\d\d:\d\d:\d\d", line):
            continue
        t = ts(line)
        t0 = t if t0 is None else t0
        t -= t0
        for kind, rx in P.items():
            m = rx.search(line)
            if m:
                ev.append((t, kind, m.groups(), line[13:].strip()))
                break
    return ev


class Replay:
    def __init__(self, ev):
        self.ev = ev
        self.i = 0
        self.mode = "MIRROR"
        self.status = "centre your finger"
        self.routine = None
        self.gesture = None          # (cls, conf, count, n) of the newest model result
        self.chips_fired = {}        # gesture -> t fired
        self.events = []             # (t, name, gesture)
        self.target = None
        self.path = []               # (t, x, y, z) streamed + routine targets
        self.grabs = self.places = self.freezes = self.handshakes = self.homes = 0
        self.vetoes = 0
        self.holding = False
        self.ticker = []
        self.last_result_t = -9

    def advance(self, t):
        while self.i < len(self.ev) and self.ev[self.i][0] <= t:
            tt, k, g, raw = self.ev[self.i]
            self.i += 1
            if k == "event":
                name, gest = g[0], g[1]
                self.events.append((tt, name, gest))
                self.chips_fired[gest] = tt
                if name == "GRAB": self.grabs += 1
                elif name == "RELEASE" and self.holding: self.places += 1
                elif name == "FLOURISH": self.handshakes += 1
                elif name == "HOME": self.homes += 1
                elif name == "FREEZE": self.freezes += 1
                self.ticker.append((tt, f"{EVENT_NAME.get(name, name)}  ({gest} {float(g[2]):.2f})", config.GESTURE_COLOURS.get(gest, WHITE)))
            elif k == "mode":
                self.mode = g[1]
                if g[1] == "FROZEN":
                    self.status = "halted - open palm to resume"
                elif g[1] == "MIRROR":
                    self.status = "following the fingertip"
                    self.routine = None
            elif k == "routine":
                name, what = g[0], g[1]
                if what == "start":
                    self.routine = name
                    self.status = {"GRAB": "grab: descending", "PLACE": "place: descending", "FLOURISH": "handshake",
                                   "HOME": "going home", "PICK": "autonomous pick"}.get(name, name.lower())
                else:
                    self.ticker.append((tt, f"{EVENT_NAME.get(name, name).lower()} {what.split(' ')[0]}", GREY))
            elif k == "plan":
                self.status = f"{g[0].lower()}: descending to ({g[1]}, {g[2]})"
            elif k == "stream":
                self.target = tuple(float(v) for v in g[:3])
                self.path.append((tt,) + self.target)
            elif k == "move":
                x, y, z = (float(v) for v in g[:3])
                self.target = (x, y, z)
                self.path.append((tt, x, y, z))
                if self.routine in ("GRAB", "PLACE") and self.target is not None:
                    if z <= 110:
                        self.status = f"{self.routine.lower()}: cup on the block"
                    elif self.path[-2][3] < z if len(self.path) > 1 else False:
                        self.status = f"{self.routine.lower()}: lifting"
            elif k == "accepted":
                self.gesture = (g[0], float(g[1]), int(g[2]), int(g[3])); self.last_result_t = tt
            elif k == "seen":
                self.gesture = (g[0], float(g[1]), 0, config.DEBOUNCE_N); self.last_result_t = tt
            elif k == "rejected":
                self.gesture = (g[0], float(g[1]), 0, config.DEBOUNCE_N); self.last_result_t = tt
            elif k == "nodet":
                self.gesture = None; self.last_result_t = tt
            elif k == "veto":
                self.vetoes += 1
            elif k == "suction":
                self.holding = g[0] == "ON"
                self.status = "suction on - lifting" if self.holding else self.status.replace("cup on the block", "released")
                self.ticker.append((tt, "pump on" if self.holding else "vent, valve close", GREY))
            elif k == "halt":
                self.ticker.append((tt, "HALT - motion inhibited", config.GESTURE_COLOURS["fist"]))
            elif k == "recentred":
                if self.mode == "MIRROR":
                    self.status = "following the fingertip"
                self.ticker.append((tt, "finger re-centred", GREY))
            elif k == "resumed":
                if self.mode == "MIRROR":
                    self.status = "centre your finger" if "recenter=True" in raw else "following the fingertip"
            elif k == "handshake":
                self.status = "handshake: 4 pumps"
        self.ticker = self.ticker[-4:]


# ---------------------------------------------------------------- drawing

def card(img, x, y, w, h, label, value, colour=WHITE, sub="", big=1.5, thick=2):
    pill(img, x, y, x + w, y + h, fill=(38, 32, 30), alpha=0.92)
    cv2.line(img, (x, y + 8), (x, y + h - 8), ACCENT, 3, cv2.LINE_AA)
    _text(img, label, (x + 18, y + 26), 0.5, GREY, 1)
    vw, vh = _text_w(str(value), big, thick)
    _text(img, str(value), (x + 18, y + 34 + vh), big, colour, thick)
    if sub:
        _text(img, sub, (x + 18, y + h - 12), 0.46, GREY, 1)


def gesture_chips(img, x, y, w, rep: Replay, t):
    """Six chips; the model's current class is lit, a fired one flashes for FLASH_S."""
    cw, gap = (w - 2 * 8) // 3, 8
    cur = rep.gesture[0] if rep.gesture and t - rep.last_result_t < 1.0 else None
    for k, g in enumerate(GESTURES):
        cx = x + (k % 3) * (cw + gap)
        y = y if k < 3 else y   # rows below
        yy = y + (k // 3) * 52
        colour = config.GESTURE_COLOURS[g]
        fired = g in rep.chips_fired and 0 <= t - rep.chips_fired[g] < FLASH_S
        lit = g == cur
        if fired:
            cv2.rectangle(img, (cx, yy), (cx + cw, yy + 44), colour, -1)
            _text(img, g, (cx + 10, yy + 29), 0.52, INK, 1)
        elif lit:
            pill(img, cx, yy, cx + cw, yy + 44, fill=colour, alpha=0.35, border=colour, border_thick=2)
            _text(img, g, (cx + 10, yy + 29), 0.52, WHITE, 1)
        else:
            pill(img, cx, yy, cx + cw, yy + 44, fill=(38, 32, 30), alpha=0.9)
            _text(img, g, (cx + 10, yy + 29), 0.52, MUTED, 1)


def timeline(img, x, y, w, h, rep: Replay, t_clip, duration, offset):
    """Event markers along the clip's duration with a playhead."""
    _text(img, "events", (x, y + 16), 0.5, GREY, 1)
    ty = y + 52
    cv2.line(img, (x, ty), (x + w, ty), MUTED, 2, cv2.LINE_AA)
    for s in range(0, int(duration) + 1, 5):
        px = int(x + w * s / duration)
        cv2.line(img, (px, ty - 5), (px, ty + 5), MUTED, 1)
        _text(img, f"{s}s", (px - 8, ty + 24), 0.4, MUTED, 1)
    for tt, name, gest in rep.events:
        tc = tt - offset
        if tc < 0 or tc > duration:
            continue
        px = int(x + w * tc / duration)
        colour = config.GESTURE_COLOURS.get(gest, WHITE)
        cv2.circle(img, (px, ty), 9, INK, -1, cv2.LINE_AA)
        cv2.circle(img, (px, ty), 7, colour, -1, cv2.LINE_AA)
        label = EVENT_NAME.get(name, name)
        lw, lh = _text_w(label, 0.5, 1)
        lx = min(max(px - lw // 2, x), x + w - lw)      # keep the label inside the track
        _text(img, label, (lx, ty - 16), 0.5, colour, 1)
    px = int(x + w * min(max(t_clip, 0), duration) / duration)
    cv2.line(img, (px, ty - 30), (px, ty + 30), WHITE, 2, cv2.LINE_AA)


def arm_path(img, x, y, w, h, rep: Replay, t):
    """x/z view of every commanded target so far (the steering path), newest point highlighted."""
    _text(img, "arm path  x / z", (x, y + 16), 0.5, GREY, 1)
    (xlo, xhi), _, (zlo, zhi) = (config.MIRROR_X_MM, config.MIRROR_Y_MM, (config.MIRROR_Z_MM[0] - 40, config.MIRROR_Z_MM[1]))
    xlo, xhi = min(xlo, -215), max(xhi, 215)
    px0, py0, pw, ph = x, y + 28, w, h - 34
    cv2.rectangle(img, (px0, py0), (px0 + pw, py0 + ph), MUTED, 1)

    def P(px, pz):
        return (int(px0 + (px - xlo) / (xhi - xlo) * pw), int(py0 + (1 - (pz - zlo) / (zhi - zlo)) * ph))

    pts = [(tt, xx, zz) for tt, xx, yy, zz in rep.path]
    for (t1, x1, z1), (t2, x2, z2) in zip(pts[:-1], pts[1:]):
        age = t - t2
        c = tuple(int(v * max(0.25, 1 - age / 25)) for v in ACCENT)
        cv2.line(img, P(x1, z1), P(x2, z2), c, 2, cv2.LINE_AA)
    if pts:
        _, xx, zz = pts[-1]
        cv2.circle(img, P(xx, zz), 6, WHITE, -1, cv2.LINE_AA)
        cv2.circle(img, P(xx, zz), 6, ACCENT, 2, cv2.LINE_AA)
    _text(img, "table", (px0 + 6, py0 + ph - 6), 0.4, MUTED, 1)


def title_card(w, h, title, subtitle):
    img = np.full((h, w, 3), INK, np.uint8)
    y = 120
    for ln in title.split("|"):
        _text(img, ln.strip(), (60, y), 2.0, WHITE, 3)
        y += 84
    cv2.line(img, (60, y - 56), (60 + _text_w(title.split("|")[-1].strip(), 2.0, 3)[0], y - 56), ACCENT, 5)
    _text(img, subtitle, (60, y + 4), 0.8, GREY, 1)
    y += 70
    x = 60
    for k, (g, what) in enumerate([("point", "steer"), ("pinch", "grab"), ("open-palm", "place"), ("peace", "handshake"),
                                   ("thumbs-up", "home"), ("fist", "freeze")]):
        colour = config.GESTURE_COLOURS[g]
        tw, th = _text_w(f"{g}  {what}", 0.7, 1)
        pill(img, x, y, x + tw + 30, y + th + 24, fill=colour, alpha=0.28, border=colour, border_thick=2)
        _text(img, f"{g}  {what}", (x + 15, y + th + 12), 0.7, WHITE, 1)
        x += tw + 44
    return img


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip")
    ap.add_argument("log")
    ap.add_argument("--offset", type=float, required=True, help="log seconds at the clip's first frame")
    ap.add_argument("--out", default="Demo/demo-ui.mp4")
    ap.add_argument("--title", default="POINT. PINCH. PLACE.|A robot arm that follows your finger")
    ap.add_argument("--subtitle", default="MediaPipe hand landmarks + RF-DETR gesture model (Roboflow)  -  Hiwonder MaxArm  -  all local")
    ap.add_argument("--rail-title", default="POINT. PINCH. PLACE.")
    args = ap.parse_args()

    ev = parse_log(args.log)
    rep = Replay(ev)
    cap = cv2.VideoCapture(args.clip)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = n / fps
    out = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    cv2.imwrite(args.out + ".title.png", title_card(W, H, args.title, args.subtitle))
    t_render = time.time()
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t_clip = i / fps
        rep.advance(t_clip + args.offset)
        canvas = np.full((H, W, 3), INK, np.uint8)
        canvas[0:VID_H, 0:VID_W] = cv2.resize(frame, (VID_W, VID_H), interpolation=cv2.INTER_AREA)

        # ---- right rail
        x, w = RAIL_X + 18, W - RAIL_X - 36
        y = 14
        _text(canvas, args.rail_title, (x, y + 30), 0.95, WHITE, 2)
        cv2.line(canvas, (x, y + 42), (x + _text_w(args.rail_title, 0.95, 2)[0], y + 42), ACCENT, 3)
        _text(canvas, "gesture-steered MaxArm", (x, y + 66), 0.5, GREY, 1)
        y += 84
        mode_colour = config.MODE_COLOURS.get(rep.mode, GREY)
        if rep.mode == "FROZEN":
            cv2.rectangle(canvas, (x, y), (x + w, y + 64), mode_colour, -1)
            _text(canvas, "FROZEN", (x + 18, y + 44), 1.1, WHITE, 2)
        else:
            pill(canvas, x, y, x + w, y + 64, fill=(38, 32, 30), alpha=0.92, border=mode_colour, border_thick=2)
            _text(canvas, rep.mode, (x + 18, y + 44), 1.1, mode_colour, 2)
        y += 72
        _text(canvas, rep.status, (x, y + 18), 0.52, WHITE, 1)
        y += 40
        # gesture the model sees right now
        g = rep.gesture if rep.gesture and t_clip + args.offset - rep.last_result_t < 1.0 else None
        gcol = config.GESTURE_COLOURS.get(g[0], GREY) if g else GREY
        charge = min(g[2], g[3]) if g else 0
        card(canvas, x, y, w, 108, "MODEL SEES", f"{g[0]}  {g[1]:.2f}" if g else "no hand", gcol, big=1.1,
             sub=(f"debounce {charge}/{g[3]}" + ("  fired" if g[2] >= g[3] else "")) if g else "RF-DETR top-1, every 2nd frame")
        if g and charge > 0:
            bw = int((w - 36) * charge / g[3])
            cv2.rectangle(canvas, (x + 18, y + 74), (x + 18 + bw, y + 79), gcol, -1)
        y += 118
        gesture_chips(canvas, x, y, w, rep, t_clip + args.offset)
        y += 110
        tgt = rep.target
        card(canvas, x, y, w, 96, "ARM TARGET  mm", f"{tgt[0]:.0f}, {tgt[1]:.0f}, {tgt[2]:.0f}" if tgt else "-", WHITE,
             big=1.05, sub="x, y, z  -  commanded, 15 Hz")
        y += 106
        cw = (w - 10) // 2
        card(canvas, x, y, cw, 92, "GRABBED", rep.grabs, config.GESTURE_COLOURS["pinch"] if rep.grabs else GREY, big=1.6,
             sub="holding" if rep.holding else "")
        card(canvas, x + cw + 10, y, cw, 92, "PLACED", rep.places, config.GESTURE_COLOURS["open-palm"] if rep.places else GREY, big=1.6)
        y += 102
        card(canvas, x, y, cw, 92, "FREEZES", rep.freezes, config.GESTURE_COLOURS["fist"] if rep.freezes else GREY, big=1.6)
        card(canvas, x + cw + 10, y, cw, 92, "VETOED", rep.vetoes, WHITE, big=1.6, sub="peace -> point")
        y += 102
        _text(canvas, "MediaPipe  -  RF-DETR on Roboflow  -  Hiwonder MaxArm", (x, VID_H - 12), 0.42, MUTED, 1)

        # ---- bottom strip
        cv2.line(canvas, (0, STRIP_Y), (W, STRIP_Y), ACCENT, 2)
        timeline(canvas, 24, STRIP_Y + 14, 1380, 100, rep, t_clip, duration, args.offset)
        # ticker
        ty = STRIP_Y + 130
        _text(canvas, "log", (24, ty + 16), 0.5, GREY, 1)
        for k, (tt, text, colour) in enumerate(rep.ticker[-4:]):
            _text(canvas, f"{max(0, tt - args.offset):5.1f}s  {text}", (24, ty + 44 + k * 26), 0.5, colour, 1)
        arm_path(canvas, 1480, STRIP_Y + 14, 420, H - STRIP_Y - 30, rep, t_clip + args.offset)
        _text(canvas, f"{t_clip:5.1f} s   frame {i:04d}", (1060, H - 16), 0.5, MUTED, 1)

        out.write(canvas)
        i += 1
        if i % 300 == 0:
            print(f"  render {i}/{n} ({time.time() - t_render:.0f}s)")
    cap.release()
    out.release()
    print(f"wrote {args.out}: {i} frames, {i / fps:.1f}s, and {args.out}.title.png")


if __name__ == "__main__":
    main()
