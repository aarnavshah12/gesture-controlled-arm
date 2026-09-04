"""Article figures rendered from the real take, the real logs and the dataset frames (nothing synthetic).

    python article_figs.py --clip "demo draft.mov" --log logs/20260903-125002.log --offset 12.64 \
        --samples /path/to/samples --out Demo/article-figs

Writes: 01-debounce.png, 02-veto.png, 03-steering.png, 04-pick-place.png, 05-fist.png, 06-results.png
and contact.jpg (all six tiled, for a quick look).
"""

from __future__ import annotations

import argparse
import glob
import os
import re

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import patches  # noqa: E402

from gesture import config  # noqa: E402

RGB = {k: tuple(c / 255 for c in reversed(v)) for k, v in config.GESTURE_COLOURS.items()}   # BGR -> RGB 0..1
INK = "#1B2028"
GREY = "#6B7280"
ACCENT = "#2F8FD6"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": GREY, "axes.labelcolor": INK,
                     "xtick.color": GREY, "ytick.color": GREY, "text.color": INK, "axes.titleweight": "bold",
                     "axes.titlesize": 13, "figure.dpi": 150})


def ts(line):
    h, m, s = line[:12].split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def read_log(path):
    lines = [l for l in open(path).read().splitlines() if re.match(r"\d\d:\d\d:\d\d", l)]
    t0 = ts(lines[0])
    return [(ts(l) - t0, l[13:]) for l in lines]


def frame_at(cap, t_clip):
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t_clip * fps)))
    ok, f = cap.read()
    return f


def rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ------------------------------------------------------------------ 1. debounce

def fig_debounce(log, out):
    """Every model result around the first GRAB: confidence, class, the charge count, the event."""
    t_ev = next(t for t, l in log if "EVENT GRAB" in l)
    lo, hi = t_ev - 4.2, t_ev + 1.2
    pts = []
    count = [(lo, 0)]
    for t, l in log:
        if not (lo <= t <= hi):
            continue
        m = re.search(r"gesture (seen|accepted|rejected): (\S+) ([\d.]+)(?: count=(\d+)/(\d+))?", l)
        if m:
            kind, cls, conf = m.group(1), m.group(2), float(m.group(3))
            pts.append((t, cls, conf, kind, int(m.group(4)) if m.group(4) else 0))
            count.append((t, min(int(m.group(4)), 5) if m.group(4) else 0))
        if "gesture rejected: no detection" in l:
            pts.append((t, "none", 0.0, "rejected", 0)); count.append((t, 0))
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 5.2), sharex=True, gridspec_kw={"height_ratios": [3, 1.4], "hspace": 0.12})
    ax.axhspan(0, config.CONFIDENCE, color="#F1F3F6", zorder=0)
    ax.axhline(config.CONFIDENCE, color=GREY, ls="--", lw=1)
    ax.text(-4.15, config.CONFIDENCE + 0.02, "0.70 threshold", color=GREY, fontsize=9)
    ax.set_xlim(-4.2, 1.2)
    for t, cls, conf, kind, c in pts:
        ax.scatter(t - t_ev, conf, s=70, color=RGB.get(cls, "#999"), edgecolor="white", linewidth=1, zorder=3)
    ax.axvline(0, color=RGB["pinch"], lw=2, alpha=0.8)
    ax.text(0.06, 0.28, "EVENT: GRAB fires\n(5th accepted pinch)", color=RGB["pinch"], fontsize=10, fontweight="bold")
    first_pinch = next(t for t, cls, conf, kind, c in pts if cls == "pinch") - t_ev
    ax.annotate("pinch first seen\ncount 1/5, target frozen", (first_pinch, 0.75), (first_pinch - 1.9, 0.86),
                arrowprops=dict(arrowstyle="->", color=INK), fontsize=9)
    ax.text(-4.0, 0.955, "point = steering pose: recognised, drawn, never charges", color=RGB["point"], fontsize=9)
    ax.set_ylim(0.4, 1.02); ax.set_ylabel("model confidence")
    ax.set_title("One gesture, frame by frame: the model's results around the first grab (12:50 session)")
    for cls in ("point", "pinch"):
        ax.scatter([], [], color=RGB[cls], s=60, label=cls)
    ax.legend(loc="lower left", frameon=False, fontsize=9)
    xs, ys = zip(*count)
    ax2.step([x - t_ev for x in xs] + [hi - t_ev], list(ys) + [ys[-1]], where="post", color=RGB["pinch"], lw=2)
    ax2.fill_between([x - t_ev for x in xs] + [hi - t_ev], list(ys) + [ys[-1]], step="post", color=RGB["pinch"], alpha=0.15)
    ax2.set_yticks([0, 5]); ax2.set_ylim(-0.3, 6); ax2.set_ylabel("charge")
    ax2.set_xlabel("seconds before / after the event")
    ax2.text(-4.0, 4.6, "debounce: 5 consecutive accepted results of the same class", fontsize=9, color=GREY)
    for a in (ax, ax2):
        a.spines[["top", "right"]].set_visible(False)
    fig.savefig(out, bbox_inches="tight", facecolor="white"); plt.close(fig)


# ------------------------------------------------------------------ 2. point vs peace

def finger_panel(ax, img_bgr, title):
    """Real frame + the landmark test: wrist->PIP vs wrist->tip for the four fingers."""
    from gesture.perception import FINGERS, HandTracker, finger_states, two_fingers_up

    tr = HandTracker()
    h = None
    for k in range(3):
        h = tr.process(img_bgr, k * 0.1) or h
    tr.close()
    if h is None:
        ax.imshow(rgb(img_bgr)); ax.axis("off"); ax.set_title(title + " (no hand)"); return
    # crop to the hand (plus margin) so the finger test is legible
    x0, y0 = h.pts.min(axis=0); x1, y1 = h.pts.max(axis=0)
    m = 0.45 * max(x1 - x0, y1 - y0)
    cx0, cy0 = int(max(0, x0 - m)), int(max(0, y0 - m))
    cx1, cy1 = int(min(img_bgr.shape[1], x1 + m)), int(min(img_bgr.shape[0], y1 + m))
    ax.imshow(rgb(img_bgr[cy0:cy1, cx0:cx1])); ax.axis("off")
    pts = h.pts - np.array([cx0, cy0])
    for a, b in tr.connections:
        ax.plot([pts[a, 0], pts[b, 0]], [pts[a, 1], pts[b, 1]], color="white", lw=1.2, alpha=0.8)
    st = finger_states(h.norm)
    w = pts[0]
    for name, (tip, pip) in FINGERS.items():
        col = "#22C55E" if st[name] else "#EF4444"
        ax.plot([w[0], pts[pip, 0]], [w[1], pts[pip, 1]], color="#FACC15", lw=2.2)
        ax.plot([w[0], pts[tip, 0]], [w[1], pts[tip, 1]], color=col, lw=2.2)
        ax.scatter([pts[tip, 0]], [pts[tip, 1]], color=col, s=60, zorder=5, edgecolor="white")
    verdict = "two fingers up: peace stands" if two_fingers_up(st) else "not two fingers: peace -> point"
    code = "  ".join(f"{k} {'up' if v else 'down'}" for k, v in st.items())
    ax.set_title(f"{title}\n{verdict}", fontsize=10)
    ax.text(0.02, 0.03, code, transform=ax.transAxes, fontsize=8.5, color="white",
            bbox=dict(facecolor=INK, alpha=0.75, edgecolor="none", pad=4))


def fig_veto(cap, log, offset, samples, out):
    """One live moment, three panels: what RF-DETR said (quoted from the log), what the landmarks said
    (from the same log line), and the label the app actually showed and used."""
    vet = [(t, l) for t, l in log if "landmark veto" in l]
    t_v, line = max(vet, key=lambda tl: float(re.search(r"model peace ([\d.]+)", tl[1]).group(1)))
    conf = re.search(r"model peace ([\d.]+)", line).group(1)
    code = re.search(r"fingers (\S+)", line).group(1)
    names = ["index", "middle", "ring", "pinky"]
    states = ", ".join(f"{n} {'up' if c.isupper() else 'down'}" for n, c in zip(names, code))
    fr = frame_at(cap, t_v - offset + 0.15)
    H, W = fr.shape[:2]
    x0, x1 = int(W * 0.28), int(W * 0.74)
    full = fr[int(H * 0.16):int(H * 0.98), x0:x1]          # includes the app's own label pill
    below = fr[int(H * 0.275):int(H * 0.98), x0:x1]        # label pill cropped away
    pink, purple = RGB["peace"], RGB["point"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.6))
    for ax in axes:
        ax.axis("off")
    axes[0].imshow(rgb(below))
    axes[0].set_title("1. What RF-DETR said", fontsize=12)
    axes[0].text(0.03, 0.96, f"peace  {conf}", transform=axes[0].transAxes, va="top", fontsize=15, fontweight="bold", color="white",
                 bbox=dict(facecolor=pink, edgecolor="none", pad=8))
    axes[0].text(0.03, 0.80, "top-1 class at this instant\n(quoted from the session log)", transform=axes[0].transAxes, va="top",
                 fontsize=9, color=INK, bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=5))
    axes[1].imshow(rgb(below))
    axes[1].set_title("2. What the landmarks said", fontsize=12)
    axes[1].text(0.03, 0.96, "not two fingers up", transform=axes[1].transAxes, va="top", fontsize=15, fontweight="bold", color="white",
                 bbox=dict(facecolor=INK, edgecolor="none", pad=8))
    axes[1].text(0.03, 0.80, f"{states}\n(same log line: peace needs index + middle up,\nring + pinky down)", transform=axes[1].transAxes,
                 va="top", fontsize=9, color=INK, bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=5))
    axes[2].imshow(rgb(full))
    axes[2].set_title("3. What the app showed and used", fontsize=12)
    axes[2].text(0.03, 0.04, f"point  {conf}  (lm)", transform=axes[2].transAxes, va="bottom", fontsize=15, fontweight="bold", color="white",
                 bbox=dict(facecolor=purple, edgecolor="none", pad=8))
    axes[2].text(0.03, 0.20, "the label on screen (top of the box): peace vetoed -> point,\nthe steering pose. No handshake fires. (lm) marks the correction.",
                 transform=axes[2].transAxes, va="bottom", fontsize=9, color=INK, bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=5))
    fig.text(0.5, 0.01, f"Log line: {line.replace('INFO gesture ', '').strip()}     Landmarks can veto a label, never fire an event. "
             f"{len(vet)} vetoes in this take, 459 across the day. On clean validation frames the model also hedges: a pointing hand "
             f"gets both a point 0.63 and a peace 0.37 box.", ha="center", fontsize=8.5, color=GREY, wrap=True)
    fig.subplots_adjust(wspace=0.04, bottom=0.1)
    fig.savefig(out, bbox_inches="tight", facecolor="white"); plt.close(fig)


# ------------------------------------------------------------------ 3. finger -> coordinate

def fig_steering(cap, log, offset, out):
    """Left: the steering frame with the reference and the fingertip vector. Right: the workspace box with
    every streamed target of the session, from the log."""
    from gesture.perception import HandTracker

    ref = next(re.search(r"re-centred at \(([\d.]+), ([\d.]+)\)", l) for t, l in log if "re-centred" in l)
    ref = (float(ref.group(1)), float(ref.group(2)))
    t_clip = next(t for t, l in log if "stream_to" in l) - offset + 2.3
    fr = frame_at(cap, t_clip)
    H, W = fr.shape[:2]
    small = cv2.resize(fr, (1280, 720))
    tr = HandTracker(); hand = None
    for k in range(3):
        hand = tr.process(small, k * 0.1) or hand
    tr.close()
    fig = plt.figure(figsize=(14, 5.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1], wspace=0.1)
    ax = fig.add_subplot(gs[0]); ax.imshow(rgb(small)); ax.axis("off")
    rx, ry = ref[0] * 1280, ref[1] * 720
    ax.scatter([rx], [ry], s=120, facecolor="none", edgecolor=ACCENT, linewidth=2)
    ax.text(rx + 14, ry - 40, "reference (re-centred here)", color=ACCENT, fontsize=9, fontweight="bold")
    if hand is not None:
        fx, fy = hand.pts[config.TRACK_LANDMARK]
        ax.annotate("", (fx, fy), (rx, ry), arrowprops=dict(arrowstyle="->", color="#FACC15", lw=2.5))
        ax.scatter([fx], [fy], s=90, color="#FACC15", edgecolor=INK, zorder=5)
        ax.text(fx + 12, fy + 26, "fingertip (landmark 8)", color="#FACC15", fontsize=9, fontweight="bold")
    ax.set_title("target = origin + gain x (fingertip - reference)", fontsize=12)
    ax2 = fig.add_subplot(gs[1])
    (xlo, xhi), _, (zlo, zhi) = config.MIRROR_X_MM, config.MIRROR_Y_MM, config.MIRROR_Z_MM
    ax2.add_patch(patches.Rectangle((xlo, zlo), xhi - xlo, zhi - zlo, facecolor="#F1F3F6", edgecolor=GREY, lw=1.5))
    xs, zs = [], []
    for t, l in log:
        m = re.search(r"stream_to\(([-\d.]+), ([-\d.]+), ([-\d.]+)\)", l)
        if m:
            xs.append(float(m.group(1))); zs.append(float(m.group(3)))
    ax2.plot(xs, zs, color=ACCENT, lw=1.5, alpha=0.9)
    ax2.scatter(xs[0], zs[0], color=ACCENT, s=40, zorder=4); ax2.text(xs[0] + 6, zs[0] + 3, "start", fontsize=8, color=ACCENT)
    ox, oy, oz = config.MIRROR_ORIGIN_XYZ_MM
    ax2.scatter([ox], [oz], marker="+", s=160, color=INK, zorder=5); ax2.text(ox + 6, oz + 4, "origin", fontsize=9)
    ax2.axhline(87, color=RGB["fist"], ls=":", lw=1); ax2.text(xhi - 4, 89, "block tops 87 mm", fontsize=8, color=RGB["fist"], ha="right")
    ax2.annotate("", (xhi - 10, zlo - 8), (xlo + 10, zlo - 8), arrowprops=dict(arrowstyle="<->", color=GREY))
    ax2.text(0, zlo - 17, "finger left / right  ->  arm x", ha="center", fontsize=9, color=GREY)
    ax2.annotate("", (xhi + 18, zhi - 5), (xhi + 18, zlo + 5), arrowprops=dict(arrowstyle="<->", color=GREY))
    ax2.text(xhi + 24, (zlo + zhi) / 2, "finger up / down\n->  arm z", fontsize=9, color=GREY, va="center")
    ax2.set_xlim(xlo - 30, xhi + 90); ax2.set_ylim(78, zhi + 20)
    ax2.set_xlabel("arm x (mm)"); ax2.set_ylabel("arm z (mm)"); ax2.set_aspect("equal")
    ax2.set_title(f"Workspace box x {xlo:.0f}..{xhi:.0f}, z {zlo:.0f}..{zhi:.0f} mm, y fixed at {oy:.0f}\n{len(xs)} streamed targets in this session (blue)", fontsize=10.5)
    ax2.spines[["top", "right"]].set_visible(False)
    fig.savefig(out, bbox_inches="tight", facecolor="white"); plt.close(fig)


# ------------------------------------------------------------------ 4. pick and place filmstrip

def fig_pick(cap, log, offset, out):
    """The arm camera through the GRAB, at the log's own timestamps, plus the rewind moment from the log."""
    t_ev = next(t for t, l in log if "EVENT GRAB" in l)
    moves = [(t, l) for t, l in log if t_ev <= t <= t_ev + 8 and ("move_to" in l or "suction ON" in l or "routine GRAB: done" in l)]
    steps = [(t_ev - 0.6, "steering, cup at z 125"), ]
    for t, l in moves:
        m = re.search(r"move_to\([-\d.]+, [-\d.]+, ([-\d.]+)\) (\d+) ms", l)
        if m:
            z, ms = float(m.group(1)), int(m.group(2))
            n_hover = sum(1 for _, c in steps if c.startswith("hover")) + sum(1 for _, c in steps if c.startswith("lift"))
            label = {84.0: "down to z 84 (cup on the block)"}.get(z)
            if label is None:
                label = ["hover at z 124", "lift back to z 124", "up: holding the block"][min(n_hover, 2)] if z == 124.0 else f"z {z:.0f}"
            steps.append((t + ms / 1000 + 0.25, label))
        elif "suction ON" in l:
            steps.append((t + 0.6, "pump on, 1 s to seal"))
    steps = steps[:6]
    H, W = 2160, 3840
    fig, axes = plt.subplots(1, len(steps), figsize=(2.6 * len(steps), 4.2))
    for ax, (t, cap_text) in zip(axes, steps):
        fr = frame_at(cap, t - offset)
        pip = fr[0:int(H * 0.57), int(W * 0.74):W]
        ax.imshow(rgb(pip)); ax.axis("off")
        ax.set_title(f"{t - t_ev:+.1f} s", fontsize=10, color=GREY)
        ax.text(0.5, -0.06, cap_text, transform=ax.transAxes, ha="center", fontsize=9)
    t_rew, rew = next(((t, l) for t, l in log if "rewound to" in l and abs(t - t_ev) < 2), (t_ev, ""))
    before = [l for t, l in log if "stream_to" in l and t < t_rew][-1]
    bx = re.search(r"stream_to\(([-\d.]+),", before).group(1)
    rx = re.search(r"rewound to \((-?\d+),", rew).group(1) if rew else "?"
    fig.suptitle(f"pinch = GRAB: hover, slow descent, suction, lift  |  the take's log: while the pinch formed the target drifted to x {float(bx):.0f}; "
                 f"the first pinch frame rewound it to x {rx}, where the finger had pointed", fontsize=10.5, y=1.02)
    fig.savefig(out, bbox_inches="tight", facecolor="white"); plt.close(fig)


# ------------------------------------------------------------------ 5. the fist

def fig_fist(cap, log, offset, out):
    t_ev = next(t for t, l in log if "EVENT FREEZE" in l)
    fr = frame_at(cap, t_ev - offset + 0.5)
    H, W = fr.shape[:2]
    fig = plt.figure(figsize=(14, 5.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1], wspace=0.06)
    ax = fig.add_subplot(gs[0]); ax.imshow(rgb(fr[0:int(H * 0.86), int(W * 0.12):int(W * 0.74)])); ax.axis("off")
    ax.set_title("Fist: streaming stops within a tick, the halt re-commands\nthe verified position, motion is inhibited", fontsize=10.5)
    ax2 = fig.add_subplot(gs[1]); ax2.axis("off")
    lines = [(t, l) for t, l in log if t_ev - 1.2 <= t <= t_ev + 6 and any(k in l for k in ("EVENT", "mode ", "HALT", "ABORT", "inhibit", "ignored", "count=5/5", "count=1/5"))]
    txt = "\n".join(f"{t - t_ev:+6.2f}s  {l.replace('INFO gesture ', '').replace('INFO blockpicker ', 'arm  ')[:100]}" for t, l in lines[:12])
    ax2.text(0, 1, "the log, seconds relative to the fist event\n\n" + txt, va="top", fontsize=7.6, family="monospace",
             bbox=dict(facecolor="#F1F3F6", edgecolor="none", pad=10))
    fig.savefig(out, bbox_inches="tight", facecolor="white"); plt.close(fig)


# ------------------------------------------------------------------ 6. results

def fig_results(log_dir, out):
    logs = sorted(glob.glob(os.path.join(log_dir, "20260903-*.log")))
    live = [f for f in logs if "arm: connected" in open(f).read()]
    tot = {"GRAB": [0, 0], "PLACE": [0, 0], "FLOURISH": [0, 0], "HOME": [0, 0]}
    freezes = events = streams = 0
    best = (None, 0)
    for f in live:
        s = open(f).read()
        for k in tot:
            done = len(re.findall(rf"routine {k}: done", s))
            att = done + len(re.findall(rf"routine {k}: failed", s))   # a fist abort is a demo, not a failure
            tot[k][0] += done; tot[k][1] += att
        freezes += len(re.findall(r"EVENT FREEZE", s)); events += len(re.findall(r" EVENT ", s)); streams += len(re.findall(r"stream_to\(", s))
        g = len(re.findall(r"routine GRAB: done", s))
        if g > best[1]:
            best = (f, g)
    fig = plt.figure(figsize=(14, 5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.5], wspace=0.25)
    ax = fig.add_subplot(gs[0])
    names = [("GRAB", "pinch"), ("PLACE", "open-palm"), ("HANDSHAKE", "peace"), ("HOME", "thumbs-up")]
    keys = ["GRAB", "PLACE", "FLOURISH", "HOME"]
    for i, ((label, g), k) in enumerate(zip(names, keys)):
        done, att = tot[k]
        ax.barh(i, att, color="#E5E7EB"); ax.barh(i, done, color=RGB[g])
        ax.text(att + 0.3, i, f"{done} / {att}", va="center", fontsize=10)
    ax.barh(4, freezes, color=RGB["fist"]); ax.text(freezes + 0.3, 4, f"{freezes}", va="center", fontsize=10)
    ax.set_yticks(range(5)); ax.set_yticklabels([n for n, _ in names] + ["FREEZE"]); ax.invert_yaxis()
    ax.set_xlim(0, 17); ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(f"{len(live)} live sessions on 3 Sep: routines completed / attempted\n{events} events, {streams:,} streamed commands, 0 blocked steps  (fist aborts not counted)", fontsize=10.5)
    # best session timeline
    ax2 = fig.add_subplot(gs[1])
    lg = read_log(best[0])
    t_conn = next(t for t, l in lg if "arm: connected" in l)
    evs = [(t - t_conn, re.search(r"EVENT (\S+) from (\S+)", l).groups()) for t, l in lg if " EVENT " in l]
    name_map = {"GRAB": "grab", "RELEASE": "place", "FLOURISH": "handshake", "HOME": "home", "FREEZE": "freeze"}
    holding = False
    for k, (t, (name, gest)) in enumerate(evs):
        label = name_map.get(name, name.lower())
        if name == "RELEASE":
            label = "place" if holding else "release"
        holding = name == "GRAB" or (holding and name != "RELEASE")
        ax2.scatter([t], [0], s=150, color=RGB.get(gest, "#999"), edgecolor="white", zorder=3)
    for gest, label in (("pinch", "grab"), ("open-palm", "place"), ("peace", "handshake"), ("thumbs-up", "home"), ("fist", "freeze")):
        ax2.scatter([], [], s=80, color=RGB[gest], label=label)
    ax2.legend(loc="upper center", ncol=5, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.92))
    ax2.axhline(0, color="#E5E7EB", lw=3, zorder=1)
    ax2.set_yticks([]); ax2.set_ylim(-0.6, 0.6); ax2.set_xlabel("seconds since the arm connected")
    ax2.spines[["top", "right", "left"]].set_visible(False)
    ax2.set_title(f"Best session ({os.path.basename(best[0])[9:11]}:{os.path.basename(best[0])[11:13]}): {best[1]} pick-and-place cycles back to back, {len(evs)} events", fontsize=11)
    fig.savefig(out, bbox_inches="tight", facecolor="white"); plt.close(fig)


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", default="demo draft.mov")
    ap.add_argument("--log", default="logs/20260903-125002.log")
    ap.add_argument("--offset", type=float, default=12.64, help="log seconds at the clip's first frame")
    ap.add_argument("--samples", required=True, help="folder with the dataset sample frames (NN-class.jpg)")
    ap.add_argument("--out", default="Demo/article-figs")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    log = read_log(args.log)
    cap = cv2.VideoCapture(args.clip)
    fig_debounce(log, os.path.join(args.out, "01-debounce.png"))
    fig_veto(cap, log, args.offset, args.samples, os.path.join(args.out, "02-veto.png"))
    fig_steering(cap, log, args.offset, os.path.join(args.out, "03-steering.png"))
    fig_pick(cap, log, args.offset, os.path.join(args.out, "04-pick-place.png"))
    fig_fist(cap, log, args.offset, os.path.join(args.out, "05-fist.png"))
    fig_results(os.path.dirname(args.log), os.path.join(args.out, "06-results.png"))
    tiles = [cv2.imread(os.path.join(args.out, f)) for f in sorted(os.listdir(args.out)) if f.endswith(".png")]
    tiles = [cv2.resize(t, (960, int(960 * t.shape[0] / t.shape[1]))) for t in tiles]
    h = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for t in tiles]
    rows = [np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]
    cv2.imwrite(os.path.join(args.out, "contact.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 80])
    print("wrote", sorted(os.listdir(args.out)))


if __name__ == "__main__":
    main()
