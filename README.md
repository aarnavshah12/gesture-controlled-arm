# Gesture Arm

A hand in front of the Mac's camera drives a Hiwonder MaxArm. MediaPipe hand landmarks (21 points, every
frame) decide **where** the arm goes: your index fingertip is mirrored onto the end effector. An RF-DETR gesture
model trained on [Roboflow](https://roboflow.com) and run locally decides **what** the arm does: six
gestures, debounced into events. Everything runs on the Mac; the arm's own firmware does the inverse
kinematics and only ever receives end-effector (x, y, z) and suction on/off over USB serial.

```
Mac camera ─► brightness fix ─┬─► MediaPipe landmarks (every frame) ─► smoothed wrist ─► 10 Hz mirror loop ─┐
                              └─► RF-DETR gesture model (every 2nd frame, worker thread) ─► debounce ─► events ─┤
                                                                                                                ▼
                                                                          state machine (MIRROR / FROZEN / ROUTINE) ─► MaxArm
```

The demo is the overlay: the gesture box with its confidence pill and charging arc, the full hand skeleton,
the wrist trail, the mode banner, event toasts and a status strip, all drawn every frame by `gesture/viz.py`.

## Gestures

| Gesture (model class) | Event | What the arm does |
|---|---|---|
| `point` | (steer) | The steering pose: the arm follows your index fingertip. Never fires an event. |
| `pinch` | GRAB | At the current spot: descend to hover, slowly onto the block, suction on, lift back up. The arm stops following while the pinch charges, so it lands where you pointed. Refused (grey toast says why) until your finger is re-centred and the arm is inside the steering box. Mirroring continues without re-centring afterwards. |
| `open-palm` | RELEASE / PLACE / RESUME | Holding something: descend, release just above the block, lift back up. Otherwise a plain release. From FROZEN: the only way out (plain release). |
| `fist` | FREEZE | Dead-man switch. Halts where it is, ignores everything except `open-palm`. Aborts any routine. |
| `thumbs-up` | HOME | Goes to the home pose. Mirroring resumes once your finger is back in the centre of the frame. |
| `peace` | FLOURISH | A scripted wave and nod. Accepted only if the landmarks agree it is two fingers (index + middle up, ring + pinky down); otherwise it counts as `point`, labelled `point (lm)`. (Map `"peace": "PICK"` in `gesture/config.py` for the block picker's autonomous pick instead.) |

An event fires after 5 consecutive accepted predictions above 0.7 confidence (`gesture/config.py`), once
per hold; a flickering prediction fires nothing. The arc on the box shows the charge.

## Setup

```bash
./setup.sh                                   # uv venv (Python 3.12) + pinned deps + MediaPipe hand model
export ROBOFLOW_API_KEY=...                  # or put ROBOFLOW_API_KEY=... in a gitignored .env
cd tests && ../.venv/bin/python -m unittest  # 40 offline tests, no hardware needed
```

The block-picker project must be at `~/Documents/Defect-detect bot` (`gesture/config.py: BLOCK_PICKER_DIR`).
It supplies the arm driver, the rig's measured values (serial port, reach limits, home, table Z) and, for
PICK, its `calibration.npy`, model and overhead camera. Nothing from it is copied or re-implemented.

## Run

```bash
.venv/bin/python gesture_arm.py --dry-run     # full pipeline + full overlay, planned targets drawn, NO motion
.venv/bin/python gesture_arm.py               # live: asks "Workspace clear?" once, then goes
.venv/bin/python gesture_arm.py --clean       # no status strip (filming); everything else stays
.venv/bin/python gesture_arm.py --windowed    # not full screen
.venv/bin/python gesture_arm.py --record demo.mp4
```

Keys: `q`/Esc quit, `c` toggle the status strip, `f` toggle full screen, `r` re-centre the hand reference
(only in MIRROR; it never un-freezes).

Steering starts with the ring in the middle of the frame: hold your index fingertip inside it and it fills
green in under half a second. Then finger left/right = arm x, finger up/down = arm z, inside a fixed box in
front of the arm. No hand for 1 s = the arm holds. After HOME, FLOURISH or any freeze you re-centre again;
after GRAB / PLACE you just carry on. The overlay is deliberately quiet: the box, the skeleton with a ring on
the steering fingertip, a short trail, the mode pill (FROZEN is a solid red banner), one toast at a time, the
bottom-right map of the arm's box, and a thin status strip that `--clean` removes.
Optional depth (hand closer to the camera = arm forward) is `MIRROR_DEPTH` in `gesture/config.py`, off by
default because it is the noisiest axis.

## Bring-up, in the plan's order

1. **Perception** - `--dry-run`, no arm needed. Each of the six gestures should get a correctly labelled box
   and the full 21-point skeleton at the same time. (Brightness correction is identity: the dataset was
   recorded with the camera's default output.)
2. **State machine** - still `--dry-run`: each gesture fires exactly one `EVENT` line in the log; the toast
   and the arc show it on screen. Flicker fires nothing.
3. **Mirroring** - arm plugged in and powered 15 s. `--dry-run` shows the planned target on the map in the
   corner; then live. Check the mirror box (`MIRROR_*` in config, intersected with the block picker's reach
   limits) and that a fist freezes instantly.
4. **Grip / release / home** live.
5. **Routines** - point (needs the overhead camera and a valid block-picker calibration), peace, and a fist
   mid-routine.
6. **Filming** - `--clean`, see `DEMO.md`.

## Safety

- Every commanded target is clamped to the mirror box, which is itself inside the block picker's measured
  reach limits and above table Z; targets that would need more than 88 % of the arm's stretch are pulled
  back (the firmware silently ignores those). The driver checks again before any byte is sent.
- The commanded point moves at most 150 mm/s; the control loop runs at a fixed 10 Hz regardless of camera FPS.
- `fist` halts the arm at its read-back position within one control tick, aborts routines and inhibits every
  further move frame at the serial layer until `open-palm` resumes. No hand for 1 s = hold. If the arm's
  position cannot be read, mirroring stays disabled (the strip says so) until thumbs-up HOME re-syncs.
- GRAB descends to the block picker's pick height (table Z + block height - cup press = 84 mm), i.e. it assumes
  a 40 mm kit block under the cup; steer above the block before pinching. While holding, the steering box
  floor rises to the travel height (160 mm) so the carried block clears the blocks still on the table, and
  a second pinch is ignored until you place.
- A fist mid-GRAB or mid-carry keeps the pump on; the following `open-palm` vents where the arm stopped
  (logged as a warning). Every freeze requires a re-centre before the arm follows again. Quitting while
  FROZEN vents and leaves the arm where it halted.
- The first live run of a session asks `Workspace clear? [y/N]` before anything moves. `--dry-run` never
  opens the serial port. Do not leave it running unattended.
- Every accepted/rejected prediction (class, confidence, debounce count), every event, every mode transition
  and every serial frame goes to `logs/<timestamp>.log`.

## Files

| File | What it owns |
|---|---|
| `gesture_arm.py` | The loop: camera -> landmarks + detector worker -> debounce -> state machine -> overlay. |
| `gesture/config.py` | Every tunable and physical value; class strings verbatim from the model. |
| `gesture/camera.py` | Capture, selfie flip, `cv2.convertScaleAbs` brightness correction. |
| `gesture/perception.py` | `HandTracker` (MediaPipe), `GestureDetector` (RF-DETR via `inference`), `DetectorWorker`. |
| `gesture/gestures.py` | `Debouncer`, `StateMachine` (MIRROR / FROZEN / ROUTINE), the `Actions` interface. |
| `gesture/motion.py` | Mirror math, clamps, velocity cap, the 10 Hz `MotionController`. |
| `gesture/arm.py` | `GestureArm` over the block picker's driver: `stream_to`, `halt`, `grip`, `release`. |
| `gesture/routines.py` | HOME, FLOURISH, PICK (the block picker's `PickLoop`, once), abortable. |
| `gesture/viz.py` | All drawing. |
| `gesture/bp.py` | Imports the block-picker project as-is. |
| `gesture/runlog.py` | Per-session log; the block picker's logger writes to the same file. |
| `tests/` | Offline tests: debounce, state machine, motion, arm (fake serial), overlay. |
