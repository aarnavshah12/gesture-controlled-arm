"""Every tunable and every physical value for the gesture arm, in one place.

Values marked OWNER are physical facts about the rig. Anything the owner has not confirmed yet is
flagged with a *_CONFIRMED = False so the code can warn instead of silently guessing. Arm reach
limits, home pose, table Z, serial port and baud are NOT duplicated here: they are read from the
block-picker project's config.py (single source of truth for the rig) via gesture.bp.
"""

from __future__ import annotations

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --------------------------------------------------------------------------
# Roboflow gesture model (runs locally through the `inference` package)
# --------------------------------------------------------------------------
# Project aarnavs-space/robotic-gestures, version 1 (the only version), rfdetr-medium, training
# finished 2026-09-01. The slug already carries the version number, and `inference.get_model`
# takes it as-is (a "/1" suffix on a "-t1" slug is rejected - block-picker finding, 2026-08-25).
MODEL_ID = "aarnavs-space/robotic-gestures-1-rfdetr-medium-t1"
API_KEY_ENV = "ROBOFLOW_API_KEY"
# Class strings exactly as the trained model reports them (read back from the loaded model on
# 2026-09-02: ['fist', 'null', 'open-palm', 'peace', 'pinch', 'point', 'thumbs-up']). Hyphens, not
# underscores. The model's own class list is checked against these at start-up.
CLASSES = ("fist", "open-palm", "pinch", "point", "peace", "thumbs-up")
NULL_CLASS = "null"  # the model's negative class: never an event; counts as "no gesture"
# Owner decision 2026-09-03: the arm follows the INDEX FINGERTIP, so `point` is the steering pose and
# fires nothing; `pinch` = GRAB (descend, suction on, lift at the current spot); `open-palm` = RELEASE,
# which becomes PLACE (descend, vent, lift) while an object is held. The block picker's autonomous
# pick routine is still available: map it here, e.g. "peace": "PICK", if it is wanted on a gesture.
GESTURE_EVENTS = {
    "fist": "FREEZE",
    "open-palm": "RELEASE",
    "pinch": "GRAB",
    "thumbs-up": "HOME",
    "peace": "FLOURISH",
}
NO_EVENT_CLASSES = ("point",)   # recognised and drawn, never an event, resets the debounce
CONFIDENCE = 0.7      # a prediction below this is rejected (logged) and resets the debounce
DEBOUNCE_N = 5        # consecutive accepted predictions of the same class before the event fires
DETECT_EVERY_N = 2    # run the gesture model on every Nth camera frame (in a worker thread)
DETECT_MIN_CONF = 0.3 # model floor: lower than CONFIDENCE so rejected (below-threshold) predictions are still logged and drawn

# --------------------------------------------------------------------------
# Mac camera (the one that sees the hand)
# --------------------------------------------------------------------------
# OWNER: None = auto-detect the built-in camera. With the overhead "HD Web Camera" plugged in the
# block picker measured index 0 = external, index 1 = built-in; alone, the built-in is index 0.
CAMERA_INDEX: int | None = None
CAMERA_PROBE_MAX = 3      # indices 0..CAMERA_PROBE_MAX are probed when CAMERA_INDEX is None
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
CAMERA_FPS = 30           # requested; the camera reports what it actually delivers in the log
MIRROR_VIEW = True        # selfie flip so the overlay moves with the hand (model is flip-augmented)
# OWNER: brightness correction applied right after capture, cv2.convertScaleAbs(frame, alpha, beta).
# Owner confirmed 2026-09-02: the dataset was recorded with the camera's default output, no correction,
# so the identity values are the right ones (correct() is then a no-op).
BRIGHTNESS_ALPHA = 1.0
BRIGHTNESS_BETA = 0.0
BRIGHTNESS_CONFIRMED = True

# --------------------------------------------------------------------------
# MediaPipe hand landmarker (continuous positioning signal)
# --------------------------------------------------------------------------
HAND_MODEL_PATH = os.path.join(ROOT, "models", "hand_landmarker.task")
HAND_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/"
                  "float16/1/hand_landmarker.task")
HAND_NUM = 1
HAND_MIN_DETECTION_CONF = 0.5
HAND_MIN_PRESENCE_CONF = 0.5
HAND_MIN_TRACKING_CONF = 0.5
# The landmark that drives the arm: 8 = index fingertip (owner decision 2026-09-03; 0 = wrist).
TRACK_LANDMARK = 8
WRIST_SMOOTHING = 0.35    # EMA weight of the newest tracked-point sample (1.0 = raw, 0.1 = very smooth)
WRIST_TRAIL_LEN = 30      # smoothed tracked-point positions kept for the on-screen trail
NO_HAND_HOLD_S = 1.0      # no hand for this long in MIRROR = hold position

# --------------------------------------------------------------------------
# Arm: mirroring envelope and control loop (mm, arm frame: +z up, front of the arm is -y)
# --------------------------------------------------------------------------
# OWNER: the block-picker project. Its config.py supplies SERIAL_PORT, BAUDRATE, REACH_*_MM,
# TABLE_Z_MM, HOME_XYZ_MM, TRAVEL_Z_MM; its calibration.npy + detector drive the PICK routine.
BLOCK_PICKER_DIR = os.path.expanduser("~/Documents/Defect-detect bot")
# Mirroring box. Taken from the block picker's calibrated pick area (x -115..145, y -250..-100, the
# region the arm has demonstrably reached) and heights between pick hover (122) and travel (160)
# with margin. Always intersected with the block picker's REACH_* at start-up; never wider.
MIRROR_X_MM = (-115.0, 145.0)
MIRROR_Y_MM = (-250.0, -130.0)   # near edge kept outside the block picker's 120 mm base keep-out radius
MIRROR_Z_MM = (100.0, 200.0)     # floor clears a 40 mm block on the table (47 + 40) with margin
# Where the cup sits when the hand is at its reference point: centre of the box, at a height the
# arm reached throughout calibration (61 % of full stretch).
MIRROR_ORIGIN_XYZ_MM = (0.0, -175.0, 150.0)
# Hand -> arm mapping. Wrist position is normalised to the frame (0..1). A full frame width of hand
# travel = MIRROR_GAIN_X_MM of arm travel; full frame height = MIRROR_GAIN_Z_MM. Screen y grows
# downwards, arm z grows upwards, hence the -1. Targets are clamped to the box regardless.
MIRROR_GAIN_X_MM = 320.0
MIRROR_GAIN_Z_MM = 220.0
MIRROR_X_SIGN = +1.0
MIRROR_Z_SIGN = -1.0
# Depth (arm y, toward/away from the front of the arm) from the apparent palm size: a hand closer to
# the camera looks bigger. Off by default: it is the noisiest axis. Palm size = wrist -> middle-finger
# base distance, which does not change with the finger pose. ratio = size / size at re-centre.
MIRROR_DEPTH = False
MIRROR_GAIN_Y_MM = 250.0   # arm mm per +100 % apparent palm size
MIRROR_Y_SIGN = -1.0       # bigger hand (closer to the camera) -> more negative y (toward the arm's front)
DEPTH_DEADBAND = 0.05      # ignore size ratios within +-5 % of the reference
DEPTH_SMOOTHING = 0.15
RECENTER_RADIUS = 0.12     # normalised distance from the frame centre that counts as "re-centred"
RECENTER_HOLD_S = 0.3
CONTROL_HZ = 10.0
VELOCITY_CAP_MM_S = 150.0  # commanded target may move at most this fast (per control tick)
STREAM_MOVE_MS = 150       # duration sent with each streamed target (a little over one tick)
EXTENSION_MAX = 0.88       # targets needing more of the arm's full stretch are pulled back (IK refuses ~0.9+)
READBACK_EVERY_S = 2.0     # how often the control loop reads the real position for the status strip
HALT_MOVE_MS = 300         # duration of the "stop where you are" command on FREEZE / abort

# Gripper = the MaxArm suction nozzle: GRAB = pump on, RELEASE = pump off + vent + valve close.
# GRAB / PLACE routines happen at the arm's current (x, y): descend to hover, then slowly to the pick
# height, act, lift back to hover, then to the height the arm came from. None = the block picker's value
# (pick height = TABLE_Z + BLOCK_HEIGHT - CUP_PRESS, i.e. the cup pressed onto a 40 mm block's top).
GRAB_Z_MM: float | None = None
GRAB_HOVER_MM: float | None = None      # None = block picker HOVER_OFFSET_MM (40)
PLACE_LIFT_MM: float | None = None      # release this far above the pick height (block picker RELEASE_LIFT_MM, 20)
GRAB_SUCTION_PAUSE_S = 1.0              # let the vacuum build before lifting
# While holding a block the mirror box floor rises to this height so the carried block (bottom = z - 40)
# clears blocks still on the table (tops at 87). None = block picker TRAVEL_Z_MM (160).
CARRY_Z_FLOOR_MM: float | None = None
GRAB_DESCENT_MS = 700                   # slow final descent
# Whether mirroring asks for a re-centred hand after each routine. GRAB/PLACE return to the height they
# left from, so the finger -> arm mapping is still valid and mirroring just continues.
RESUME_RECENTER = {"HOME": True, "PICK": True, "FLOURISH": True, "GRAB": False, "PLACE": False}

# FLOURISH: scripted wave then nod, as offsets (dx, dy, dz, ms) from MIRROR_ORIGIN_XYZ_MM.
FLOURISH_STEPS = [
    (60.0, 0.0, 0.0, 600), (-60.0, 0.0, 0.0, 600), (60.0, 0.0, 0.0, 600), (-60.0, 0.0, 0.0, 600),
    (0.0, 0.0, 0.0, 600), (0.0, 0.0, 30.0, 500), (0.0, 0.0, -20.0, 500), (0.0, 0.0, 30.0, 500),
    (0.0, 0.0, 0.0, 600),
]

# PICK routine: the block picker's own camera, model, homography and loop, run once.
PICK_CAMERA_INDEX: int | None = None   # None = block-picker config.WEBCAM_INDEX
PICK_MAX_CYCLES = 1

# --------------------------------------------------------------------------
# Overlay (BGR)
# --------------------------------------------------------------------------
GESTURE_COLOURS = {
    "fist": (68, 68, 239),        # red      FREEZE
    "pinch": (0, 140, 255),       # orange   GRIP
    "open-palm": (129, 185, 16),  # green    RELEASE
    "thumbs-up": (255, 144, 30),  # blue     HOME
    "point": (237, 58, 124),      # purple   PICK
    "peace": (153, 72, 236),      # pink     FLOURISH
    "null": (140, 140, 140),      # grey     no gesture
}
MODE_COLOURS = {
    "MIRROR": (129, 185, 16),     # green
    "FROZEN": (68, 68, 239),      # red (filled banner)
    "ROUTINE": (237, 58, 124),    # purple
}
TOAST_S = 1.0
WINDOW_NAME = "Gesture Arm"
LOG_DIR = os.path.join(ROOT, "logs")


class ConfigError(RuntimeError):
    pass


def api_key() -> str:
    """ROBOFLOW_API_KEY from the environment, else a gitignored .env next to the repo root."""
    key = os.environ.get(API_KEY_ENV)
    if key:
        return key
    env_path = os.path.join(ROOT, ".env")
    try:
        for line in open(env_path):
            if line.strip().startswith(API_KEY_ENV + "="):
                key = line.split("=", 1)[1].strip().strip('"')
    except OSError:
        pass
    if key:
        return key
    raise ConfigError(f"{API_KEY_ENV} is not set: export it, or put {API_KEY_ENV}=... in {env_path}")
