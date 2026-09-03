"""The two perception streams, fixed roles.

- HandTracker: MediaPipe Hand Landmarker (VIDEO mode) on every frame -> 21 landmarks. Drives WHERE
  the arm goes (the wrist feeds the mirroring controller).
- GestureDetector: the Roboflow RF-DETR model, run locally through `inference`. Drives WHAT the arm
  does (one top-1 class per inference, debounced into events). It runs in DetectorWorker, a thread
  that takes every Nth camera frame when idle, so the camera / landmark / draw loop keeps its FPS.

Neither stream is ever used for the other's job: the RF-DETR box is drawn and debounced, never used
for positioning; landmarks never produce gesture events.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

from . import config, runlog
from .gestures import Prediction

WRIST = 0
FINGERTIPS = (4, 8, 12, 16, 20)
# (tip, PIP) landmark pairs for the four fingers; the thumb is not needed for point-vs-peace
FINGERS = {"index": (8, 6), "middle": (12, 10), "ring": (16, 14), "pinky": (20, 18)}
# Hand skeleton edges (MediaPipe HandLandmarksConnections.HAND_CONNECTIONS, 21 edges, verified
# against mediapipe 0.10.21 at start-up in HandTracker).
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),            # thumb
    (1, 5), (5, 6), (6, 7), (7, 8),            # index (palm edge thumb CMC -> index MCP, as in mediapipe 0.10.21)
    (5, 9), (9, 10), (10, 11), (11, 12),       # middle
    (9, 13), (13, 14), (14, 15), (15, 16),     # ring
    (13, 17), (17, 18), (18, 19), (19, 20),    # pinky
    (0, 17),                                   # palm
)


@dataclass
class Hand:
    pts: np.ndarray        # (21, 2) pixel coordinates in the displayed frame
    norm: np.ndarray       # (21, 2) normalised 0..1
    handedness: str
    score: float
    t: float

    @property
    def wrist_px(self) -> tuple[float, float]:
        return float(self.pts[WRIST, 0]), float(self.pts[WRIST, 1])

    @property
    def wrist_norm(self) -> tuple[float, float]:
        return float(self.norm[WRIST, 0]), float(self.norm[WRIST, 1])


def finger_states(norm: np.ndarray, ratio: float | None = None) -> dict[str, bool]:
    """Which of index / middle / ring / pinky are extended, from the 21 normalised landmarks.

    A finger is extended when its tip is farther from the wrist than its PIP joint (by `ratio`); a
    curled finger brings the tip back toward the palm. Rotation-invariant, good enough to tell one
    raised finger from two, which is all it is used for.
    """
    ratio = config.FINGER_EXTENDED_RATIO if ratio is None else ratio
    w = norm[WRIST]
    out = {}
    for name, (tip, pip) in FINGERS.items():
        d_tip = float(np.hypot(*(norm[tip] - w)))
        d_pip = float(np.hypot(*(norm[pip] - w)))
        out[name] = d_tip > d_pip * ratio
    return out


def two_fingers_up(states: dict[str, bool]) -> bool:
    """The peace sign: index + middle extended, ring + pinky folded."""
    return states["index"] and states["middle"] and not states["ring"] and not states["pinky"]


class HandTracker:
    """MediaPipe hand landmarker, one hand, VIDEO running mode (timestamps must increase)."""

    def __init__(self, model_path: str | None = None, num_hands: int | None = None):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp = mp
        self.model_path = model_path or config.HAND_MODEL_PATH
        opts = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=self.model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands or config.HAND_NUM,
            min_hand_detection_confidence=config.HAND_MIN_DETECTION_CONF,
            min_hand_presence_confidence=config.HAND_MIN_PRESENCE_CONF,
            min_tracking_confidence=config.HAND_MIN_TRACKING_CONF,
        )
        t0 = time.time()
        self.landmarker = vision.HandLandmarker.create_from_options(opts)
        self.load_seconds = time.time() - t0
        try:
            from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarksConnections

            conns = tuple((c.start, c.end) for c in HandLandmarksConnections.HAND_CONNECTIONS)
            if {frozenset(c) for c in conns} != {frozenset(c) for c in HAND_CONNECTIONS}:
                runlog.get_logger().warning("mediapipe HAND_CONNECTIONS differ from the built-in table; using mediapipe's")
                self.connections = conns
            else:
                self.connections = HAND_CONNECTIONS
        except Exception:  # noqa: BLE001 - older/newer mediapipe: keep the table
            self.connections = HAND_CONNECTIONS
        self._last_ts_ms = -1
        self.last_ms = 0.0

    def process(self, frame_bgr: np.ndarray, t: float) -> Hand | None:
        t0 = time.time()
        ts_ms = int(t * 1000)
        if ts_ms <= self._last_ts_ms:
            ts_ms = self._last_ts_ms + 1
        self._last_ts_ms = ts_ms
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        res = self.landmarker.detect_for_video(mp_img, ts_ms)
        self.last_ms = (time.time() - t0) * 1000.0
        if not res.hand_landmarks:
            return None
        lms = res.hand_landmarks[0]
        h, w = frame_bgr.shape[:2]
        norm = np.array([[lm.x, lm.y] for lm in lms], dtype=np.float32)
        pts = norm * np.array([w, h], dtype=np.float32)
        handedness, score = "?", 0.0
        if res.handedness and res.handedness[0]:
            handedness = res.handedness[0][0].category_name
            score = float(res.handedness[0][0].score)
        return Hand(pts=pts, norm=norm, handedness=handedness, score=score, t=t)

    def close(self) -> None:
        try:
            self.landmarker.close()
        except Exception:  # noqa: BLE001
            pass


class GestureDetector:
    """The trained RF-DETR gesture model, loaded once, run locally."""

    def __init__(self, model_id: str | None = None, api_key: str | None = None, min_conf: float | None = None):
        from inference import get_model  # heavy import; here so tests can fake the detector

        self.model_id = model_id or config.MODEL_ID
        self.min_conf = config.DETECT_MIN_CONF if min_conf is None else float(min_conf)
        log = runlog.get_logger()
        t0 = time.time()
        self.model = get_model(model_id=self.model_id, api_key=api_key or config.api_key())
        self.load_seconds = time.time() - t0
        self.class_names: list[str] = list(getattr(self.model, "class_names", None) or [])
        log.info("gesture model %s loaded in %.1fs classes=%s", self.model_id, self.load_seconds, self.class_names)
        if self.class_names:
            missing = [c for c in config.CLASSES if c not in self.class_names]
            if missing:
                raise config.ConfigError(f"model {self.model_id} reports classes {self.class_names}; config expects "
                                         f"{list(config.CLASSES)} - missing {missing}. Fix gesture/config.py to the "
                                         f"dashboard's exact strings.")
            extra = [c for c in self.class_names if c not in config.CLASSES and c != config.NULL_CLASS]
            if extra:
                log.warning("model has classes with no event mapping: %s (they will be rejected)", extra)
        else:
            log.warning("model did not report class names; class strings cannot be verified at start-up")
        self.last_ms = 0.0

    def detect(self, frame_bgr: np.ndarray) -> tuple[Prediction | None, list[Prediction]]:
        """All predictions above the model floor, and the top-1 (the only one the state machine uses)."""
        t0 = time.time()
        result = self.model.infer(frame_bgr, confidence=self.min_conf)[0]
        self.last_ms = (time.time() - t0) * 1000.0
        preds = []
        for p in result.predictions:
            x1, y1 = int(p.x - p.width / 2), int(p.y - p.height / 2)
            x2, y2 = int(p.x + p.width / 2), int(p.y + p.height / 2)
            preds.append(Prediction(p.class_name, float(p.confidence), x1, y1, x2, y2))
        top = max(preds, key=lambda d: d.conf) if preds else None
        return top, preds


@dataclass
class DetectionResult:
    frame_id: int
    top: Prediction | None
    preds: list[Prediction]
    ms: float
    t: float          # capture time of the frame it was computed on


class DetectorWorker:
    """Runs GestureDetector.detect in a thread on every Nth submitted frame (when idle)."""

    def __init__(self, detector, every_n: int | None = None):
        self.det = detector
        self.every_n = config.DETECT_EVERY_N if every_n is None else max(1, int(every_n))
        self._cv = threading.Condition()
        self._pending: tuple[int, np.ndarray, float] | None = None
        self._busy = False
        self._stop = False
        self._result: DetectionResult | None = None
        self._result_seq = 0
        self._polled_seq = 0
        self.submitted = 0
        self.skipped_busy = 0
        self.errors = 0
        self.log = runlog.get_logger()
        self._thread = threading.Thread(target=self._run, name="gesture-detector", daemon=True)

    def start(self) -> "DetectorWorker":
        self._thread.start()
        return self

    def submit(self, frame: np.ndarray, frame_id: int, t: float) -> bool:
        """Offer a frame; taken only if it is an Nth frame and the worker is idle."""
        if frame_id % self.every_n != 0:
            return False
        with self._cv:
            if self._busy or self._pending is not None:
                self.skipped_busy += 1
                return False
            self._pending = (frame_id, frame, t)
            self.submitted += 1
            self._cv.notify()
            return True

    def poll(self) -> DetectionResult | None:
        """The newest result if one arrived since the last poll, else None."""
        with self._cv:
            if self._result_seq == self._polled_seq:
                return None
            self._polled_seq = self._result_seq
            return self._result

    @property
    def latest(self) -> DetectionResult | None:
        with self._cv:
            return self._result

    def _run(self) -> None:
        while True:
            with self._cv:
                while self._pending is None and not self._stop:
                    self._cv.wait(0.2)
                if self._stop:
                    return
                frame_id, frame, t = self._pending
                self._pending = None
                self._busy = True
            try:
                top, preds = self.det.detect(frame)
                res = DetectionResult(frame_id, top, preds, self.det.last_ms, t)
            except Exception as e:  # noqa: BLE001 - a failed inference must not kill the loop
                self.errors += 1
                self.log.error("gesture inference failed on frame %d: %r", frame_id, e)
                res = DetectionResult(frame_id, None, [], 0.0, t)
            with self._cv:
                self._result = res
                self._result_seq += 1
                self._busy = False

    def stop(self) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify_all()
        self._thread.join(timeout=2.0)
