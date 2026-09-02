"""Mac camera capture with the plan's brightness correction and selfie flip.

Perception and drawing both consume the corrected frame, so the correction happens here, once,
immediately after capture.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from . import config, runlog


def correct(frame: np.ndarray, alpha: float | None = None, beta: float | None = None) -> np.ndarray:
    """cv2.convertScaleAbs(frame, alpha, beta) with the dataset-collection values from config."""
    alpha = config.BRIGHTNESS_ALPHA if alpha is None else alpha
    beta = config.BRIGHTNESS_BETA if beta is None else beta
    if alpha == 1.0 and beta == 0.0:
        return frame
    return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)


def _try_open(index: int, width: int, height: int) -> cv2.VideoCapture | None:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    ok, frame = cap.read()
    if not ok or frame is None or frame.size == 0:
        cap.release()
        return None
    return cap


class Camera:
    def __init__(self, index: int | None = None, width: int | None = None, height: int | None = None,
                 mirror: bool | None = None):
        self.index = config.CAMERA_INDEX if index is None else index
        self.width = width or config.FRAME_WIDTH
        self.height = height or config.FRAME_HEIGHT
        self.mirror = config.MIRROR_VIEW if mirror is None else mirror
        self.cap: cv2.VideoCapture | None = None
        self.log = runlog.get_logger()
        self.frames = 0
        self.t_open = None

    def open(self) -> "Camera":
        if not config.BRIGHTNESS_CONFIRMED:
            self.log.warning("camera: BRIGHTNESS_ALPHA/BETA not confirmed by the owner (using %.2f / %.1f); "
                             "live frames may not match the training frames", config.BRIGHTNESS_ALPHA,
                             config.BRIGHTNESS_BETA)
        if self.index is None:
            # The built-in camera is the LAST index that opens: alone it is 0; with the overhead
            # webcam plugged in the block picker measured external = 0, built-in = 1.
            found = []
            for i in range(config.CAMERA_PROBE_MAX + 1):
                cap = _try_open(i, self.width, self.height)
                if cap is None:
                    if found:
                        break        # indices are contiguous on macOS: first gap = end of the list
                    continue
                found.append(i)
                cap.release()
            if not found:
                raise RuntimeError("no camera opened on indices 0..%d" % config.CAMERA_PROBE_MAX)
            self.index = found[-1]
            self.log.info("camera: probed indices %s -> using %d (built-in = last)", found, self.index)
        cap = _try_open(self.index, self.width, self.height)
        if cap is None:
            raise RuntimeError(f"cannot open camera index {self.index}")
        self.cap = cap
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.log.info("camera %d opened %dx%d (asked %dx%d) mirror=%s brightness alpha=%.2f beta=%.1f",
                      self.index, w, h, self.width, self.height, self.mirror,
                      config.BRIGHTNESS_ALPHA, config.BRIGHTNESS_BETA)
        self.t_open = time.time()
        return self

    def read(self) -> np.ndarray | None:
        """Next frame, flipped (if MIRROR_VIEW) and brightness-corrected. None on a read failure."""
        if self.cap is None:
            raise RuntimeError("camera not open")
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return None
        if self.mirror:
            frame = cv2.flip(frame, 1)
        self.frames += 1
        return correct(frame)

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self) -> "Camera":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.release()
