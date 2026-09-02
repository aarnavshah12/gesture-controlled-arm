"""One timestamped log per session: logs/YYYYMMDD-HHMMSS.log (+ a frames folder next to it).

Everything the plan says must be logged goes through the "gesture" logger. The block picker's own
"blockpicker" logger (its arm driver and pick loop) is attached to the same file so a PICK routine's
serial frames land in the session log too.
"""

from __future__ import annotations

import logging
import os
import sys
import time

from . import config

_run_stamp: str | None = None
_frames_dir: str | None = None
LOGGER = "gesture"
SHARED_LOGGERS = ("blockpicker",)


def start_run(name: str = "run", level: int = logging.INFO, quiet: bool = False) -> logging.Logger:
    global _run_stamp, _frames_dir
    os.makedirs(config.LOG_DIR, exist_ok=True)
    _run_stamp = time.strftime("%Y%m%d-%H%M%S")
    _frames_dir = os.path.join(config.LOG_DIR, f"{_run_stamp}-frames")
    log_path = os.path.join(config.LOG_DIR, f"{_run_stamp}.log")
    fmt = logging.Formatter("%(asctime)s.%(msecs)03d %(levelname)s %(name)s %(message)s", "%H:%M:%S")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    handlers = [fh]
    if not quiet:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        handlers.append(sh)
    for lname in (LOGGER, *SHARED_LOGGERS):
        lg = logging.getLogger(lname)
        lg.setLevel(level)
        lg.handlers.clear()
        for h in handlers:
            lg.addHandler(h)
        lg.propagate = False
    log = logging.getLogger(LOGGER)
    log.info("run=%s name=%s log=%s", _run_stamp, name, log_path)
    return log


def get_logger() -> logging.Logger:
    log = logging.getLogger(LOGGER)
    if not log.handlers:
        return start_run()
    return log


def log_path() -> str | None:
    return None if _run_stamp is None else os.path.join(config.LOG_DIR, f"{_run_stamp}.log")


def save_frame(frame, tag: str) -> str:
    import cv2

    if _frames_dir is None:
        start_run()
    os.makedirs(_frames_dir, exist_ok=True)
    path = os.path.join(_frames_dir, f"{time.strftime('%H%M%S')}-{tag}.jpg")
    cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return path
