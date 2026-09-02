"""Bridge to the block-picker project (~/Documents/Defect-detect bot), imported AS-IS.

The block picker owns the arm driver (hardware-validated MaxArm serial protocol + safety envelope),
the rig's physical values (config.py), the overhead camera detector, the pixel->arm homography and
the pick loop. This module puts its directory on sys.path and imports those top-level modules under
one namespace. Nothing in the gesture package is allowed to shadow them: the gesture code lives in
the `gesture` package precisely so that `config`, `arm`, `detect`, `mapping`, `pick`, `runlog` and
`hud` resolve to the block picker's files.
"""

from __future__ import annotations

import importlib
import os
import sys
import types

from . import config

REQUIRED_FILES = ("arm.py", "config.py", "detect.py", "mapping.py", "pick.py", "runlog.py", "hud.py")
MODULES = ("config", "runlog", "arm", "mapping", "detect", "hud", "pick")

_ns: types.SimpleNamespace | None = None


class BlockPickerMissing(RuntimeError):
    pass


def directory() -> str:
    return os.path.abspath(config.BLOCK_PICKER_DIR)


def calibration_path() -> str:
    return os.path.join(directory(), "calibration.npy")


def check() -> list[str]:
    """Human-readable problems with the block-picker install (empty = all good)."""
    d = directory()
    problems = []
    if not os.path.isdir(d):
        return [f"block-picker directory not found: {d} (gesture.config.BLOCK_PICKER_DIR)"]
    for f in REQUIRED_FILES:
        if not os.path.exists(os.path.join(d, f)):
            problems.append(f"missing {os.path.join(d, f)}")
    if not os.path.exists(calibration_path()):
        problems.append(f"missing {calibration_path()} - ask the owner to run the block picker's calibrate.py; "
                        f"never invent a matrix")
    return problems


def load() -> types.SimpleNamespace:
    """Import the block picker's modules (once) and return them as a namespace."""
    global _ns
    if _ns is not None:
        return _ns
    d = directory()
    problems = [p for p in check() if "calibration.npy" not in p]  # calibration is checked when PICK needs it
    if problems:
        raise BlockPickerMissing("; ".join(problems))
    if d not in sys.path:
        sys.path.insert(0, d)
    for name in MODULES:
        mod = sys.modules.get(name)
        if mod is not None and not os.path.abspath(getattr(mod, "__file__", "") or "").startswith(d):
            raise BlockPickerMissing(f"module {name!r} is already imported from {mod.__file__}, not from {d}; "
                                     f"something shadows the block picker")
    ns = types.SimpleNamespace()
    for name in MODULES:
        mod = importlib.import_module(name)
        if not os.path.abspath(mod.__file__).startswith(d):
            raise BlockPickerMissing(f"{name} resolved to {mod.__file__}, expected a file under {d}")
        setattr(ns, name, mod)
    _ns = ns
    sync_logging()
    return ns


def sync_logging() -> None:
    """Point the block picker's runlog at THIS session (log dir, run stamp, frames dir).

    Its PickLoop calls its own runlog.save_frame(); if that module has no run yet it would call its
    start_run(), which clears the shared "blockpicker" logger's handlers and opens a second log file.
    """
    if _ns is None:
        return
    from . import runlog as grl
    _ns.config.LOG_DIR = config.LOG_DIR
    _ns.runlog._run_stamp = grl._run_stamp
    _ns.runlog._frames_dir = grl._frames_dir


def rig_values() -> dict:
    """The owner-measured rig values, straight from the block picker's config.py."""
    c = load().config
    return {
        "SERIAL_PORT": c.SERIAL_PORT,
        "BAUDRATE": c.BAUDRATE,
        "TABLE_Z_MM": c.TABLE_Z_MM,
        "HOME_XYZ_MM": c.HOME_XYZ_MM,
        "TRAVEL_Z_MM": c.TRAVEL_Z_MM,
        "REACH_X_MM": c.REACH_X_MM,
        "REACH_Y_MM": c.REACH_Y_MM,
        "REACH_Z_MM": c.REACH_Z_MM,
        "MIN_RADIUS_MM": c.MIN_RADIUS_MM,
        "WEBCAM_INDEX": c.WEBCAM_INDEX,
    }
