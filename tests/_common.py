"""Shared test helpers: a silent logger so tests never create session log files."""
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def silent_logger(name="gesture-test"):
    log = logging.getLogger(name)
    log.handlers.clear()
    log.addHandler(logging.NullHandler())
    log.propagate = False
    return log
