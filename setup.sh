#!/bin/sh
# One-time setup: Python 3.12 venv, pinned deps, MediaPipe hand model. Run from the repo root.
set -e
cd "$(dirname "$0")"
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
mkdir -p models
[ -f models/hand_landmarker.task ] || curl -sSL -o models/hand_landmarker.task \
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
echo "ok. Put ROBOFLOW_API_KEY in the environment or in a gitignored .env, then: .venv/bin/python gesture_arm.py --dry-run"
