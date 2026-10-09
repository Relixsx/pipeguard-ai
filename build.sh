#!/bin/sh
set -eu
python -m pip install -r requirements.txt
# Deployment must not train or change frozen thresholds.
python -c "from pathlib import Path; from backend.runtime import Runtime; Runtime(Path('deployment/models')); print('Frozen artifacts verified')"
