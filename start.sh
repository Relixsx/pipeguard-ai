#!/bin/sh
set -eu
# One worker keeps bounded ephemeral runs and the compute semaphore consistent.
exec python -m uvicorn backend.main:app --host 0.0.0.0 --port "${PORT:-8000}" --workers 1
