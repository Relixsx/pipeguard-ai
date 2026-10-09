"""Rebuild the landing-page example using the actual inference runtime."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.experiments import create_simulation
from backend.main import ScenarioRequest
from backend.runtime import Runtime

runtime = Runtime(ROOT / "deployment/models")
start = time.perf_counter()
run = create_simulation(runtime, ScenarioRequest())
run["id"] = "example"
run["compute_seconds"] = round(time.perf_counter() - start, 3)
(ROOT / "deployment/evidence/example.json").write_text(json.dumps(run, allow_nan=False, separators=(",", ":")) + "\n")
print(json.dumps(run["summaries"], indent=2))
