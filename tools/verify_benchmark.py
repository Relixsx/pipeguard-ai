"""Replay all 66 final scenarios; compare every deployed alarm with the archive.

This verifies runtime integration, not a new independent performance estimate.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.main import ScenarioRequest
from backend.experiments import create_simulation
from backend.runtime import Runtime

runtime = Runtime(ROOT / "deployment/models")
archive = json.loads((ROOT / "deployment/evidence/regime_results.json").read_text())
rows = [row for row in archive["run_results"] if row["calibration_coverage"] == "broad"]
runs = {}
start = time.perf_counter()
for row in rows:
    run_id = row["run_id"]
    if run_id not in runs:
        number = int(run_id.split("_")[1])
        runs[run_id] = create_simulation(runtime, ScenarioRequest(
            scenario=row["kind"], regime=row["regime"], seed=81000 + number,
            leak_fraction=row["leak_fraction"] or .05, location=row["location"],
        ))
    actual = runs[run_id]
    model = actual["predictions"]["models"][row["model"]]
    assert model["event_times_s"] == row["alarm_times_s"], (run_id, row["model"], model["event_times_s"], row["alarm_times_s"])
    if row["detected"] is not None:
        assert actual["summaries"][row["model"]]["detected_injected_leak"] == row["detected"]
        assert actual["summaries"][row["model"]]["delay_s"] == row["delay_s"]
result = {"runs":len(runs),"model_run_comparisons":len(rows),"alarm_timestamps_match":True,
          "seconds":round(time.perf_counter()-start,3),"scope":"Replay parity, not new validation data"}
(ROOT / "deployment/evidence/runtime-parity.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2))
