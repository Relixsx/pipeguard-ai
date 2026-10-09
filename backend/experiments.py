"""Bounded demo run storage; ground truth is used only after inference."""
import csv
import io
import threading
import time
import uuid
from collections import OrderedDict

import numpy as np

from .runtime import MODEL_NAMES
from .simulator import FEATURES, simulate

SCENARIOS = {
    "leak": {"label": "Abrupt leak", "description": "A leak opens at one injected volume."},
    "gradual_leak": {"label": "Growing leak", "description": "Leak outflow ramps up over ten minutes."},
    "healthy": {"label": "Healthy operation", "description": "Normal pressure and valve command changes."},
    "normal_transient": {"label": "Valve transition", "description": "A legitimate valve and pressure change, without a leak."},
    "sensor_bias": {"label": "Pressure sensor drift", "description": "The middle pressure sensor drifts by 10 kPa, without a leak."},
    "missing_sensor": {"label": "Missing pressure sensor", "description": "The middle sensor drops out for 200 seconds."},
    "weak_excitation": {"label": "Quiet operation", "description": "Almost constant commands; attribution remains unestablished."},
}


class RunStore:
    def __init__(self, capacity=12, ttl_s=3600):
        self.capacity, self.ttl_s = capacity, ttl_s
        self._runs = OrderedDict()
        self._lock = threading.Lock()

    def _expire(self):
        now = time.monotonic()
        for key in [k for k, (_, added) in self._runs.items() if now - added > self.ttl_s]:
            self._runs.pop(key)

    def add(self, run):
        with self._lock:
            self._expire()
            run_id = uuid.uuid4().hex
            run["id"] = run_id
            self._runs[run_id] = (run, time.monotonic())
            while len(self._runs) > self.capacity:
                self._runs.popitem(last=False)
            return run

    def get(self, run_id):
        with self._lock:
            self._expire()
            entry = self._runs.get(run_id)
            if entry:
                self._runs.move_to_end(run_id)
                return entry[0]
            return None


def serialize_readings(x):
    return [[float(v) if np.isfinite(v) else None for v in row] for row in x]


def create_simulation(runtime, request):
    fraction = request.leak_fraction if request.scenario in ("leak", "gradual_leak") else 0.0
    sim = simulate("interactive", request.seed, request.scenario, fraction,
                   request.location, request.regime)
    # This interface deliberately supplies sensor/time arrays only.
    predictions = runtime.score_series(sim["x"], sim["time_s"])
    ground_truth = {
        "onset_s": sim["onset_s"], "injected_location": sim["location"] if fraction else None,
        "leak_present": [bool(v) for v in sim["y"]],
        "leak_outflow_kg_s": sim["truth"][:, -1].tolist(),
    }
    summaries = {}
    for name, model in predictions["models"].items():
        after_onset = [t for t in model["event_times_s"]
                       if sim["onset_s"] is not None and t >= sim["onset_s"]]
        # An alarm already active before onset receives no detection credit.
        delay = after_onset[0] - sim["onset_s"] if after_onset else None
        summaries[name] = {"new_alarm_events": len(model["event_times_s"]),
                           "detected_injected_leak": bool(after_onset) if fraction else None,
                           "delay_s": float(delay) if delay is not None else None}
    return {
        "source": "simulation", "scenario": sim["kind"], "regime": sim["regime"],
        "seed": sim["seed"], "leak_fraction": fraction, "schema": FEATURES,
        "time_s": sim["time_s"].tolist(), "readings": serialize_readings(sim["x"]),
        "ground_truth": ground_truth, "predictions": predictions, "summaries": summaries,
        "max_mass_conservation_error_kg": sim["max_mass_balance_error_kg"],
        "scope": runtime.config["scope"],
    }


def create_analysis(runtime, request):
    values = np.array([[np.nan if v is None else v for v in row.values]
                       for row in request.readings], dtype=float)
    times = np.array([row.timestamp_s for row in request.readings], dtype=float)
    result = runtime.score_series(values, times)
    return {
        "source": "uploaded_readings", "scenario": "uploaded_readings", "regime": "unknown",
        "seed": None, "leak_fraction": None, "schema": FEATURES,
        "time_s": times.tolist(), "readings": serialize_readings(values),
        "ground_truth": None, "predictions": result,
        "summaries": {name: {"new_alarm_events": len(model["event_times_s"]),
                             "detected_injected_leak": None, "delay_s": None}
                      for name, model in result["models"].items()},
        "scope": "Uploaded readings scored against a simulation reference. An alert does not establish a leak or its location.",
    }


def run_csv(run):
    output = io.StringIO()
    writer = csv.writer(output)
    headers = ["timestamp_s", *FEATURES, "data_quality"]
    for name in MODEL_NAMES:
        headers.extend([f"{name}_score", f"{name}_active"])
    headers += ["injected_leak_present", "injected_leak_outflow_kg_s"]
    writer.writerow(headers)
    predictions, truth = run["predictions"], run["ground_truth"]
    for i, (time_s, values) in enumerate(zip(run["time_s"], run["readings"])):
        row = [time_s, *values, predictions["quality"][i]]
        for name in MODEL_NAMES:
            model = predictions["models"][name]
            row.extend([model["scores"][i], int(model["active"][i])])
        row += [int(truth["leak_present"][i]), truth["leak_outflow_kg_s"][i]] if truth else ["", ""]
        writer.writerow(row)
    return output.getvalue()
