"""Regression checks for causal inference and the deployable API."""
import copy
import csv
import io
import json
import sys
import unittest
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.experiments import RunStore
from backend.main import create_app
from backend.runtime import MODEL_NAMES, Runtime
from backend.simulator import simulate


class ApplicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = Runtime(ROOT / "deployment/models")
        cls.client = TestClient(create_app())
        cls.client.__enter__()
        cls.example = json.loads((ROOT / "deployment/evidence/example.json").read_text())

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)

    def test_example_matches_actual_runtime(self):
        actual = self.runtime.score_series(self.example["readings"], self.example["time_s"])
        for name in MODEL_NAMES:
            np.testing.assert_allclose(actual["models"][name]["scores"][31:],
                                       self.example["predictions"]["models"][name]["scores"][31:], rtol=2e-6)
            self.assertEqual(actual["models"][name]["event_times_s"],
                             self.example["predictions"]["models"][name]["event_times_s"])

    def test_future_readings_cannot_change_past_scores(self):
        x = np.asarray(self.example["readings"])
        times = self.example["time_s"]
        prefix = self.runtime.score_series(x[:300], times[:300])
        changed = x.copy()
        changed[300:, :3] *= 1.02
        full = self.runtime.score_series(changed, times)
        for name in MODEL_NAMES:
            np.testing.assert_allclose(prefix["models"][name]["scores"][31:],
                                       full["models"][name]["scores"][31:300], rtol=3e-6)
            self.assertEqual(prefix["models"][name]["active"], full["models"][name]["active"][:300])

    def test_missing_sensors_restart_warmup(self):
        sim = simulate("missing", 84002, "missing_sensor")
        result = self.runtime.score_series(sim["x"], sim["time_s"])
        self.assertEqual(result["quality"].count("data_quality_alert"), 40)
        for name in MODEL_NAMES:
            self.assertTrue(all(v is None for v in result["models"][name]["scores"][280:351]))
            self.assertIsNotNone(result["models"][name]["scores"][351])

    def test_timestamp_validation_and_gap_reset(self):
        x = self.example["readings"][:100]
        times = np.array(self.example["time_s"][:100], dtype=float)
        times[60:] += 5
        result = self.runtime.score_series(x, times)
        self.assertIn("Time gap", result["quality_reasons"][60])
        self.assertTrue(all(v is None for v in result["models"]["mass_inventory"]["scores"][60:91]))
        times[10] = times[9]
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            self.runtime.score_series(x, times)

    def test_runs_do_not_share_history(self):
        first = self.runtime.score_series(self.example["readings"][:180], self.example["time_s"][:180])
        healthy = simulate("independent", 84003)
        self.runtime.score_series(healthy["x"], healthy["time_s"])
        again = self.runtime.score_series(self.example["readings"][:180], self.example["time_s"][:180])
        self.assertEqual(first, again)

    def test_new_alarm_required_after_onset(self):
        threshold = 1.0
        series = np.ones(100) * 2
        _, events = self.runtime._alarms(series, np.arange(100)*5, threshold)
        self.assertEqual(events, [10.0])
        self.assertEqual([t for t in events if t >= 200], [])

    def test_ui_and_health_are_served_together(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Experiment lab", response.text)
        self.assertIn("script-src 'self'", response.headers["Content-Security-Policy"])
        self.assertEqual(self.client.get("/assets/app.js").status_code, 200)
        self.assertEqual(self.client.get("/health").json()["models_loaded"], 4)
        docs = self.client.get("/docs")
        self.assertIn("nonce=", docs.text)
        self.assertIn("'nonce-", docs.headers["Content-Security-Policy"])

    def test_api_uses_real_scores_and_exports_every_reading(self):
        response = self.client.post("/api/runs", json={"scenario":"leak", "seed":82001})
        self.assertEqual(response.status_code, 201, response.text)
        run = response.json()
        self.assertEqual(run["predictions"], self.example["predictions"])
        self.assertEqual(self.client.get(f'/api/runs/{run["id"]}').json()["id"], run["id"])
        exported = self.client.get(f'/api/runs/{run["id"]}/export.csv')
        rows = list(csv.DictReader(io.StringIO(exported.text)))
        self.assertEqual(len(rows), 720)
        self.assertEqual(rows[0]["mass_inventory_score"], "")
        self.assertIn("injected_leak_present", rows[0])
        self.assertEqual(self.client.get(f'/api/runs/{run["id"]}/export.json').json()["predictions"], run["predictions"])

    def test_upload_validation_and_missing_values(self):
        readings = [{"timestamp_s":t,"values":v} for t,v in zip(self.example["time_s"][:90],self.example["readings"][:90])]
        readings[45]["values"][1] = None
        response = self.client.post("/api/analyze", json={"readings":readings})
        self.assertEqual(response.status_code, 201, response.text)
        result = response.json()
        self.assertIsNone(result["ground_truth"])
        self.assertEqual(result["predictions"]["quality"][45], "data_quality_alert")
        bad = copy.deepcopy(readings)
        bad[2]["timestamp_s"] = bad[1]["timestamp_s"]
        self.assertEqual(self.client.post("/api/analyze",json={"readings":bad}).status_code,422)
        bad = copy.deepcopy(readings)
        bad[0]["values"][0] = 3.9  # MPa where absolute Pa is required.
        self.assertEqual(self.client.post("/api/analyze",json={"readings":bad}).status_code,422)
        self.assertEqual(self.client.post("/api/analyze",json={"readings":readings,"labels":[1]*90}).status_code,422)

    def test_unknown_routes_and_request_bounds(self):
        self.assertEqual(self.client.get("/api/download/anything").status_code,404)
        self.assertEqual(self.client.get("/api/runs/expired").status_code,404)
        self.assertEqual(self.client.post("/api/runs",json={"seed":-1}).status_code,422)
        self.assertEqual(self.client.post("/api/runs",content=b"x"*512001).status_code,413)
        self.assertEqual(self.client.get("/api/download/report").headers["content-type"],"application/pdf")

    def test_store_is_bounded(self):
        store = RunStore(capacity=2)
        first = store.add({"source":"simulation"})["id"]
        store.add({"source":"simulation"}); last = store.add({"source":"simulation"})["id"]
        self.assertIsNone(store.get(first)); self.assertIsNotNone(store.get(last))


if __name__ == "__main__":
    unittest.main()
