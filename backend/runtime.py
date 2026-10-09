"""Causal inference from sensor readings only, using frozen research artifacts.

No labels, fault locations, onset times, or simulator truth enter this module.
Every call owns its windows and alarm state; model sessions are immutable.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .physics import inventory_window_score
from .simulator import FEATURES

MODEL_NAMES = ("mass_inventory", "dynamic_residual", "cnn_lstm", "hybrid")
MODEL_LABELS = {
    "mass_inventory": "Mass balance", "dynamic_residual": "Dynamic predictor",
    "cnn_lstm": "CNN–LSTM", "hybrid": "Residual hybrid",
}


class Runtime:
    def __init__(self, model_dir: Path):
        self.config = json.loads((model_dir / "config.json").read_text())
        self.manifest = json.loads((model_dir / "manifest.json").read_text())
        for name, entry in self.manifest["files"].items():
            if hashlib.sha256((model_dir / name).read_bytes()).hexdigest() != entry["sha256"]:
                raise RuntimeError(f"Artifact checksum failed: {name}")
        if self.config["schema"] != FEATURES:
            raise RuntimeError("Feature schema does not match the simulator")
        with np.load(model_dir / "dynamic_reference.npz", allow_pickle=False) as data:
            self.mean = data["scaler_mean"]
            self.scale = data["scaler_scale"]
            self.coef = data["coef"]
            self.residual_mean = data["residual_mean"]
            self.whitener = data["whitener"]
            self.lag = int(data["lag"])
        if self.lag != self.config["lag"]:
            raise RuntimeError("Dynamic lag does not match the frozen configuration")
        self.window_size = self.config["sequence_length"] + self.lag
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self.sessions = {
            name: ort.InferenceSession(str(model_dir / f"{name}_11.onnx"),
                                       options, providers=["CPUExecutionProvider"])
            for name in ("cnn_lstm", "hybrid")
        }
        # Readiness checks execute both models, rather than only checking files.
        for name, session in self.sessions.items():
            dimensions = 8 if name == "cnn_lstm" else 6
            output = session.run(None, {"window": np.zeros((1, 30, dimensions), np.float32)})[0]
            if output.shape != (1, 30, dimensions) or not np.isfinite(output).all():
                raise RuntimeError(f"Invalid model output: {name}")

    def _neural_scores(self, name, windows):
        chunks = []
        for start in range(0, len(windows), 32):
            x = np.ascontiguousarray(windows[start:start + 32], dtype=np.float32)
            reconstructed = self.sessions[name].run(None, {"window": x})[0]
            chunks.append(np.mean((reconstructed - x) ** 2, axis=(1, 2)))
        return np.concatenate(chunks) if chunks else np.empty(0)

    def score_series(self, values, timestamps):
        """Use trailing windows; gaps and missing values restart warm-up."""
        x = np.asarray(values, dtype=np.float64)
        times = np.asarray(timestamps, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != len(FEATURES) or len(x) != len(times):
            raise ValueError("Readings must have eight features and matching timestamps")
        if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError("Timestamps must be finite and strictly increasing")
        endpoints, statuses, reasons = [], [], []
        contiguous = 0
        for i, row in enumerate(x):
            gap = i > 0 and abs(times[i] - times[i - 1] - self.config["sample_s"]) > .5
            if gap:
                contiguous = 0
            if not np.isfinite(row).all():
                contiguous = 0
                statuses.append("data_quality_alert")
                reasons.append("Missing or nonfinite sensor; scoring suspended")
            else:
                contiguous += 1
                ready = contiguous >= self.window_size
                statuses.append("ready" if ready else "warming_up")
                reasons.append("Time gap; window restarted" if gap else None)
                if ready:
                    endpoints.append(i)
        scores = {name: np.full(len(x), np.nan) for name in MODEL_NAMES}
        if endpoints:
            blocks = np.stack([x[i - self.window_size + 1:i + 1] for i in endpoints])
            z = (blocks - self.mean) / self.scale
            design = np.concatenate(
                [z[:, self.lag - k:self.window_size - k] for k in range(1, self.lag + 1)]
                + [np.ones((len(blocks), 30, 1))], axis=2,
            )
            residual = (z[:, self.lag:, :6] - design @ self.coef - self.residual_mean) @ self.whitener
            # Preserve the float32 scoring convention of the offline benchmark.
            residual32 = residual.astype(np.float32)
            dynamic_score = np.mean(residual32 ** 2, axis=(1, 2))
            raw_score = self._neural_scores("cnn_lstm", z[:, -30:])
            neural_residual_score = self._neural_scores("hybrid", residual32)
            scores["mass_inventory"][endpoints] = [
                inventory_window_score(b[-30:], self.config["sample_s"]) for b in blocks
            ]
            scores["dynamic_residual"][endpoints] = dynamic_score
            scores["cnn_lstm"][endpoints] = raw_score
            scores["hybrid"][endpoints] = np.maximum(
                neural_residual_score / self.config["hybrid_neural_scale"],
                dynamic_score / self.config["hybrid_dynamic_scale"],
            )
        models = {}
        for name, series in scores.items():
            active, events = self._alarms(series, times, self.config["thresholds"][name])
            models[name] = {
                "label": MODEL_LABELS[name], "threshold": self.config["thresholds"][name],
                "scores": [float(s) if np.isfinite(s) else None for s in series],
                "active": active, "event_times_s": events,
            }
        return {"quality": statuses, "quality_reasons": reasons, "models": models,
                "attribution": "not_established", "warmup_readings": self.window_size,
                "sample_s": self.config["sample_s"]}

    def _alarms(self, scores, times, threshold):
        high = low = 0
        active = False
        last_event = -np.inf
        states, events = [], []
        for score, time_s in zip(scores, times):
            if not np.isfinite(score):
                high = low = 0
                active = False
            elif score > threshold:
                high += 1
                low = 0
                if high >= self.config["persistence"] and not active:
                    active = True
                    if time_s - last_event >= self.config["cooldown_s"]:
                        events.append(float(time_s))
                        last_event = time_s
            else:
                high = 0
                low += 1
                if low >= self.config["clearance"]:
                    active = False
            states.append(active)
        return states, events
