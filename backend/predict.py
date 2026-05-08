"""
predict.py
----------
PipelinePredictor — the single inference interface for the entire system.

Responsibilities:
  - Load trained model weights, MinMaxScaler, and threshold from disk
  - Scale raw sensor readings before inference
  - Run the LSTM Autoencoder reconstruction error computation
  - Classify the score as Normal / Warning / Leak Detected
  - Maintain a rolling buffer for single-reading streaming mode

main.py imports the module-level `predictor` singleton — it never
instantiates the model, scaler or threshold directly.
"""

import json
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from config import (
    MODEL_PATH,
    SCALER_PATH,
    THRESHOLD_PATH,
    FEATURES,
    SEQ_LEN,
    SENSOR_BUFFER_MAXLEN,
    RATIO_WARNING,
    RATIO_CRITICAL,
    N_FEATURES,
    HIDDEN_DIM,
    LATENT_DIM,
    NUM_LAYERS,
)
from model          import build_model
from preprocessing  import MinMaxScaler
from utils          import get_logger, classify_ratio, build_event

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  PipelinePredictor
# ─────────────────────────────────────────────────────────────────────────────

class PipelinePredictor:
    """
    Wraps the trained LSTM Autoencoder for real-time pipeline monitoring.

    Lifecycle:
        predictor = PipelinePredictor()
        predictor.load()          # called once at FastAPI startup
        result = predictor.infer_reading({"pressure": 80.1, ...})

    Modes:
        infer_reading(reading)     → adds one reading to rolling buffer;
                                     returns prediction once buffer is full
        infer_batch(readings_list) → classify a pre-built list of ≥ SEQ_LEN dicts
        infer_array(seq)           → low-level: classify a scaled (SEQ_LEN, F) ndarray
    """

    def __init__(self):
        self.model:     Optional[LSTMAutoencoder] = None
        self.scaler:    Optional[MinMaxScaler]    = None
        self.threshold: Optional[float]           = None
        self._buffer:   deque                     = deque(maxlen=SENSOR_BUFFER_MAXLEN)
        self.is_ready:  bool                      = False

    # ── Artifact Loading ──────────────────────────────────────────

    def load(self) -> bool:
        """
        Load all model artifacts from the models/ directory.

        Returns:
            True  if loading succeeded and predictor is ready
            False if model file is missing (train.py has not been run)
        """
        if not MODEL_PATH.exists():
            logger.warning("Model weights not found at: %s", MODEL_PATH)
            logger.warning("Run:  python train.py")
            return False

        # Threshold + feature config
        with open(THRESHOLD_PATH) as f:
            cfg = json.load(f)
        self.threshold = float(cfg["threshold"])
        self.features  = cfg.get("features", ["pressure", "flow_rate", "temperature"])
        n_features     = cfg.get("n_features", len(self.features))
        logger.info("Threshold loaded: %.6f | features: %s", self.threshold, self.features)

        # Scaler
        self.scaler = MinMaxScaler.load(SCALER_PATH)
        logger.info("Scaler loaded: %s", self.scaler)

        # Model — built with correct input_dim from saved config
        # Auto-detect architecture from saved config
        arch = cfg.get("architecture", "cnn_lstm")
        self.model = build_model(architecture=arch, input_dim=n_features)
        state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        logger.info("Model loaded from: %s", MODEL_PATH)

        self.is_ready = True
        logger.info("PipelinePredictor ready ✅")
        return True

    # ── Score Classification ──────────────────────────────────────

    def classify_score(self, score: float) -> dict:
        """
        Map a raw reconstruction error to a full classification result dict.

        Args:
            score : MSE reconstruction error (float)

        Returns:
            dict with keys: status, color, icon, is_anomaly, score_ratio
        """
        ratio              = score / self.threshold if self.threshold else 0.0
        status, color, icon = classify_ratio(ratio)
        return {
            "status":      status,
            "color":       color,
            "icon":        icon,
            "is_anomaly":  ratio >= RATIO_CRITICAL,
            "score_ratio": round(ratio, 4),
        }

    # ── Inference: scaled numpy array ────────────────────────────

    def infer_array(self, seq: np.ndarray) -> dict:
        """
        Run inference on a pre-scaled (SEQ_LEN, N_FEATURES) numpy array.

        This is the lowest-level inference method; all higher-level methods
        ultimately call this one.

        Args:
            seq : float32 array of shape (SEQ_LEN, N_FEATURES), already scaled

        Returns:
            dict with keys: status, color, icon, is_anomaly, score_ratio, anomaly_score
        """
        tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)  # → (1, SEQ_LEN, F)
        score  = self.model.reconstruction_error(tensor).item()
        result = self.classify_score(score)
        result["anomaly_score"] = round(score, 6)
        return result

    # ── Inference: single raw reading (rolling buffer mode) ───────

    def infer_reading(self, reading: dict) -> dict:
        """
        Add one raw sensor reading to the rolling buffer and run inference
        once the buffer holds at least SEQ_LEN readings.

        This is the method used by the SSE stream and /predict/single endpoint.

        Args:
            reading : dict with keys: pressure, flow_rate, temperature

        Returns:
            dict — either a "Collecting" placeholder or a full inference result.
        """
        scaled = self.scaler.transform_dict(reading)
        self._buffer.append(scaled.tolist())

        buf_len = len(self._buffer)

        if buf_len < SEQ_LEN:
            # Not enough history yet — return a placeholder
            return {
                "status":        "Collecting",
                "color":         "gray",
                "icon":          "⏳",
                "is_anomaly":    False,
                "anomaly_score": 0.0,
                "score_ratio":   0.0,
                "buffer_fill":   buf_len,
                "buffer_needed": SEQ_LEN,
            }

        seq    = np.array(list(self._buffer), dtype=np.float32)
        result = self.infer_array(seq)
        result["buffer_fill"]   = SEQ_LEN
        result["buffer_needed"] = SEQ_LEN
        return result

    # ── Inference: batch of raw readings ─────────────────────────

    def infer_batch(self, readings: list[dict]) -> dict:
        """
        Classify a pre-built sequence of ≥ SEQ_LEN raw sensor readings.

        Used by the /predict endpoint (the caller provides the full window).
        Scales each reading internally using the fitted scaler.

        Args:
            readings : list of dicts with keys: pressure, flow_rate, temperature
                       (must contain at least SEQ_LEN items)

        Returns:
            Full inference result dict.
        """
        window = readings[-SEQ_LEN:]
        arr    = np.array(
            [[r["pressure"], r["flow_rate"], r["temperature"]] for r in window],
            dtype=np.float32,
        )
        scaled = self.scaler.transform(arr)
        return self.infer_array(scaled)

    # ── Buffer Management ─────────────────────────────────────────

    def reset_buffer(self) -> None:
        """Clear the rolling inference buffer (e.g. after a stream reconnect)."""
        self._buffer.clear()
        logger.debug("Inference buffer cleared")

    @property
    def buffer_size(self) -> int:
        """Current number of readings in the rolling buffer."""
        return len(self._buffer)

    # ── Status / Health ───────────────────────────────────────────

    def health(self) -> dict:
        """Return a health-check dict for the /health endpoint."""
        return {
            "model_loaded":  self.is_ready,
            "threshold":     self.threshold,
            "buffer_size":   self.buffer_size,
            "seq_len":       SEQ_LEN,
            "features":      FEATURES,
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Module-level Singleton
#  main.py imports this object and calls predictor.load() at startup.
# ─────────────────────────────────────────────────────────────────────────────

predictor = PipelinePredictor()


# ─────────────────────────────────────────────────────────────────────────────
#  CLI Smoke-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.DEBUG)

    ok = predictor.load()
    if not ok:
        print("Train the model first: python train.py")
        exit(1)

    print("\n── Single-reading mode (buffer filling) ──")
    for i in range(SEQ_LEN + 2):
        from data_simulation import get_random_sample
        r = get_random_sample(anomaly=(i > SEQ_LEN))
        res = predictor.infer_reading(r)
        print(f"  [{i+1:>3}] {res['status']:<15}  score={res['anomaly_score']:.6f}")

    print("\n── Batch mode ──")
    from data_simulation import get_random_sample
    batch = [get_random_sample(anomaly=False) for _ in range(SEQ_LEN)]
    res   = predictor.infer_batch(batch)
    print("  Normal batch  :", res)

    batch_leak = [get_random_sample(anomaly=True) for _ in range(SEQ_LEN)]
    res_leak   = predictor.infer_batch(batch_leak)
    print("  Anomaly batch :", res_leak)
