"""
utils.py
--------
Shared utilities: structured logging (console + rotating file),
threshold computation, event builders, response formatters.
"""

import logging
import time
from logging.handlers import RotatingFileHandler
from typing import Optional

import numpy as np

from config import (
    OPERATING_RANGES, SIGMA_MULTIPLIER, RATIO_WARNING, RATIO_CRITICAL,
    LOG_FILE, LOG_MAX_BYTES, LOG_BACKUP_COUNT, LOG_LEVEL,
)

# ─────────────────────────────────────────────────────────────────────────────
#  Logging — writes to both console AND rotating log file
# ─────────────────────────────────────────────────────────────────────────────

_LOG_FORMAT  = "%(asctime)s  [%(levelname)-8s]  %(name)s — %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_configured_loggers: set = set()


def get_logger(name: str) -> logging.Logger:
    """
    Return a logger that writes to both the console and a rotating file.
    Safe to call multiple times — handlers are added only once per name.
    """
    logger = logging.getLogger(name)
    if name in _configured_loggers:
        return logger

    logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))

    formatter = logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Rotating file handler
    fh = RotatingFileHandler(
        LOG_FILE,
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.propagate = False
    _configured_loggers.add(name)
    return logger


# ─────────────────────────────────────────────────────────────────────────────
#  Threshold Computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_threshold(errors: np.ndarray, sigma: float = SIGMA_MULTIPLIER) -> float:
    return float(errors.mean() + sigma * errors.std())


def threshold_report(errors: np.ndarray, sigma: float = SIGMA_MULTIPLIER) -> dict:
    return {
        "threshold": compute_threshold(errors, sigma),
        "mean":      float(errors.mean()),
        "std":       float(errors.std()),
        "min":       float(errors.min()),
        "max":       float(errors.max()),
        "sigma":     sigma,
        "n_samples": int(len(errors)),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Timestamp Helpers
# ─────────────────────────────────────────────────────────────────────────────

def utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def utc_now_epoch() -> float:
    return time.time()


# ─────────────────────────────────────────────────────────────────────────────
#  Classification Helper
# ─────────────────────────────────────────────────────────────────────────────

def classify_ratio(score_ratio: float) -> tuple:
    if score_ratio < RATIO_WARNING:
        return "Normal",        "green",  "🟢"
    elif score_ratio < RATIO_CRITICAL:
        return "Warning",       "yellow", "🟡"
    else:
        return "Leak Detected", "red",    "🔴"


# ─────────────────────────────────────────────────────────────────────────────
#  Event / Response Builders
# ─────────────────────────────────────────────────────────────────────────────

def build_event(reading: dict, result: dict) -> dict:
    return {
        "timestamp":     utc_now_iso(),
        "status":        result.get("status",        "Unknown"),
        "anomaly_score": round(result.get("anomaly_score", 0.0), 6),
        "score_ratio":   round(result.get("score_ratio",   0.0), 4),
        "pressure":      round(reading.get("pressure",    0.0), 3),
        "flow_rate":     round(reading.get("flow_rate",   0.0), 3),
        "temperature":   round(reading.get("temperature", 0.0), 3),
    }


def format_prediction_response(result: dict, reading: Optional[dict] = None) -> dict:
    response = dict(result)
    if reading:
        for key, val in reading.items():
            response[key] = round(float(val), 3) if isinstance(val, float) else val
    response["timestamp"] = utc_now_iso()
    return response


def stream_payload(raw_reading: dict, result: dict, buffer_fill: int, seq_len: int) -> dict:
    payload = {
        "timestamp":     utc_now_iso(),
        **{k: round(float(v), 3) for k, v in raw_reading.items()},
        "buffer_fill":   buffer_fill,
        "buffer_needed": seq_len,
    }
    payload.update(result)
    return payload


# ─────────────────────────────────────────────────────────────────────────────
#  Validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_reading(reading: dict, ranges: dict = OPERATING_RANGES) -> list:
    warnings = []
    for feature, bounds in ranges.items():
        val = reading.get(feature)
        if val is None:
            warnings.append(f"Missing field: '{feature}'")
            continue
        if not (bounds["min"] <= val <= bounds["max"]):
            warnings.append(
                f"{feature} = {val:.2f} {bounds['unit']} outside normal range "
                f"[{bounds['min']}, {bounds['max']}]"
            )
    return warnings


# ─────────────────────────────────────────────────────────────────────────────
#  Training progress bar
# ─────────────────────────────────────────────────────────────────────────────

def log_epoch(epoch: int, total: int, train_loss: float, val_loss: float) -> None:
    bar_len = 20
    filled  = int(bar_len * epoch / total)
    bar     = "█" * filled + "░" * (bar_len - filled)
    print(f"  [{bar}] {epoch:>3}/{total}  Train: {train_loss:.6f}  │  Val: {val_loss:.6f}")
