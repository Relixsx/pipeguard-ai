"""
preprocessing.py
----------------
All data preparation logic for the pipeline leak detection system.

Responsibilities:
  - MinMaxScaler: fit on normal data, transform arrays/dicts, save/load
  - Sliding-window sequence builder for LSTM input
  - Train/validation split
  - CSV loading and DataFrame-to-records conversion

Both train.py (fitting) and predict.py (transforming) import from here.
"""

import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from config import (
    FEATURES,
    SEQ_LEN,
    SCALER_PATH,
    TRAIN_VAL_SPLIT,
)
from utils import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Min-Max Scaler
# ─────────────────────────────────────────────────────────────────────────────

class MinMaxScaler:
    """
    Lightweight Min-Max scaler.

    Scales each feature to [0, 1] using statistics fitted on the
    training (normal) data only.  Keeps anomaly statistics out of
    the scaler so the model learns truly "normal" bounds.

    Serialised as a plain pickle — no sklearn dependency required.
    """

    def __init__(self):
        self.mins:     np.ndarray | None = None
        self.maxs:     np.ndarray | None = None
        self.features: list[str]         = list(FEATURES)

    # ── Fitting ───────────────────────────────────────────────────

    def fit(
        self,
        df:       pd.DataFrame,
        features: list[str] = FEATURES,
    ) -> "MinMaxScaler":
        """
        Fit scaler statistics from a DataFrame.

        Args:
            df       : DataFrame containing the feature columns
            features : list of column names to fit on

        Returns:
            self  (allows method chaining: scaler.fit(df).transform(arr))
        """
        arr        = df[features].values.astype(np.float32)
        self.mins  = arr.min(axis=0)
        self.maxs  = arr.max(axis=0)
        self.features = list(features)
        logger.debug(
            "Scaler fitted | features: %s | mins: %s | maxs: %s",
            self.features, self.mins.tolist(), self.maxs.tolist(),
        )
        return self

    # ── Transforming ──────────────────────────────────────────────

    def transform(self, arr: np.ndarray) -> np.ndarray:
        """
        Scale a (N, F) or (F,) numpy array using fitted statistics.
        Result is clipped to [0, 1] to handle mild out-of-range values.
        """
        if self.mins is None:
            raise RuntimeError("Scaler has not been fitted. Call .fit() first.")
        scaled = (arr - self.mins) / (self.maxs - self.mins + 1e-8)
        return scaled.astype(np.float32)

    def fit_transform(
        self,
        df:       pd.DataFrame,
        features: list[str] = FEATURES,
    ) -> np.ndarray:
        """Fit and immediately transform the given DataFrame."""
        self.fit(df, features)
        return self.transform(df[features].values.astype(np.float32))

    def transform_dict(self, reading: dict) -> np.ndarray:
        """
        Scale a single sensor reading provided as a dict.

        Args:
            reading : e.g. {"pressure": 80.2, "flow_rate": 495.0, "temperature": 45.3}

        Returns:
            1-D float32 array of length N_FEATURES
        """
        arr = np.array(
            [reading[f] for f in self.features],
            dtype=np.float32,
        )
        return self.transform(arr)

    def transform_df(
        self,
        df:       pd.DataFrame,
        features: list[str] = FEATURES,
    ) -> np.ndarray:
        """Scale a full DataFrame column-subset."""
        return self.transform(df[features].values.astype(np.float32))

    # ── Persistence ───────────────────────────────────────────────

    def save(self, path: Path = SCALER_PATH) -> None:
        """Serialise scaler to disk using pickle."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("Scaler saved → %s", path)

    @classmethod
    def load(cls, path: Path = SCALER_PATH) -> "MinMaxScaler":
        """Load a previously serialised scaler from disk."""
        with open(path, "rb") as f:
            scaler = pickle.load(f)
        logger.debug("Scaler loaded from %s", path)
        return scaler

    # ── Representation ────────────────────────────────────────────

    def __repr__(self) -> str:
        if self.mins is None:
            return "MinMaxScaler(unfitted)"
        pairs = ", ".join(
            f"{f}: [{mn:.3f}, {mx:.3f}]"
            for f, mn, mx in zip(self.features, self.mins, self.maxs)
        )
        return f"MinMaxScaler({pairs})"


# ─────────────────────────────────────────────────────────────────────────────
#  Sliding-Window Sequence Builder
# ─────────────────────────────────────────────────────────────────────────────

def make_sequences(
    data:    np.ndarray,
    seq_len: int = SEQ_LEN,
) -> np.ndarray:
    """
    Convert a flat time-series array into overlapping sliding windows.

    Args:
        data    : (T, F) float32 array — T time steps, F features
        seq_len : length of each window (look-back period)

    Returns:
        (N, seq_len, F) float32 array
        where N = T - seq_len + 1

    Example:
        data    shape: (1971, 3)
        seq_len      : 30
        output  shape: (1942, 30, 3)
    """
    if len(data) < seq_len:
        raise ValueError(
            f"Data length ({len(data)}) is shorter than seq_len ({seq_len}). "
            "Provide more data or reduce seq_len."
        )
    sequences = [data[i : i + seq_len] for i in range(len(data) - seq_len + 1)]
    return np.array(sequences, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  Train / Validation Split
# ─────────────────────────────────────────────────────────────────────────────

def train_val_split(
    seqs:  np.ndarray,
    split: float = TRAIN_VAL_SPLIT,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Chronological train/validation split (no shuffling — time-series data).

    Args:
        seqs  : (N, seq_len, F) array of sequences
        split : fraction used for training (default 0.9)

    Returns:
        (train_seqs, val_seqs)
    """
    idx = int(split * len(seqs))
    logger.debug(
        "Train/val split: %d train | %d val (%.0f%% / %.0f%%)",
        idx, len(seqs) - idx, split * 100, (1 - split) * 100,
    )
    return seqs[:idx], seqs[idx:]


# ─────────────────────────────────────────────────────────────────────────────
#  CSV I/O
# ─────────────────────────────────────────────────────────────────────────────

def load_csv(path: Path) -> pd.DataFrame:
    """
    Load a pipeline dataset CSV from disk.

    Expected columns: timestamp, pressure, flow_rate, temperature, is_anomaly
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset CSV not found at {path}. "
            "Run data_simulation.py to generate it."
        )
    df = pd.read_csv(path, parse_dates=["timestamp"])
    logger.info("Loaded CSV: %s  (%d rows)", path, len(df))
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  DataFrame → API Records
# ─────────────────────────────────────────────────────────────────────────────

def df_to_records(df: pd.DataFrame) -> list[dict]:
    """
    Convert a pipeline DataFrame to a list of JSON-serialisable dicts.
    Used by the /simulate/sequence endpoint.
    """
    records = []
    for _, row in df.iterrows():
        records.append({
            "timestamp":   row["timestamp"].isoformat(),
            "pressure":    round(float(row["pressure"]),    2),
            "flow_rate":   round(float(row["flow_rate"]),   2),
            "temperature": round(float(row["temperature"]), 2),
            "is_anomaly":  int(row["is_anomaly"]),
        })
    return records


# ─────────────────────────────────────────────────────────────────────────────
#  CLI Smoke-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.DEBUG)

    from data_simulation import generate_normal_data

    # Fit scaler on normal data
    df     = generate_normal_data(n_samples=200)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)
    print("Scaler   :", scaler)
    print("Scaled   :", scaled.shape, "  min:", scaled.min(), " max:", scaled.max())

    # Build sequences
    seqs = make_sequences(scaled, seq_len=30)
    print("Sequences:", seqs.shape)

    # Split
    tr, va = train_val_split(seqs)
    print(f"Train: {tr.shape}  |  Val: {va.shape}")

    # Single-reading transform
    sample  = {"pressure": 80.1, "flow_rate": 501.2, "temperature": 44.9}
    s_scaled = scaler.transform_dict(sample)
    print("Single scaled:", s_scaled)
