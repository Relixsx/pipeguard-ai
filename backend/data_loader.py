"""
data_loader.py
--------------
Loader for the SCADA Pipeline Operations Dataset.

Exact column names confirmed from dataset:
  timestamp, segment_id, pressure, flow_rate, temperature,
  valve_status, pump_state, pump_speed, compressor_state,
  energy_consumption, alarm_triggered, event_type, target

Event types: normal, leak, surge, blockage, degradation
Binary target: 0=normal, 1=fault
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from config import SCADA_CSV_PATH
from utils  import get_logger

logger = get_logger(__name__)

# Exact column names from the real dataset
TIMESTAMP_COL   = "timestamp"
PRESSURE_COL    = "pressure"
FLOW_COL        = "flow_rate"
TEMP_COL        = "temperature"
VALVE_COL       = "valve_status"
PUMP_STATE_COL  = "pump_state"
PUMP_SPEED_COL  = "pump_speed"
COMPRESSOR_COL  = "compressor_state"
ENERGY_COL      = "energy_consumption"
ALARM_COL       = "alarm_triggered"
EVENT_COL       = "event_type"
TARGET_COL      = "target"

CORE_FEATURES     = [PRESSURE_COL, FLOW_COL, TEMP_COL]
EXTENDED_FEATURES = [PRESSURE_COL, FLOW_COL, TEMP_COL,
                     VALVE_COL, PUMP_STATE_COL, PUMP_SPEED_COL,
                     COMPRESSOR_COL, ENERGY_COL]


def load_real_data(
    path:      Path = SCADA_CSV_PATH,
    extended:  bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the real SCADA dataset (1,000 rows).

    Returns:
        normal_df : 694 normal rows for training baseline
        full_df   : all 1,000 rows with is_anomaly + event_label
    """
    if not path.exists():
        raise FileNotFoundError(f"SCADA CSV not found at: {path}")

    df = pd.read_csv(path)
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], dayfirst=True, errors="coerce")
    df = df.sort_values(TIMESTAMP_COL).reset_index(drop=True)

    # Rename to standard names
    df = df.rename(columns={TARGET_COL: "is_anomaly", EVENT_COL: "event_label"})

    features = EXTENDED_FEATURES if extended else CORE_FEATURES
    keep     = [TIMESTAMP_COL] + features + ["is_anomaly", "event_label"]
    df       = df[[c for c in keep if c in df.columns]].copy()

    # Normalise pump_speed when pump is off
    if PUMP_SPEED_COL in df.columns and PUMP_STATE_COL in df.columns:
        df.loc[df[PUMP_STATE_COL] == 0, PUMP_SPEED_COL] = 0.0

    normal_df = df[df["is_anomaly"] == 0].copy().reset_index(drop=True)

    logger.info(
        "Real SCADA loaded | total=%d | normal=%d | fault=%d",
        len(df), len(normal_df), int(df["is_anomaly"].sum()),
    )
    return normal_df, df


def build_hybrid_dataset(
    n_synthetic: int  = 5000,
    extended:    bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Combine real SCADA data with calibrated synthetic data.

    Training set  = 694 real normal rows + 5,000 synthetic normal rows
    Evaluation    = 306 real fault rows  + injected synthetic faults

    This gives the model enough data to converge while ensuring
    it has seen the exact real sensor distributions.

    Returns:
        normal_df : all normal rows (real + synthetic) → for LSTM training
        full_df   : everything with labels             → for evaluation
    """
    from data_simulation import generate_normal_data, inject_anomalies

    # Real data
    real_normal, real_full = load_real_data(extended=extended)

    # Synthetic data calibrated to real statistics
    synth_normal = generate_normal_data(n_samples=n_synthetic)
    synth_full   = inject_anomalies(synth_normal)

    # Keep only core features for synthetic (extended not available)
    features = EXTENDED_FEATURES if extended else CORE_FEATURES

    # Align columns
    for df in [real_normal, real_full]:
        for col in ["event_label", "is_anomaly"]:
            if col not in df.columns:
                df[col] = "normal" if col == "event_label" else 0

    # Combine normal rows (used for training)
    synth_normal_aligned = synth_normal[[
        "timestamp", "pressure", "flow_rate", "temperature", "is_anomaly", "event_label"
    ]].copy()
    real_normal_aligned = real_normal[[
        "timestamp", "pressure", "flow_rate", "temperature", "is_anomaly", "event_label"
    ]].copy()
    combined_normal = pd.concat(
        [real_normal_aligned, synth_normal_aligned], ignore_index=True
    )

    # Full dataset for evaluation
    real_aligned = real_full[[
        "timestamp", "pressure", "flow_rate", "temperature", "is_anomaly", "event_label"
    ]].copy()
    synth_aligned = synth_full[[
        "timestamp", "pressure", "flow_rate", "temperature", "is_anomaly", "event_label"
    ]].copy()
    combined_full = pd.concat([real_aligned, synth_aligned], ignore_index=True)

    logger.info(
        "Hybrid dataset | normal=%d (real=%d + synth=%d) | total=%d | fault=%d",
        len(combined_normal), len(real_normal), len(synth_normal),
        len(combined_full), int(combined_full["is_anomaly"].sum()),
    )

    return combined_normal, combined_full
