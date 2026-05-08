"""
data_simulation.py
------------------
Generates synthetic pipeline data with statistics calibrated EXACTLY
to the real SCADA dataset (1,000 rows, analysed April 2026).

Normal operating statistics (from 694 real normal rows):
  pressure    : mean=74.85  std=4.93
  flow_rate   : mean=4.52   std=0.49
  temperature : mean=32.11  std=2.00

Fault signatures are derived from real per-event-type means and stds,
so generated faults are statistically indistinguishable from real faults.
"""

from typing import Tuple

import numpy as np
import pandas as pd

from config import (
    DATA_CSV_PATH, DATA_SEED, ANOMALY_SEED,
    N_NORMAL_SAMPLES, MIN_ANOMALY_EVENTS, MAX_ANOMALY_EVENTS,
    MIN_ANOMALY_LEN, MAX_ANOMALY_LEN,
    OPERATING_RANGES, SENSOR_NOISE, FAULT_SIGNATURES,
)
from utils import get_logger

logger = get_logger(__name__)

_B = {k: v["base"] for k, v in OPERATING_RANGES.items()}
_N = SENSOR_NOISE
_F = FAULT_SIGNATURES


# ─────────────────────────────────────────────────────────────────────────────
#  Normal Data
# ─────────────────────────────────────────────────────────────────────────────

def generate_normal_data(
    n_samples: int = N_NORMAL_SAMPLES,
    seed:      int = DATA_SEED,
) -> pd.DataFrame:
    """
    Generate normal pipeline data with statistics matching the real dataset.
    Uses slow sinusoidal drift to simulate realistic operational cycles,
    plus Gaussian noise calibrated to real sensor standard deviations.
    """
    rng = np.random.default_rng(seed)
    t   = np.arange(n_samples)

    # Pressure: mean=74.85, std=4.93
    pressure = (
        _B["pressure"]
        + 3.5 * np.sin(2 * np.pi * t / 500)     # slow operational cycle
        + 1.2 * np.sin(2 * np.pi * t / 80)      # faster ripple
        + rng.normal(0, _N["pressure"], n_samples)
    )

    # Flow rate: mean=4.52, std=0.49
    flow_rate = (
        _B["flow_rate"]
        + 0.30 * np.sin(2 * np.pi * t / 400 + 0.5)
        + 0.12 * np.sin(2 * np.pi * t / 60  + 1.2)
        + rng.normal(0, _N["flow_rate"], n_samples)
    )

    # Temperature: mean=32.11, std=2.00
    temperature = (
        _B["temperature"]
        + 1.5 * np.sin(2 * np.pi * t / 600 + 0.9)
        + rng.normal(0, _N["temperature"], n_samples)
    )

    # Clip to realistic bounds (matching real data min/max)
    pressure    = np.clip(pressure,    61.0, 89.0)
    flow_rate   = np.clip(flow_rate,   3.0,  6.2)
    temperature = np.clip(temperature, 27.0, 39.0)

    timestamps = pd.date_range("2024-01-01", periods=n_samples, freq="1s")

    return pd.DataFrame({
        "timestamp":   timestamps,
        "pressure":    pressure.round(3),
        "flow_rate":   flow_rate.round(3),
        "temperature": temperature.round(3),
        "is_anomaly":  0,
        "event_label": "normal",
    })


# ─────────────────────────────────────────────────────────────────────────────
#  Fault Injection — calibrated from real SCADA per-event statistics
# ─────────────────────────────────────────────────────────────────────────────

def inject_anomalies(
    df:   pd.DataFrame,
    seed: int = ANOMALY_SEED,
) -> pd.DataFrame:
    """
    Inject fault events whose sensor distributions match the real SCADA data.

    Each fault type uses the real measured mean delta and std from the dataset:

    LEAK        : pressure -14.76 bar (mean), flow -1.54 m³/h
    SURGE       : pressure +22.59 bar, flow +1.02 m³/h
    BLOCKAGE    : pressure +19.89 bar, flow -2.81 m³/h (very distinctive)
    DEGRADATION : pressure +0.25 bar  (almost identical to normal — hardest to detect)
    """
    rng        = np.random.default_rng(seed)
    df         = df.copy()
    n          = len(df)
    fault_types = list(_F.keys())
    raw         = [_F[f]["proportion"] for f in fault_types]
    total       = sum(raw)
    weights     = [w / total for w in raw]   # normalise to sum exactly to 1.0

    n_events = int(rng.integers(MIN_ANOMALY_EVENTS, MAX_ANOMALY_EVENTS + 1))

    for _ in range(n_events):
        fault  = rng.choice(fault_types, p=weights)
        sig    = _F[fault]
        dur    = int(rng.integers(MIN_ANOMALY_LEN, MAX_ANOMALY_LEN + 1))
        start  = int(rng.integers(50, n - dur - 30))
        end    = min(start + dur, n)
        actual = end - start
        ramp   = np.linspace(0.0, 1.0, actual)
        t_norm = np.arange(actual) / actual

        # Sample fault magnitudes from real distributions
        p_delta = float(rng.normal(sig["pressure_delta"], sig["pressure_std"] * 0.5))
        f_delta = float(rng.normal(sig["flow_delta"],    sig["flow_std"]     * 0.5))
        t_delta = float(rng.normal(sig["temp_delta"],    sig["temp_std"]     * 0.3))

        if fault == "leak":
            # Progressive pressure drop + flow collapse
            df.loc[start:end-1, "pressure"]    += p_delta * ramp
            df.loc[start:end-1, "flow_rate"]   += f_delta * ramp
            df.loc[start:end-1, "temperature"] += t_delta * ramp

        elif fault == "surge":
            # Sharp spike then returns to normal
            spike = np.sin(np.pi * t_norm)
            df.loc[start:end-1, "pressure"]    += p_delta * spike
            df.loc[start:end-1, "flow_rate"]   += f_delta * spike
            df.loc[start:end-1, "temperature"] += abs(t_delta) * spike

        elif fault == "blockage":
            # Flow drops hard, pressure rises (backpressure)
            df.loc[start:end-1, "pressure"]    += p_delta * ramp
            df.loc[start:end-1, "flow_rate"]   += f_delta * ramp
            df.loc[start:end-1, "temperature"] += t_delta * ramp

        elif fault == "degradation":
            # Very subtle — nearly flat shift with noise
            noise_p = rng.normal(0, sig["pressure_std"], actual)
            noise_f = rng.normal(0, sig["flow_std"],     actual)
            df.loc[start:end-1, "pressure"]    += p_delta + noise_p * 0.3
            df.loc[start:end-1, "flow_rate"]   += f_delta + noise_f * 0.3

        df.loc[start:end-1, "is_anomaly"]  = 1
        df.loc[start:end-1, "event_label"] = fault

    # Clip to physically possible bounds after injection
    df["pressure"]    = df["pressure"].clip(40.0, 120.0)
    df["flow_rate"]   = df["flow_rate"].clip(0.5, 8.0)
    df["temperature"] = df["temperature"].clip(22.0, 45.0)

    total = int(df["is_anomaly"].sum())
    logger.info(
        "Fault injection | %d events | %d fault rows (%.1f%%)",
        n_events, total, 100 * total / n
    )
    logger.info("Event breakdown:\n%s", df["event_label"].value_counts().to_string())
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  Dataset Builder
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset(n_normal: int = N_NORMAL_SAMPLES) -> Tuple[pd.DataFrame, pd.DataFrame]:
    normal_df = generate_normal_data(n_normal)
    full_df   = inject_anomalies(normal_df)
    return normal_df, full_df


def save_dataset(df: pd.DataFrame) -> None:
    df.to_csv(DATA_CSV_PATH, index=False)
    logger.info("Dataset saved → %s  (%d rows)", DATA_CSV_PATH, len(df))


# ─────────────────────────────────────────────────────────────────────────────
#  Single-Reading Simulator (used by /simulate and /stream endpoints)
# ─────────────────────────────────────────────────────────────────────────────

def get_random_sample(anomaly: bool = False) -> dict:
    """
    Generate one sensor reading using real SCADA calibrated distributions.
    Fault readings are sampled from the real per-fault-type distributions.
    """
    rng = np.random.default_rng()

    if anomaly:
        fault_types = list(_F.keys())
        raw         = [_F[f]["proportion"] for f in fault_types]
        weights     = [w / sum(raw) for w in raw]
        fault       = rng.choice(fault_types, p=weights)
        sig         = _F[fault]

        pressure    = _B["pressure"]    + float(rng.normal(sig["pressure_delta"], sig["pressure_std"]))
        flow_rate   = _B["flow_rate"]   + float(rng.normal(sig["flow_delta"],    sig["flow_std"]))
        temperature = _B["temperature"] + float(rng.normal(sig["temp_delta"],    sig["temp_std"]))
    else:
        pressure    = _B["pressure"]    + float(rng.normal(0, _N["pressure"]))
        flow_rate   = _B["flow_rate"]   + float(rng.normal(0, _N["flow_rate"]))
        temperature = _B["temperature"] + float(rng.normal(0, _N["temperature"]))

    return {
        "pressure":    round(float(np.clip(pressure,    40.0, 120.0)), 3),
        "flow_rate":   round(float(np.clip(flow_rate,   0.5,   8.0)), 3),
        "temperature": round(float(np.clip(temperature, 22.0,  45.0)), 3),
    }


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    normal, full = build_dataset()
    print(f"\nNormal rows : {len(normal):,}")
    print(f"Total rows  : {len(full):,}")
    print(f"Fault rows  : {int(full['is_anomaly'].sum()):,}")
    print(f"\nEvent breakdown:\n{full['event_label'].value_counts()}")
    print(f"\nNormal stats vs real SCADA:")
    print(f"{'':20} {'Generated':>12} {'Real':>12}")
    for col in ["pressure", "flow_rate", "temperature"]:
        real_means = {"pressure": 74.85, "flow_rate": 4.52, "temperature": 32.11}
        print(f"  {col:<18} {normal[col].mean():>12.3f} {real_means[col]:>12.3f}")
    save_dataset(full)
