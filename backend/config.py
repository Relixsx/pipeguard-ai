"""
config.py
---------
All constants and hyperparameters.
Sensor statistics are calibrated exactly from the real SCADA dataset.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
#  Directories
# ─────────────────────────────────────────────────────────────────────────────

BACKEND_DIR : Path = Path(__file__).resolve().parent
ROOT_DIR    : Path = BACKEND_DIR.parent
DATA_DIR    : Path = ROOT_DIR / "data"
MODELS_DIR  : Path = ROOT_DIR / "models"
LOGS_DIR    : Path = ROOT_DIR / "logs"

for _dir in (DATA_DIR, MODELS_DIR, LOGS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
#  Artifact Paths
# ─────────────────────────────────────────────────────────────────────────────

DATA_CSV_PATH   : Path = DATA_DIR   / "simulated_pipeline_data.csv"
SCADA_CSV_PATH  : Path = DATA_DIR   / "scada_pipeline.csv"
MODEL_PATH      : Path = MODELS_DIR / "lstm_autoencoder.pth"
SCALER_PATH     : Path = MODELS_DIR / "scaler.pkl"
THRESHOLD_PATH  : Path = MODELS_DIR / "threshold.json"

# ─────────────────────────────────────────────────────────────────────────────
#  Database
# ─────────────────────────────────────────────────────────────────────────────

DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    f"sqlite:///{DATA_DIR / 'pipeguard.db'}"
)

# ─────────────────────────────────────────────────────────────────────────────
#  Auth
# ─────────────────────────────────────────────────────────────────────────────

SECRET_KEY     : str = os.getenv("SECRET_KEY", "CHANGE-THIS-SECRET-KEY-IN-PRODUCTION-NOW")
ALGORITHM      : str = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "480"))

DEFAULT_ADMIN_USERNAME: str = os.getenv("ADMIN_USERNAME", "admin")
DEFAULT_ADMIN_PASSWORD: str = os.getenv("ADMIN_PASSWORD", "PipeGuard2024!")
DEFAULT_ADMIN_FULLNAME: str = os.getenv("ADMIN_FULLNAME", "System Administrator")

# ─────────────────────────────────────────────────────────────────────────────
#  Logging
# ─────────────────────────────────────────────────────────────────────────────

LOG_LEVEL       : str  = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE        : Path = LOGS_DIR / "pipeguard.log"
LOG_MAX_BYTES   : int  = 10 * 1024 * 1024
LOG_BACKUP_COUNT: int  = 5

# ─────────────────────────────────────────────────────────────────────────────
#  Rate Limiting
# ─────────────────────────────────────────────────────────────────────────────

RATE_LIMIT_DEFAULT : str = os.getenv("RATE_LIMIT_DEFAULT", "120/minute")
RATE_LIMIT_PREDICT : str = os.getenv("RATE_LIMIT_PREDICT", "60/minute")
RATE_LIMIT_STREAM  : str = os.getenv("RATE_LIMIT_STREAM",  "10/minute")
RATE_LIMIT_AUTH    : str = os.getenv("RATE_LIMIT_AUTH",    "10/minute")

# ─────────────────────────────────────────────────────────────────────────────
#  Application
# ─────────────────────────────────────────────────────────────────────────────

API_TITLE       : str = "PipeGuard AI — Pipeline Leak Detection API"
API_DESCRIPTION : str = "Production-grade AI anomaly detection for oil & gas pipeline sensor data."
API_VERSION     : str = "2.0.0"
ENVIRONMENT     : str = os.getenv("ENVIRONMENT", "development")

# ─────────────────────────────────────────────────────────────────────────────
#  Features
# ─────────────────────────────────────────────────────────────────────────────

FEATURES   : list = ["pressure", "flow_rate", "temperature"]
N_FEATURES : int  = len(FEATURES)

# ─────────────────────────────────────────────────────────────────────────────
#  Operating Ranges — derived from real SCADA dataset (694 normal rows)
#
#  pressure    : mean=74.85  std=4.93  min=61.37  max=88.82
#  flow_rate   : mean=4.52   std=0.49  min=3.07   max=6.15
#  temperature : mean=32.11  std=2.00  min=27.08  max=38.52
# ─────────────────────────────────────────────────────────────────────────────

OPERATING_RANGES: dict = {
    "pressure": {
        "min":  61.0,
        "max":  89.0,
        "base": 74.85,
        "unit": "bar",
    },
    "flow_rate": {
        "min":  3.0,
        "max":  6.2,
        "base": 4.52,
        "unit": "m³/h",
    },
    "temperature": {
        "min":  27.0,
        "max":  39.0,
        "base": 32.11,
        "unit": "°C",
    },
}

# Noise from real data standard deviations (normal rows only)
SENSOR_NOISE: dict = {
    "pressure":    4.93,
    "flow_rate":   0.49,
    "temperature": 2.00,
}

# ─────────────────────────────────────────────────────────────────────────────
#  Fault Signatures — measured directly from real SCADA data
#
#  Each entry: (mean_delta, std)  where delta = fault_mean - normal_mean
#  Used by data_simulation.py to generate realistic fault events
# ─────────────────────────────────────────────────────────────────────────────

FAULT_SIGNATURES: dict = {
    # Real: pressure mean=60.10, flow mean=2.98
    "leak": {
        "pressure_delta": -14.76,   "pressure_std": 5.71,
        "flow_delta":     -1.54,    "flow_std":     0.58,
        "temp_delta":      0.07,    "temp_std":     1.81,
        "proportion":      0.213,
    },
    # Real: pressure mean=97.45, flow mean=5.54
    "surge": {
        "pressure_delta": +22.59,   "pressure_std": 7.01,
        "flow_delta":     +1.02,    "flow_std":     0.62,
        "temp_delta":      0.01,    "temp_std":     2.23,
        "proportion":      0.200,
    },
    # Real: pressure mean=94.75, flow mean=1.72
    "blockage": {
        "pressure_delta": +19.89,   "pressure_std": 8.95,
        "flow_delta":     -2.81,    "flow_std":     0.50,
        "temp_delta":     -0.11,    "temp_std":     1.83,
        "proportion":      0.148,
    },
    # Real: pressure mean=75.11, flow mean=4.53 (very close to normal — hardest to detect)
    "degradation": {
        "pressure_delta": +0.25,    "pressure_std": 4.82,
        "flow_delta":     +0.00,    "flow_std":     0.50,
        "temp_delta":     -0.10,    "temp_std":     1.90,
        "proportion":      0.443,
    },
}

# ─────────────────────────────────────────────────────────────────────────────
#  Data Simulation
# ─────────────────────────────────────────────────────────────────────────────

N_NORMAL_SAMPLES   : int   = 5000    # 5k synthetic + 694 real = 5694 training rows
DATA_SEED          : int   = 42
ANOMALY_SEED       : int   = 7
MIN_ANOMALY_EVENTS : int   = 5
MAX_ANOMALY_EVENTS : int   = 12
MIN_ANOMALY_LEN    : int   = 20
MAX_ANOMALY_LEN    : int   = 50

# ─────────────────────────────────────────────────────────────────────────────
#  Model
# ─────────────────────────────────────────────────────────────────────────────

SEQ_LEN    : int   = 30
HIDDEN_DIM : int   = 64
LATENT_DIM : int   = 16
NUM_LAYERS : int   = 2

# ─────────────────────────────────────────────────────────────────────────────
#  Training
# ─────────────────────────────────────────────────────────────────────────────

EPOCHS          : int   = 60
BATCH_SIZE      : int   = 32
LEARNING_RATE   : float = 1e-3
LR_STEP_SIZE    : int   = 20
LR_GAMMA        : float = 0.5
GRAD_CLIP_NORM  : float = 1.0
TRAIN_VAL_SPLIT : float = 0.9

# ─────────────────────────────────────────────────────────────────────────────
#  Anomaly Detection
# ─────────────────────────────────────────────────────────────────────────────

SIGMA_MULTIPLIER : float = 3.0
RATIO_WARNING    : float = 0.6
RATIO_CRITICAL   : float = 1.0

# ─────────────────────────────────────────────────────────────────────────────
#  Streaming
# ─────────────────────────────────────────────────────────────────────────────

STREAM_INTERVAL      : float = 1.0
DEFAULT_ANOMALY_PROB : float = 0.12
HISTORY_MAXLEN       : int   = 200
SENSOR_BUFFER_MAXLEN : int   = SEQ_LEN
