# ⬡ PipeGuard AI — Pipeline Leak Detection System
### AI-Powered Anomaly Detection for Oil & Gas Infrastructure

> Detects leaks in oil pipelines by learning normal pressure / flow / temperature
> patterns with an LSTM Autoencoder, then flagging deviations in real time.

---

## 📂 Project Structure

```
pipeline-leak-detection/
│
├── backend/
│   ├── config.py              ← Single source of truth: all paths, constants, hyperparameters
│   ├── utils.py               ← Shared helpers: logging, threshold computation, event builders
│   ├── data_simulation.py     ← Sensor data generator + anomaly injection
│   ├── preprocessing.py       ← MinMaxScaler, sliding-window sequencer, CSV I/O
│   ├── model.py               ← LSTM Autoencoder architecture (Encoder + Decoder)
│   ├── train.py               ← Training loop, threshold selection, artifact saving
│   ├── predict.py             ← PipelinePredictor: inference singleton used by main.py
│   └── main.py                ← FastAPI app: routing only, delegates ML to predict.py
│
├── frontend/
│   ├── index.html             ← Dashboard layout
│   ├── styles.css             ← Industrial dark-theme UI
│   └── app.js                 ← Chart.js, SSE client, API integration
│
├── models/                    ← Auto-created by train.py
│   ├── lstm_autoencoder.pth   ← Trained model weights
│   ├── scaler.pkl             ← Fitted MinMaxScaler
│   └── threshold.json         ← Anomaly threshold + statistics
│
├── data/                      ← Auto-created by train.py
│   └── simulated_pipeline_data.csv
│
├── requirements.txt
└── README.md
```

### Module Dependency Graph

```
config.py  ←──────────────────────────────────────────────────┐
    ↑                                                           │
utils.py   ←─────────────────────────────────┐               │
    ↑                                         │               │
data_simulation.py   preprocessing.py   model.py             │
         ↑                  ↑               ↑                 │
         └──────────────────┴───────────────┘                 │
                          train.py                            │
                                                              │
         predict.py  (imports model, preprocessing, utils, config)
              ↑
           main.py  (imports predict, data_simulation, preprocessing, utils, config)
```

Each module has one clear job. `main.py` contains **zero ML logic** — it only routes HTTP traffic.

---

## 🚀 Quick Start

### Prerequisites

| Tool    | Minimum Version |
|---------|----------------|
| Python  | 3.10           |
| pip     | 23+            |

---

### Step 1 — Install Dependencies

```bash
# From the project root
pip install -r requirements.txt
```

> **Apple Silicon / Windows GPU:** Visit https://pytorch.org/get-started/locally/ for the correct torch install command.

---

### Step 2 — Train the Model

```bash
cd backend/
python train.py
```

**What happens:**
1. Simulates 2,000 normal pipeline readings → saved to `data/simulated_pipeline_data.csv`
2. Fits a MinMaxScaler on normal data → saved to `models/scaler.pkl`
3. Trains LSTM Autoencoder for 40 epochs → saved to `models/lstm_autoencoder.pth`
4. Computes 3-sigma anomaly threshold → saved to `models/threshold.json`

**Expected output:**
```
════════════════════════════════════════════════════════════
  ⬡  PipeGuard AI — LSTM Autoencoder Training
════════════════════════════════════════════════════════════

  [1 / 5]  Simulating pipeline sensor data …
           Normal rows   : 2,000
           Total rows    : 2,000
           Anomaly rows  : 162  (8.1%)

  [2 / 5]  Fitting MinMaxScaler on normal data …

  [3 / 5]  Building sliding-window sequences (window = 30) …
           Sequences shape : (1971, 30, 3)
           Train / Val     : 1773 / 198

  [4 / 5]  Training LSTM Autoencoder (40 epochs) …
           Device      : cpu
           Parameters  : 74,947

           [████░░░░░░░░░░░░░░░░]   5/40  Train: 0.018234  │  Val: 0.017891
           ...
           [████████████████████]  40/40  Train: 0.001234  │  Val: 0.001456

  [5 / 5]  Computing anomaly threshold (σ = 3.0) …
           Threshold  : 0.008742

✅  Training complete!
```

Training takes **1–5 minutes** on CPU.

---

### Step 3 — Start the API Server

```bash
# Still inside backend/
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

| URL                              | Purpose                    |
|----------------------------------|----------------------------|
| http://127.0.0.1:8000            | API root                   |
| http://127.0.0.1:8000/docs       | Interactive Swagger UI     |
| http://127.0.0.1:8000/health     | System health check        |

---

### Step 4 — Open the Dashboard

```bash
# macOS
open ../frontend/index.html

# Linux
xdg-open ../frontend/index.html

# Windows
start ..\frontend\index.html
```

Or serve it properly (avoids any browser CORS quirks):
```bash
cd ../frontend && python -m http.server 3000
# Then open http://localhost:3000
```

---

## 🔌 API Reference

### `GET /health`
```json
{
  "status": "ok",
  "model_loaded": true,
  "threshold": 0.008742,
  "buffer_size": 15,
  "seq_len": 30,
  "features": ["pressure", "flow_rate", "temperature"],
  "history_count": 3
}
```

### `GET /simulate?anomaly=false`
```json
{
  "reading": { "pressure": 80.3, "flow_rate": 501.2, "temperature": 44.9 },
  "forced_anomaly": false
}
```

### `GET /simulate/sequence?n=80&with_anomaly=true`
```json
{
  "data": [
    { "timestamp": "2024-01-01T00:00:00", "pressure": 80.1, "flow_rate": 498.3,
      "temperature": 45.0, "is_anomaly": 0 },
    ...
  ],
  "count": 80
}
```

### `POST /predict`
**Request body:**
```json
{
  "readings": [
    { "pressure": 80.2, "flow_rate": 498.3, "temperature": 45.1 },
    "... (at least 30 readings)"
  ]
}
```
**Response:**
```json
{
  "status": "Normal",
  "color": "green",
  "icon": "🟢",
  "is_anomaly": false,
  "score_ratio": 0.2341,
  "anomaly_score": 0.001923,
  "pressure": 80.2,
  "flow_rate": 498.3,
  "temperature": 45.1,
  "timestamp": "2024-01-15T08:30:00Z"
}
```

### `GET /stream?anomaly_prob=0.12`
Server-Sent Events — subscribes to a continuous 1-reading/second stream.
Each `data:` event is a JSON object identical in shape to the /predict response.

### `GET /history?limit=50`
```json
{
  "events": [
    {
      "timestamp": "2024-01-15T08:30:00Z",
      "status": "Leak Detected",
      "anomaly_score": 0.014231,
      "score_ratio": 1.628,
      "pressure": 61.2,
      "flow_rate": 412.0,
      "temperature": 47.8
    }
  ],
  "total": 3,
  "limit": 50
}
```

---

## 🧠 How the AI Works

```
Normal Pipeline Data (training only)
         │
         ▼
┌────────────────────────────────────┐
│       LSTM Autoencoder             │
│                                    │
│  (seq, 3) → Encoder → (latent, 16) │
│           → Decoder → (seq, 3)     │
└────────────────────────────────────┘
         │
         ▼
   Reconstruction MSE
         │
         ├─ score < 0.6 × threshold  → 🟢 Normal
         ├─ score < 1.0 × threshold  → 🟡 Warning
         └─ score ≥ 1.0 × threshold  → 🔴 Leak Detected
```

**Core insight:** The autoencoder is trained to reconstruct normal sensor sequences.
When a leak occurs, the pattern is unfamiliar — reconstruction error spikes above the threshold.

**Threshold** is set at `mean + 3σ` of normal training errors (the 3-sigma rule).

---

## ⚙️ Configuration

All tuneable parameters are in **`backend/config.py`**:

| Parameter          | Default | Effect                                          |
|--------------------|---------|--------------------------------------------------|
| `EPOCHS`           | 40      | More epochs → better model (diminishing returns) |
| `SEQ_LEN`          | 30      | Longer window → more context, slower inference  |
| `HIDDEN_DIM`       | 64      | Wider LSTM → more capacity, more compute         |
| `SIGMA_MULTIPLIER` | 3.0     | Lower → more sensitive; Higher → fewer alarms   |
| `STREAM_INTERVAL`  | 1.0 s   | Seconds between SSE events                      |
| `DEFAULT_ANOMALY_PROB` | 0.12 | Leak probability in simulated stream           |

---

## 🛠 Troubleshooting

| Symptom                         | Fix                                                       |
|---------------------------------|-----------------------------------------------------------|
| `Model not loaded` on startup   | Run `python train.py` first                              |
| Dashboard shows "Reconnecting"  | Start the server: `uvicorn main:app --port 8000`         |
| All readings show "Normal"      | Lower `SIGMA_MULTIPLIER` in `config.py` and retrain      |
| Too many false alarms           | Raise `SIGMA_MULTIPLIER` in `config.py` and retrain      |
| `torch` install fails           | See https://pytorch.org/get-started/locally/             |
| Browser CORS error              | Serve frontend via `python -m http.server 3000`          |

---

## 🌍 Context

Built for Nigeria's petroleum sector to enable early detection of oil pipeline leaks —
reducing environmental damage, preventing revenue loss, and improving operator situational
awareness across the NPN-GRID-07 pipeline monitoring network.

---

## 📄 License

MIT — free to use, modify, and deploy.
