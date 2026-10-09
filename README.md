# PipeGuard Research Lab

A deployable pipeline-monitoring research prototype. The UI and API run together, without a visitor login, and compare **four frozen methods on the same sensor stream**. This update integrates the executed research into the original PipeGuard application.

The models were trained on a three-volume, isothermal, quasi-steady gas simulator. This is a scholarship/interview demonstration and a reproducible research starting point. Field performance, leak cause and leak localization are unestablished.

## Try it locally

Python 3.12:

```sh
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
sh start.sh
```

Open **http://localhost:8000**. The dashboard and inference API share the same origin. Do not open the HTML directly from the filesystem. No model training, GPU, account signup or credentials are required.

With Docker:

```sh
docker compose up --build
```

For Render or another host, see [DEPLOYMENT.md](DEPLOYMENT.md). The same application binds the host-provided `PORT` and exposes `/health` for readiness checks.

## What is implemented

- Actual CNN–LSTM and residual CNN–LSTM checkpoint inference through ONNX Runtime, alongside the regularized dynamic predictor and gas inventory balance.
- Seven repeatable scenarios: abrupt/growing leaks, healthy operation, valve transition, sensor drift, missing sensors and weak excitation. Size, location, command range and random seed are configurable.
- Causal trailing windows, isolated per-run histories, warm-up after missing readings or time gaps, and persistence/clearance/cooldown alarm logic.
- Replay and inspect the sensor and score traces. Scenario ground truth is displayed separately and never enters model inference.
- Import numeric sensor CSVs in the explicit eight-feature SI schema. The reference remains simulation-trained; imports do not confer field validation.
- Download all readings, scores and alert states, including healthy periods. Review the full report, ten-paper mapping, original-source audit and saved benchmark results in the UI.
- Bounded compute and ephemeral run storage. No external chart libraries or hard-coded backend URLs in the dashboard.

## Frozen models and results

The deployed neural seed is 11. The full research includes seeds 11, 23 and 37; broader calibration was executed only for seed 11. All four methods receive the same eight input channels, but monitor different objectives: the raw autoencoder reconstructs eight channels, the predictor forecasts six physical channels, and the hybrid checks six innovations.

Exploratory broader-calibration evaluation on **66 fresh simulated runs**, including **36 injected leak events**, with **28.5806 scored healthy test hours**:

| Method | New post-onset detections | False alarm events/hour | Median delay among detected events | Pressure-drift cases alerted |
|---|---:|---:|---:|---:|
| Mass balance | 36/36 | 0.245 | 94.5 s | 0/6 |
| Dynamic predictor | 25/36 | 0.210 | 212 s | 6/6 |
| CNN–LSTM | 0/36 | 0.070 | No detections | 0/6 |
| Residual hybrid | 25/36 | 0.210 | 212 s | 6/6 |

Only a new confirmed alarm at or after leak onset earns detection credit. Pre-existing false alarms receive no credit. Missed events remain in recall. The dashboard also shows confidence intervals and the narrow-calibration comparison on these same test runs.

Mass balance is promising in this controlled simulator, with known volumes/gas properties. Its false alarm rate still exceeds the 0.1/hour calibration target, and it missed all six pressure-drift cases. Hybrid superiority is not established. Do not interpret the simulated 36/36 result as a field accuracy claim.

A separate small real gas-acoustic pilot used 150 unique one-second clips, including 30 test clips. It is not SCADA, not session-held-out field validation, and not multichannel source separation. Raw audio is not bundled because redistribution rights and acquisition independence are unclear.

## Code layout

| Path | Purpose |
|---|---|
| `backend/main.py` | Same-origin web/API, validation, bounds, downloads and readiness |
| `backend/runtime.py` | Frozen ONNX/ARX/physics inference; sensor readings and timestamps only |
| `backend/experiments.py` | Scenarios, bounded storage, exports and post-inference truth comparisons |
| `backend/simulator.py`, `backend/physics.py` | Audited simulator and inventory calculation |
| `frontend/` | Dashboard, local SVG plots, replay and CSV import |
| `deployment/models/` | Two ONNX models, dynamic reference, frozen config and SHA-256 manifest |
| `deployment/evidence/` | Executed results, report, immutable example and replay parity evidence |
| `research/` | Reproducible training/calibration scripts, original PyTorch checkpoints and reading record |
| `legacy/` | Preserved original application, data and weights; excluded from deployment |
| `tests/`, `tools/` | Integration checks, export verification and benchmark replay |

The legacy label-driven score demo, account gate, global stream buffer and anomaly-only persistence are not the serving path in version 2.

## Verification and reproduction

```sh
pip install -r requirements-dev.txt
python -m unittest discover -s tests -p 'test_*.py' -v
python tools/verify_benchmark.py
```

`verify_benchmark.py` regenerates all 66 final scenarios and verifies the deployed methods against **all 264 archived model/run alarm traces**. It is a parity check, not new validation data. Tests also check causal prefixes, missing/gapped readings, isolation, real API scores, complete exports and request bounds.

For browser checks, run the server in another terminal:

```sh
python -m playwright install chromium
python tools/browser_smoke.py --url http://localhost:8000
```

To re-export the supplied neural checkpoints, install PyTorch CPU from its official wheel index, then the research dependencies:

```sh
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-research.txt
python tools/export_models.py
python tools/build_example.py
```

The export checks reconstructions and scores against PyTorch for batches 1, 7 and 32. Checkpoint/artifact hashes and error magnitudes are recorded in `deployment/models/manifest.json`.

Training and calibration commands are in [research/README.md](research/README.md). The large generated cohort is not committed here; `run_benchmark.py` regenerates it by default. To replay archived checkpoints without retraining, generate that cohort first with `pipeguard.simulator.generate_cohort`.

## Data and research provenance

The paper mapping and reading ledger cover the ten supplied Dankers/collaborator papers. Implemented adaptations are distinguished from theorem assumptions, diagnostic approximations and proposed acoustic extensions. Read [research/PAPER_MAPPING.md](research/PAPER_MAPPING.md) and [research/DATA_ACCESS.md](research/DATA_ACCESS.md).

NGPOD is a substantially better conceptual fit for gas pressure/flow monitoring, but its raw arrays were absent from the linked repository when audited. OLGA multiphase data require author/provider access. Neither is presented as acquired or used to train this demo.

## Public demo boundaries

The default single-worker server permits one experiment computation at a time, keeps at most 12 ephemeral runs for at most one hour, and limits new computations per client. Restarts and free-host sleep can discard those runs. Export a run if it matters. CSV uploads are held in memory, not written to disk or used for training. Run URLs are unguessable identifiers; this public research demo is not a sensitive operational-data portal.

The optional Swagger API docs load their viewer from jsDelivr; the dashboard itself has no external runtime dependencies. A deployment of this repository is not an industrial leak alarm or a substitute for validated safety instrumentation.
