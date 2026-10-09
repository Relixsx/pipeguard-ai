# PipeGuard: executed research prototype

This package contains **executed experiments**, trained checkpoints, a local API,
and an auditable data-access record. It extends the ideas in
[Relixsx/pipeguard-ai](https://github.com/Relixsx/pipeguard-ai), audited at commit
`f3acfde210161a396cb1835bcabd2a413ba20e78`. It has not been pushed to that repository.

## What was executed

1. Qualified oil/gas pipeline datasets and downloaded **150 unique gas-acoustic WAV files**.
   The closest field pressure/flow dataset (NGPOD) and a multiphase OLGA dataset
   were located, but their raw data were **not acquired**. See `DATA_ACCESS.md`.
2. Converted the ten supplied Dankers papers into a method with explicit
   assumptions. See `PAPER_MAPPING.md`; the reading ledger records all 119 paper pages.
3. Trained the reference CNN-LSTM architecture, a dynamic ARX residual model,
   and a residual CNN-LSTM hybrid. Both neural comparisons use three initialization
   seeds. All three main models receive the same eight observed variables.
   Raw CNN-LSTM reconstructs eight variables; ARX predicts six physical
   measurements; the hybrid monitors six innovations. Input access is matched,
   while objectives differ. This compares monitoring designs, rather than
   isolating an architecture's superiority.
4. Evaluated independent runs, an original-three-feature ablation, a separate
   real-audio pilot, a physical inventory residual, and broader healthy calibration.
5. Saved weights, results, simulation data, integrity tests, a working local API,
   and an interview brief.

## Measured findings

The primary simulation test contains 36 leak events. With narrow healthy
calibration, CNN-LSTM detects 28/36, 29/36 and 31/36 events across seeds;
the dynamic predictor and hybrid each detect 35/36. Their false alarm rates
remain too high under operating changes. **The hybrid did not demonstrate an
improvement over the dynamic predictor.**

An exploratory follow-up uses frozen weights, additional independent healthy
regimes, and another 66-run test cohort generated only after threshold fitting:

| Model, broader calibration | Detected leak events | False alarm events / healthy exposure | Median delay of detected events |
|---|---:|---:|---:|
| CNN-LSTM, seed 11 | 0/36 | 2 / 28.58 h | No detections |
| Dynamic residual | 25/36 | 6 / 28.58 h | 212 s |
| Hybrid, seed 11 | 25/36 | 6 / 28.58 h | 212 s |
| Mass-inventory residual | 36/36 | 7 / 28.58 h | 94.5 s |

The inventory model still exceeds the chosen 0.1-alarm/h target. Its 36/36
event recall has a conditional exact 95% interval of approximately 90.3%-100%.
Geometry and gas properties match the simulator, which makes this an
optimistic physics experiment. It is **not field validation**.
Broader inventory calibration did not flag any of the six sensor-bias cases;
the dynamic/hybrid models flagged all six. A separate sensor-quality branch
therefore remains necessary to validate. Detection and sensor diagnosis have
different evidence requirements.

The real-audio pilot has only 30 test clips: 20 leaks and 10 healthy clips.
The dynamic model detects 20/20 with 1 false positive; CNN-LSTM detects 8/20
with 0 false positives; the hybrid detects 20/20 with 3 false positives.
Filename blocks are provisional groups, not verified acquisition sessions.
No leak-event delay or field false-alarm rate can be inferred from this pilot.

## Install

Tested on Python 3.12, Linux CPU, PyTorch 2.14.1+cpu. From this directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

Other dependency versions may change numerical results slightly. The archived
weights and machine-readable results are the exact outputs of this execution.

## Inspect and verify the saved results

```bash
python -m unittest discover -s tests -v
python -c "from pipeguard.simulator import generate_cohort; generate_cohort('data/simulation')"
python reevaluate_saved.py --data-dir data/simulation --result-dir results
python serve_research.py --result-dir results --model mass_inventory --calibration broad
```

The server binds to `127.0.0.1:8001`. `/schema` provides the input order.
`POST /predict` accepts one ordered reading; `/readings` returns all logged
accepted readings, including healthy, warming-up and data-quality statuses.
Buffers and alarm states are separate for each asset. Scores use the actual
saved model, without label-driven demo scores.

```json
{
  "asset_id": "research-segment-A",
  "timestamp_s": 0,
  "values": [3900000, 3800000, 3700000, 2.0, 2.0, 288.0, 4000000, 1.0]
}
```

Use increasing timestamps at five-second intervals. A null sensor value raises
a data-quality alert and resets that asset's window. All eight values are
required by this research schema:

1. inlet absolute pressure, Pa;
2. midpoint absolute pressure, Pa;
3. outlet absolute pressure, Pa;
4. inlet mass flow, kg/s;
5. outlet mass flow, kg/s;
6. gas temperature, K;
7. realized upstream pressure command, Pa;
8. outlet valve command, dimensionless.

The legacy three-float API cannot supply this schema without additional sensors
and explicit unit conversion. Acoustic data use a separate feature schema and
are not accepted by this SCADA monitor. Alarm output is an anomaly indication;
component attribution remains unestablished.

## Reproduce training and the follow-up experiments

Keep the archived `results` directory intact if you want to compare against it.
To reproduce into a separate directory:

```bash
python run_benchmark.py --data-dir data/simulation --output-dir reproduced --epochs 16 --seeds 11 23 37
python run_inventory_check.py --data-dir data/simulation --output-dir reproduced
python run_regime_calibration.py --data-dir data/simulation --result-dir reproduced
```

`run_benchmark.py` regenerates the primary cohort if `--skip-generation` is
omitted. Splits use complete independent runs. Standardization uses healthy
training data only. ARX order and ridge penalty, and neural early stopping,
use healthy validation data. Alarm thresholds use healthy calibration data.
No leak labels enter model fitting or calibration.

The generator has three fixed-volume gas storage cells, ideal isothermal
storage with constant compressibility, quasi-steady square-pressure flow,
upstream feedback, external excitation and outlet valve changes. Leaks remove
mass from a declared cell. It conserves mass to floating-point precision.
It does not solve the full gas momentum/energy PDE, reproduce OLGA,
simulate multiphase flow, or validate pressure-wave localization.

## Reproduce the real-recording pilot

Third-party recordings are not included. The download utility fetches the
public listings and validates WAV headers and hashes; it does not assume that
the listings cover the entire dataset.

```bash
python download_acoustic.py --data-dir data/acoustic
python run_acoustic_pilot.py --data-dir data/acoustic --output-dir reproduced/acoustic
```

For the exact archived pilot, compare downloaded file IDs and hashes with
`evidence/acoustic_acquisition_manifest.json`. The pilot uses prefixes 0-14
for training, 15-19 for validation, and 20-24 for testing, keeping both suffix
files and all classes of a prefix in the same split. New listings may contain
additional files; they should not silently change a published benchmark.
Dataset reuse terms and acquisition-session identifiers require clarification
from the providers before a larger or commercial study.

## Event accounting and scope

A confirmed new alarm at or after physical leak onset counts as detection.
An alarm already active before onset receives no detection credit. False alarms
are counted over healthy and pre-leak exposure; sensor-bias and missing-sensor
cases are reported separately. Persistence is three scores; clearing requires
five normal scores; notifications have a 120-second cooldown.

The inventory and broad-calibration studies were added after the primary
results. Their fresh-seed cohorts are exploratory replications within the same
simulator, not independent physical validation. Three neural initialization
seeds reuse the same leak events and must not be counted as 108 independent leaks.

The recommended next research step is aligned, multi-point pipeline telemetry
with known controls, geometry, healthy operating changes, event timestamps and
held-out acquisition runs. Then test whether target-specific information and
appropriate dynamic-network estimators improve diagnosis and uncertainty.

## Rebuild the report and scientific figures

The report is included at `../deployment/evidence/research-report.pdf`. The archived
cohort manifest is included; generate the array files before replaying the benchmark.
On Linux with DejaVu Sans installed,
you can rebuild it from the archived result JSONs:

```bash
python -m pip install reportlab==4.4.9
python tools/build_report.py
```

This does not retrain models or change thresholds.
