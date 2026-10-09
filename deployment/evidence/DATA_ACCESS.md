# Qualified pipeline data and the remaining access gap

## Closest field SCADA: NGPOD

Primary description: [Early Prediction of Natural Gas Pipeline Leaks Using the MKTCN Model](https://arxiv.org/html/2411.06214v1).
It describes a 17.95 km natural-gas pipeline, four stations, pressure and
temperature channels, boundary flow, and approximately four months of
20-second measurements. This is a substantially closer modality than
oil-well operational data or water-network telemetry.

Acquisition check: [MrLiXv/MKTCN](https://github.com/MrLiXv/MKTCN), commit
`6d4a00fcd5bdb4fdc885f2e2c2b5bb78786e7b20`, contains a README and notebook.
The notebook loads six arrays from `./pipeline_datasets/`; that folder is
absent. No raw download was acquired. Do not describe it as an available
training dataset. Pre-windowed arrays alone would also be insufficient for
auditing overlap, event boundaries and causal calibration.

**Request route:** the paper authors, including Xuguang Li and Zhonglin Zuo;
use the manuscript's corresponding-author contact. Ask for raw chronological
records, station/run identifiers, confirmed event timestamps, original labels
before early-warning relabelling, sampling/aggregation definitions, units,
operating commands, geometry and research reuse terms. No request was sent.

## Closest multiphase simulation data: OLGA study

Primary record: [Naanouh and Henry, Digital 2026, 6(2), 45](https://doi.org/10.3390/digital6020045).
The study uses pressure, temperature and inlet/outlet mass flow from controlled
multiphase leak cases. Its data-availability statement directs requests for
processed data/code to the corresponding author and notes possible
redistribution restrictions associated with the original simulator/provider.
The raw files were not acquired.

**Request route:** the corresponding author, Manus Henry, and the original
providers identified in the article, Vandrangi et al. Request original
time-ordered runs, normal baseline, leak onset/magnitude/location, boundary
conditions, operating-change negatives, train/test provenance, and explicit
reuse permission. This is simulated multiphase data, not a field gas benchmark.

## Acquired real gas-acoustic pilot

Source: [Meng et al. dataset repository](https://github.com/mengdinet/Gas-pipeline-leakage-data-set),
which links its [public folder](https://drive.google.com/drive/folders/1BLjQHB5L0z9Ov71AKilmN2eURnIdPnte).
This execution downloaded 50 files from each visible `blower`, `hole_blower`
and `valve_blower` listing: 150 different hashes, mono 16-bit PCM, 96 kHz,
one second each. Folder names supply provisional healthy/hole/valve labels.
The 2025 [MSFAT paper](https://doi.org/10.3390/s25206390) describes a larger
22,500-sample use of this source; that entire release and its grouping metadata
were not verified or acquired here.

This data supports a single-channel acoustic pilot. It does not provide
pressure, flow, temperature, asset geometry, confirmed independent sessions,
or paired simultaneous channels for Dankers-style source separation. A
filename prefix is not proof of an independent experiment. No explicit dataset
license was found in the source repository. Audio bytes are excluded from
the delivered archive; file identifiers and SHA-256 hashes are retained.

**Request route:** dataset providers. Ask for the complete release, original
recording/session IDs and cut times, pressure and noise settings, synchronized
microphone channels, machinery-only negatives, and clear research reuse terms.

## Minimum acceptance contract for the main model

The required unit of evidence is an independent run or event, not a large row count.

- Natural gas or explicitly identified oil/multiphase pipeline, rather than an oil well.
- Synchronized pressure at multiple positions and inlet/outlet mass flow; gas temperature.
- Absolute vs gauge pressure, mass vs volume flow, composition and unit definitions.
- Asset/run IDs, monotone timestamps, aggregation intervals and sensor-quality flags.
- Leak onset, duration, severity and confirmation evidence; healthy control changes.
- Realized controls, known external excitation, valve/compressor status and topology.
- Geometry/volume, uncertainty and physical parameter assumptions.
- Independent events/runs retained for validation and final evaluation.
- Research access and redistribution terms stated by the provider.

Until this contract is met, the delivered SCADA comparisons remain simulations.
The public acoustic recordings are relevant to the gas-leak task but represent
a separate sensing modality. They should not replace pressure/flow telemetry
without changing the project's sensing design.
