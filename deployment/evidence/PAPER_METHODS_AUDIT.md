# PipeGuard v3: executable paper methods and scientific audit

This upgrade implements finite specializations of D2, D4, D6 and D8, and transfers
D9's cluster-refinement framework into the gas detector. The ten supplied papers
and page-reading record are in `publication_references.json` and
`paper_reading_ledger.json` (119 paper pages). This document describes executable
algorithms, rather than the earlier conceptual connections.

Dankers' network research asks whether measurements can reveal a particular
physical connection despite feedback, alternative paths and disturbances.
Accurate prediction alone does not establish that connection: a biased model
may predict well. The acoustic work addresses sources arriving as delayed,
reflected mixtures. The energy forecasting papers address context and the
consistency of transformations available during training and deployment.

## Audit of all ten supplied papers

| Paper | Research problem | What the code implements and what remains |
|---|---|---|
| D1, 2013 | Consistent module estimation in feedback networks; direct and two-stage prediction-error methods. | D2's finite two-stage specialization below. Original gas ARX remains a forecasting baseline, with no module-consistency claim. |
| D2, 2016 | Preserve the target module when choosing predictor inputs and eliminating hidden nodes. | Algorithm 2 projection; Proposition 3 exact frequency immersion; Proposition 4 path/loop preservation; minimum input-set search in a bounded small graph. |
| D3, 2018 | Model-set identifiability depends on known/unknown dynamics, parameter restrictions and excitation. | Explicit known-topology/independent-parameter contracts and D4 target graph test. General D3 model-set theorem checker is not implemented. |
| D4, supplied 2018 manuscript | Identify one module even if its entire incoming row is ambiguous. | Target rank increment via vertex-disjoint maximum flow; Figure 1 reproduction. Generic interpretation assumes independent modules and admissible excitation. |
| D5, 2017 | Handle shared disturbances without introducing unblocked confounding through the chosen measured nodes. | Colored shared-disturbance stress test with valid instruments. General Property 3 blocking-node algorithm is not implemented; D2 is not relabelled as that algorithm. |
| D6, supplied v5 / 2026 | Target information, disconnecting sets and excitation location can suffice without full-model information. | Definitions 4–5/Corollary 6 excitation exclusion and disjoint paths; Section 6 example; known-transfer spectral checks. Finite regression ranks are separately labelled. |
| D7, 2016 | Identify sparse long-echo acoustic mixing systems to recover sources. | Assumptions motivate the acoustic branch implemented through D8. Full D7 sparse long-echo optimization is not reproduced. |
| D8, 2025 | Local delayed spatial FIR filtering and verification for real-time pipeline source estimation. | Eq.8 design, minimum-singular-value ridge, stable inverse, healthy-frozen monitoring, held-out verification, synchronized uploads and recorded-source controlled mixing. Original field acquisition is not reproduced. |
| D9, 2025 | Refine forecasts using clustered operating context, balanced assignment and augmented forecasts. | Causal chunk statistics, DBSCAN, exact minimum-cost bijection, augmented ridge refiners and L1 routing in fifth gas detector. Ridge replaces CatBoost; past numeric commands replace outage/text context. No SBERT/TSDAE implementation. |
| D10, 2025 | Batched decomposition should match what is available at the forecast origin. | Past-only context, frozen preprocessing and prefix-invariance tests. Full batched VMD/decomposition framework is not implemented. |

## D2: graph selection, immersion and two-stage identification

The controlled network has target w1→w2, parallel route w1→w3→w2, output loop
w2→w4→w2, and feedback w2→w1. Proposition 4 requires the target input to be
retained, every alternative target-to-output route to pass a retained predictor
input, and every output loop to pass one. The minimum selected set is
**w1,w3,w4**. Omitting w3 or w4 can change the immersed target.

`immerse_frequency` eliminates hidden nodes with
Grr + Grh(I−Ghh)^−1Ghr, then normalizes induced self-loops, following Proposition 3.
Tests reproduce the paper's loop-denominator example. This is an exact calculation
for a declared known model, not recovery of an unknown field graph.

`identify_two_stage` specializes Algorithm 2 to strictly delayed one-tap modules:
project selected measurements at t−1 onto independent external-reference
histories t−1…t−12; fit the output using projected inputs. No noise model is used
(D2 Remark 1). HAC intervals use the IV structural residual. The stable generator
supplies independent r1,r3,r4, no omitted direct output reference, and colored
shared disturbances independent of those references.

Attribution requires graph preservation, generic target rank, finite projected
target rank, and reference/conditional-target energy ≥0.01 in normalized lab
units. The energy levels are practical guards, not D6 theorem thresholds.
Realized feedback commands in the gas simulator are **not automatically valid
external instruments**, so ordinary SCADA commands are not given this claim.

In 80 independent 4,096-sample runs per condition (true target gain 0.65):

| Condition | Two-stage RMSE | Ordinary regression RMSE | Approximate 95% interval coverage | Attribution supported |
|---|---:|---:|---:|---:|
| Independent noise | 0.00418 | 0.00399 | 92.5% | 80/80 |
| Colored shared disturbance | 0.01431 | 0.31664 | 95.0% | 80/80 |
| Very weak excitation | 0.59999 | 0.65405 | 11.25% | 0/80 |

With shared disturbances, ordinary regression averages 0.9664; two-stage averages
0.6513. This is a demonstrated controlled-model bias reduction. Weak-reference
estimates and poor interval coverage remain displayed as negative evidence,
with attribution withheld.

Rolling 1,024-sample estimates, stride 256, compare against an independent
healthy run using approximate HAC difference intervals and Bonferroni correction.
Windows spanning a parameter change violate stationarity. The default
other-feedback-change example produces one change flag: perfect false-positive
control is not claimed. The Monte Carlo above evaluates fixed-module estimation,
not calibrated event-level fault diagnosis.

## D4/D6: target information and excitation location

D4 Figure 1 is reproduced: three unknown incoming modules and two excitations
leave the whole row ambiguous while the target is unique. Maximum flow has
unit vertex capacity, including excitation and target endpoints.

D6 Section 6 is reproduced with u1→u2, u2→u3, u3→u2, u4→u3. The target input
is u1, disconnecting set {u2}, remainder {u3,u4}. Independent x1+x2 supports
the sufficient target condition. x1+x3 fails: x3 reaches the remainder without
passing u2 and is excluded. The full four-module row remains ambiguous.

A stable declared transfer specialization is checked at 64 frequencies. Minimum
eligible target-block spectral eigenvalue is 0.6033 for x1+x2 and zero for x1+x3.
The finite grid supplements the graph condition; it is not a proof at every
frequency for arbitrary transfer functions. More rows at the wrong excitation
location do not repair a structural information deficiency.

## D8: spatial acoustic filtering and verification audit

Neighbor lags follow Eq.8: distance*d + 1…distance*d + n. Eq.5 uses a different
starting-lag convention; the code explicitly chooses Eq.8. The monic inverse
mixing FIR M is fitted by per-channel prediction-error ridge regression. Lambda
is the smallest nonzero design singular value, as stated on supplied PDF page 4,
not its square. This is an empirical heuristic, not universally optimal ridge.

Default n=4, radius=1,d=3 samples on five synchronized simulated channels is a
small specialization, not reproduction of the field n=150,d=500,8 kHz setup.
Healthy training sets mean/std and coefficients; these remain frozen during
monitoring. M is stable FIR; M^-1 needs the separately computed companion
spectral radius <1. Prefix and unit-rescaling tests check causal transformation
and saved normalization.

Independent training, healthy verification and monitoring runs are used. The
paper's pointwise 99% correlation bound is reported. An added max-over-pairs/lags
circular-shift screen uses 99 surrogates under stationarity. Neither screen
certifies independence or physically correct recovery.

| Sources; 12 replications each | Mean raw → residual maximum cross-correlation | Mean known-source correlation | Screen rejections |
|---|---:|---:|---:|
| Independent white | 0.357 → 0.055 | 0.9987 | 2/12 |
| Correlated | 0.890 → 0.948 | 0.9987 | 12/12 |
| Independent colored | 0.456 → 0.054 | 0.7235 | 2/12 |
| Dependent but uncorrelated | 0.353 → 0.054 | 0.9987 | 2/12 |

The last case uses Gaussian s and (s²−1)/sqrt(2): uncorrelated yet dependent.
Most runs pass, exposing why correlation is not an independence certificate.
Colored sources can yield innovations rather than original waveforms. Rejection
of 2/12 independent runs also means the nominal screen level is not established
finite-sample calibration. These failures remain visible in the app.

Uploads accept finite simultaneous 2–8-channel data. The first half of a
user-declared healthy reference fits the separator, the second half verifies it,
and only later samples are monitored. No physical sampling rate, geometry,
healthy status or source location is inferred from row indices.

## Recorded data acquired and actually used

GPLA-12 v1 is pinned at `afec271f4eb445a976826c6a45485892e5fbab53`; file hashes
and the repository license are retained by the downloader. The advertised v3
folder contains a one-byte placeholder. Actual v1 CSV arrays contain **684 unique
rows × 1,460 values**, 12 conditions of 57 rows. The conditions are gas-leak
pressure/noise/microphone settings, not healthy/leak labels. Synchronized release
pairing and independent acquisition sessions are unavailable.

Every GPLA row, unused in fitting, is placed into an imposed five-channel stable
mixing model with recorded Meng blower interference. The separator was fitted
on independent simulated white-source measurements. Artificial common sample
indices are imposed; no GPLA physical rate is assumed. Known recorded references
are used only for evaluation. Median correlation improves **0.360→0.981**, with
median scale-invariant SDR improvement **22.4 dB**. Per-row metrics, condition
summaries and hashes are saved.

This is recorded-waveform transfer under controlled mixing, not unknown field
mixing, leak-event accuracy or localization. Matching training/test propagation
is optimistic; the reference waveform itself may already contain recording noise.
684 rows are not claimed to be 684 independent physical acquisitions. GPLA has
no healthy class, so it is not used for fabricated leak/no-leak classification.

NGPOD/MKTCN is a closer pressure/flow lead, but the linked raw arrays remain
absent. OLGA multiphase data require author/provider access. Neither was acquired
or used for training; water data are not substituted as gas field validation.

## D9: context refinement in the actual gas runtime

`backend/context_refinement.py` executes these stages:

1. Base ARX forecast: 40 healthy runs, seeds 101000–101039.
2. Eq.8 context: mean/std/max/min of the previous 30 samples of two command
   channels; DBSCAN centers fitted to healthy training contexts.
3. Separate refinement: 32 healthy runs, seeds 102000–102031. Groups of k chunks
   minimize L1 context-to-center cost with one-to-one assignment (Eq.12). For k≤6,
   exhaustive permutations solve the exact assignment objective; this solver
   is not labelled the Hungarian algorithm.
4. Eq.15 augmented lag features and base forecasts feed ridge refiners. Candidate
   clustering is selected using 12 healthy validation runs (103000–103011).
   Three clusters are selected; residual means/scales use refinement data.
5. Eq.18 live L1 routing uses only past commands. The score averages 30 whitened
   forecast errors. Independent healthy calibration uses 24 runs (104000–104023)
   before any test generation.

Ridge replaces CatBoost; numeric past command context replaces daily outage/text
features. This is a framework adaptation. It needs 60 consecutive valid readings;
the four older methods retain 32. Missing values and cadence gaps reset windows.
No retrospective operating-mode labels or leak labels enter inference.

First new test cohort, seeds 105000–105065:

| Detector | Leak detections | False events / healthy exposure | Median detected delay |
|---|---:|---:|---:|
| Inventory | 36/36 | 5 / 28.685 h | 92.5 s |
| Original dynamic | 24/36 | 0 / 28.685 h | 246.5 s |
| CNN–LSTM | 0/36 | 2 / 28.685 h | No detections |
| Original hybrid | 24/36 | 0 / 28.685 h | 246.5 s |
| Context refiner | 26/36 | 0 / 26.585 h | 199 s |

Expanded healthy training and longer warm-up prevent an architecture-superiority
claim. Zero false events over 26.585 h does not imply zero population rate;
one-sided 95% Poisson upper bound ≈0.113/hour under representative Poisson exposure.

A matched-data global refiner uses the same base model, training/calibration runs,
features, penalty and warm-up. Both are frozen before seeds 106000–106065:

| Matched-data method | Detections | False events / exposure | Median detected delay |
|---|---:|---:|---:|
| Three-context refiner | 27/36 | 4 / 26.6625 h | 203 s |
| Global refiner | 26/36 | 4 / 26.6625 h | 184 s |

Only one discordant pair favors context; exact two-sided McNemar p=1.0.
**Clustering superiority is not established.** Both exceed the 0.1/hour calibration
budget on this fresh cohort (~0.150/hour). They are not retuned after inspecting
the results. The context method is an experimental fifth detector, not a claimed
replacement for the strongest baseline.

## Reproduction and interview demonstration

From repository root with pinned serving/development dependencies:

```sh
OPENBLAS_NUM_THREADS=1 python research/run_paper_methods.py
OPENBLAS_NUM_THREADS=1 python research/run_context_refinement.py
OPENBLAS_NUM_THREADS=1 python research/run_context_ablation.py
python research/download_gpla.py --data-dir /tmp/pipeguard-gpla
OPENBLAS_NUM_THREADS=1 python research/run_recorded_source_transfer.py --gpla /tmp/pipeguard-gpla --meng /path/to/acquired/meng
python tools/register_context.py
python tools/build_example.py
python -m unittest discover -s tests -v
python tools/verify_benchmark.py
python tools/verify_context_runtime.py
python tools/browser_smoke.py --start-server
```

Research commands deliberately refit/write artifacts; none run at server startup.
`register_context.py` synchronizes the calibrated threshold and hashes after a
deliberate refit. Original results/PDF remain archived separately. The web server
loads frozen artifacts. Novelty and scholarship outcomes are not asserted.

A defensible interview demonstration: show ordinary regression's confounding
bias; show D2's corrected estimate and selected sensors; move D6 excitation to
the wrong location; expose the acoustic correlation-certificate limitation;
finish with D9's matched-data result and the missing field acquisition experiment.
The central research hypothesis is whether information-aware module estimation
adds reliable attribution beyond generic anomaly detection under confounding and
weak excitation. These experiments provide a starting point, not a settled result.
