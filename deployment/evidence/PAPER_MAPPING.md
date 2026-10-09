# From the ten papers to this implementation

The source papers were reviewed in full (119 pages), alongside 45 pages of
supporting project/study material. The page-reading ledger is in `evidence/`.
The following distinguishes implemented approximations from theorem-level
claims and work that needs additional data.

| Paper | Transfer to PipeGuard | Executed implementation and limit |
|---|---|---|
| D1: Basic methods for consistent module estimates, 2013 | Predict a measured variable from justified network inputs and retain a frozen healthy reference. | Regularized multivariate ARX forecasting, fitted on healthy runs. This is not a proof of consistency with feedback and colored measurement noise. |
| D2: Predictor input selection, 2016 | Preserve relevant paths and feedback relationships when selecting sensors. | All three pressure nodes, both boundary flows and known controls are retained. A three-feature input ablation is evaluated separately. No field graph-reduction proof is claimed. |
| D3: Identifiability of linear dynamic networks, 2018 | Establish that the experiment can distinguish the desired model before interpreting parameters. | Topology and simulation assumptions are declared. No physical field-module identifiability claim is made. |
| D4: Single-module identifiability, 2018 | A target can be recoverable even if a larger parameterization is ambiguous. | Target-block partial-regression diagnostics supplement the full-design condition number. They describe the chosen regression, not an unknown physical module. |
| D5: Handling confounding variables, 2017 | Shared disturbances and feedback can bias component inference. | Simulator exposes realized feedback commands and separate external excitation. Actual control commands are not called exogenous instruments. Alerts remain generic anomalies. |
| D6: Data-informativity conditions, supplied version/published 2026 | Structural possibility differs from information in the observed data; target validity does not validate the full predictor. | Healthy/weak-excitation design diagnostics are reported. Neither an SVD cutoff nor a neural reconstruction score is presented as D6's theorem. Detection remains active when attribution is unsupported. |
| D7: Blind acoustic source separation, 2016 | Separate mixed acoustic sources using appropriate synchronized measurements and system structure. | Single-channel gas-audio pilot executed. Full BSS is blocked by missing synchronized channels; temporal AR forecasting is not labelled source separation. |
| D8: Real-time pipeline field study, 2025 | Spatial FIR filtering and regularization may help machinery/leak source separation. | Input normalization, saved preprocessing, causal scoring and explicit runtime boundaries adopted. The spatial inverse-FIR experiment cannot be reproduced from unpaired mono clips. Mutual information is not used as a p-value. |
| D9: Cluster refinement and outage context, 2025 | Operating context can change normal behaviour and forecast validity. | Additional healthy operating regimes calibrated before another fresh test cohort. No future event labels or retrospective regime tags enter model features. |
| D10: Batched decomposition, 2025 | Training transformations must be available at the time a forecast/alarm is issued. | No future-looking smoothing or decomposition. Every scored window ends at the current reading, and saved preprocessing is shared by offline and streamed inference. |

## Five concrete solution paths

1. **Dynamic-network residual monitoring.** Start with the fitted predictor and
   the observed input graph. It improved leak detection over the neural-only
   benchmark, but operating changes caused excess alarms. Advance to an estimator
   suited to the graph/noise/feedback assumptions when real telemetry is obtained.
2. **Mass-inventory monitoring with operating coverage.** Include gas stored in
   the pipe when interpreting inlet/outlet imbalance. This was the strongest
   executed simulation follow-up, with known geometry and isothermal assumptions.
   Field evaluation must quantify geometry, compressibility, thermal and flow-meter uncertainty.
3. **Neural residual hybrid.** Train CNN-LSTM on standardized dynamic innovations
   and keep the dynamic channel active. The executed hybrid did not improve the
   simpler dynamic model, so retaining it needs further independent evidence.
4. **Information-aware diagnosis and sensor design.** Specify the target module,
   identify necessary sensors/excitation and attach justified uncertainty before
   attributing a cause. A diagnostic panel and separate attribution status are
   implemented; calibrated component confidence requires a validated estimator
   and independently labelled fault mechanisms.
5. **Multichannel acoustic monitoring.** Use gas recordings as a separate sensing
   branch and test D7/D8-compatible separation against an unchanged classifier
   on held-out interference/sessions. The pilot is executed; full spatial
   separation/localization requires simultaneous channels and session metadata.

## Narrow research hypothesis to discuss with Dr. Dankers

Can a target-specific finite-data information diagnostic, paired with an
appropriate dynamic-network estimator and frozen healthy reference, reduce
unsupported component attribution under weak excitation while preserving
leak detection through independent residual channels?

This is a proposed experiment, not an established contribution. Dynamic-network
fault diagnosis already has prior art (Shi, Fonken and Van den Hof, 2024,
DOI 10.1016/j.ifacol.2024.08.559). Originality would need a new and demonstrated
diagnostic, uncertainty result, acquisition design or validated robustness gain.
