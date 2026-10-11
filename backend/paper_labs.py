"""Paper-method laboratories. Inference sees measurements, references and graph.

Generator truth is confined to the final evaluation fields. Experiments are
declared linear-network/acoustic simulations, distinct from the gas SCADA demo.
"""
from statistics import NormalDist

import numpy as np

from .paper_methods import (SpatialFIRSeparator, cross_correlations, finite_target_rank,
                            identify_two_stage, immerse_frequency, preservation_check,
                            select_inputs, structural_target, target_excitation, verify_sources)

NODES = ["w1", "w2", "w3", "w4"]
EDGES = [("w1", "w2"), ("w1", "w3"), ("w3", "w2"),
         ("w2", "w1"), ("w2", "w4"), ("w4", "w2")]
EXTERNAL_EDGES = [("r1", "w1"), ("r3", "w3"), ("r4", "w4")]


def network_data(seed, samples=4096, excitation=1., confounding=.9,
                 fault=False, control_change=False):
    rng = np.random.default_rng(seed)
    burn = 256
    r = excitation*rng.normal(size=(samples+burn, 4))
    r[:, 1] = 0  # No omitted directly injected output excitation in Algorithm 2.
    noise = rng.normal(scale=.25, size=(samples+burn, 4))
    common = np.zeros(samples+burn)
    for t in range(1, len(common)):
        common[t] = .8*common[t-1]+rng.normal(scale=.6)
    noise[:, 0] += confounding*common
    noise[:, 1] += confounding*common
    w = np.zeros_like(r)
    gains = np.full(samples+burn, .65)
    onset = burn+samples//2
    if fault:
        gains[onset:] = .25
    for t in range(1, len(w)):
        A = np.array([[0, .15, 0, 0], [gains[t], 0, .35, .2],
                      [.4, 0, 0, 0], [0, .25, 0, 0]])
        if control_change and t >= onset:
            A[0, 1], A[3, 1] = .3, .1
        w[t] = A @ w[t-1]+r[t]+noise[t]
    return {"measurements": w[burn:], "external_references": r[burn:],
            "truth_target_gain": gains[burn:], "fault_onset_sample": samples//2 if fault else None}


def network_lab(seed=91001, scenario="confounded", sensor_mode="selected"):
    chosen = select_inputs(EDGES, NODES, "w1", "w2")
    inputs = chosen if sensor_mode == "selected" else ["w1"]
    indices = [NODES.index(v) for v in inputs]
    guard = preservation_check(EDGES, "w1", "w2", inputs)
    excitation = .005 if scenario == "weak_excitation" else 1.
    confounding = 0. if scenario == "independent_noise" else .9
    reference = network_data(seed, excitation=excitation, confounding=confounding)
    observed = network_data(seed+1, excitation=excitation, confounding=confounding,
                            fault=scenario == "target_change", control_change=scenario == "control_change")
    baseline = identify_two_stage(reference["measurements"], reference["external_references"], indices, 1)
    full = identify_two_stage(observed["measurements"][-2048:], observed["external_references"][-2048:], [0, 2, 3], 1)
    omitted = identify_two_stage(observed["measurements"][-2048:], observed["external_references"][-2048:], [0], 1)
    structural = structural_target(EDGES+EXTERNAL_EDGES, ["r1", "r3", "r4"], inputs, "w1")
    endpoints = list(range(1024, len(observed["measurements"])+1, 256))
    z_guard = NormalDist().inv_cdf(1-.01/(2*len(endpoints)))
    records = []
    for end in endpoints:
        fit = identify_two_stage(observed["measurements"][end-1024:end],
                                 observed["external_references"][end-1024:end], indices, 1)
        supported = (guard["target_preserved"] and structural["target_generically_identifiable"]
                     and fit["attribution_supported"] and baseline["attribution_supported"])
        base_se = (baseline["target_interval_95"][1]-baseline["target_estimate"])/1.959963984540054
        se = (fit["target_interval_95"][1]-fit["target_estimate"])/1.959963984540054
        delta = fit["target_estimate"]-baseline["target_estimate"]
        changed = bool(supported and abs(delta)>z_guard*np.sqrt(base_se**2+se**2))
        records.append({"endpoint_sample": end-1, "estimate": fit["target_estimate"],
                        "interval_95": fit["target_interval_95"], "supported": supported,
                        "target_change_flag": changed, "reference_energy": fit["reference_energy"]})
    # Frequency-domain immersion check uses the declared simulation graph for
    # explanatory evaluation only, never the estimator's parameter values.
    A = np.array([[0,.15,0,0],[.65,0,.35,.2],[.4,0,0,0],[0,.25,0,0]])
    frequency = np.linspace(.05, np.pi, 64)
    curves = []
    for omega in frequency:
        G = A*np.exp(-1j*omega)
        retained = [1, *indices]
        compressed = immerse_frequency(G, retained)
        curves.append(float(abs(compressed[0, 1]-G[1, 0])))
    stride = 16
    return {"lab": "network", "scenario": scenario, "seed": seed,
            "scope": "Known-topology linear dynamic-network experiment; not field pipeline diagnosis",
            "papers": ["D1", "D2", "D3", "D4", "D5", "D6"],
            "graph": {"nodes": NODES, "edges": EDGES, "external_edges": EXTERNAL_EDGES,
                      "target": ["w1", "w2"], "selected_inputs": chosen, "used_inputs": inputs},
            "preservation": guard, "structural": structural,
            "baseline": baseline, "full_two_stage": full, "omitted_two_stage": omitted,
            "comparison_sample_scope":"Last 2048 observed samples; baseline is a separate healthy run",
            "rolling": records, "sample_indices": list(range(0, len(observed["measurements"]), stride)),
            "measurements": observed["measurements"][::stride].tolist(),
            "evaluation": {"known_healthy_target": .65, "fault_target": .25 if scenario=="target_change" else None,
                           "fault_onset_sample": observed["fault_onset_sample"],
                           "max_target_immersion_error": max(curves)},
            "diagnosis_guard": "Known preserved target, generic structural rank, external reference energy, projected target information and approximate HAC change interval",
            "change_interval_scope": "Bonferroni guard over displayed windows using asymptotic HAC intervals; finite-sample coverage not guaranteed"}


def information_lab(excitation="target_and_separator"):
    # D6 Section 6: P21, P32, P23, P34; independently parameterized output modules.
    internal = [("u1","u2"),("u2","u3"),("u3","u2"),("u4","u3")]
    active = ["x1", "x2"] if excitation=="target_and_separator" else ["x1", "x3"]
    edges = internal+[(x,"u"+x[1:]) for x in active]
    condition = target_excitation(edges, "u1", ["u2","u3","u4"],
                                  ["u1","u2","u3","u4"], active)
    structural = structural_target(edges, active, ["u1","u2","u3","u4"], "u1")
    spectra, rank_tests = [], []
    needed = [0]+[int(v[1:])-1 for v in condition["disconnecting_set"]]
    eligible = [int(x[1:])-1 for x in condition["eligible_excitations"]]
    for omega in np.linspace(.01, np.pi, 64):
        delay = np.exp(-1j*omega)
        P = np.zeros((4,4), complex)
        P[1,0],P[2,1],P[1,2],P[2,3] = .4*delay,.6*delay,.2*delay,.3*delay
        T = np.linalg.inv(np.eye(4)-P)
        F = T[np.ix_(needed,eligible)]
        spectra.append(float(max(0.,np.linalg.eigvalsh(F@F.conj().T)[0])))
        all_transfer = T[:, [int(x[1:])-1 for x in active]]
        full = np.linalg.matrix_rank(all_transfer, tol=1e-9)
        other = np.linalg.matrix_rank(all_transfer[1:], tol=1e-9)
        rank_tests.append(bool(full>other))
    # D4 Figure 1: three incoming modules, only two excitations. G41 unique.
    d4_edges = [("r1","w1"),("r2","w2"),("w2","w3"),
                ("w1","w4"),("w2","w4"),("w3","w4")]
    d4 = structural_target(d4_edges,["r1","r2"],["w1","w2","w3"],"w1")
    # A finite regression analog is shown separately from the transfer theorem.
    rng = np.random.default_rng(91119)
    a,b = rng.normal(size=(2,1024))
    finite = finite_target_rank(np.column_stack([a,b,2*b]),[0])
    return {"lab":"information", "scope":"Reproduced graph/rank examples; known transfer-function specialization",
            "excitation":excitation, "active_excitations":active, "d6":condition,
            "structural":structural, "d4_paper_example":d4, "finite_regression_analogue":finite,
            "minimum_target_spectral_eigenvalue":min(spectra),
            "target_transfer_rank_condition_all_grid_points":all(rank_tests),
            "spectral_grid":spectra, "frequency_grid_count":64,
            "paper_references":["D4 Theorem 2 / Proposition 3 / Figure 1", "D6 Sections 5-7, Theorem 4 and Corollary 6"],
            "assumptions":"Independent external white excitation, declared stable input network, independent module parameters; finite frequency checks supplement the graph condition"}


def acoustic_data(seed, samples=4096, scenario="independent", channels=5):
    rng = np.random.default_rng(seed)
    burn = 128
    sources = rng.normal(size=(samples+burn, channels))
    if scenario=="correlated":
        sources[:,1] = .9*sources[:,0]+.3*sources[:,1]
    elif scenario=="dependent_uncorrelated":
        sources[:,1] = (sources[:,0]**2-1)/np.sqrt(2)
    elif scenario=="colored":
        for t in range(1,len(sources)):
            sources[t] += .7*sources[t-1]
    onset = burn+samples//2
    if scenario=="source_change":
        sources[onset:,channels//2] *= 3.
    w = np.zeros_like(sources)
    # Known stable local propagation/reflection system G=(I-A(q))^-1.
    for t in range(4,len(w)):
        w[t] = .25*w[t-1]+sources[t]
        w[t,1:] += .28*w[t-4,:-1]
        w[t,:-1] += .28*w[t-4,1:]
    return {"measurements":w[burn:],"sources":sources[burn:],
            "onset_sample":samples//2 if scenario=="source_change" else None}


def source_fidelity(estimated, truth):
    estimated, truth = np.asarray(estimated), np.asarray(truth)
    correlations = []
    for j in range(estimated.shape[1]):
        correlations.append(float(np.corrcoef(estimated[:,j],truth[:,j])[0,1]))
    return correlations


def summarize_acoustics(separator, validation, observed, seed):
    sources = separator.transform(observed)
    checks = verify_sources(separator.transform(validation),seed=seed)
    raw_check = cross_correlations((validation-separator.mean)/separator.scale)
    before = max(abs(v) for p in raw_check for v in p["correlations"])
    scaled = (observed-separator.mean)/separator.scale
    bins = 128
    energies, raw_energies, centers = [],[],[]
    for start in range(0,len(sources)-bins+1,bins):
        energies.append(np.sqrt(np.mean(sources[start:start+bins]**2,axis=0)).tolist())
        raw_energies.append(np.sqrt(np.mean(scaled[separator.warmup+start:separator.warmup+start+bins]**2,axis=0)).tolist())
        centers.append(separator.warmup+start+bins-1)
    return {"separator":separator.details(),"verification":checks,
            "raw_maximum_absolute_correlation":before,
            "rms_sample_indices":centers,"source_rms":energies,"measured_rms":raw_energies,
            "dependence_screen_passed":separator.inverse_spectral_radius<1 and not checks["familywise_rejected"],
            "support_scope":"Assumptions and held-out dependence screen; not proof of physical source recovery",
            "channels":observed.shape[1]}


def acoustic_lab(seed=92001, scenario="source_change"):
    training_scenario = scenario if scenario in ("correlated","colored","dependent_uncorrelated") else "independent"
    train = acoustic_data(seed,scenario=training_scenario)
    validation = acoustic_data(seed+1,scenario=training_scenario)
    test = acoustic_data(seed+2,scenario=scenario)
    separator = SpatialFIRSeparator().fit(train["measurements"])
    result = summarize_acoustics(separator,validation["measurements"],test["measurements"],seed+3)
    residual = separator.transform(test["measurements"])
    result.update(lab="acoustic",scenario=scenario,seed=seed,papers=["D7","D8"],
                  scope="Five synchronized simulated channels with declared mixing; no field DAS data",
                  evaluation={"known_source_correlations":source_fidelity(residual,test["sources"][separator.warmup:]),
                              "injected_source_index":2 if scenario=="source_change" else None,
                              "source_change_onset_sample":test["onset_sample"],
                              "true_sources_dependent":scenario in ("correlated","dependent_uncorrelated"),
                              "true_sources_white":scenario not in ("colored",),
                              "note":"Known sources used only for evaluation, not fitting or verification"})
    return result


def analyze_acoustic(measurements, healthy_samples, order=4, radius=1, adjacent_delay=3, seed=93101):
    """Fit on first healthy half, verify on its second half, then monitor future."""
    x = np.asarray(measurements, float)
    train_end = healthy_samples//2
    separator = SpatialFIRSeparator(order,radius,adjacent_delay).fit(x[:train_end])
    result = summarize_acoustics(separator,x[train_end:healthy_samples],x[healthy_samples:],seed)
    result["rms_sample_indices"] = [healthy_samples+i for i in result["rms_sample_indices"]]
    result.update(lab="acoustic_upload",source="user-provided synchronized channels",
                  scope="Healthy-reference assumption supplied by uploader; physical source interpretation unvalidated",
                  healthy_reference_samples=healthy_samples,monitoring_start_sample=healthy_samples,
                  papers=["D7","D8"])
    return result
