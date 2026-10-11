"""Executable specializations of the supplied papers, with explicit contracts.

D2: Algorithm 2 (two-stage projection), Proposition 3 (network immersion),
    Proposition 4 (parallel-path and loop conditions).
D4: Theorem 2 / Proposition 3 (target rank increment / vertex-disjoint paths).
D6: Section 5-7 (disconnecting sets and target excitation).
D8: Equations 8, 13-17 (delayed spatial FIR, automatic ridge, verification).

The numerical rank tests concern the supplied finite model. Graph tests concern
generic structural identifiability under the paper's independent-parameter,
known-topology assumptions. Neither certifies an arbitrary physical pipeline.
"""
from collections import deque
from itertools import combinations
from statistics import NormalDist

import numpy as np


def path(edges, start, end, removed=(), excluded_edge=None):
    """Return a witness path, or None; start=end means a nonempty loop."""
    removed = set(removed)
    if start in removed or end in removed:
        return None
    adjacency = {}
    for a, b in edges:
        if a not in removed and b not in removed and (a, b) != excluded_edge:
            adjacency.setdefault(a, []).append(b)
    queue = deque([(start, [start])])
    visited = {start}
    while queue:
        node, route = queue.popleft()
        for nxt in sorted(adjacency.get(node, [])):
            if nxt == end:
                return route + [nxt]
            if nxt not in visited:
                visited.add(nxt)
                queue.append((nxt, route + [nxt]))
    return None


def preservation_check(edges, source, output, inputs):
    """D2 Proposition 4, sufficient graph conditions for target preservation."""
    inputs = set(inputs)
    alternative = path(edges, source, output, inputs - {source}, (source, output))
    loop = path(edges, output, output, inputs)
    valid_membership = source in inputs and output not in inputs
    return {"target_preserved": valid_membership and alternative is None and loop is None,
            "unblocked_parallel_path": alternative, "unblocked_output_loop": loop,
            "inputs": sorted(inputs), "reference": "D2 Proposition 4"}


def select_inputs(edges, nodes, source, output, available=None):
    """Minimum-cardinality measured set meeting D2's path/loop conditions.

    Exhaustive selection is deliberately limited to the small laboratory graph;
    this is not advertised as an efficient large-network search algorithm.
    """
    available = set(nodes if available is None else available) - {output}
    if source not in available or len(available) > 12:
        raise ValueError("Target input must be measured; selector supports at most 12 candidates")
    others = sorted(available - {source})
    for size in range(len(others) + 1):
        for subset in combinations(others, size):
            selected = [source, *subset]
            if preservation_check(edges, source, output, selected)["target_preserved"]:
                return selected
    return None


def immerse_frequency(G, retained):
    """D2 Proposition 3: exact Schur complement at one frequency.

    G[output,input]. Eliminate hidden nodes, then normalize away self-loops.
    Stability/well-posedness and nonzero denominators are caller assumptions.
    """
    G = np.asarray(G, dtype=complex)
    retained = list(retained)
    hidden = [i for i in range(len(G)) if i not in retained]
    A = G[np.ix_(retained, retained)].copy()
    if hidden:
        A += G[np.ix_(retained, hidden)] @ np.linalg.solve(
            np.eye(len(hidden)) - G[np.ix_(hidden, hidden)], G[np.ix_(hidden, retained)])
    denominator = 1 - np.diag(A)
    if np.any(np.abs(denominator) < 1e-10):
        raise ValueError("Immersed network is not well posed at this frequency")
    np.fill_diagonal(A, 0)
    return A / denominator[:, None]


def disjoint_paths(edges, sources, targets):
    """Vertex-capacitated max-flow: every vertex, including endpoints, has cap 1."""
    vertices = sorted(set(v for e in edges for v in e) | set(sources) | set(targets))
    index = {v: i for i, v in enumerate(vertices)}
    n = len(vertices)
    capacity = np.zeros((2*n+2, 2*n+2), dtype=int)
    origin, sink = 2*n, 2*n+1
    for v, i in index.items():
        capacity[2*i, 2*i+1] = 1
    for a, b in edges:
        capacity[2*index[a]+1, 2*index[b]] = n+1
    for v in sources:
        capacity[origin, 2*index[v]] = 1
    for v in targets:
        capacity[2*index[v]+1, sink] = 1
    residual = capacity.copy()
    count = 0
    while True:
        parent = {origin: None}
        queue = deque([origin])
        while queue and sink not in parent:
            a = queue.popleft()
            for b in np.flatnonzero(residual[a] > 0):
                b = int(b)
                if b not in parent:
                    parent[b] = a
                    queue.append(b)
        if sink not in parent:
            break
        b = sink
        while parent[b] is not None:
            a = parent[b]
            residual[a, b] -= 1
            residual[b, a] += 1
            b = a
        count += 1
    return count


def structural_target(edges, excitations, predictor_inputs, target_input):
    full = disjoint_paths(edges, excitations, predictor_inputs)
    nuisance = disjoint_paths(edges, excitations,
                              [v for v in predictor_inputs if v != target_input])
    return {"full_row_generic_rank": full, "nuisance_generic_rank": nuisance,
            "target_generically_identifiable": full > nuisance,
            "whole_row_generically_identifiable": full == len(predictor_inputs),
            "reference": "D4 Proposition 3; independent modules and admissible excitations"}


def disconnecting_set(edges, source, other_inputs, candidates):
    candidates = sorted(set(candidates) - {source})
    for size in range(len(candidates)+1):
        for chosen in combinations(candidates, size):
            if all(path(edges, source, v, chosen) is None for v in other_inputs):
                return list(chosen)
    return None


def target_excitation(edges, target, other_inputs, input_nodes, excitation_nodes):
    """D6 Definitions 4-5 / Corollary 6, for the declared input-generating graph."""
    separating = disconnecting_set(edges, target, other_inputs, input_nodes)
    if separating is None:
        return {"sufficient_condition": False, "reason": "No admissible disconnecting set"}
    remainder = sorted(set(other_inputs) - set(separating))
    excluded = [x for x in excitation_nodes
                if any(path(edges, x, v, separating) is not None for v in remainder)]
    eligible = sorted(set(excitation_nodes) - set(excluded))
    needed = [target, *separating]
    rank = disjoint_paths(edges, eligible, needed)
    return {"disconnecting_set": separating, "remainder": remainder,
            "excluded_excitations": excluded, "eligible_excitations": eligible,
            "required_disjoint_paths": len(needed), "available_disjoint_paths": rank,
            "sufficient_condition": rank == len(needed),
            "reference": "D6 Definitions 4-5 and Corollary 6",
            "assumptions": "Known graph, independent modules, independent persistently exciting external signals"}


def finite_target_rank(X, target_columns, rtol=1e-9):
    """Finite FIR specialization: target uniqueness despite nuisance ambiguity.

    This is a finite linear-algebra result, not D6's unrestricted transfer theorem.
    """
    X = np.asarray(X, float)
    scale = np.linalg.norm(X, axis=0)
    X = X / np.maximum(scale, 1e-15)
    nuisance = np.delete(X, target_columns, axis=1)
    threshold = rtol * max(float(np.linalg.svd(X, compute_uv=False)[0]), 1.)
    full = int(np.sum(np.linalg.svd(X, compute_uv=False) > threshold))
    rest = int(np.sum(np.linalg.svd(nuisance, compute_uv=False) > threshold)) if nuisance.shape[1] else 0
    return {"rank": full, "columns": X.shape[1], "nuisance_rank": rest,
            "target_rank_increment": full-rest,
            "target_identifiable_in_finite_model": full-rest == len(target_columns),
            "whole_design_identifiable": full == X.shape[1]}


def lagged_references(references, endpoints, order):
    """Causal external instruments only; no realized feedback command substitutes."""
    return np.column_stack([references[endpoints-k] for k in range(1, order+1)])


def hac_interval(X, residual, target=0, max_lag=12, confidence=.95):
    """Asymptotic HAC sandwich for a fixed finite model, not exact finite coverage."""
    bread = np.linalg.pinv(X.T @ X, rcond=1e-10)
    moments = X * residual[:, None]
    meat = moments.T @ moments
    for k in range(1, min(max_lag, len(X)-1)+1):
        gamma = moments[k:].T @ moments[:-k]
        meat += (1-k/(max_lag+1)) * (gamma+gamma.T)
    covariance = bread @ meat @ bread
    return float(NormalDist().inv_cdf((1+confidence)/2) *
                 np.sqrt(max(0., covariance[target, target])))


def identify_two_stage(values, references, inputs, output, projection_order=12):
    """D2 Algorithm 2, FIR specialization with a one-sample target-module delay.

    Step 1: project the selected delayed node measurements onto causal histories
    of known external references. Step 2: fit output to these projected inputs.
    No noise model is used (per D2 Remark 1); valid independent references,
    sufficient excitation, correct one-tap modules, and retained paths are required.
    """
    values, references = np.asarray(values, float), np.asarray(references, float)
    if values.ndim != 2 or references.ndim != 2 or len(values) != len(references):
        raise ValueError("Node measurements and external references must be synchronized matrices")
    if not np.isfinite(values).all() or not np.isfinite(references).all():
        raise ValueError("Identification requires finite synchronized data")
    endpoints = np.arange(projection_order, len(values))
    if len(endpoints) < 128:
        raise ValueError("At least 128 usable samples are required")
    X = values[endpoints-1][:, inputs]
    y = values[endpoints, output]
    Z = lagged_references(references, endpoints, projection_order)
    X, y, Z = X-X.mean(axis=0), y-y.mean(), Z-Z.mean(axis=0)
    projection = np.linalg.lstsq(Z, X, rcond=1e-10)[0]
    Xp = Z @ projection
    finite = finite_target_rank(Xp, [0])
    target_residual = Xp[:, 0] - Xp[:, 1:] @ np.linalg.lstsq(Xp[:, 1:], Xp[:, 0], rcond=1e-10)[0]
    target_energy = float(np.mean(target_residual**2))
    signal_energy = float(np.mean(X[:, 0]**2))
    # Practical guard declared before evaluation; not a D6 theorem threshold.
    reference_energy = float(np.mean(Z**2))
    adequate = (finite["target_identifiable_in_finite_model"] and target_energy >= .01
                and reference_energy >= .01)
    coef = np.linalg.lstsq(Xp, y, rcond=1e-10)[0]
    # IV structural residual; projected regressors provide valid score moments.
    residual = y-X @ coef
    half = hac_interval(Xp, residual)
    ols = np.linalg.lstsq(X, y, rcond=1e-10)[0]
    return {"coefficients": coef.tolist(), "target_estimate": float(coef[0]),
            "target_interval_95": [float(coef[0]-half), float(coef[0]+half)],
            "naive_ols_estimate": float(ols[0]), "finite_information": finite,
            "target_projected_energy": target_energy,
            "reference_energy": reference_energy,
            "practical_guard": "Normalized laboratory units: reference and conditional target energy >= 0.01; not a theorem threshold",
            "projected_energy_fraction": target_energy/max(signal_energy, 1e-15),
            "attribution_supported": adequate,
            "attribution_status": "supported_in_declared_model" if adequate else "insufficient_external_excitation",
            "confidence_scope": "Asymptotic HAC interval under valid-reference and correct-model assumptions",
            "usable_samples": len(endpoints), "projection_order": projection_order,
            "method": "D2 Algorithm 2: external-reference projection then module estimation"}


class SpatialFIRSeparator:
    """D8 inverse-FIR/PEM source estimator, using Eq.8's delayed design.

    The paper's Eq.5 and Eq.8 use different starting-lag conventions; this code
    explicitly uses Eq.8: neighbor lags distance*d + 1 through distance*d + n.
    lambda = minimum nonzero singular value of Phi, as stated on PDF page 4.
    Normalize with healthy training statistics; keep parameters frozen at test.
    """
    def __init__(self, order=4, radius=1, adjacent_delay=3, ridge_mode="paper"):
        self.order, self.radius, self.adjacent_delay = order, radius, adjacent_delay
        self.ridge_mode = ridge_mode

    def _design(self, z, j):
        labels = [(i, k+self.adjacent_delay*abs(i-j))
                  for i in range(max(0, j-self.radius), min(z.shape[1], j+self.radius+1))
                  for k in range(1, self.order+1)]
        endpoints = np.arange(self.warmup, len(z))
        return np.column_stack([z[endpoints-lag, i] for i, lag in labels]), labels

    def fit(self, measurements):
        x = np.asarray(measurements, float)
        if x.ndim != 2 or not 2 <= x.shape[1] <= 8 or not np.isfinite(x).all():
            raise ValueError("Source separation needs 2-8 finite synchronized channels")
        if np.any(np.ptp(x, axis=0) == 0):
            raise ValueError("Every healthy channel needs observed variation; constant channels cannot identify mixing")
        self.mean = x.mean(axis=0)
        self.scale = np.maximum(x.std(axis=0), 1e-12)
        self.warmup = self.order+self.radius*self.adjacent_delay
        if len(x)-self.warmup < 4*self.order*(2*self.radius+1):
            raise ValueError("Too few samples for the spatial FIR parameter count")
        z = (x-self.mean)/self.scale
        self.coef, self.labels, self.penalties = [], [], []
        for j in range(x.shape[1]):
            X, labels = self._design(z, j)
            y = z[self.warmup:, j]
            u, s, vt = np.linalg.svd(X, full_matrices=False)
            nonzero = s[s > max(s[0]*1e-10, 1e-12)]
            penalty = float(nonzero[-1]) if self.ridge_mode == "paper" and len(nonzero) else 0.
            factor = s/(s*s+penalty) if penalty else np.divide(1., s, out=np.zeros_like(s), where=s>max(s[0]*1e-10, 1e-12))
            self.coef.append(vt.T @ (factor*(u.T @ y)))
            self.labels.append(labels)
            self.penalties.append(penalty)
        # M is an FIR, hence stable. M^-1 stability requires this AR companion test.
        channels, lag_count = x.shape[1], self.warmup
        companion = np.zeros((channels*lag_count, channels*lag_count))
        for j, (labels, coeff) in enumerate(zip(self.labels, self.coef)):
            for (i, lag), value in zip(labels, coeff):
                companion[j, (lag-1)*channels+i] = value
        if lag_count > 1:
            companion[channels:, :-channels] = np.eye(channels*(lag_count-1))
        self.inverse_spectral_radius = float(np.max(np.abs(np.linalg.eigvals(companion))))
        return self

    def transform(self, measurements):
        x = np.asarray(measurements, float)
        if x.ndim != 2 or x.shape[1] != len(self.mean) or not np.isfinite(x).all():
            raise ValueError("All synchronized channels must match the fitted separator")
        z = (x-self.mean)/self.scale
        if len(z) <= self.warmup:
            return np.empty((0, z.shape[1]))
        result = z[self.warmup:].copy()
        for j in range(z.shape[1]):
            X, _ = self._design(z, j)
            result[:, j] -= X @ self.coef[j]
        return result

    def details(self):
        return {"order": self.order, "radius": self.radius, "adjacent_delay_samples": self.adjacent_delay,
                "warmup_samples": self.warmup, "ridge_rule": self.ridge_mode,
                "regularization_by_channel": self.penalties,
                "inverse_spectral_radius": self.inverse_spectral_radius,
                "stable_inverse": self.inverse_spectral_radius < 1,
                "reference": "D8 Equations 8, 13-16; healthy-frozen monitoring adaptation"}


def cross_correlations(x, max_lag=8):
    x = np.asarray(x, float)
    x = (x-x.mean(axis=0))/np.maximum(x.std(axis=0), 1e-12)
    pairs = []
    for a in range(x.shape[1]):
        for b in range(a+1, x.shape[1]):
            corr = []
            for lag in range(-max_lag, max_lag+1):
                if lag < 0:
                    value = np.mean(x[-lag:, a]*x[:lag, b])
                elif lag > 0:
                    value = np.mean(x[:-lag, a]*x[lag:, b])
                else:
                    value = np.mean(x[:, a]*x[:, b])
                corr.append(float(value))
            pairs.append({"channels": [a, b], "correlations": corr})
    return pairs


def verify_sources(sources, max_lag=8, surrogates=99, seed=7123, alpha=.01):
    """Held-out dependence screen, not an independence/correctness certificate.

    D8 pointwise 99% correlation bounds are shown for comparison. The added
    max-over-pairs/lags circular-shift surrogate screen preserves each channel's
    autocorrelation and addresses multiplicity, under stationarity assumptions.
    Failure to reject correlation does not prove independence or true recovery.
    """
    x = np.asarray(sources, float)
    if len(x) < 128:
        raise ValueError("At least 128 held-out source samples are required")
    pairs = cross_correlations(x, max_lag)
    maximum = max(abs(v) for pair in pairs for v in pair["correlations"])
    pointwise = 2.576/np.sqrt(len(x))
    rng = np.random.default_rng(seed)
    null = []
    low = max(max_lag+1, len(x)//8)
    for _ in range(surrogates):
        shifted = np.column_stack([x[:, 0], *[np.roll(x[:, j], int(rng.integers(low, len(x)-low)))
                                               for j in range(1, x.shape[1])]])
        null.append(max(abs(v) for pair in cross_correlations(shifted, max_lag)
                        for v in pair["correlations"]))
    p = (1+sum(v >= maximum for v in null))/(surrogates+1)
    return {"pairs": pairs, "lags": list(range(-max_lag, max_lag+1)),
            "maximum_absolute_correlation": maximum, "paper_pointwise_99_bound": float(pointwise),
            "paper_pointwise_within_bounds": bool(maximum <= pointwise),
            "familywise_surrogate_p": p, "familywise_rejected": p <= alpha,
            "surrogates": surrogates, "alpha": alpha,
            "interpretation": "Dependence screen only; uncorrelated sources can still be dependent"}
