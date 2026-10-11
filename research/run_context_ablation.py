"""Matched-data D9 ablation: frozen three-context model versus global refinement.

All training/calibration seeds match the original context experiment. No test
labels select thresholds. A new 66-run cohort is generated only after freezing
the global comparator, so the original exploratory test remains archived.
"""
import hashlib
import json
from math import comb, log
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.context_refinement import ContextRefinement, forecast_design, ridge
from backend.simulator import simulate
from research.run_context_refinement import alarm_events, calibration, healthy_cohort


def run():
    context = ContextRefinement.load(ROOT / 'deployment/models/context_refinement.npz')
    original = json.loads((ROOT / 'deployment/evidence/context_refinement_protocol.json').read_text())
    refinement = healthy_cohort(102000, 32)
    xs, ys = [], []
    for run in refinement:
        z = (run['x'] - context.mean) / context.scale
        X = forecast_design(z)[28:]
        xs.append(np.column_stack([X[:, :-1], 2*(X @ context.base_coef), np.ones(len(X))]))
        ys.append(z[30:, :6])
    coef = ridge(np.concatenate(xs), np.concatenate(ys), 10.)
    global_model = ContextRefinement(context.mean, context.scale, context.base_coef,
                                    context.context_mean, context.context_scale, np.zeros((1, 8)),
                                    coef[None], np.zeros((1, 6)), np.eye(6)[None])
    errors = np.concatenate([global_model.residuals(r['x'])[0] for r in refinement])
    global_model.residual_means[0] = errors.mean(axis=0)
    eig, vec = np.linalg.eigh(np.cov(errors, rowvar=False))
    eig = np.maximum(eig, max(eig.max()*1e-4, 1e-10))
    global_model.whiteners[0] = vec @ np.diag(1/np.sqrt(eig)) @ vec.T
    cal = healthy_cohort(104000, 24)
    threshold, cal_info = calibration([global_model.score_series(r['x'], r['time_s']) for r in cal])
    model_path = ROOT / 'deployment/models/global_refinement_ablation.npz'
    global_model.save(model_path)
    protocol = {'scope': 'Matched-data ablation on new simulator seeds; no field validation',
                'test_seed_start': 106000, 'test_runs': 66, 'minimum_readings_both_models': 60,
                'same_initial_runs': 40, 'same_refinement_runs': 32, 'same_calibration_runs': 24,
                'same_base_forecast': True, 'same_ridge_penalty': 10, 'same_augmented_features': True,
                'context_clusters': len(context.centers), 'global_clusters': 1,
                'context_threshold': original['selected_threshold'], 'global_threshold': threshold,
                'global_calibration': cal_info, 'thresholds_frozen_before_test_generation': True,
                'global_model_sha256': hashlib.sha256(model_path.read_bytes()).hexdigest(),
                'context_model_sha256': original['model_sha256']}
    dest = ROOT / 'deployment/evidence/context_ablation_results.json'
    dest.write_text(json.dumps({'protocol': protocol}, indent=2))
    cases = [(kind, fraction, location, regime)
             for regime in ['in_range', 'shifted'] for kind in ['leak', 'gradual_leak']
             for fraction in [.02, .05, .1] for location in range(3)]
    cases += [(kind, 0., i % 3, 'shifted' if i % 2 else 'in_range')
              for kind in ['healthy', 'normal_transient', 'weak_excitation', 'sensor_bias', 'missing_sensor']
              for i in range(6)]
    rows = []
    for i, (kind, fraction, location, regime) in enumerate(cases):
        r = simulate(f'context_ablation_{i}', 106000+i, kind, fraction, location, regime)
        for name, model, cut in [('context_refinement', context, original['selected_threshold']),
                                 ('global_refinement', global_model, threshold)]:
            scores = model.score_series(r['x'], r['time_s'])
            events = alarm_events(scores, cut, r['time_s'])
            ordinary = kind in ('healthy', 'normal_transient', 'weak_excitation', 'leak', 'gradual_leak')
            healthy = (r['y'] == 0) & np.isfinite(scores) if ordinary else np.zeros(len(scores), bool)
            post = [t for t in events if r['onset_s'] is not None and t >= r['onset_s']]
            rows.append({'method': name, 'seed': r['seed'], 'kind': kind, 'regime': regime,
                         'leak_fraction': fraction, 'location': location, 'leak_case': fraction > 0,
                         'detected': bool(post) if fraction else None,
                         'delay_s': post[0]-r['onset_s'] if post else None,
                         'false_events': sum(bool(healthy[int(t/5)]) for t in events),
                         'healthy_hours': float(healthy.sum()*5/3600), 'alarm_times_s': events})
        if i % 12 == 0:
            print('Matched-data test', i+1, '/66', flush=True)
    summaries = []
    for name in ['context_refinement', 'global_refinement']:
        block = [r for r in rows if r['method'] == name]
        leaks = [r for r in block if r['leak_case']]
        hours = sum(r['healthy_hours'] for r in block)
        false = sum(r['false_events'] for r in block)
        delays = [r['delay_s'] for r in leaks if r['detected']]
        summaries.append({'method': name, 'leak_events': len(leaks),
                          'detected_events': sum(r['detected'] for r in leaks),
                          'false_alarm_events': false, 'healthy_hours': hours,
                          'false_alarms_per_hour': false/hours,
                          'zero_false_event_poisson_upper_95_per_hour': -log(.05)/hours if false == 0 else None,
                          'median_detected_delay_s': float(np.median(delays)) if delays else None})
    leak_rows = [r for r in rows if r['leak_case']]
    pairs = [(leak_rows[i]['detected'], leak_rows[i+1]['detected']) for i in range(0, len(leak_rows), 2)]
    context_only = sum(a and not b for a, b in pairs)
    global_only = sum(b and not a for a, b in pairs)
    n = context_only+global_only
    paired_p = min(1., 2*sum(comb(n, j) for j in range(min(context_only, global_only)+1))/2**n) if n else 1.
    result = {'protocol': protocol, 'summaries': summaries, 'run_results': rows,
              'paired_leak_detection': {'context_only': context_only, 'global_only': global_only,
                                       'two_sided_exact_mcnemar_p': paired_p,
                                       'interpretation': 'Paired fixed cohort comparison; does not establish field superiority'}}
    dest.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({k: v for k, v in result.items() if k != 'run_results'}, indent=2), flush=True)


if __name__ == '__main__':
    run()
