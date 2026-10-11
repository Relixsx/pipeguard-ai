"""Replay the frozen fifth detector against both archived fresh test cohorts."""
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.runtime import Runtime
from backend.simulator import simulate

runtime = Runtime(ROOT / 'deployment/models')
cases = [(kind, fraction, location, regime)
         for regime in ['in_range', 'shifted'] for kind in ['leak', 'gradual_leak']
         for fraction in [.02, .05, .1] for location in range(3)]
cases += [(kind, 0., i % 3, 'shifted' if i % 2 else 'in_range')
          for kind in ['healthy', 'normal_transient', 'weak_excitation', 'sensor_bias', 'missing_sensor']
          for i in range(6)]
started = time.perf_counter()
for filename, seed_start in [('context_refinement_results.json', 105000), ('context_ablation_results.json', 106000)]:
    archive = json.loads((ROOT / 'deployment/evidence' / filename).read_text())
    rows = {r['seed']: r for r in archive['run_results'] if r['method'] == 'context_refinement'}
    for i, (kind, fraction, location, regime) in enumerate(cases):
        r = simulate(f'parity_{i}', seed_start+i, kind, fraction, location, regime)
        actual = runtime.score_series(r['x'], r['time_s'])['models']['context_refinement']
        assert actual['event_times_s'] == rows[seed_start+i]['alarm_times_s'], (filename, i)
result = {'runs': 132, 'context_alarm_timestamps_match': True,
          'scope': 'Frozen serving integration parity; not new validation data',
          'seconds': round(time.perf_counter()-started, 3)}
(ROOT / 'deployment/evidence/context-runtime-parity.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
