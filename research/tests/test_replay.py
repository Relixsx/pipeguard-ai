import unittest
from pathlib import Path
import numpy as np
import torch
from pipeguard.monitor import ResearchMonitor
from pipeguard.simulator import simulate
from reevaluate_saved import load_scorer
from run_inventory_check import inventory_score

ROOT=Path(__file__).resolve().parents[1]

class ReplayTests(unittest.TestCase):
    @unittest.skipUnless((ROOT/'results/benchmark_results.json').exists(),'Trained checkpoint required')
    def test_saved_models_match_streaming_scores(self):
        torch.set_num_threads(2)
        run=simulate('replay_check',99001)
        for name in ['dynamic_residual','cnn_lstm','hybrid','mass_inventory']:
            monitor=ResearchMonitor(ROOT/'results',name,11,'narrow')
            fn=inventory_score if name=='mass_inventory' else load_scorer(ROOT/'results',name,11)[0]
            scores,_,_=fn(run)
            readings=[monitor.push('a',int(t),x) for t,x in zip(run['time_s'][:50],run['x'][:50])]
            actual=[r['score'] for r in readings if 'score' in r]
            np.testing.assert_allclose(actual,scores[:len(actual)],rtol=2e-5,atol=1e-6)

if __name__=='__main__':unittest.main()
