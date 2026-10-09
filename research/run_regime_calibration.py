"""Independent healthy-regime calibration and a third, fresh-seed test cohort.

This is an exploratory follow-up to the narrow-regime failure. Model weights
are unchanged. All threshold choices use healthy runs only, before generating
the final cohort. It is still an in-simulator experiment, not field validation.
"""
from pathlib import Path
import argparse,json
import numpy as np
import torch
from pipeguard.simulator import simulate,load_cohort
from reevaluate_saved import load_scorer
from run_inventory_check import inventory_score
from run_benchmark import calibrate_threshold,evaluate

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-dir',default='data/simulation')
    parser.add_argument('--result-dir',default='results')
    args=parser.parse_args();torch.set_num_threads(3)
    root=Path(args.result_dir)
    _,runs=load_cohort(args.data_dir)
    calibration=[r for r in runs if r['split']=='calibration']
    additions=[]
    for kind,regime,count in [('healthy','shifted',12),('normal_transient','shifted',12),('weak_excitation','in_range',6)]:
        for i in range(count):
            additions.append((kind,regime))
    protocol={'healthy_additional_seed_start':61000,'extra_healthy_runs':len(additions),
              'fixed_methods':['dynamic_residual','cnn_lstm_11','hybrid_11','mass_inventory_nominal'],
              'test_seed_start':81000,'test_runs':66,'alarm_budget_per_healthy_hour':.1,
              'weight_updates':False,'purpose':'Check calibration coverage, after initial operating-regime failures.',
              'scope':'Exploratory follow-up with independent seeds from the same simulator.'}
    (root/'regime_protocol.json').write_text(json.dumps(protocol,indent=2))
    for i,(kind,regime) in enumerate(additions):
        calibration.append(simulate(f'wide_cal_{i:03d}_{kind}',61000+i,kind,regime=regime))
    methods={}
    for name in ['dynamic_residual','cnn_lstm','hybrid']:
        fn,old_threshold=load_scorer(root,name,11)
        methods[name]=(fn,old_threshold)
    inventory_cal=json.loads((root/'inventory_calibration.json').read_text())
    methods['mass_inventory']=(inventory_score,inventory_cal['1.0']['threshold'])
    # Fit thresholds before any final test readings or labels are generated.
    calibration_results={}
    score_cal={}
    for name,(fn,old_threshold) in methods.items():
        score_cal[name]=[fn(r)[0] for r in calibration]
        threshold,meta=calibrate_threshold(score_cal[name])
        calibration_results[name]=meta|{'threshold':threshold,'old_threshold':old_threshold}
    (root/'regime_calibration.json').write_text(json.dumps(calibration_results,indent=2))
    scenarios=[]
    for regime in ['in_range','shifted']:
        for kind in ['leak','gradual_leak']:
            for fraction in [.02,.05,.10]:
                for location in range(3):scenarios.append((kind,fraction,location,regime))
    for kind in ['healthy','normal_transient','weak_excitation','sensor_bias','missing_sensor']:
        for i in range(6):scenarios.append((kind,0.,i%3,'shifted' if i%2 else 'in_range'))
    test=[simulate(f'final_{i:03d}_{kind}',81000+i,kind,fraction,location,regime)
          for i,(kind,fraction,location,regime) in enumerate(scenarios)]
    summaries,details=[],[]
    for name,(fn,old_threshold) in methods.items():
        # Same final runs and scores, with narrow vs broad healthy calibration.
        cached={r['run_id']:fn(r) for r in test}
        for mode,threshold in [('narrow',old_threshold),('broad',calibration_results[name]['threshold'])]:
            summary,rows=evaluate(name,11 if name in ('cnn_lstm','hybrid') else 0,test,
                                  lambda r:tuple(np.copy(a) for a in cached[r['run_id']]),threshold)
            summary['calibration_coverage']=mode;summaries.append(summary)
            details.extend([row|{'calibration_coverage':mode} for row in rows])
    result={'summaries':summaries,'run_results':details,'calibration':calibration_results,'protocol':protocol,
            'test_seed_range':[81000,81065],'event_definition':'Only new alarms after leak onset; no credit for preexisting alarms.'}
    (root/'regime_results.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(summaries),flush=True)

if __name__=='__main__':main()
