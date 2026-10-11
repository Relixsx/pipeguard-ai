"""Reproduce paper-method experiments; no API labels are used by the estimators."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from backend.paper_labs import (acoustic_data,acoustic_lab,information_lab,network_data,network_lab,
                                source_fidelity)
from backend.paper_methods import SpatialFIRSeparator,identify_two_stage,verify_sources


def run(output, replications=80):
    started=time.perf_counter()
    output.mkdir(parents=True,exist_ok=True)
    network=[]
    for scenario,excitation,confounding in [('independent_noise',1.,0.),('confounded',1.,.9),('weak_excitation',.005,.9)]:
        fits=[]
        for i in range(replications):
            observed=network_data(94000+i,excitation=excitation,confounding=confounding)
            fit=identify_two_stage(observed['measurements'],observed['external_references'],[0,2,3],1)
            fits.append(fit)
        two=np.array([f['target_estimate'] for f in fits]);ols=np.array([f['naive_ols_estimate'] for f in fits])
        network.append({'scenario':scenario,'replications':replications,'true_target_gain':.65,
                        'two_stage_mean':float(two.mean()),'two_stage_bias':float(two.mean()-.65),
                        'two_stage_rmse':float(np.sqrt(np.mean((two-.65)**2))),
                        'naive_ols_mean':float(ols.mean()),'naive_ols_bias':float(ols.mean()-.65),
                        'naive_ols_rmse':float(np.sqrt(np.mean((ols-.65)**2))),
                        'hac_interval_coverage':float(np.mean([f['target_interval_95'][0]<=.65<=f['target_interval_95'][1] for f in fits])),
                        'attribution_supported_runs':sum(f['attribution_supported'] for f in fits)})
        print('Network completed',scenario,flush=True)
    acoustic=[]
    for scenario in ['independent','correlated','colored','dependent_uncorrelated']:
        details=[]
        for i in range(12):
            r=acoustic_lab(95000+4*i,scenario)
            details.append({'screen_rejected':r['verification']['familywise_rejected'],
                            'raw_correlation':r['raw_maximum_absolute_correlation'],
                            'residual_correlation':r['verification']['maximum_absolute_correlation'],
                            'mean_known_source_correlation':float(np.mean(r['evaluation']['known_source_correlations']))})
        acoustic.append({'scenario':scenario,'replications':len(details),
                         'dependence_screen_rejections':sum(x['screen_rejected'] for x in details),
                         'mean_raw_cross_correlation':float(np.mean([x['raw_correlation'] for x in details])),
                         'mean_residual_cross_correlation':float(np.mean([x['residual_correlation'] for x in details])),
                         'mean_known_source_correlation':float(np.mean([x['mean_known_source_correlation'] for x in details])),
                         'interpretation':'Uncorrelated-but-dependent and colored-source cases expose limits of a correlation certificate'})
        print('Acoustic completed',scenario,flush=True)
    results={'network':network,'acoustic':acoustic,
             'information_examples':{k:information_lab(k) for k in ['target_and_separator','wrong_location']},
             'protocol':{'network_seeds':[94000,94000+replications-1],'acoustic_seeds':[95000,95044],
                         'network_samples_per_run':4096,'projection_order':12,
                         'acoustic_train_validation_test_independent':True,
                         'source_verification':'Held-out max-correlation circular-shift screen, 99 surrogates, alpha=.01',
                         'scope':'Declared linear-network and synchronized acoustic simulations; no field validation'},
             'wall_seconds':round(time.perf_counter()-started,3)}
    (output/'paper_methods_results.json').write_text(json.dumps(results,indent=2,allow_nan=False))
    examples={'network':network_lab(),'acoustic':acoustic_lab(),'information':information_lab()}
    (output/'paper_methods_examples.json').write_text(json.dumps(examples,indent=2,allow_nan=False))
    print(json.dumps(results['network'],indent=2),flush=True)
    return results


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'deployment/evidence')
    parser.add_argument('--replications',type=int,default=80)
    args=parser.parse_args();run(args.output,args.replications)
