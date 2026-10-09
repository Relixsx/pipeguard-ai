"""Re-score frozen weights and thresholds, including corrected event accounting."""
from pathlib import Path
import argparse,json,csv
import numpy as np
import torch
from pipeguard.models import Standardizer,DynamicPredictor,CNNLSTMAutoencoder,windows,autoencoder_scores
from pipeguard.simulator import load_cohort
from run_benchmark import evaluate,clean_for_forecast

def load_scorer(result_dir,name,seed=11):
    root=Path(result_dir)
    result=json.loads((root/'benchmark_results.json').read_text())
    with np.load(root/'models/dynamic_reference.npz') as p:
        scaler=Standardizer();scaler.mean=p['scaler_mean'];scaler.scale=p['scaler_scale']
        dynamic=DynamicPredictor(int(p['lag']),float(p['alpha']))
        dynamic.coef=p['coef'];dynamic.whitener=p['whitener'];dynamic.residual_mean=p['residual_mean']
    key='dynamic_residual' if name=='dynamic_residual' else 'original_three_features' if name=='original_three_features' else f'{name}_{seed}'
    calibration=result['calibration'][key]
    model=None
    if name!='dynamic_residual':
        filename='original_three_features.pt' if name=='original_three_features' else f'{name}_{seed}.pt'
        checkpoint=torch.load(root/'models'/filename,map_location='cpu',weights_only=True)
        model=CNNLSTMAutoencoder(checkpoint['input_dim']);model.load_state_dict(checkpoint['state_dict']);model.eval()
    def score_fn(run):
        clean,invalid=clean_for_forecast(run['x'])
        z=scaler.transform(clean)
        residual=dynamic.residuals(z)
        endpoints=np.arange(dynamic.lag+29,len(z))
        mask=np.array([invalid[max(0,e-29-dynamic.lag):e+1].any() for e in endpoints])
        si_score=np.mean(windows(residual)**2,axis=(1,2))
        scores=si_score
        if model is not None:
            x=residual if name=='hybrid' else z[dynamic.lag:,[0,4,5]] if name=='original_three_features' else z[dynamic.lag:]
            scores=autoencoder_scores(model,windows(x))
            if name=='hybrid':scores=np.maximum(scores/calibration['neural_scale'],si_score/calibration['si_scale'])
        return scores,endpoints,mask
    return score_fn,calibration['threshold']

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data-dir',default='data/simulation');parser.add_argument('--result-dir',default='results')
    args=parser.parse_args();torch.set_num_threads(3)
    root=Path(args.result_dir);result=json.loads((root/'benchmark_results.json').read_text())
    _,runs=load_cohort(args.data_dir);test=[r for r in runs if r['split']=='test']
    summaries,details=[],[]
    for old in result['summaries']:
        fn,threshold=load_scorer(root,old['model'],old['seed'])
        summary,rows=evaluate(old['model'],old['seed'],test,fn,threshold)
        summaries.append(summary);details.extend(rows)
    result.update(summaries=summaries,run_results=details,
                  event_definition='Only a new alarm event at or after physical leak onset counts. Alarms already active beforehand receive no detection credit.')
    (root/'benchmark_results.json').write_text(json.dumps(result,indent=2))
    with (root/'benchmark_summary.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=summaries[0].keys());writer.writeheader();writer.writerows(summaries)
    print(json.dumps([{'model':r['model'],'seed':r['seed'],'detected':r['detected_events'],'preexisting':r['leaks_with_preexisting_alarm']} for r in summaries]),flush=True)

if __name__=='__main__':main()
