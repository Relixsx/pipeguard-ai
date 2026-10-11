"""D9 framework transfer: healthy train/refine/validate/calibrate, then fresh test."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from backend.context_refinement import fit_context_refinement
from backend.simulator import simulate
from backend.runtime import Runtime


def alarm_events(scores,threshold,times):
    high=low=0;active=False;last=-1e20;events=[]
    for score,t in zip(scores,times):
        if not np.isfinite(score):high=low=0;active=False;continue
        if score>threshold:
            high+=1;low=0
            if high>=3 and not active:
                active=True
                if t-last>=120:events.append(float(t));last=t
        else:
            high=0;low+=1
            if low>=5:active=False
    return events


def calibration(score_arrays):
    all_scores=np.concatenate([x[np.isfinite(x)] for x in score_arrays])
    hours=sum(np.isfinite(x).sum()*5/3600 for x in score_arrays)
    for threshold in np.unique(np.r_[np.quantile(all_scores,np.linspace(.5,1,251)),np.nextafter(all_scores.max(),np.inf)]):
        events=sum(len(alarm_events(x,threshold,np.arange(len(x))*5)) for x in score_arrays)
        if events/hours<=.1:return float(threshold),{'events':events,'hours':hours,'alarms_per_hour':events/hours}
    raise RuntimeError('No calibration threshold')


def healthy_cohort(seed,count):
    return [simulate(f'context_healthy_{seed+i}',seed+i,
                     'normal_transient' if i%4>=2 else 'healthy',
                     regime='shifted' if i%2 else 'in_range') for i in range(count)]


def run(model_dir,output):
    started=time.perf_counter()
    initial=healthy_cohort(101000,40);refinement=healthy_cohort(102000,32)
    validation=healthy_cohort(103000,12)
    print('Healthy cohorts generated',flush=True)
    model,fit=fit_context_refinement([r['x'] for r in initial],[r['x'] for r in refinement],[r['x'] for r in validation])
    print('Refiner fitted',json.dumps(fit),flush=True)
    cal=healthy_cohort(104000,24)
    scores=[model.score_series(r['x'],r['time_s']) for r in cal]
    threshold,calibration_info=calibration(scores)
    model_dir.mkdir(parents=True,exist_ok=True)
    path=model_dir/'context_refinement.npz';model.save(path)
    protocol={'initial_healthy_runs':40,'refinement_healthy_runs':32,'validation_healthy_runs':12,
              'calibration_healthy_runs':24,'test_seed_start':105000,'test_runs':66,
              'selected_threshold':threshold,'fit':fit,'calibration':calibration_info,
              'training_seed_ranges':[[101000,101039],[102000,102031],[103000,103011],[104000,104023]],
              'test_scope':'Fresh seeds from the same gas simulator; exploratory method transfer',
              'comparability':'Older four baselines retain their original weights/thresholds; refiner has expanded healthy training. No architecture-superiority claim.',
              'threshold_fitted_before_test_generation':True,
              'model_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    output.mkdir(parents=True,exist_ok=True)
    (output/'context_refinement_protocol.json').write_text(json.dumps(protocol,indent=2))
    # No test runs were generated or evaluated before the above freeze.
    cases=[]
    for regime in ['in_range','shifted']:
        for kind in ['leak','gradual_leak']:
            for fraction in [.02,.05,.1]:
                for location in range(3):cases.append((kind,fraction,location,regime))
    for kind in ['healthy','normal_transient','weak_excitation','sensor_bias','missing_sensor']:
        for i in range(6):cases.append((kind,0.,i%3,'shifted' if i%2 else 'in_range'))
    runtime=Runtime(model_dir)
    rows=[]
    for i,(kind,fraction,location,regime) in enumerate(cases):
        r=simulate(f'paper_context_test_{i}',105000+i,kind,fraction,location,regime)
        predictions=runtime.score_series(r['x'],r['time_s'])
        ctx=model.score_series(r['x'],r['time_s'])
        methods={k:(np.array([np.nan if s is None else s for s in v['scores']]),v['event_times_s'])
                 for k,v in predictions['models'].items() if k!='context_refinement'}
        methods['context_refinement']=(ctx,alarm_events(ctx,threshold,r['time_s']))
        for name,(values,events) in methods.items():
            ordinary=kind in ('healthy','normal_transient','weak_excitation','leak','gradual_leak')
            healthy=(r['y']==0)&np.isfinite(values) if ordinary else np.zeros(len(values),bool)
            false=sum(bool(healthy[int(t/5)]) for t in events)
            post=[t for t in events if r['onset_s'] is not None and t>=r['onset_s']]
            rows.append({'method':name,'run_id':r['run_id'],'seed':r['seed'],'kind':kind,
                         'leak_case':fraction>0,'detected':bool(post) if fraction else None,
                         'delay_s':post[0]-r['onset_s'] if post else None,
                         'false_events':false,'healthy_hours':float(healthy.sum()*5/3600),
                         'bias_alerted':bool(events) if kind=='sensor_bias' else None,
                         'alarm_times_s':events})
        if i%12==0:print('Fresh test',i+1,'/66',flush=True)
    summaries=[]
    for name in sorted(set(x['method'] for x in rows)):
        block=[r for r in rows if r['method']==name];leaks=[r for r in block if r['leak_case']]
        hours=sum(r['healthy_hours'] for r in block);false=sum(r['false_events'] for r in block)
        delays=[r['delay_s'] for r in leaks if r['detected']]
        summaries.append({'method':name,'leak_events':len(leaks),'detected_events':sum(r['detected'] for r in leaks),
                          'false_alarm_events':false,'healthy_hours':hours,'false_alarms_per_hour':false/hours,
                          'median_detected_delay_s':float(np.median(delays)) if delays else None,
                          'sensor_bias_alerted_runs':sum(r['bias_alerted'] is True for r in block)})
    result={'protocol':protocol,'summaries':summaries,'run_results':rows,
            'wall_seconds':round(time.perf_counter()-started,3)}
    (output/'context_refinement_results.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(summaries,indent=2),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--model-dir',type=Path,default=ROOT/'deployment/models')
    parser.add_argument('--output',type=Path,default=ROOT/'deployment/evidence')
    args=parser.parse_args();run(args.model_dir,args.output)
