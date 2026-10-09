"""Generate, train, calibrate, then evaluate without tuning on test runs."""
from pathlib import Path
import argparse, copy, hashlib, json, time
import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
from scipy.stats import binomtest
from pipeguard.simulator import generate_cohort, load_cohort, FEATURES
from pipeguard.models import (Standardizer, DynamicPredictor, arx_design, windows,
                             train_autoencoder, autoencoder_scores)

def clean_for_forecast(x):
    result = x.copy()
    invalid = ~np.isfinite(result).all(axis=1)
    # Causal last observation carried forward for numeric execution ONLY.
    # Affected windows are suppressed, with a separate data quality alert.
    for t in range(len(result)):
        missing = ~np.isfinite(result[t])
        if missing.any():
            if t == 0:
                raise ValueError('No initial value for missing sensor')
            result[t,missing] = result[t-1,missing]
    return result,invalid

def alarm_logic(scores,threshold,sample_s=5,persistence=3,clearance=5,cooldown_s=120):
    alerts = np.zeros(len(scores),dtype=bool)
    events = []
    high,low,active,last_event = 0,0,False,-10**9
    for i,score in enumerate(scores):
        if not np.isfinite(score):
            high,low,active = 0,0,False
            continue
        if score > threshold:
            high += 1
            low = 0
            if high >= persistence and not active:
                active = True
                if (i-last_event)*sample_s >= cooldown_s:
                    events.append(i)
                    last_event = i
        else:
            high = 0
            low += 1
            if low >= clearance:
                active = False
        alerts[i] = active
    return alerts,events

def calibrate_threshold(score_arrays,target_far=.1,sample_s=5):
    total = np.concatenate(score_arrays)
    hours = sum(np.isfinite(s).sum()*sample_s/3600 for s in score_arrays)
    quantiles = np.r_[np.linspace(.50,.97,100),np.linspace(.97,1.0,151)]
    candidates = np.unique(np.quantile(total,quantiles))
    candidates = np.r_[candidates,np.nextafter(total.max(),np.inf)]
    # Lowest candidate meeting the pre-declared HEALTHY calibration alarm budget.
    for threshold in candidates:
        count = sum(len(alarm_logic(s,threshold,sample_s)[1]) for s in score_arrays)
        if count/hours <= target_far:
            return float(threshold),{'events':count,'healthy_hours':hours,'alarms_per_hour':count/hours,
                                     'target_alarms_per_hour':target_far,'candidate_count':len(candidates)}
    raise RuntimeError('Calibration failed')

def target_information(z_runs,lag=3):
    X = np.concatenate([arx_design(z,lag)[0] for z in z_runs])
    # Partial-regression diagnostic for a pressure-midpoint lag block.
    target_indices = [1+8*k for k in range(lag)]
    target = X[:,target_indices]
    nuisance = np.delete(X,target_indices,axis=1)
    residualized = target-nuisance@np.linalg.lstsq(nuisance,target,rcond=1e-9)[0]
    sv = np.linalg.svd(X,compute_uv=False)
    eigenvalues = np.linalg.eigvalsh(residualized.T@residualized/len(X))
    return {'full_design_condition':float(sv[0]/sv[-1]),
            'target_block_conditional_eigenvalues':eigenvalues.tolist(),
            'rows':len(X),
            'interpretation':'Finite-data regression diagnostic; not proof of physical-module identifiability.'}

def evaluate(name,seed,test_runs,score_fn,threshold):
    rows,all_y,all_alert,all_score = [],[],[],[]
    normal_kinds = {'healthy','normal_transient','weak_excitation'}
    for run in test_runs:
        score,endpoints,invalid = score_fn(run)
        score[invalid] = np.nan
        alerts,event_indices = alarm_logic(score,threshold,run['sample_s'])
        times = run['time_s'][endpoints]
        y = run['y'][endpoints]
        finite = np.isfinite(score)
        leak_case = run['kind'] in ('leak','gradual_leak')
        is_normal = run['kind'] in normal_kinds or leak_case
        healthy = (y==0)&finite if is_normal else np.zeros(len(y),dtype=bool)
        false_events = sum(bool(healthy[i]) for i in event_indices)
        delay = None
        preexisting_alarm = False
        if leak_case:
            onset_index = np.searchsorted(times,run['onset_s'])
            preexisting_alarm = bool(onset_index>0 and alerts[onset_index-1])
            # A false alarm active before onset is not credited as leak detection.
            detected = [i for i in event_indices if times[i]>=run['onset_s'] and finite[i]]
            if detected:
                delay = float(times[detected[0]]-run['onset_s'])
        row = {key:run[key] for key in ['run_id','kind','regime','leak_fraction','location','onset_s']}
        row.update(model=name,seed=seed,detected=(delay is not None) if leak_case else None,
                   delay_s=delay,preexisting_alarm_at_onset=preexisting_alarm,
                   false_events=false_events,healthy_hours=float(healthy.sum()*run['sample_s']/3600),
                   alarm_fraction=float(alerts[finite].mean()) if finite.any() else None,
                   anomaly_alert=bool(alerts.any()),suppressed_windows=int(invalid.sum()),
                   observed_endpoints=int(finite.sum()),alarm_times_s=[int(times[i]) for i in event_indices])
        rows.append(row)
        if is_normal:
            all_y.append(y[finite]);all_alert.append(alerts[finite]);all_score.append(score[finite])
    leaks = [r for r in rows if r['detected'] is not None]
    detected = sum(r['detected'] for r in leaks)
    interval = binomtest(detected,len(leaks)).proportion_ci()
    y,alert,score = np.concatenate(all_y),np.concatenate(all_alert),np.concatenate(all_score)
    precision,recall,f1,_ = precision_recall_fscore_support(y,alert,average='binary',zero_division=0)
    false_events = sum(r['false_events'] for r in rows)
    hours = sum(r['healthy_hours'] for r in rows)
    delays = [r['delay_s'] for r in leaks if r['detected']]
    result = {'model':name,'seed':seed,'leak_events':len(leaks),'detected_events':detected,
              'event_recall':detected/len(leaks),'event_recall_95pct_ci':[interval.low,interval.high],
              'leaks_with_preexisting_alarm':sum(r['preexisting_alarm_at_onset'] for r in leaks),
              'false_alarm_events':false_events,'healthy_hours':hours,'false_alarms_per_hour':false_events/hours,
              'median_detected_delay_s':float(np.median(delays)) if delays else None,
              'p90_detected_delay_s':float(np.quantile(delays,.9)) if delays else None,
              'point_precision':float(precision),'point_recall':float(recall),'point_f1':float(f1),
              'average_precision':float(average_precision_score(y,score)),
              'roc_auc':float(roc_auc_score(y,score)),
              'sensor_bias_alerted_runs':sum(r['anomaly_alert'] for r in rows if r['kind']=='sensor_bias'),
              'missing_sensor_suppressed_windows':sum(r['suppressed_windows'] for r in rows if r['kind']=='missing_sensor')}
    return result,rows

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir',default='results')
    parser.add_argument('--data-dir',default='data/simulation')
    parser.add_argument('--epochs',type=int,default=16)
    parser.add_argument('--seeds',type=int,nargs='+',default=[11,23,37])
    parser.add_argument('--threads',type=int,default=3)
    parser.add_argument('--skip-generation',action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    output = Path(args.output_dir)
    (output/'models').mkdir(parents=True,exist_ok=True)
    if not args.skip_generation:
        generate_cohort(args.data_dir)
    manifest,runs = load_cohort(args.data_dir)
    by_split = {s:[r for r in runs if r['split']==s] for s in ['train','validation','calibration','test']}
    train,val,cal,test = [by_split[s] for s in ['train','validation','calibration','test']]
    scaler = Standardizer().fit([r['x'] for r in train])
    ztrain,zval = [[scaler.transform(r['x']) for r in group] for group in [train,val]]
    selection = []
    best,best_model = np.inf,None
    for lag in [1,2,3]:
        for alpha in [.01,1.0,100.0]:
            dynamic = DynamicPredictor(lag,alpha).fit(ztrain)
            # Select by unwhitened healthy prediction error, not leak labels or test data.
            losses = []
            for z in zval:
                X,Y = arx_design(z,lag)
                losses.append(np.mean((Y-X@dynamic.coef)**2))
            loss = float(np.mean(losses))
            selection.append({'lag':lag,'alpha':alpha,'validation_prediction_mse':loss})
            if loss < best:
                best,best_model = loss,dynamic
    dynamic = best_model
    lag = dynamic.lag
    np.savez(output/'models/dynamic_reference.npz',coef=dynamic.coef,
             residual_mean=dynamic.residual_mean,whitener=dynamic.whitener,
             scaler_mean=scaler.mean,scaler_scale=scaler.scale,lag=lag,alpha=dynamic.alpha)
    info = {'training':target_information(ztrain,lag),
            'weak_excitation_test_diagnostic':target_information([scaler.transform(r['x']) for r in test if r['kind']=='weak_excitation'],lag),
            'note':'Test diagnostic does not alter models, thresholds, or anomaly detection.'}
    arrays = {}
    for group_name,group in by_split.items():
        arrays[group_name] = []
        for run in group:
            clean,invalid = clean_for_forecast(run['x'])
            z = scaler.transform(clean)
            residual = dynamic.residuals(z)
            endpoints = np.arange(lag+29,len(z))
            # Any missing sensor in the prediction lags or monitored window suppresses scoring.
            invalid_windows = np.array([invalid[max(0,end-29-lag):end+1].any() for end in endpoints])
            arrays[group_name].append({'z':z,'residual':residual,'endpoints':endpoints,'invalid':invalid_windows})
    lookups = {r['run_id']:a for group in by_split for r,a in zip(by_split[group],arrays[group])}
    def si_score(run):
        a = lookups[run['run_id']]
        score = np.mean(windows(a['residual'])**2,axis=(1,2))
        return score,a['endpoints'],a['invalid']
    si_cal = [si_score(r)[0] for r in cal]
    si_threshold,si_calibration = calibrate_threshold(si_cal)
    # Neural residual scaling is fitted on validation healthy scores.
    si_scale = float(np.quantile(np.concatenate([si_score(r)[0] for r in val]),.95))
    summaries,run_results,training_logs,calibrations = [],[],{},{}
    summary,rows = evaluate('dynamic_residual',0,test,si_score,si_threshold)
    summaries.append(summary);run_results.extend(rows)
    calibrations['dynamic_residual'] = dict(threshold=si_threshold,**si_calibration)
    print('SI reference selected:',lag,dynamic.alpha,summary,flush=True)
    settings = {'schema':FEATURES,'sequence_length':30,'sample_s':5,'seeds':args.seeds,
                'epochs_limit':args.epochs,'threshold_target_far':.1,'alarm_persistence':3,
                'clearance':5,'cooldown_s':120,'arx_selection':selection,'selected_lag':lag,
                'selected_alpha':dynamic.alpha,'split_run_counts':{k:len(v) for k,v in by_split.items()},
                'source_commit':'f3acfde210161a396cb1835bcabd2a413ba20e78',
                'baseline_note':'Same reference architecture, retrained; expanded input dimension 8 for equal-context comparison.'}
    (output/'protocol.json').write_text(json.dumps(settings,indent=2))
    for seed in args.seeds:
        for name,mode,columns in [('cnn_lstm','raw',None),('hybrid','residual',None)]:
            def train_windows(group,stride):
                rows = [a['z'][lag:] if mode=='raw' else a['residual'] for a in arrays[group]]
                return np.concatenate([windows(x,stride=stride) for x in rows])
            training,validation = train_windows('train',6),train_windows('validation',6)
            started = time.perf_counter()
            model,history = train_autoencoder(training,validation,training.shape[-1],seed,args.epochs)
            elapsed = time.perf_counter()-started
            torch.save({'state_dict':model.state_dict(),'input_dim':training.shape[-1],'seq_len':30,
                        'seed':seed,'schema':FEATURES if mode=='raw' else ['whitened_innovation_'+s for s in FEATURES[:6]]},
                       output/f'models/{name}_{seed}.pt')
            def neural(run):
                a = lookups[run['run_id']]
                x = a['z'][lag:] if mode=='raw' else a['residual']
                return autoencoder_scores(model,windows(x))
            neural_scale = float(np.quantile(np.concatenate([neural(r) for r in val]),.95)) if mode=='residual' else 1.0
            def score_fn(run):
                a = lookups[run['run_id']]
                scores = neural(run)
                if mode=='residual':
                    # Do not disable the dynamic channel when attribution is uncertain.
                    scores = np.maximum(scores/neural_scale,si_score(run)[0]/si_scale)
                return scores,a['endpoints'],a['invalid']
            threshold,calibration = calibrate_threshold([score_fn(r)[0] for r in cal])
            key = f'{name}_{seed}'
            training_logs[key] = {'epochs':history,'training_seconds':elapsed,'train_windows':len(training),
                                  'validation_windows':len(validation),'parameters':sum(p.numel() for p in model.parameters()),
                                  'input_dim':training.shape[-1]}
            calibrations[key] = dict(threshold=threshold,neural_scale=neural_scale,si_scale=si_scale,**calibration)
            summary,rows = evaluate(name,seed,test,score_fn,threshold)
            summaries.append(summary);run_results.extend(rows)
            print('Completed',key,'seconds',round(elapsed,1),json.dumps(summary),flush=True)
            # Incremental checkpoint: results survive a long-running experiment.
            (output/'benchmark_results.json').write_text(json.dumps({'summaries':summaries,'run_results':run_results,
                                                                    'calibration':calibrations,'training':training_logs,
                                                                    'information':info},indent=2))
    # Access ablation: original three feature semantics, only first initialization seed.
    seed = args.seeds[0]
    cols = [0,4,5]
    training = np.concatenate([windows(a['z'][lag:,cols],stride=6) for a in arrays['train']])
    validation = np.concatenate([windows(a['z'][lag:,cols],stride=6) for a in arrays['validation']])
    started = time.perf_counter()
    model,history = train_autoencoder(training,validation,3,seed,args.epochs)
    def score_three(run):
        a = lookups[run['run_id']]
        return autoencoder_scores(model,windows(a['z'][lag:,cols])),a['endpoints'],a['invalid']
    threshold,calibration = calibrate_threshold([score_three(r)[0] for r in cal])
    summary,rows = evaluate('original_three_features',seed,test,score_three,threshold)
    summaries.append(summary);run_results.extend(rows)
    calibrations['original_three_features'] = dict(threshold=threshold,**calibration)
    training_logs['original_three_features'] = {'epochs':history,'training_seconds':time.perf_counter()-started,
                                               'parameters':sum(p.numel() for p in model.parameters()),'input_dim':3,
                                               'train_windows':len(training),'validation_windows':len(validation)}
    torch.save({'state_dict':model.state_dict(),'input_dim':3,'seq_len':30,'seed':seed,
                'schema':[FEATURES[c] for c in cols]},output/'models/original_three_features.pt')
    result = {'summaries':summaries,'run_results':run_results,'calibration':calibrations,'training':training_logs,
              'information':info,'physical_audit':{'max_conservation_error_kg':max(r['max_mass_balance_error_kg'] for r in runs)},
              'scope':'SIMULATION ONLY: independent generated runs; no field performance claim.'}
    (output/'benchmark_results.json').write_text(json.dumps(result,indent=2))
    import csv
    with (output/'benchmark_summary.csv').open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=summaries[0].keys())
        writer.writeheader();writer.writerows(summaries)
    print('BENCHMARK COMPLETE',json.dumps(summaries),flush=True)

if __name__ == '__main__':
    main()
