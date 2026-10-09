"""Small blocked gas-leak recording pilot; no acquisition-session generalization claim."""
from pathlib import Path
import argparse, json, wave, csv
import numpy as np
from scipy.signal import stft, resample_poly
from sklearn.metrics import confusion_matrix, average_precision_score, roc_auc_score
from scipy.stats import binomtest
import torch
from pipeguard.models import Standardizer, windows, train_autoencoder, autoencoder_scores

def extract(path):
    with wave.open(str(path)) as audio:
        if audio.getnchannels()!=1 or audio.getsampwidth()!=2 or audio.getframerate()!=96000:
            raise ValueError('Unexpected audio schema; cannot silently reinterpret recording')
        raw = np.frombuffer(audio.readframes(audio.getnframes()),dtype='<i2')
    signal = raw.astype(float)/32768.0
    # Preserve the full acquired 0-48 kHz bandwidth: no unexplained ultrasonic truncation.
    _,_,spectrum = stft(signal,fs=96000,nperseg=2048,noverlap=1536,boundary=None,padded=False)
    frequency = np.fft.rfftfreq(2048,1/96000)
    edges = np.geomspace(100,48000,17)
    power = np.abs(spectrum)**2
    features = np.stack([np.log10(power[(frequency>=lo)&(frequency<hi)].mean(axis=0)+1e-12)
                         for lo,hi in zip(edges[:-1],edges[1:])],axis=1)
    return features,{'rms':float(np.sqrt(np.mean(signal**2))),
                     'clipped_fraction':float((np.abs(raw.astype(np.int32))>=32767).mean()),
                     'frames':len(features)}

def design(z,lag):
    return np.column_stack([z[lag-k:len(z)-k] for k in range(1,lag+1)]+[np.ones(len(z)-lag)]),z[lag:]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir',default='data/acoustic')
    parser.add_argument('--output-dir',default='results/acoustic')
    parser.add_argument('--epochs',type=int,default=16)
    parser.add_argument('--threads',type=int,default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    output = Path(args.output_dir);output.mkdir(parents=True,exist_ok=True)
    manifest = json.loads((Path(args.data_dir)/'acquisition_manifest.json').read_text())
    clips = []
    for item in manifest['recordings']:
        group = int(Path(item['filename']).stem.split('-')[0])
        split = 'train' if group<=14 else 'validation' if group<=19 else 'test'
        features,audit = extract(Path(args.data_dir)/item['category']/item['filename'])
        clips.append(item | {'features':features,'audit':audit,'group':group,'split':split,
                             'label':int(item['category']!='blower')})
    healthy_train = [r for r in clips if r['split']=='train' and r['label']==0]
    healthy_val = [r for r in clips if r['split']=='validation' and r['label']==0]
    test = [r for r in clips if r['split']=='test']
    scaler = Standardizer().fit([r['features'] for r in healthy_train])
    for r in clips:r['z']=scaler.transform(r['features'])
    best,best_parameters = np.inf,None
    selection = []
    for lag in [1,2,3]:
        train_parts = [design(r['z'],lag) for r in healthy_train]
        X,Y = [np.concatenate([p[k] for p in train_parts]) for k in [0,1]]
        for alpha in [1.0,10.0,100.0]:
            reg=np.eye(X.shape[1])*alpha;reg[-1,-1]=0
            coef=np.linalg.solve(X.T@X+reg,X.T@Y)
            mse=float(np.mean([np.mean((design(r['z'],lag)[1]-design(r['z'],lag)[0]@coef)**2) for r in healthy_val]))
            selection.append({'lag':lag,'alpha':alpha,'healthy_validation_mse':mse})
            if mse<best:best,best_parameters=mse,(lag,alpha,coef,Y-X@coef)
    lag,alpha,coef,train_residual = best_parameters
    residual_mean=train_residual.mean(axis=0)
    eig,vec=np.linalg.eigh(np.cov(train_residual,rowvar=False))
    whiten=vec@np.diag(1/np.sqrt(np.maximum(eig,eig.max()*1e-5)))@vec.T
    for r in clips:
        X,Y=design(r['z'],lag)
        r['residual']=(Y-X@coef-residual_mean)@whiten
        r['dynamic_score']=float(np.mean(windows(r['residual'])**2))
    np.savez(output/'acoustic_reference.npz',lag=lag,alpha=alpha,coef=coef,
             residual_mean=residual_mean,whitener=whiten,scaler_mean=scaler.mean,scaler_scale=scaler.scale)
    records,training = [],{}
    all_model_scores = {'dynamic_residual':{id(r):r['dynamic_score'] for r in clips}}
    si_scale=max(r['dynamic_score'] for r in healthy_val)
    for name,key in [('cnn_lstm','z'),('hybrid','residual')]:
        def build(group):
            return np.concatenate([windows(r[key][lag:] if key=='z' else r[key],stride=5) for r in group])
        model,history=train_autoencoder(build(healthy_train),build(healthy_val),16,11,args.epochs)
        torch.save({'state_dict':model.state_dict(),'input_dim':16,'seq_len':30,'seed':11,
                    'schema':'16 log-power acoustic bands, 100-48000 Hz; innovations for hybrid'},output/f'{name}.pt')
        scores={id(r):float(autoencoder_scores(model,windows(r[key][lag:] if key=='z' else r[key])).mean()) for r in clips}
        if name=='hybrid':
            nn_scale=max(scores[id(r)] for r in healthy_val)
            scores={id(r):max(scores[id(r)]/nn_scale,r['dynamic_score']/si_scale) for r in clips}
        all_model_scores[name]=scores
        training[name]={'history':history,'train_healthy_clips':len(healthy_train),
                        'train_windows':len(build(healthy_train)),'validation_healthy_clips':len(healthy_val)}
    summaries=[]
    for name,scores in all_model_scores.items():
        threshold=max(scores[id(r)] for r in healthy_val)
        labels=np.array([r['label'] for r in test])
        test_scores=np.array([scores[id(r)] for r in test])
        prediction=test_scores>threshold
        tn,fp,fn,tp=confusion_matrix(labels,prediction,labels=[0,1]).ravel()
        recall_interval=binomtest(int(tp),int(tp+fn)).proportion_ci()
        fpr_interval=binomtest(int(fp),int(fp+tn)).proportion_ci()
        summaries.append({'model':name,'threshold':threshold,'test_clips':len(test),
                          'healthy_test_clips':int(tn+fp),'leak_test_clips':int(tp+fn),
                          'true_negative':int(tn),'false_positive':int(fp),'false_negative':int(fn),'true_positive':int(tp),
                          'leak_clip_recall':float(tp/(tp+fn)),'recall_95pct_ci':[recall_interval.low,recall_interval.high],
                          'healthy_clip_false_positive_rate':float(fp/(fp+tn)),
                          'false_positive_rate_95pct_ci':[fpr_interval.low,fpr_interval.high],
                          'average_precision':float(average_precision_score(labels,test_scores)),
                          'roc_auc':float(roc_auc_score(labels,test_scores))})
        for r,score,pred in zip(test,test_scores,prediction):
            records.append({'model':name,'category':r['category'],'filename':r['filename'],'group':r['group'],
                            'label':r['label'],'score':float(score),'prediction':int(pred),'sha256':r['sha256']})
    audit={'clips':len(clips),'unique_hashes':len({r['sha256'] for r in clips}),
           'duration_s':len(clips),'train_prefixes':list(range(15)),
           'validation_prefixes':list(range(15,20)),'test_prefixes':list(range(20,25)),
           'per_split_counts':{s:sum(r['split']==s for r in clips) for s in ['train','validation','test']},
           'feature_frame_seconds':(2048-1536)/96000,'maximum_clipped_fraction':max(r['audit']['clipped_fraction'] for r in clips),
           'limitation':'Prefix blocks are provisional. Acquisition sessions, pressure and SNR metadata are unavailable. This is not a session-held-out or field test.'}
    result={'scope':'SMALL REAL GAS ACOUSTIC PILOT, not pressure/flow SCADA',
            'audit':audit,'summaries':summaries,'test_predictions':records,'training':training,
            'ar_selection':selection,'selected_lag':lag,'selected_alpha':alpha,
            'threshold_protocol':'Maximum healthy-validation clip score. Ten clips cannot establish a low operational false-alarm rate.',
            'single_channel_warning':'No BSS, spatial localization or dynamic-network causal-module estimate was performed.'}
    (output/'acoustic_results.json').write_text(json.dumps(result,indent=2))
    safe_clips=[{k:v for k,v in r.items() if k not in ('z','features','residual')} for r in clips]
    (output/'acoustic_manifest.json').write_text(json.dumps(safe_clips,indent=2))
    print(json.dumps({'audit':audit,'summaries':summaries}),flush=True)

if __name__=='__main__':main()
