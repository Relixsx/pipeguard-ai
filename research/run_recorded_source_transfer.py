"""Recordings + imposed synchronized mixing: waveform-recovery transfer study.

GPLA-12 contains leakage-condition classes, not a healthy class. No leak recall,
field localization or real multichannel acquisition is inferred from this study.
The separator is fitted on independent simulated white-source measurements.
All 684 actual GPLA waveforms are unseen test sources; Meng machinery waveforms
provide interference. Sampling indices/delays are artificial, not travel times.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import wave

import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from backend.paper_methods import SpatialFIRSeparator
from backend.paper_labs import acoustic_data


def standardize(x):
    x=np.asarray(x,float);return (x-x.mean())/max(x.std(),1e-12)


def mix(sources):
    w=np.zeros_like(sources)
    for t in range(4,len(w)):
        w[t]=.25*w[t-1]+sources[t]
        w[t,1:]+=.28*w[t-4,:-1]
        w[t,:-1]+=.28*w[t-4,1:]
    return w


def si_sdr(estimate,reference):
    e,r=estimate-estimate.mean(),reference-reference.mean()
    projection=r*np.dot(e,r)/max(np.dot(r,r),1e-15)
    return float(10*np.log10(max(np.dot(projection,projection),1e-15)/max(np.sum((e-projection)**2),1e-15)))


def run(gpla,meng,output):
    x=np.loadtxt(gpla/'data.csv',delimiter=',');labels=np.loadtxt(gpla/'label.csv',delimiter=',').astype(int)
    machinery=[];provenance=[]
    for p in sorted((meng/'blower').glob('*.wav')):
        with wave.open(str(p),'rb') as reader:
            if reader.getnchannels()!=1 or reader.getsampwidth()!=2:raise ValueError('Expected mono 16-bit PCM')
            raw=np.frombuffer(reader.readframes(reader.getnframes()),dtype='<i2').astype(float)
        # Index resampling solely for imposed synthetic mixtures. No common Hz is claimed.
        resampled=np.interp(np.linspace(0,len(raw)-1,x.shape[1]),np.arange(len(raw)),raw)
        machinery.append(standardize(resampled));provenance.append({'file':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    if len(machinery)<2:raise ValueError('At least two acquired machinery recordings are needed')
    train=acoustic_data(97000,samples=8192)
    separator=SpatialFIRSeparator().fit(train['measurements'])
    rows=[]
    for i,(record,label) in enumerate(zip(x,labels)):
        rng=np.random.default_rng(97100+i)
        sources=rng.normal(scale=.25,size=(len(record),5))
        sources[:,2]=standardize(record)
        sources[:,1]=6*machinery[i%len(machinery)]
        sources[:,3]=4*machinery[(i+17)%len(machinery)]
        measured=mix(sources);estimated=separator.transform(measured)
        start=64
        truth=sources[start+separator.warmup:,2]
        raw=measured[start+separator.warmup:,2]
        recovered=estimated[start:,2]
        rows.append({'row':i,'condition_class':int(label),
                     'raw_reference_correlation':float(np.corrcoef(raw,truth)[0,1]),
                     'separated_reference_correlation':float(np.corrcoef(recovered,truth)[0,1]),
                     'raw_si_sdr_db':si_sdr(raw,truth),'separated_si_sdr_db':si_sdr(recovered,truth),
                     'record_sha256':hashlib.sha256(record.tobytes()).hexdigest()})
        if i%120==0:print('Recorded transfer',i+1,'/684',flush=True)
    grouped=[]
    for label in sorted(set(labels)):
        block=[r for r in rows if r['condition_class']==label]
        grouped.append({'condition_class':int(label),'waveforms':len(block),
                        'median_raw_correlation':float(np.median([r['raw_reference_correlation'] for r in block])),
                        'median_separated_correlation':float(np.median([r['separated_reference_correlation'] for r in block])),
                        'median_si_sdr_improvement_db':float(np.median([r['separated_si_sdr_db']-r['raw_si_sdr_db'] for r in block]))})
    result={'scope':'Semi-synthetic recorded-source transfer; imposed mixing, not field multichannel validation',
            'gpla':json.loads((gpla/'acquisition.json').read_text()),'machinery_files':provenance,
            'waveforms':len(x),'unique_recording_rows':len(set(r['record_sha256'] for r in rows)),
            'source_conditions':'12 pressure/noise/microphone conditions; all leakage recordings; no healthy class',
            'separator_training':'Independent simulated white-source measurements, seed 97000; no GPLA row used in fitting',
            'mixture':'Five imposed channels, recorded source at index 2, machinery interference at 1 and 3',
            'sampling':'Artificial common sample index; physical source rates/travel distances are not inferred',
            'evaluation':'Scale-invariant SDR and reference-waveform correlation; not leak-event accuracy',
            'session_independence':'Unavailable; 684 rows are not asserted to be 684 independent physical experiments',
            'median_raw_correlation':float(np.median([r['raw_reference_correlation'] for r in rows])),
            'median_separated_correlation':float(np.median([r['separated_reference_correlation'] for r in rows])),
            'median_si_sdr_improvement_db':float(np.median([r['separated_si_sdr_db']-r['raw_si_sdr_db'] for r in rows])),
            'condition_summaries':grouped,'row_results':rows,'separator':separator.details()}
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps({k:result[k] for k in ['waveforms','median_raw_correlation','median_separated_correlation','median_si_sdr_improvement_db']},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--gpla',type=Path,required=True)
    parser.add_argument('--meng',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=ROOT/'deployment/evidence/recorded_source_transfer.json')
    a=parser.parse_args();run(a.gpla,a.meng,a.output)
