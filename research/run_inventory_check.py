"""Exploratory physics residual, with a fresh-seed replication cohort.

Added after seeing the first ARX false-alarm result. Known simulator geometry
is used explicitly; therefore this is an optimistic physics benchmark, not
evidence of performance with uncertain field geometry or varying temperature.
"""
from pathlib import Path
import argparse,json
import numpy as np
from pipeguard.simulator import SimConfig,load_cohort,simulate
from pipeguard.physics import inventory_window_score
from run_benchmark import evaluate,calibrate_threshold,clean_for_forecast

def inventory_score(run,volume_scale=1.0):
    cfg=SimConfig()
    x,invalid=clean_for_forecast(run['x'])
    endpoints=np.arange(31,len(x)) # Same start as selected ARX lag 2 + window 30.
    score,mask=[],[]
    for end in endpoints:
        block=x[end-29:end+1]
        score.append(inventory_window_score(block,run['sample_s'],volume_scale))
        mask.append(invalid[end-31:end+1].any())
    return np.asarray(score),endpoints,np.asarray(mask)

def fresh_runs():
    scenarios=[]
    for regime in ['in_range','shifted']:
        for kind in ['leak','gradual_leak']:
            for fraction in [.02,.05,.10]:
                for location in range(3):scenarios.append((kind,fraction,location,regime))
    for kind in ['healthy','normal_transient','weak_excitation','sensor_bias','missing_sensor']:
        for i in range(6):scenarios.append((kind,0.,i%3,'shifted' if i%2 else 'in_range'))
    return [simulate(f'replication_{i:03d}_{kind}',41000+i,kind,fraction,location,regime)
            for i,(kind,fraction,location,regime) in enumerate(scenarios)]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-dir',default='data/simulation')
    parser.add_argument('--output-dir',default='results')
    args=parser.parse_args()
    output=Path(args.output_dir);output.mkdir(parents=True,exist_ok=True)
    manifest,runs=load_cohort(args.data_dir)
    calibration=[r for r in runs if r['split']=='calibration']
    test=[r for r in runs if r['split']=='test']
    # Freeze formula, geometry sensitivity and calibration before fresh replication.
    settings={'window':30,'volume_scales':[.8,1.,1.2],'calibration_alarm_budget_per_hour':.1,
              'fresh_seed_start':41000,'fresh_cohort_size':66,
              'method_scope':'Known geometry, constant gas temperature and Z; optimistic simulation physics baseline.',
              'exploratory_note':'Added after inspecting initial ARX results. Fresh seeds provide replication within the same simulator, not external validation.'}
    (output/'inventory_protocol.json').write_text(json.dumps(settings,indent=2))
    calibrations={}
    for scale in settings['volume_scales']:
        threshold,meta=calibrate_threshold([inventory_score(r,scale)[0] for r in calibration])
        calibrations[str(scale)]=meta|{'threshold':threshold}
    (output/'inventory_calibration.json').write_text(json.dumps(calibrations,indent=2))
    fresh=fresh_runs()
    summaries,details=[],[]
    for cohort_name,cohort in [('original_exploratory',test),('fresh_replication',fresh)]:
        for scale in settings['volume_scales']:
            threshold=calibrations[str(scale)]['threshold']
            summary,rows=evaluate('mass_inventory',0,cohort,lambda r:inventory_score(r,scale),threshold)
            summary.update(cohort=cohort_name,volume_scale=scale)
            summaries.append(summary)
            details.extend([r|{'cohort':cohort_name,'volume_scale':scale} for r in rows])
    result={'summaries':summaries,'run_results':details,'calibration':calibrations,'protocol':settings}
    (output/'inventory_results.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(summaries),flush=True)

if __name__=='__main__':main()
