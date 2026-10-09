"""Three-volume, isothermal, quasi-steady natural-gas linepack simulator.

Mass is conserved at every integration step. Pressure is absolute Pa; flows
are kg/s. This is an intentionally low-order surrogate, not an OLGA solver,
negative-pressure-wave model, or validated replica of a field pipeline.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import numpy as np

FEATURES = ['pressure_in_pa', 'pressure_mid_pa', 'pressure_out_pa',
            'inlet_mass_flow_kg_s', 'outlet_mass_flow_kg_s', 'temperature_k',
            'upstream_command_pa', 'outlet_valve_command']

@dataclass
class SimConfig:
    duration_s: int = 3600
    burn_in_s: int = 600
    integration_s: float = 1.0
    sample_s: int = 5
    volumes_m3: tuple = (21.2, 21.2, 21.2)
    specific_gas_constant: float = 518.3
    compressibility: float = 0.9
    nominal_flow_kg_s: float = 2.0
    pressure_noise_pa: float = 150.0
    flow_noise_kg_s: float = 0.004

def simulate(run_id, seed, kind='healthy', leak_fraction=0.0, location=1,
             regime='in_range', config=None):
    cfg = config or SimConfig()
    rng = np.random.default_rng(seed)
    volume = np.array(cfg.volumes_m3)
    temperature = 288.0 + rng.uniform(-2, 2)
    rt = cfg.compressibility*cfg.specific_gas_constant*temperature
    p_nominal = np.array([4.0, 3.9, 3.8, 3.7, 3.6])*1e6
    conductance = cfg.nominal_flow_kg_s/np.sqrt(p_nominal[:-1]**2-p_nominal[1:]**2)
    pressure = p_nominal[1:-1].copy()
    amp = 22000.0 if regime == 'in_range' else 60000.0
    valve_amp = 0.07 if regime == 'in_range' else 0.15
    if kind == 'weak_excitation':
        amp, valve_amp = 200.0, 0.0005
    period = int(rng.integers(140, 260))
    n_seconds = cfg.duration_s+cfg.burn_in_s
    upstream_steps = rng.uniform(-amp, amp, size=n_seconds//period+2)
    valve_steps = rng.uniform(-valve_amp, valve_amp, size=n_seconds//period+2)
    phase = rng.uniform(0, 2*np.pi)
    leak_onset = int(rng.integers(1100, 1500))
    records, truth, labels, states = [], [], [], []
    qout_initial, max_balance_error = 2.0, 0.0
    for second in range(n_seconds):
        t = second-cfg.burn_in_s
        base_upstream = 4e6 + upstream_steps[second//period]
        # Known realized control includes feedback; it is not an exogenous IV.
        upstream = base_upstream + 0.2*(3.9e6-pressure[0])
        valve = 1.0+valve_steps[second//period]+0.012*np.sin(second/170+phase)
        if kind == 'normal_transient' and t >= 1200:
            valve *= 1.15
            upstream += 45000.0
        nodes = np.r_[upstream, pressure, 3.6e6]
        square_difference = nodes[:-1]**2-nodes[1:]**2
        flow = conductance*np.sign(square_difference)*np.sqrt(np.abs(square_difference))
        flow[-1] *= valve
        leak = np.zeros(3)
        if kind in ('leak', 'gradual_leak') and t >= leak_onset:
            ramp = min(1.0, (t-leak_onset)/600.0) if kind == 'gradual_leak' else 1.0
            # Approximate choked outflow at fixed gas temperature and ambient backpressure.
            leak[location] = cfg.nominal_flow_kg_s*leak_fraction*ramp*pressure[location]/p_nominal[location+1]
        if t >= 0 and t % cfg.sample_s == 0:
            observed_p = pressure + rng.normal(0, cfg.pressure_noise_pa, 3)
            observed_flow = flow[[0,-1]]+rng.normal(0, cfg.flow_noise_kg_s, 2)
            observed_temp = temperature+rng.normal(0, 0.035)
            if kind == 'sensor_bias' and t >= 1200:
                observed_p[1] += 10000.0*min(1.0, (t-1200)/120.0)
            if kind == 'missing_sensor' and 1400 <= t < 1600:
                observed_p[1] = np.nan
            records.append(np.r_[observed_p, observed_flow, observed_temp, upstream, valve])
            truth.append(np.r_[pressure, flow[[0,-1]], leak.sum()])
            states.append(volume*pressure/rt)
            labels.append(int(leak.sum() > 0))
        # Fixed-volume ideal-gas storage, with constant T and Z within each run.
        mass_before = volume*pressure/rt
        net = flow[:-1]-flow[1:]-leak
        pressure = pressure+cfg.integration_s*rt/volume*net
        mass_after = volume*pressure/rt
        err = abs((mass_after.sum()-mass_before.sum())-
                  cfg.integration_s*(flow[0]-flow[-1]-leak.sum()))
        max_balance_error = max(max_balance_error, float(err))
        if np.any(pressure <= 0) or not np.isfinite(pressure).all():
            raise ArithmeticError('Nonphysical pressure or unstable integration')
    x = np.asarray(records, dtype=np.float64)
    return dict(run_id=run_id, seed=seed, kind=kind, regime=regime,
                leak_fraction=leak_fraction, location=location,
                onset_s=leak_onset if kind in ('leak','gradual_leak') else None,
                sample_s=cfg.sample_s, x=x, y=np.array(labels, dtype=np.int8),
                time_s=np.arange(len(x))*cfg.sample_s, truth=np.asarray(truth),
                inventory_kg=np.asarray(states), temperature_true_k=temperature,
                max_mass_balance_error_kg=max_balance_error)

def generate_cohort(output_dir, config=None):
    cfg = config or SimConfig()
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    assignments = []
    for split, count in [('train',40), ('validation',12), ('calibration',24)]:
        for i in range(count):
            assignments.append((split, 'healthy', 0.0, i%3, 'in_range'))
    # Entire independent runs, including all their windows, remain in one split.
    for regime in ['in_range','shifted']:
        for kind in ['leak','gradual_leak']:
            for fraction in [.02,.05,.10]:
                for location in range(3):
                    assignments.append(('test',kind,fraction,location,regime))
    for kind in ['healthy','normal_transient','weak_excitation','sensor_bias','missing_sensor']:
        for i in range(6):
            assignments.append(('test',kind,0.0,i%3,'shifted' if i%2 else 'in_range'))
    metadata = []
    for number, (split,kind,fraction,location,regime) in enumerate(assignments):
        run_id = f'{split}_{number:03d}_{kind}'
        run = simulate(run_id, 17000+number, kind, fraction, location, regime, cfg)
        np.savez_compressed(root/f'{run_id}.npz', x=run['x'], y=run['y'],
                            time_s=run['time_s'], truth=run['truth'],
                            inventory_kg=run['inventory_kg'])
        metadata.append({k:v for k,v in run.items() if not isinstance(v,np.ndarray)} | {'split':split})
    manifest = {'config':asdict(cfg), 'features':FEATURES, 'runs':metadata,
                'physical_scope':'Isothermal quasi-steady linepack; no wave, energy, multiphase, or field validation.'}
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return manifest

def load_cohort(directory):
    root = Path(directory)
    manifest = json.loads((root/'manifest.json').read_text())
    runs = []
    for meta in manifest['runs']:
        with np.load(root/f"{meta['run_id']}.npz") as arrays:
            runs.append(meta | {key:arrays[key] for key in arrays.files})
    return manifest, runs
