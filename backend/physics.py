"""Isothermal inventory balance with explicitly assumed gas properties/geometry."""
import numpy as np
from .simulator import SimConfig

def inventory_window_score(block,sample_s=5,volume_scale=1.0):
    cfg=SimConfig()
    volume=np.asarray(cfg.volumes_m3)*volume_scale
    temperature=block[:,5].mean()
    mass_start=np.dot(volume,block[0,:3])/(cfg.compressibility*cfg.specific_gas_constant*temperature)
    mass_end=np.dot(volume,block[-1,:3])/(cfg.compressibility*cfg.specific_gas_constant*temperature)
    interval=(len(block)-1)*sample_s
    net=np.trapezoid(block[:,3]-block[:,4],dx=sample_s)/interval
    return float(abs(net-(mass_end-mass_start)/interval))
