"""Load calibrated research models and score ordered, per-asset readings."""
from pathlib import Path
import json
import numpy as np
import torch
from .models import CNNLSTMAutoencoder, DynamicPredictor, Standardizer
from .simulator import FEATURES
from .streaming import StreamWindows
from .physics import inventory_window_score

class ResearchMonitor:
    def __init__(self,result_dir,model_name='mass_inventory',seed=11,calibration_coverage='broad'):
        root = Path(result_dir)
        protocol = json.loads((root/'protocol.json').read_text())
        results = json.loads((root/'benchmark_results.json').read_text())
        with np.load(root/'models/dynamic_reference.npz') as reference:
            self.scaler = Standardizer()
            self.scaler.mean,self.scaler.scale = reference['scaler_mean'],reference['scaler_scale']
            self.dynamic = DynamicPredictor(int(reference['lag']),float(reference['alpha']))
            self.dynamic.coef,self.dynamic.whitener = reference['coef'],reference['whitener']
            self.dynamic.residual_mean = reference['residual_mean']
        self.name = model_name
        if model_name not in ['cnn_lstm','dynamic_residual','hybrid','mass_inventory']:
            raise ValueError('Unsupported full-schema monitor')
        calibration_key = model_name if model_name=='dynamic_residual' else f'{model_name}_{seed}'
        if model_name=='mass_inventory':
            self.calibration=json.loads((root/'inventory_calibration.json').read_text())['1.0']
        else:
            self.calibration = results['calibration'][calibration_key]
        if calibration_coverage=='broad':
            if model_name in ('cnn_lstm','hybrid') and seed!=11:
                raise ValueError('Broad calibration was executed only for neural seed 11')
            broad=json.loads((root/'regime_calibration.json').read_text())[model_name]
            self.calibration=self.calibration|{'threshold':broad['threshold']}
        elif calibration_coverage!='narrow':
            raise ValueError('Calibration coverage must be broad or narrow')
        self.threshold = self.calibration['threshold']
        self.model = None
        if model_name in ('cnn_lstm','hybrid'):
            checkpoint = torch.load(root/f'models/{model_name}_{seed}.pt',map_location='cpu',weights_only=True)
            self.model = CNNLSTMAutoencoder(checkpoint['input_dim'])
            self.model.load_state_dict(checkpoint['state_dict'])
            self.model.eval()
        self.windows = StreamWindows(FEATURES,seq_len=30+self.dynamic.lag,sample_s=protocol['sample_s'])
        self.alarm_states = {}

    def push(self,asset_id,timestamp_s,values):
        result = self.windows.push(asset_id,timestamp_s,values)
        if result['status']!='ready':
            self.alarm_states.pop(asset_id,None)
            return {k:v for k,v in result.items() if k!='window'} | {'asset_id':asset_id,'timestamp_s':timestamp_s}
        z = self.scaler.transform(result['window'])
        residual = self.dynamic.residuals(z)
        si_score = float(np.mean(residual**2))
        score = inventory_window_score(result['window'][-30:],self.windows.sample_s) if self.name=='mass_inventory' else si_score
        if self.model is not None:
            x = z[-30:] if self.name=='cnn_lstm' else residual
            tensor = torch.from_numpy(x.astype(np.float32)).unsqueeze(0)
            with torch.no_grad():score = float(((self.model(tensor)-tensor)**2).mean())
            if self.name=='hybrid':
                score = max(score/self.calibration['neural_scale'],si_score/self.calibration['si_scale'])
        state = self.alarm_states.setdefault(asset_id,{'high':0,'low':0,'active':False,'last_event':-1e12})
        event = False
        if score>self.threshold:
            state['high']+=1;state['low']=0
            if state['high']>=3 and not state['active']:
                state['active']=True
                if timestamp_s-state['last_event']>=120:
                    event=True;state['last_event']=timestamp_s
        else:
            state['high']=0;state['low']+=1
            if state['low']>=5:state['active']=False
        return {'status':'anomaly_alert' if state['active'] else 'healthy_within_reference',
                'asset_id':asset_id,'timestamp_s':timestamp_s,'score':score,'threshold':self.threshold,
                'new_alarm_event':event,'attribution':'not_established',
                'scope':'simulation-trained research model'}
