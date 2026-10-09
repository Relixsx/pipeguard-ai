"""Per-asset stream validation, with no cross-asset window state."""
from collections import deque
import numpy as np

class StreamWindows:
    def __init__(self,feature_names,seq_len=30,sample_s=5):
        self.feature_names = tuple(feature_names)
        self.seq_len,self.sample_s = seq_len,sample_s
        self.assets = {}
        self.last_timestamps = {}
    def push(self,asset_id,timestamp_s,values):
        if not asset_id:
            raise ValueError('asset_id is required')
        x = np.asarray(values,dtype=float)
        if x.shape != (len(self.feature_names),):
            raise ValueError('Feature count does not match model schema')
        if not np.isfinite(timestamp_s):
            raise ValueError('Timestamp must be finite')
        previous = self.last_timestamps.get(asset_id)
        if previous is not None and timestamp_s <= previous:
            raise ValueError('Timestamp must increase within each asset')
        self.last_timestamps[asset_id] = timestamp_s
        if not np.isfinite(x).all():
            self.assets.pop(asset_id,None)
            return {'status':'data_quality_alert','reason':'nonfinite_sensor','window':None}
        state = self.assets.get(asset_id)
        if state and abs(timestamp_s-state['timestamp']-self.sample_s)>self.sample_s*.1:
            self.assets.pop(asset_id,None)
            state = None
        if state is None:
            state = {'timestamp':timestamp_s,'buffer':deque(maxlen=self.seq_len)}
            self.assets[asset_id] = state
        state['buffer'].append(x)
        state['timestamp'] = timestamp_s
        if len(state['buffer']) < self.seq_len:
            return {'status':'warming_up','window':None}
        return {'status':'ready','window':np.stack(state['buffer'])}

def attribution_status(informative):
    # Detection stays active independently of attribution confidence.
    return 'candidate_component' if informative else 'insufficient_information_for_attribution'
