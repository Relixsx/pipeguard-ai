"""Local research API with true model scores and a log of ALL accepted readings.

Run on loopback for inspection. Models were trained on the declared simulator;
this adapter is not an authorization to use them for operational safety.
"""
from pathlib import Path
from threading import Lock
import argparse,json,sqlite3
import numpy as np
import torch
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel,Field
from pipeguard.monitor import ResearchMonitor
from pipeguard.simulator import FEATURES

class Reading(BaseModel):
    asset_id:str=Field(min_length=1,max_length=100)
    timestamp_s:float=Field(allow_inf_nan=False)
    values:list[float|None]=Field(min_length=8,max_length=8)

def create_app(result_dir='results',model='mass_inventory',seed=11,log_path='reading_log.sqlite',calibration_coverage='broad'):
    torch.set_num_threads(2)
    monitor=ResearchMonitor(result_dir,model,seed,calibration_coverage)
    lock=Lock()
    database=sqlite3.connect(str(log_path),check_same_thread=False)
    database.execute('CREATE TABLE IF NOT EXISTS readings (id INTEGER PRIMARY KEY, asset_id TEXT, timestamp_s REAL, result_json TEXT)')
    app=FastAPI(title='PipeGuard research monitor')
    @app.get('/schema')
    def schema():
        return {'features':FEATURES,'sample_s':5,'model':model,
                'scope':'Simulation-trained model; not field validated.'}
    @app.post('/predict')
    def predict(reading:Reading):
        with lock:
            try:
                values=np.asarray([np.nan if x is None else x for x in reading.values])
                result=monitor.push(reading.asset_id,reading.timestamp_s,values)
            except ValueError as error:
                raise HTTPException(status_code=422,detail=str(error)) from error
            database.execute('INSERT INTO readings (asset_id,timestamp_s,result_json) VALUES (?,?,?)',
                             (reading.asset_id,reading.timestamp_s,json.dumps(result)))
            database.commit()
        return result
    @app.get('/readings')
    def readings(limit:int=100):
        with lock:
            rows=database.execute('SELECT result_json FROM readings ORDER BY id DESC LIMIT ?',
                                  (max(1,min(limit,1000)),)).fetchall()
        return [json.loads(row[0]) for row in rows]
    return app

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--result-dir',default='results')
    parser.add_argument('--model',choices=['dynamic_residual','cnn_lstm','hybrid','mass_inventory'],default='mass_inventory')
    parser.add_argument('--calibration',choices=['narrow','broad'],default='broad')
    parser.add_argument('--seed',type=int,default=11)
    parser.add_argument('--port',type=int,default=8001)
    args=parser.parse_args()
    import uvicorn
    uvicorn.run(create_app(args.result_dir,args.model,args.seed,calibration_coverage=args.calibration),host='127.0.0.1',port=args.port)
