import unittest,tempfile,json
from pathlib import Path
from fastapi.testclient import TestClient
from serve_research import create_app

ROOT=Path(__file__).resolve().parents[1]

class APITests(unittest.TestCase):
    @unittest.skipUnless((ROOT/'results/benchmark_results.json').exists(),'Trained checkpoint required')
    def test_model_endpoint_logs_healthy_and_keeps_assets_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            client=TestClient(create_app(ROOT/'results','dynamic_residual',log_path=Path(tmp)/'log.sqlite'))
            values=[3.9e6,3.8e6,3.7e6,2.,2.,288.,4e6,1.]
            for t in range(0,160,5):
                response=client.post('/predict',json={'asset_id':'a','timestamp_s':t,'values':values})
                self.assertEqual(response.status_code,200)
            response=client.post('/predict',json={'asset_id':'b','timestamp_s':0,'values':values})
            self.assertEqual(response.json()['status'],'warming_up')
            missing=values.copy();missing[1]=None
            response=client.post('/predict',json={'asset_id':'a','timestamp_s':160,'values':missing})
            self.assertEqual(response.json()['status'],'data_quality_alert')
            rows=client.get('/readings').json()
            self.assertEqual(len(rows),34)
            self.assertTrue(any(r['status']=='healthy_within_reference' for r in rows))
            self.assertEqual(client.post('/predict',json={'asset_id':'b','timestamp_s':0,'values':values}).status_code,422)

if __name__=='__main__':unittest.main()
