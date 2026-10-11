"""Verify scientific guards survive the public request and export boundary."""
import unittest
import numpy as np
from fastapi.testclient import TestClient
from backend.main import create_app
from backend.paper_labs import acoustic_data


class PaperAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(create_app())
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)

    def test_supported_and_omitted_graphs_have_different_attribution(self):
        selected = self.client.post('/api/labs/network', json={'sensor_mode': 'selected'}).json()
        omitted = self.client.post('/api/labs/network', json={'sensor_mode': 'target_only'}).json()
        self.assertTrue(any(r['supported'] for r in selected['rolling']))
        self.assertFalse(any(r['supported'] for r in omitted['rolling']))
        self.assertIsNotNone(omitted['preservation']['unblocked_parallel_path'])
        exported = self.client.get(f"/api/runs/{selected['id']}/export.json")
        self.assertEqual(exported.json()['rolling'], selected['rolling'])
        self.assertEqual(self.client.get(f"/api/runs/{selected['id']}/export.csv").status_code, 422)

    def test_weak_excitation_blocks_attribution(self):
        run = self.client.post('/api/labs/network', json={'scenario': 'weak_excitation'}).json()
        self.assertFalse(any(r['supported'] for r in run['rolling']))
        self.assertFalse(any(r['target_change_flag'] for r in run['rolling']))

    def test_excitation_location_changes_information(self):
        for excitation, supported in [('target_and_separator', True), ('wrong_location', False)]:
            run = self.client.post('/api/labs/information', json={'excitation': excitation}).json()
            self.assertEqual(run['d6']['sufficient_condition'], supported)
            self.assertFalse(run['structural']['whole_row_generically_identifiable'])

    def test_correlated_sources_reject_screen_without_independence_claim(self):
        response = self.client.post('/api/labs/acoustic', json={'scenario': 'correlated'})
        self.assertEqual(response.status_code, 201, response.text)
        self.assertTrue(response.json()['verification']['familywise_rejected'])
        self.assertIn('not proof', response.json()['support_scope'])

    def test_synchronized_upload_and_invalid_channels(self):
        x = acoustic_data(95001, samples=1536)['measurements']
        response = self.client.post('/api/labs/acoustic/analyze', json={'measurements': x.tolist(), 'healthy_samples': 1024})
        self.assertEqual(response.status_code, 201, response.text)
        self.assertEqual(response.json()['monitoring_start_sample'], 1024)
        self.assertNotIn('evaluation', response.json())
        self.assertEqual(self.client.post('/api/labs/acoustic/analyze', json={'measurements': x[:, :1].tolist()}).status_code, 422)
        self.assertEqual(self.client.post('/api/labs/acoustic/analyze', json={'measurements': x.tolist(), 'healthy_samples': 1400}).status_code, 422)
        self.assertEqual(self.client.post('/api/labs/acoustic', json={'truth': [1]}).status_code, 422)

    def test_executed_evidence_is_available(self):
        evidence = self.client.get('/api/paper-methods').json()
        self.assertEqual(evidence['recorded_transfer']['waveforms'], 684)
        self.assertEqual(evidence['ablation']['protocol']['test_runs'], 66)
        self.assertEqual(evidence['benchmarks']['network'][0]['replications'], 80)


if __name__ == '__main__':
    unittest.main()
