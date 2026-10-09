"""Meaningful scientific/data integrity checks, without duplicated model tests."""
import unittest
import numpy as np
from pipeguard.simulator import simulate
from pipeguard.streaming import StreamWindows, attribution_status
from pipeguard.models import Standardizer, arx_design
from run_benchmark import clean_for_forecast, alarm_logic, evaluate

class IntegrityTests(unittest.TestCase):
    def test_mass_conservation_and_positivity(self):
        for fraction in [.0,.1]:
            r = simulate('check',7,'leak',fraction,2)
            self.assertLess(r['max_mass_balance_error_kg'],1e-8)
            self.assertTrue((r['truth'][:,:3]>0).all())

    def test_disjoint_reproducible_runs(self):
        a,b = simulate('a',91),simulate('b',92)
        np.testing.assert_array_equal(a['x'],simulate('a',91)['x'])
        self.assertFalse(np.array_equal(a['x'],b['x']))

    def test_clients_cannot_share_windows(self):
        state = StreamWindows(['pressure'],seq_len=3,sample_s=5)
        state.push('a',0,[10]);state.push('b',0,[100])
        state.push('a',5,[11]);state.push('b',5,[101])
        result = state.push('a',10,[12])
        np.testing.assert_array_equal(result['window'].ravel(),[10,11,12])
        result = state.push('b',10,[102])
        np.testing.assert_array_equal(result['window'].ravel(),[100,101,102])

    def test_time_gap_and_missing_reset(self):
        state = StreamWindows(['pressure'],seq_len=2,sample_s=5)
        state.push('a',0,[10])
        self.assertEqual(state.push('a',20,[11])['status'],'warming_up')
        self.assertEqual(state.push('a',25,[np.nan])['status'],'data_quality_alert')
        with self.assertRaises(ValueError):state.push('a',20,[12])
        self.assertEqual(state.push('a',30,[12])['status'],'warming_up')
        with self.assertRaises(ValueError):state.push('a',30,[13])

    def test_predictor_has_no_future_input(self):
        z = np.arange(80,dtype=float).reshape(10,8)
        X,Y = arx_design(z,lag=2)
        np.testing.assert_array_equal(X[0,:8],z[1])
        np.testing.assert_array_equal(X[0,8:16],z[0])
        np.testing.assert_array_equal(Y[0],z[2,:6])

    def test_scaler_freezes_and_invalid_values_are_flagged(self):
        s = Standardizer().fit([np.array([[1.,2.],[3.,4.]])])
        mean = s.mean.copy()
        s.transform(np.array([[1000.,1000.]]))
        np.testing.assert_array_equal(mean,s.mean)
        clean,invalid = clean_for_forecast(np.array([[1.,2.],[np.nan,3.],[4.,5.]]))
        self.assertTrue(invalid[1]);self.assertEqual(clean[1,0],1.)

    def test_detection_is_not_gated_by_information(self):
        self.assertEqual(attribution_status(False),'insufficient_information_for_attribution')
        alert,events = alarm_logic(np.array([2.,2.,2.,2.]),1.)
        self.assertTrue(alert[-1]);self.assertEqual(events,[2])

    def test_alarm_before_leak_cannot_count_as_detection(self):
        run={'run_id':'case','kind':'leak','regime':'in_range','leak_fraction':.1,
             'location':1,'onset_s':15,'sample_s':5,'time_s':np.arange(10)*5,
             'y':np.r_[np.zeros(3,dtype=int),np.ones(7,dtype=int)]}
        result,rows=evaluate('check',0,[run],lambda r:(np.full(10,2.),np.arange(10),np.zeros(10,dtype=bool)),1.)
        self.assertEqual(result['detected_events'],0)
        self.assertEqual(result['leaks_with_preexisting_alarm'],1)
        self.assertIsNone(rows[0]['delay_s'])

if __name__ == '__main__':unittest.main()
