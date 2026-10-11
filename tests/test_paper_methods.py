"""Analytical paper examples and failure cases, rather than UI-only checks."""
import json
from pathlib import Path
import unittest

import numpy as np

from backend.context_refinement import ContextRefinement, balanced_assignment
from backend.paper_methods import (SpatialFIRSeparator,identify_two_stage,immerse_frequency,
                                    preservation_check,select_inputs,structural_target)
from backend.paper_labs import EDGES,NODES,acoustic_data,information_lab,network_data

ROOT=Path(__file__).resolve().parents[1]


class GraphMethods(unittest.TestCase):
    def test_parallel_path_and_loop_each_require_observation(self):
        selected=select_inputs(EDGES,NODES,'w1','w2')
        self.assertEqual(selected,['w1','w3','w4'])
        omitted=preservation_check(EDGES,'w1','w2',['w1'])
        self.assertIsNotNone(omitted['unblocked_parallel_path'])
        self.assertIsNotNone(omitted['unblocked_output_loop'])
        self.assertFalse(omitted['target_preserved'])

    def test_immersion_matches_paper_loop_example(self):
        # D2 Example 4: immersed G21 = G21/(1-G23*G32).
        G=np.array([[0,.1,0],[.7,0,.2],[0,.3,0]],complex)
        immersed=immerse_frequency(G,[1,0])
        self.assertAlmostEqual(immersed[0,1].real,.7/(1-.2*.3))
        kept=immerse_frequency(G,[1,0,2])
        self.assertAlmostEqual(kept[0,1].real,.7)

    def test_target_uniqueness_does_not_require_whole_row_uniqueness(self):
        r=information_lab()['d4_paper_example']
        self.assertTrue(r['target_generically_identifiable'])
        self.assertFalse(r['whole_row_generically_identifiable'])
        self.assertEqual((r['full_row_generic_rank'],r['nuisance_generic_rank']),(2,1))

    def test_excitation_location_changes_target_identifiability(self):
        good=information_lab('target_and_separator');bad=information_lab('wrong_location')
        self.assertTrue(good['d6']['sufficient_condition'])
        self.assertTrue(good['target_transfer_rank_condition_all_grid_points'])
        self.assertFalse(bad['d6']['sufficient_condition'])
        self.assertFalse(bad['target_transfer_rank_condition_all_grid_points'])
        self.assertIn('x3',bad['d6']['excluded_excitations'])


class Identification(unittest.TestCase):
    def test_two_stage_estimate_resists_known_colored_confounder(self):
        r=network_data(94001)
        fit=identify_two_stage(r['measurements'],r['external_references'],[0,2,3],1)
        self.assertLess(abs(fit['target_estimate']-.65),.1)
        self.assertGreater(abs(fit['naive_ols_estimate']-.65),.2)

    def test_zero_references_cannot_support_attribution(self):
        r=network_data(94002,excitation=0)
        fit=identify_two_stage(r['measurements'],r['external_references'],[0,2,3],1)
        self.assertFalse(fit['attribution_supported'])

    def test_missing_measurement_is_rejected(self):
        r=network_data(94002);r['measurements'][10,0]=np.nan
        with self.assertRaises(ValueError):
            identify_two_stage(r['measurements'],r['external_references'],[0,2,3],1)


class AcousticMethods(unittest.TestCase):
    def test_constant_channels_cannot_identify_acoustic_mixing(self):
        with self.assertRaisesRegex(ValueError, "observed variation"):
            SpatialFIRSeparator().fit(np.zeros((1024, 5)))

    def setUp(self):
        self.training=acoustic_data(95100)
        self.model=SpatialFIRSeparator().fit(self.training['measurements'])

    def test_sources_recovered_on_independent_known_mixing(self):
        r=acoustic_data(95101)
        recovered=self.model.transform(r['measurements'])
        for j in range(5):
            corr=np.corrcoef(recovered[:,j],r['sources'][self.model.warmup:,j])[0,1]
            self.assertGreater(corr,.98)
        self.assertLess(self.model.inverse_spectral_radius,1)

    def test_source_estimates_do_not_change_when_future_samples_added(self):
        r=acoustic_data(95102)['measurements']
        short=self.model.transform(r[:600]);long=self.model.transform(r)
        np.testing.assert_allclose(short,long[:len(short)],atol=1e-12,rtol=1e-12)

    def test_channel_unit_change_is_removed_by_healthy_normalization(self):
        factors=np.array([1.,100.,.01,5.,2.])
        alternate=SpatialFIRSeparator().fit(self.training['measurements']*factors)
        r=acoustic_data(95103)['measurements']
        np.testing.assert_allclose(self.model.transform(r),alternate.transform(r*factors),atol=1e-10,rtol=1e-10)

    def test_single_channel_is_not_labelled_spatial_source_separation(self):
        with self.assertRaises(ValueError):SpatialFIRSeparator().fit(np.ones((512,1)))


class ContextMethods(unittest.TestCase):
    def test_balanced_assignment_minimizes_total_cost(self):
        cost=np.array([[1.,2.,9.],[1.,8.,9.],[9.,1.,2.]])
        selected=balanced_assignment(cost)
        self.assertEqual(len(set(selected)),3)
        self.assertEqual(sum(cost[i,j] for i,j in enumerate(selected)),5.)

    def test_frozen_context_scores_are_causal_and_reset_after_missing(self):
        path=ROOT/'deployment/models/context_refinement.npz'
        self.assertTrue(path.exists(),'Train the supplied frozen context artifact first')
        model=ContextRefinement.load(path)
        from backend.simulator import simulate
        r=simulate('test',93199,'healthy')
        short=model.score_series(r['x'][:400],r['time_s'][:400])
        complete=model.score_series(r['x'],r['time_s'])
        np.testing.assert_allclose(short,complete[:400],atol=1e-10,rtol=1e-10,equal_nan=True)
        self.assertTrue(np.isnan(short[:59]).all())
        damaged=r['x'].copy();damaged[200,1]=np.nan
        gap=model.score_series(damaged,r['time_s'])
        self.assertTrue(np.isnan(gap[200:260]).all())
        self.assertTrue(np.isfinite(gap[260]))


if __name__=='__main__':unittest.main()
