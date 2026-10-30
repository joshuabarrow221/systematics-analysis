"""Small semantic regression tests; no CAF files or remote reads required."""
import unittest
import numpy as np

from atmospheric_response.v2_physics import (derive_observables, reduce_particles, classify_containment,
                        choose_direction, geometric_baseline_km, unit_direction)


class PhysicsTests(unittest.TestCase):
    def test_containment_matches_cpp_min_loop(self):
        # Type3's NaN is expected; +inf/NaN on other objects are ignored only
        # if another object's finite wall establishes a valid minimum.
        d = reduce_particles([3, 2, 1, 2, 1, 1, 0],
            [1, 2, 3, 1, 2, 1, 1, 2, 3, 1],
            [11, np.inf, np.nan, 10, np.nan, -1, 12, -np.inf, 4, np.inf])
        np.testing.assert_array_equal(d["n_particles"], [2, 2, 1, 2, 0, 1, 0])
        np.testing.assert_array_equal(classify_containment(d["min_wall_cm"], d["wall_valid"]),
                                      [1, 0, -1, -1, -1, -1, -1])
        np.testing.assert_array_equal(d["invalid_wall_particles"], [1, 1, 1, 1, 0, 1, 0])
        with self.assertRaises(ValueError):
            reduce_particles([2], [1], [10])

    def test_literal_hybrid_switch_and_horizontal_validity(self):
        cat = np.array([0, 1, 0, 1, 2, 3, 4])
        n = np.array([1, 2, 2, 2, 9, 2, 2])
        e = np.array([0.5, 1.2999, 1.3, np.nan, 0.2, 0.2, 0.2])
        heshw = np.tile([2., 0., 0.], (7, 1))
        lngtrk = np.tile([0., 2., 0.], (7, 1))
        calo = np.tile([0., 0., 3.], (7, 1))
        _, valid, method = choose_direction(cat, n, e, heshw, lngtrk, calo)
        np.testing.assert_array_equal(method, [2, 1, 2, 2, 2, 0, 1])
        self.assertTrue(valid.all())  # y=0 is a valid horizontal direction.
        _, _, old = choose_direction(cat, n, e, heshw, lngtrk, calo, "all_particles")
        self.assertEqual(old[4], 0)
        _, good = unit_direction([[0, 0, 0], [1, np.nan, 0], [1, 0, 0]])
        np.testing.assert_array_equal(good, [False, False, True])

    def test_geometry_and_metres_conversion(self):
        # User-selected current convention: 15 km production height. The
        # explicit 25 km case below remains a check that overrides still work.
        np.testing.assert_allclose(geometric_baseline_km([1., -1.]), [15., 12757.])
        R, h, depth = 6371., 25., 1.5
        L = geometric_baseline_km(np.array([1., 0., -1.]), R, h, depth)
        np.testing.assert_allclose(L, [h+depth, np.sqrt((R+h)**2-(R-depth)**2),
                                       2*R+h-depth], rtol=1e-12)
        self.assertTrue(np.isnan(geometric_baseline_km([1.01]))[0])
        with self.assertRaises(ValueError):
            geometric_baseline_km([0], detector_depth_km=-1)
        d = derive_observables(self.raw())
        np.testing.assert_allclose(d["true_baseline_stored_km"], [1, np.nan, 2], equal_nan=True)
        np.testing.assert_array_equal(d["masks"]["true_l_over_e"], [True, False, True])
        proxy = derive_observables(self.raw(), {"true_baseline_mode": "geometric"})
        self.assertTrue(proxy["masks"]["true_l_over_e"].all())
        np.testing.assert_allclose(proxy["true_baseline_geometric_km"], [15., 15., 15.])

    def test_previous_method_has_no_switch_and_preserves_old_validity(self):
        # Both charge signs; NC uses heshw even above the hybrid switch.
        category = np.array([0, 1, 2, 3, 4, 0, 0])
        heshw = np.tile([0., -1., 0.], (7, 1))
        lngtrk = np.tile([0., 0., 1.], (7, 1))
        calo = np.tile([1., 0., 0.], (7, 1))
        heshw[5] = [0., -2., 0.]  # normalized by hybrid, invalid historically
        heshw[6] = [0., 0., 0.]
        unit, valid, method = choose_direction(category, np.ones(7),
            np.array([.1, 1.3, 3., np.nan, 100., .2, .2]),
            heshw, lngtrk, calo, "all_particles")
        np.testing.assert_array_equal(method, [0, 1, 0, 0, 1, 0, 0])
        np.testing.assert_array_equal(valid, [True]*5+[False, False])
        np.testing.assert_allclose(-unit[:5, 1], [1., 0., 1., 1., 0.])
        _, hybrid_valid, _ = choose_direction(category, np.full(7, 2),
            np.full(7, .2), heshw, lngtrk, calo, "hybrid")
        self.assertTrue(hybrid_valid[5])

    @staticmethod
    def raw():
        return {
            "E": np.array([2., 4., 3.]), "q0": np.array([.5, 1., 1.]),
            "inelasticity": np.array([.2, .25, .4]), "baseline_m": np.array([1000., np.nan, 2000.]),
            "pdg": np.array([12, 14, 16]), "iscc": np.array([1, 1, 0]),
            "enu_calo": np.array([.2, .2, .2]), "enu_e_calo": np.array([2., 5., 3.]),
            "enu_e_had": np.array([.4, 1., .5]), "enu_lep_calo": np.array([8., 4., 6.]),
            "enu_mu_range": np.array([10., 2., 10.]), "enu_mu_had": np.array([3., .6, 3.]),
            "n_particles": np.array([2, 2, 2]), "min_wall_cm": np.array([10., 11., -1.]),
            "true_momentum": np.tile([0., -2., 0.], (3, 1)),
            "reco_heshw": np.tile([0., -1., 0.], (3, 1)),
            "reco_lngtrk": np.tile([0., -1., 0.], (3, 1)),
            "reco_calo": np.tile([0., -1., 0.], (3, 1)),
        }

    def test_fitter_energy_y_and_nc_no_fake_lepton(self):
        d = derive_observables(self.raw())
        np.testing.assert_allclose(d["values"]["reco_neutrino_energy"], [2., 4., .2])
        np.testing.assert_allclose(d["values"]["reco_inelasticity"], [.2, .25, np.nan], equal_nan=True)
        self.assertFalse(d["masks"]["reco_inelasticity"][2])
        self.assertFalse(d["masks"]["true_lepton_energy"][2])
        # The stored generator Y is preserved, rather than replaced by q0/E.
        self.assertEqual(d["values"]["true_inelasticity"][0], .2)
        self.assertAlmostEqual(d["true_y_minus_q0_over_E"][0], -.05)
        alt = derive_observables(self.raw(), {"energy_estimator": "matched_range"})
        self.assertEqual(alt["values"]["reco_neutrino_energy"][1], 2)
        self.assertAlmostEqual(alt["values"]["reco_inelasticity"][1], .3)

    def test_bad_reconstruction_does_not_cut_true_observables(self):
        raw = self.raw()
        raw["reco_heshw"][0] = 0
        raw["enu_e_had"][0] = np.nan
        d = derive_observables(raw)
        self.assertTrue(d["masks"]["true_neutrino_energy"][0])
        self.assertTrue(d["masks"]["true_coszenith"][0])
        self.assertTrue(d["masks"]["reco_neutrino_energy"][0])
        self.assertFalse(d["masks"]["reco_lepton_energy"][0])
        self.assertFalse(d["masks"]["reco_coszenith"][0])
        self.assertFalse(d["masks"]["angular_resolution"][0])

    def test_fitter_y_keeps_flow_and_nonpositive_total_convention(self):
        raw = self.raw()
        raw["enu_e_calo"][0] = 0
        raw["enu_lep_calo"][1] = .5
        d = derive_observables(raw)
        np.testing.assert_allclose(d["values"]["reco_inelasticity"][:2], [0., 2.])
        self.assertTrue(d["masks"]["reco_inelasticity"][:2].all())
        self.assertFalse(d["masks"]["reco_l_over_e"][0])


if __name__ == "__main__":
    unittest.main()
