"""Protect supplied physical bin boundaries and explicit unsupported samples.

The expected Miles lists below deliberately duplicate the user's specification:
reading them back out of the configuration would not catch transcription errors.
"""
import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import yaml

from atmospheric_response.v2_config import (edges_from_spec, load_config, resolve_edges,
                        resolved_profile, profile_has_samples, SELECTIONS)


CONFIG = Path(__file__).parents[1] / "atmospheric_response" / "configs" / "analysis_v2.yaml"
COSZ = [-1, -.8, -.6, -.4, -.2, 0, .2, .4, .6, .8, 1]
MILES = {
    "PC_nue": {
        "RecoENu": [0, .2, .4, .6, .8, 1.05, 1.4, 2, 3.25, 100],
        "RecoY": [0, .1, .2, .3, .4, .5, 1],
    },
    "FC_numu": {
        "RecoENu": [0, .4, .6, .8, 1, 1.1, 1.2, 1.45, 1.65, 1.9, 100],
        "RecoY": [0, .1, .2, .3, .4, .5, 1],
    },
    "FC_nue": {
        "RecoENu": [0, .2, .4, .6, .8, 1, 1.1, 1.2, 1.35, 1.5, 1.7,
                    1.95, 2.25, 3.25, 100],
        "RecoY": [0, .1, .2, .3, .4, .5, 1],
    },
    "PC_numu": {
        "RecoENu": [0, .4, .6, .8, 1, 1.15, 1.3, 1.45, 1.6, 1.8, 2,
                    2.2, 2.45, 2.7, 3, 3.35, 3.75, 4.25, 4.85, 6.85, 100],
        "RecoY": [0, .1, .2, .3, 1],
    },
}


class BinningConfigTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(CONFIG)

    def load_modified(self, cfg):
        """Round-trip a modified user YAML through the public load boundary."""
        cfg = copy.deepcopy(cfg)
        cfg.pop('_path', None)
        cfg.pop('_sha256', None)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'analysis.yaml'
            path.write_text(yaml.safe_dump(cfg))
            return load_config(path)

    def test_exact_supplied_edges_for_both_signs_and_true_reco(self):
        fields = {"CosZ": ("true_coszenith", "reco_coszenith"),
                  "RecoENu": ("true_neutrino_energy", "reco_neutrino_energy"),
                  "RecoY": ("true_inelasticity", "reco_inelasticity")}
        for sample, supplied in MILES.items():
            containment, family = sample.split("_")
            for selection in [s for s in SELECTIONS if s["family"] == family]:
                for field, expected in {"CosZ": COSZ, **supplied}.items():
                    for observable in fields[field]:
                        with self.subTest(sample=sample, selection=selection["id"], observable=observable):
                            actual, note = resolve_edges(self.cfg, "miles", containment,
                                                         selection, observable)
                            np.testing.assert_array_equal(actual, expected)
                            self.assertIn(sample, note)
                            if field == "RecoENu":
                                self.assertEqual(actual[0], 0.)
                                self.assertEqual(actual[-1], 100.)

    def test_no_invented_all_unknown_or_nc_miles_samples(self):
        for containment in ("all", "unknown", "fc", "pc"):
            for selection in SELECTIONS:
                if containment in ("fc", "pc") and selection["family"] != "nc":
                    continue
                for observable in ("reco_neutrino_energy", "true_coszenith", "angular_resolution"):
                    actual, note = resolve_edges(self.cfg, "miles", containment, selection, observable)
                    self.assertIsNone(actual)
                    self.assertIn("no supplied", note)
                actual, _ = resolve_edges(self.cfg, "fine", containment, selection,
                                          "reco_neutrino_energy")
                self.assertIsNotNone(actual)

    def test_user_profile_axes_override_and_inherit_independently(self):
        cfg = copy.deepcopy(self.cfg)
        profile_edges = [.1, .3, 1., 3., 10., 100.]
        cfg["binning_profiles"]["custom"] = {
            "inherits": "fine", "axes": {"profile_energy": {"edges": profile_edges}}}
        actual, _ = resolve_edges(cfg, "custom", "all", SELECTIONS[1], "angular_resolution")
        np.testing.assert_array_equal(actual, profile_edges)
        fine, _ = resolve_edges(cfg, "fine", "all", SELECTIONS[1], "reco_coszenith")
        inherited, _ = resolve_edges(cfg, "custom", "all", SELECTIONS[1], "reco_coszenith")
        np.testing.assert_array_equal(inherited, fine)
        # A Miles sample override takes priority over its generic parent axis.
        cfg["binning_profiles"]["miles"]["samples"]["FC_nue"]["RecoENu"] = [0, 1, 5, 100]
        actual, _ = resolve_edges(cfg, "miles", "fc", SELECTIONS[0], "reco_neutrino_energy")
        np.testing.assert_array_equal(actual, [0, 1, 5, 100])
        # Observables without supplied sample-specific edges inherit fine bins.
        inherited, _ = resolve_edges(cfg, "miles", "fc", SELECTIONS[0], "angular_resolution")
        fine, _ = resolve_edges(cfg, "fine", "fc", SELECTIONS[0], "angular_resolution")
        np.testing.assert_array_equal(inherited, fine)

    def test_invalid_edges_rejected_without_sorting_or_dropping(self):
        for invalid in ([0, 2, 1, 3], [0, 1, 1, 2], [0, np.nan, 2],
                        [0, np.inf], [1], [[0, 1], [2, 3]]):
            with self.subTest(edges=invalid), self.assertRaises(ValueError):
                edges_from_spec(invalid)
        for invalid in ({"spacing": "log", "min": 0, "max": 10, "bins": 4},
                        {"min": 0, "max": 1, "bins": 0},
                        {"min": 0, "max": 1, "bins": True},
                        {"min": 0, "max": 1, "bins": 2.5}):
            with self.subTest(spec=invalid), self.assertRaises(ValueError):
                edges_from_spec(invalid)

    def test_invalid_yaml_sample_is_rejected_at_load(self):
        cfg = copy.deepcopy(self.cfg)
        cfg.pop("_path")
        cfg.pop("_sha256")
        cfg["binning_profiles"]["miles"]["samples"]["PC_numu"]["RecoENu"] = [0, 1, .5, 100]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.yaml"
            path.write_text(yaml.safe_dump(cfg))
            with self.assertRaises(ValueError):
                load_config(path)

    def test_explicit_and_generated_axes_preserve_endpoints(self):
        np.testing.assert_array_equal(edges_from_spec({"edges": [0, .2, 100]}), [0, .2, 100])
        linear = edges_from_spec({"min": -1, "max": 1, "bins": 10})
        log = edges_from_spec({"spacing": "log", "min": .1, "max": 100, "bins": 3})
        self.assertEqual(len(linear), 11)
        np.testing.assert_array_equal(linear[[0, -1]], [-1, 1])
        np.testing.assert_allclose(log, [.1, 1, 10, 100], atol=0, rtol=1e-15)

    def test_nested_miles_inheritance_merges_fields_and_keeps_sample_scope(self):
        cfg = copy.deepcopy(self.cfg)
        cfg['binning_profiles']['tuned'] = {
            'inherits': 'miles', 'samples': {'FC_nue': {'RecoENu': [0, .5, 2, 100]}}}
        cfg['binning_profiles']['nested'] = {
            'inherits': 'tuned', 'axes': {'profile_energy': [.1, 1, 10, 100]}}
        cfg = self.load_modified(cfg)
        self.assertFalse(profile_has_samples(cfg, 'fine'))
        self.assertTrue(profile_has_samples(cfg, 'nested'))
        actual, _ = resolve_edges(cfg, 'nested', 'fc', SELECTIONS[0], 'reco_neutrino_energy')
        np.testing.assert_array_equal(actual, [0, .5, 2, 100])
        actual, _ = resolve_edges(cfg, 'nested', 'fc', SELECTIONS[0], 'reco_inelasticity')
        np.testing.assert_array_equal(actual, MILES['FC_nue']['RecoY'])
        actual, _ = resolve_edges(cfg, 'nested', 'pc', SELECTIONS[1], 'reco_neutrino_energy')
        np.testing.assert_array_equal(actual, MILES['PC_numu']['RecoENu'])
        actual, _ = resolve_edges(cfg, 'nested', 'fc', SELECTIONS[0], 'angular_resolution')
        np.testing.assert_array_equal(actual, [.1, 1, 10, 100])
        self.assertIsNone(resolve_edges(cfg, 'nested', 'all', SELECTIONS[0], 'reco_coszenith')[0])
        self.assertIsNone(resolve_edges(cfg, 'nested', 'fc', SELECTIONS[2], 'reco_coszenith')[0])
        # Resolved dictionaries are caller-owned; changing one cannot mutate YAML.
        merged = resolved_profile(cfg, 'nested')
        merged['samples']['PC_numu']['RecoENu'][0] = -99
        self.assertEqual(cfg['binning_profiles']['miles']['samples']['PC_numu']['RecoENu'][0], 0)

    def test_missing_parents_cycles_and_malformed_profiles_rejected(self):
        for extra in ({'broken': {'inherits': 'missing'}},
                      {'a': {'inherits': 'b'}, 'b': {'inherits': 'a'}},
                      {'self': {'inherits': 'self'}},
                      {'broken': {'inherits': 'fine', 'axes': []}},
                      {'broken': {'inherits': 'miles', 'samples': {'FC_nue': None}}}):
            cfg = copy.deepcopy(self.cfg)
            cfg['binning_profiles'].update(extra)
            with self.subTest(profiles=extra), self.assertRaises(ValueError):
                self.load_modified(cfg)

    def test_conditional_edges_and_effective_count_are_validated(self):
        invalid = [('minimum_effective_count', -1), ('minimum_effective_count', np.nan),
                   ('minimum_effective_count', np.inf), ('minimum_effective_count', True),
                   ('conditional_energy_edges', [.1, 10, 1, 100]),
                   ('conditional_coszenith_edges', [-1, 0, 0, 1]),
                   ('conditional_energy_edges', [.1, np.nan, 100])]
        for key, value in invalid:
            cfg = copy.deepcopy(self.cfg)
            cfg['statistics'][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.load_modified(cfg)
        cfg = copy.deepcopy(self.cfg)
        cfg['statistics']['minimum_effective_count'] = 0
        cfg['statistics']['conditional_energy_edges'] = {'spacing': 'log', 'min': .1, 'max': 100, 'bins': 3}
        loaded = self.load_modified(cfg)
        self.assertEqual(loaded['statistics']['minimum_effective_count'], 0)
        np.testing.assert_allclose(loaded['statistics']['conditional_energy_edges'], [.1, 1, 10, 100])

    def test_weight_policy_and_physics_modes_are_validated(self):
        for key in ('nonfinite_knot_replacement', 'knot_min', 'knot_max'):
            for invalid in (np.nan, np.inf, -np.inf, True):
                cfg = copy.deepcopy(self.cfg)
                cfg['weight_policy'][key] = invalid
                with self.subTest(key=key, invalid=invalid), self.assertRaises(ValueError):
                    self.load_modified(cfg)
        cfg = copy.deepcopy(self.cfg)
        cfg['weight_policy'].update(knot_min=2, knot_max=1)
        with self.assertRaises(ValueError):
            self.load_modified(cfg)
        for key in ('direction_method', 'true_baseline_mode'):
            cfg = copy.deepcopy(self.cfg)
            cfg['physics'][key] = 'misspelled-choice'
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.load_modified(cfg)
        for direction in ('hybrid', 'hits', 'all_particles'):
            for baseline in ('geometric', 'stored'):
                cfg = copy.deepcopy(self.cfg)
                cfg['physics'].update(direction_method=direction, true_baseline_mode=baseline)
                self.load_modified(cfg)


if __name__ == "__main__":
    unittest.main(verbosity=2)
