"""Statistical validation of paired shape diagnostics (no ROOT inputs).

Run: python -m pytest tests/test_v2_statistics.py -q
The seeded Monte Carlo checks validate coverage and shared-event covariance,
not just that the implementation reproduces its own algebra.
"""
import json
import math
from pathlib import Path
import unittest

import numpy as np

from atmospheric_response.v2_statistics import (
    paired_shape_flatness, moments_from_events, energy_stratified_flatness, _chi2_sf_fallback,
)


class PairedShapeTests(unittest.TestCase):
    def test_same_events_cancel_and_nonunity_cv_is_reference(self):
        rng = np.random.default_rng(490)
        index = rng.integers(0, 15, 12000)
        reference = rng.lognormal(.2, .6, len(index))
        for scale in (1, 2.7, .001):
            result = paired_shape_flatness(**moments_from_events(index, scale * reference, reference, 15))
            self.assertEqual(result["status"], "proportional_within_precision")
            self.assertEqual(result["ndof"], 0)
            self.assertIsNone(result["p_value"])
            np.testing.assert_allclose(result["ratio_shape"], 1, atol=1e-14)
            np.testing.assert_allclose(result["ratio_shape_se"], 0, atol=1e-14)
            np.testing.assert_allclose(result["ratio_rate"], scale, atol=1e-14)

    def test_dense_covariance_pseudoinverse_and_scale_invariance(self):
        rng = np.random.default_rng(733)
        index = rng.integers(0, 35, 30000)
        b = rng.lognormal(0, .4, len(index))
        a = b * rng.lognormal(.03 + .002 * index, .2)
        answer = None
        for scale in (1, 1e-9, 1e9):
            result = paired_shape_flatness(**moments_from_events(index, a * scale, b * scale, 35),
                                           return_covariance=True)
            self.assertEqual(result["status"], "ok_asymptotic")
            d, v = np.array(result["difference"]), np.array(result["difference_covariance"])
            dense = d @ np.linalg.pinv(v, rcond=1e-12) @ d
            self.assertAlmostEqual(dense / result["statistic"], 1, places=11)
            self.assertEqual(result["ndof"], 34)
            if answer is None:
                answer = result["statistic"]
            self.assertAlmostEqual(result["statistic"] / answer, 1, places=10)
            json.dumps(result, allow_nan=False)

    def test_shape_is_invariant_under_extreme_relative_rate_scales(self):
        rng = np.random.default_rng(118)
        index = rng.integers(0, 8, 12000)
        b = rng.uniform(.5, 1.5, len(index))
        a = b * rng.uniform(.8, 1.4, len(index))
        nominal = paired_shape_flatness(**moments_from_events(index, a, b, 8))
        for factor in (1e-20, 1e-12, 1e12, 1e20):
            result = paired_shape_flatness(**moments_from_events(index, a * factor, b, 8))
            self.assertEqual(result["status"], "ok_asymptotic")
            self.assertAlmostEqual(result["statistic"] / nominal["statistic"], 1, places=9)
            np.testing.assert_allclose(result["ratio_shape_se"], nominal["ratio_shape_se"], rtol=1e-10)

    def test_singular_covariance_keeps_supported_contrast(self):
        index = np.repeat(np.arange(3), 200)
        b = np.ones(600)
        a = b.copy()
        a[:100] += .2
        a[100:200] -= .2
        result = paired_shape_flatness(**moments_from_events(index, a, b, 3), return_covariance=True)
        self.assertEqual(result["status"], "ok_asymptotic")
        self.assertEqual(result["ndof"], 1)
        self.assertAlmostEqual(result["statistic"], 0, places=20)
        self.assertAlmostEqual(result["p_value"], 1, places=10)

    def test_nonfinite_signed_zero_support_and_sparse_are_explicit(self):
        i = np.repeat(np.arange(3), 100)
        b = np.ones(300)
        a = b.copy()
        a[1] = np.nan
        result = paired_shape_flatness(**moments_from_events(i, a, b, 3))
        self.assertEqual(result["status"], "unavailable_nonfinite")
        self.assertIsNone(result["p_value"])
        a[1] = -1
        self.assertEqual(paired_shape_flatness(**moments_from_events(i, a, b, 3))["status"],
                         "unavailable_signed_weights")
        a[1] = 1
        b[:100] = 0
        result = paired_shape_flatness(**moments_from_events(i, a, b, 3))
        self.assertEqual(result["status"], "unavailable_reference_support")
        self.assertIsNone(result["ratio_shape"][0])
        sparse = paired_shape_flatness(**moments_from_events(np.array([0, 0, 1, 1]),
                                                             [1, 2, 1, 1], [1, 1, 1, 1], 2))
        self.assertEqual(sparse["status"], "low_statistics")
        self.assertIsNone(sparse["p_value"])

    def test_direct_difference_moments_resolve_tiny_response(self):
        rng = np.random.default_rng(491)
        index = rng.integers(0, 8, 12000)
        b = rng.uniform(.5, 1.5, len(index))
        a = b * (1 + 1e-8 * (rng.normal(size=len(index)) + .05 * index))
        result = paired_shape_flatness(**moments_from_events(index, a, b, 8))
        self.assertEqual(result["status"], "ok_asymptotic")
        self.assertTrue(all(x > 0 for x in result["ratio_shape_se"]))
        self.assertGreater(result["statistic"], 0)

    def test_inconsistent_reference_cross_moment_is_rejected(self):
        rng = np.random.default_rng(12)
        index = rng.integers(0, 5, 3000)
        b = rng.lognormal(.2, .3, len(index))
        a = b * rng.uniform(.8, 1.4, len(index))
        moments = moments_from_events(index, a, b, 5)
        moments["sumw_cross"] *= .95
        result = paired_shape_flatness(**moments)
        self.assertEqual(result["status"], "invalid_moments")
        self.assertIn("same event/reference pair", result["notes"][-1])

    def test_chi_square_tail_fallback(self):
        for statistic in [0, .01, 1, 5, 25, 500, 1000]:
            self.assertAlmostEqual(_chi2_sf_fallback(statistic, 1), math.erfc(math.sqrt(statistic / 2)), places=14)
            self.assertAlmostEqual(_chi2_sf_fallback(statistic, 2), math.exp(-statistic / 2), places=14)
        # Published standard critical values, not generated by this module.
        for ndof, critical in [(1, 3.841458820694124), (2, 5.991464547107979),
                               (10, 18.307038053275146), (100, 124.34211340400407)]:
            self.assertAlmostEqual(_chi2_sf_fallback(critical, ndof), .05, places=12)
        try:
            from scipy.special import gammaincc
        except ImportError:
            return
        for ndof in [1, 2, 3, 9, 20, 99, 199, 400]:
            for statistic in np.geomspace(.001, 3000, 80):
                expected = gammaincc(ndof / 2, statistic / 2)
                actual = _chi2_sf_fallback(statistic, ndof)
                np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=1e-300)

    def test_seeded_null_false_positive_rate(self):
        rng = np.random.default_rng(261030)
        p_values = []
        for _ in range(1200):
            # The response is a deterministic function of an independently
            # sampled event variable. Its expected multiplier is angle-independent.
            index = rng.integers(0, 10, 6000)
            b = rng.lognormal(0, .35, len(index))
            a = b * rng.lognormal(-.2**2 / 2, .2, len(index))
            result = paired_shape_flatness(**moments_from_events(index, a, b, 10))
            self.assertEqual(result["status"], "ok_asymptotic")
            p_values.append(result["p_value"])
        rejection_fraction = float(np.mean(np.array(p_values) < .05))
        # Deliberately wider than ordinary Monte Carlo variation, fixed seed.
        self.assertGreater(rejection_fraction, .025)
        self.assertLess(rejection_fraction, .08)
        self.assertGreater(float(np.mean(p_values)), .46)
        self.assertLess(float(np.mean(p_values)), .54)
        print(f"Null calibration: 1200 trials; p<.05 fraction={rejection_fraction:.4f}; mean p={np.mean(p_values):.4f}")

    def test_shared_poisson_bootstrap_covariance_and_ratio_errors(self):
        rng = np.random.default_rng(991)
        index = rng.integers(0, 10, 9000)
        b = rng.lognormal(0, .35, len(index))
        a = b * rng.lognormal(.01 * index, .25)
        result = paired_shape_flatness(**moments_from_events(index, a, b, 10), return_covariance=True)
        d_samples, ratio_samples = [], []
        for _ in range(2400):
            # CRITICAL: the same event multiplicity multiplies both weights.
            count = rng.poisson(1, len(index))
            aa = np.bincount(index, weights=a * count, minlength=10)
            bb = np.bincount(index, weights=b * count, minlength=10)
            scale = aa.sum() / bb.sum()
            d_samples.append(aa - scale * bb)
            ratio_samples.append(aa / bb / scale)
        empirical = np.cov(np.array(d_samples).T, ddof=1)
        analytic = np.array(result["difference_covariance"])
        frobenius_relative = np.linalg.norm(empirical - analytic) / np.linalg.norm(analytic)
        self.assertLess(frobenius_relative, .10)
        ratio_empirical = np.std(ratio_samples, axis=0, ddof=1)
        ratio_analytic = np.array(result["ratio_shape_se"])
        np.testing.assert_allclose(ratio_analytic, ratio_empirical, rtol=.10)
        print(f"Shared Poisson bootstrap: 2400 replicas; covariance relative Frobenius difference={frobenius_relative:.4f}")

    def test_energy_mixing_can_change_marginal_zenith_without_angular_dial(self):
        rng = np.random.default_rng(519)
        ne, nz, n = 4, 8, 50000
        energy = rng.integers(0, ne, n)
        # Different energy mixtures at different angles, as flux/acceptance
        # can produce. The response below depends ONLY on the energy stratum.
        angle = (rng.integers(0, 5, n) + energy) % nz
        reference = rng.uniform(.8, 1.2, n)
        varied = reference * np.array([.5, .9, 1.7, 2.2])[energy]
        joint = moments_from_events(energy * nz + angle, varied, reference, ne * nz)
        conditioned_inputs = {k: v.reshape(ne, nz) for k, v in joint.items() if isinstance(v, np.ndarray)}
        result = energy_stratified_flatness(**conditioned_inputs)
        self.assertEqual(result["status"], "proportional_within_precision")
        np.testing.assert_allclose(result["standardized_ratio"], 1, atol=1e-13)
        marginal = paired_shape_flatness(**moments_from_events(angle, varied, reference, nz))
        self.assertEqual(marginal["status"], "ok_asymptotic")
        self.assertLess(marginal["p_value"], 1e-20)
        self.assertGreater(max(abs(x - 1) for x in marginal["ratio_shape"]), .1)

    def test_energy_conditioned_missing_stratum_is_not_silently_dropped(self):
        rng = np.random.default_rng(18)
        i = rng.integers(0, 8, 1000)
        m = moments_from_events(i, rng.uniform(.5, 1.5, len(i)), np.ones(len(i)), 8)
        kwargs = {k: v.reshape(2, 4) for k, v in m.items() if isinstance(v, np.ndarray)}
        result = energy_stratified_flatness(**kwargs, nonfinite_events=[0, 1])
        self.assertEqual(result["status"], "unavailable_strata")
        self.assertEqual(result["problem_strata"], [1])
        self.assertIsNone(result["p_value"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
