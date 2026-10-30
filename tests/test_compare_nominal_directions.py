"""Protect all truth channels in the same-event direction comparison."""
import unittest

import numpy as np

from atmospheric_response.compare_nominal_directions import (
    describe_interior_peaks, encode_profile_groups, weighted_profiles,
)


class CommonDirectionProfileTests(unittest.TestCase):
    def test_int8_categories_retain_antineutrinos_and_event_closure(self):
        # The real cache category dtype is int8. Multiplying it by 50 before
        # widening used to wrap categories 3/4 and silently lose their profiles.
        category = np.array([4, 0, 3, 1, 2, 4, 3, 0, 1, 2], dtype=np.int8)
        energy_bin = np.array([49, 0, 49, 0, 49, 49, 49, 0, 0, 49])
        weights = np.array([1., 2., 3., 4., 5., 9., 7., 8., 6., 5.])
        residual = np.column_stack([np.arange(10.)*3-12, np.arange(10.)-4])
        group = encode_profile_groups(category, energy_bin, 50)
        quantiles, neff, counts = weighted_profiles(group, weights, residual, 50)

        # Independent masks, not decoded group IDs, define the reference.
        np.testing.assert_array_equal(counts.sum(axis=1), np.bincount(category, minlength=5))
        self.assertEqual(counts.sum(), len(category))
        for truth in range(5):
            for energy in range(50):
                mask = (category == truth) & (energy_bin == energy)
                self.assertEqual(counts[truth, energy], mask.sum())
                if not mask.any():
                    self.assertTrue(np.isnan(quantiles[:, truth, energy]).all())
                    continue
                self.assertAlmostEqual(neff[truth, energy], weights[mask].sum()**2/np.square(weights[mask]).sum())
                for method in (0, 1):
                    expected = np.quantile(residual[mask, method], [.16, .84],
                                           weights=weights[mask], method="inverted_cdf")
                    np.testing.assert_array_equal(quantiles[method, truth, energy], expected)

    def test_custom_grid_can_exceed_int16_group_range(self):
        groups = encode_profile_groups(np.array([3, 4], dtype=np.int8),
                                       np.array([69999, 69998]), 70000)
        np.testing.assert_array_equal(groups, [279999, 349998])

    def test_invalid_group_is_rejected_instead_of_lost(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            encode_profile_groups(np.array([4], dtype=np.int8), np.array([50]), 50)
        with self.assertRaisesRegex(ValueError, "grid"):
            weighted_profiles(np.array([-7]), np.ones(1), np.ones((1, 2)), 50)

    def test_empty_cohort_has_no_profile_values_or_claimed_peaks(self):
        quantiles, neff, counts = weighted_profiles(np.array([], dtype=np.int16),
                                                   np.array([]), np.empty((0, 2)), 1)
        self.assertTrue(np.isnan(quantiles).all())
        self.assertEqual(neff.sum(), 0)
        self.assertEqual(counts.sum(), 0)
        edges = np.linspace(-1, 1, 201)
        spectra = np.zeros((2, 5, 200))
        sources = np.zeros((3, 5, 200))
        detail = np.zeros((5, 200, 1))
        self.assertEqual(describe_interior_peaks(edges, spectra, spectra, sources,
                                                sources, detail, detail), [])


if __name__ == "__main__":
    unittest.main()
