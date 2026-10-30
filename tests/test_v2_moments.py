"""Dense-event checks for sparse paired moments and raw quality unions."""
import unittest
import numpy as np
from atmospheric_response.v2_moments import get_difference_moments, get_response_quality_counts


class SparsePairedMomentTests(unittest.TestCase):
    def test_dense_equivalence_with_nonunity_cv_and_masked_rows(self):
        rng = np.random.default_rng(313)
        n, nc, width, nv = 19000, 3, 12, 7
        group = rng.integers(-1, nc * width, n, dtype=np.int16)
        base = rng.lognormal(-3, .8, n)
        dense = np.ones((n, nv), dtype=np.float32)
        changed = rng.choice(n, 6017, replace=False)
        dense[changed] = rng.lognormal(.1, .5, (len(changed), nv))
        rows = np.flatnonzero(np.any(dense != 1, axis=1))
        for reference in (None, 4):
            actual = get_difference_moments(group, rows, base, dense[rows], reference, (nc, width))
            bw = base if reference is None else base * dense[:, reference]
            for point in range(nv):
                aw = base * dense[:, point]
                delta = aw - bw
                keep = group >= 0
                for key, weights in (("sumw2_difference", delta * delta),
                                     ("sumw_ref_difference", bw * delta),
                                     ("sumw_cross", aw * bw)):
                    expected = np.bincount(group[keep], weights=weights[keep], minlength=nc * width).reshape(nc, width)
                    np.testing.assert_allclose(actual[key][:, point, :], expected, atol=1e-12, rtol=1e-13)
            if reference is not None:
                np.testing.assert_array_equal(actual["sumw2_difference"][:, reference, :], 0)

    def test_direct_moments_keep_near_cv_precision(self):
        rng = np.random.default_rng(612)
        n, nc, width = 7000, 2, 10
        group = rng.integers(0, nc * width, n)
        base = rng.uniform(.5, 2.5, n)
        response = np.column_stack([np.ones(n), 1 + 1e-10 * rng.normal(size=n)])
        result = get_difference_moments(group, np.arange(n), base, response, 0, (nc, width))
        exact = np.bincount(group, weights=(base * response[:, 1] - base)**2, minlength=nc * width)
        np.testing.assert_allclose(result["sumw2_difference"][:, 1, :].ravel(), exact, rtol=0, atol=0)
        self.assertTrue(np.all(exact > 0))
        a2 = np.bincount(group, weights=(base * response[:, 1])**2, minlength=nc * width)
        b2 = np.bincount(group, weights=base**2, minlength=nc * width)
        cross = np.bincount(group, weights=base**2 * response[:, 1], minlength=nc * width)
        unstable = a2 + b2 - 2 * cross
        self.assertGreater(np.max(np.abs(unstable - exact) / exact), 1000)

    def test_quality_union_matches_dense_actual_reference(self):
        group = np.array([0, 0, 1, 1, 2, 3, -1, 3])
        response = np.array([[1, np.nan, 2], [1, -2, -3], [np.inf, 1, 1],
                             [1, 1, np.nan], [1, 1, 1], [-1, 1, -2],
                             [np.nan, -1, -2], [1, 1, 1]])
        rows = np.flatnonzero(np.any(response != 1, axis=1))
        for reference in (None, 2):
            actual = get_response_quality_counts(group, rows, response[rows], reference, (2, 2))
            ref = np.ones(len(group)) if reference is None else response[:, reference]
            for point in range(3):
                var = response[:, point]
                bad = (~np.isfinite(var) | ~np.isfinite(ref)) & (group >= 0)
                neg = ((np.isfinite(var) & (var < 0)) | (np.isfinite(ref) & (ref < 0))) & (group >= 0)
                for key, mask in (("nonfinite_events", bad), ("negative_events", neg)):
                    expected = np.bincount(group[mask], minlength=4).reshape(2, 2)
                    np.testing.assert_array_equal(actual[key][:, point, :], expected)

    def test_empty_sparse_response_and_bad_alignment(self):
        group = np.array([0, 1, -1])
        for helper in [get_difference_moments, get_response_quality_counts]:
            args = [group, np.array([], dtype=int)]
            if helper is get_difference_moments:
                args.append(np.ones(3))
            args.append(np.ones((0, 2)))
            out = helper(*args, reference_index=1, group_shape=(1, 2))
            for value in out.values():
                self.assertEqual(value.shape, (1, 2, 2))
            if helper is get_difference_moments:
                np.testing.assert_array_equal(out["sumw_cross"], np.ones((1, 2, 2)))
                self.assertEqual(out["sumw2_difference"].sum(), 0)
                self.assertEqual(out["sumw_ref_difference"].sum(), 0)
            else:
                for value in out.values():
                    self.assertEqual(value.sum(), 0)
        with self.assertRaises(ValueError):
            get_difference_moments(group, np.array([0, 0]), np.ones(3), np.ones((2, 2)), group_shape=(1, 2))
        with self.assertRaises(ValueError):
            get_difference_moments(group, np.array([0]), np.ones(3), np.array([[1., np.nan]]), group_shape=(1, 2))

    def test_direct_cross_preserves_extreme_relative_rates(self):
        from atmospheric_response.v2_statistics import paired_shape_flatness
        rng = np.random.default_rng(719)
        n, width = 18000, 9
        group = rng.integers(0, width, n)
        base = rng.lognormal(-1, .5, n)
        reference = rng.uniform(.7, 1.4, n)
        response = np.column_stack([reference, reference * 1e-20,
                                    reference * 1e20])
        moments = get_difference_moments(group, np.arange(n), base, response,
                                         reference_index=0, group_shape=(1, width))
        bw = base * reference
        B = np.bincount(group, weights=bw, minlength=width)
        B2 = np.bincount(group, weights=bw * bw, minlength=width)
        for point in (1, 2):
            aw = base * response[:, point]
            cross = np.bincount(group, weights=aw * bw, minlength=width)
            np.testing.assert_allclose(moments["sumw_cross"][0, point], cross, atol=0, rtol=1e-15)
            result = paired_shape_flatness(
                np.bincount(group, weights=aw, minlength=width), B,
                np.bincount(group, weights=aw * aw, minlength=width), B2,
                moments["sumw_cross"][0, point],
                sumw2_difference=moments["sumw2_difference"][0, point],
                sumw_ref_difference=moments["sumw_ref_difference"][0, point])
            self.assertEqual(result["status"], "proportional_within_precision")
        # Reconstructing this cross moment from B2+BDelta would erase it.
        reconstructed = B2 + moments["sumw_ref_difference"][0, 1]
        self.assertTrue(np.all(reconstructed == 0))
        self.assertTrue(np.all(moments["sumw_cross"][0, 1] > 0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
