"""Regression boundaries for the source policy and source-row alignment."""
from pathlib import Path
import tempfile

import numpy as np
import pytest

from atmospheric_response.v2_aggregate import make_policy, direct_bin_codes, load_dial
from atmospheric_response.v2_config import load_config


CONFIG = Path(__file__).parents[1] / "atmospheric_response/configs/analysis_v2.yaml"


def test_nonfinite_is_unity_and_negative_final_weight_is_zero():
    config = load_config(CONFIG)
    raw = np.array([[np.nan, np.inf, -np.inf, -2., 0., 1., 2., 1e6, -1e6]], dtype=np.float32)
    handled = make_policy(raw, "fitter_policy", config)
    np.testing.assert_array_equal(handled, [[1, 1, 1, 0, 0, 1, 2, 9999, 0]])
    # The input remains raw and available for the audited diagnostic view.
    assert np.isnan(raw[0, 0]) and np.isposinf(raw[0, 1]) and raw[0, 3] == -2


def test_exact_sample_edges_including_zero_final_edge_and_flows():
    # Different sample grids share internal storage, but no physical bin edge
    # may be shifted or inferred from another sample's binning.
    values = np.array([-.1, 0, .2, 100., 101., .4, 100., np.nan])
    category = np.array([0, 0, 0, 0, 0, 1, 1, 1])
    edges = {0: np.array([0, .2, 100.]), 1: np.array([0, .4, .6, 100.])}
    codes, valid, _ = direct_bin_codes(values, np.isfinite(values), category, edges)
    np.testing.assert_array_equal(codes[:7], [-1, 0, 1, 1, 3, 1, 2])
    assert not valid[-1]


@pytest.mark.parametrize("bad_rows", [np.array([-1]), np.array([2]), np.array([0.5])])
def test_corrupt_sparse_row_cannot_wrap_or_silently_coerce(bad_rows):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)
        np.save(path / "selection.npy", [0, 1])
        np.savez(path / "part.npz", d000_rows=bad_rows, d000_responses=np.ones((1, 1)))
        sources = [{"cache": str(path / "part.npz"), "selection_map": str(path / "selection.npy")}]
        with pytest.raises(ValueError, match="row"):
            load_dial(sources, np.array([[1]]), 0, 1, path)
