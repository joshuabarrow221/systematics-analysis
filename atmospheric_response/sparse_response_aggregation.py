"""Sparse, event-level response aggregation without rereading source CAFs.

The response adapter supplies only rows where at least one actual dial point is
different from unity. This module neither selects CV points nor renormalizes a
response. It keeps raw signed histograms and declares signed-weight quantiles
undefined. ``clip_to_zero`` is an explicitly different calculation.
"""

from __future__ import annotations

import numpy as np

from .response_aggregation import _edges, _require, inverted_cdf_sorted


def sparse_inverted_cdf(sorted_values, baseline_cdf, active_ranks, delta_weight,
                        quantiles=(0.16, 0.84)):
    """Invert a positive weighted empirical CDF using sparse weight increments.

    The caller verifies nonnegative final weights. Ranks must be unique, sorted,
    and local to this group. There is no residual histogram approximation: every
    search result is an original event value. Floating summation differs from a
    full NumPy cumsum; threshold ties within rounding error can select a neighbor.
    """
    x, cdf = np.asarray(sorted_values), np.asarray(baseline_cdf)
    ranks = np.asarray(active_ranks)
    delta = np.asarray(delta_weight, dtype=np.float64)
    q = np.asarray(quantiles, dtype=np.float64)
    _require(x.ndim == 1 and len(x) and cdf.shape == x.shape, "Invalid baseline CDF")
    _require(ranks.ndim == 1 and ranks.dtype.kind in "iu" and delta.shape == ranks.shape,
             "Sparse ranks/deltas are not aligned")
    _require(not len(ranks) or (ranks[0] >= 0 and ranks[-1] < len(x)
             and np.all(ranks[1:] > ranks[:-1])), "Sparse ranks must be unique, sorted, in range")
    _require(np.all(np.isfinite(delta)), "Nonfinite sparse increments")
    _require(q.ndim == 1 and np.all(np.isfinite(q)) and np.all((q >= 0) & (q <= 1)), "Invalid quantiles")
    dcdf = np.cumsum(delta, dtype=np.float64)
    total = float(cdf[-1] + (dcdf[-1] if len(dcdf) else 0.0))
    _require(np.isfinite(total) and total > 0, "Final CDF total must be finite and positive")
    answer = np.empty(len(q), dtype=x.dtype)
    for iq, level in enumerate(q):
        low, high = 0, len(x) - 1
        while low < high:
            mid = (low + high) // 2
            pos = int(np.searchsorted(ranks, mid, side="right")) - 1
            cumulative = float(cdf[mid] + (dcdf[pos] if pos >= 0 else 0.0))
            # NumPy ignores leading zero weights even for q=0.
            if cumulative <= 0 or cumulative / total < level:
                low = mid + 1
            else:
                high = mid
        answer[iq] = x[low]
    return answer


class SparseResponseStudy:
    """Precompute a common baseline, then process one sparse dial at a time.

    Arguments match ResponseStudy: category/base_weight are aligned N-vectors;
    histograms map names to values/mask/edges; profile contains energy, residual,
    mask and edges. Large source vectors may be read-only local memmaps.

    aggregate(rows, responses) takes UNIQUE global event rows and a matrix
    (len(rows), number_of_actual_values), in the actual metadata value order.
    Omitted rows have response exactly one. No CV index/value is assumed.

    Retained indexing memory is approximately N*(2*number_of_histograms+4) plus
    N_profile*20 bytes, baseline weights, and small aggregate arrays. Arrays of
    input observables are not retained. One response column is evaluated at once.
    """

    def __init__(self, category, base_weight, histograms, profile, category_codes=(0, 1, 2, 3, 4)):
        category = np.asarray(category)
        self.base = np.asarray(base_weight, dtype=np.float64)
        self.n = len(category)
        _require(category.ndim == 1 and self.base.shape == (self.n,), "Baseline arrays are not aligned")
        _require(np.all(np.isfinite(self.base) & (self.base >= 0)), "Invalid baseline weights")
        rank_dtype = np.int32 if self.n <= np.iinfo(np.int32).max else np.int64
        self.category_codes = tuple(category_codes)
        _require(self.category_codes and len(set(self.category_codes)) == len(self.category_codes), "Invalid categories")
        self.nc = len(self.category_codes)
        ci = np.full(self.n, -1, dtype=np.int16)
        for i, code in enumerate(self.category_codes):
            ci[category == code] = i
        self.histograms = {}
        for name, spec in histograms.items():
            values, mask = np.asarray(spec["values"]), np.asarray(spec["mask"])
            edges = _edges(spec["edges"], name)
            _require(values.shape == mask.shape == (self.n,) and mask.dtype == bool, f"{name}: invalid values/mask")
            selected = mask & (ci >= 0)
            _require(np.all(np.isfinite(values[selected])), f"{name}: nonfinite selected values")
            width = len(edges) + 1  # visible bins plus underflow/overflow
            dtype = np.int16 if self.nc * width <= np.iinfo(np.int16).max else np.int32
            group = np.full(self.n, -1, dtype=dtype)
            rows = np.flatnonzero(selected).astype(rank_dtype)
            slot = np.searchsorted(edges, values[rows], side="right")
            slot[values[rows] == edges[-1]] = len(edges) - 1
            group[rows] = ci[rows] * width + slot
            group_rows = group[rows]
            size, shape = self.nc * width, (self.nc, width)
            w = self.base[rows]
            self.histograms[name] = {"group": group, "edges": edges, "width": width,
                "sumw": np.bincount(group_rows, weights=w, minlength=size).reshape(shape),
                "sumw2": np.bincount(group_rows, weights=w * w, minlength=size).reshape(shape),
                "counts": np.bincount(group_rows, minlength=size).reshape(shape),
                "positive_counts": np.bincount(group_rows[w > 0], minlength=size).reshape(shape)}
            del values, mask, selected, rows, slot, group_rows, w
        energy, residual, mask = (np.asarray(profile[k]) for k in ("energy", "residual", "mask"))
        _require(energy.shape == residual.shape == mask.shape == (self.n,) and mask.dtype == bool, "Invalid profile fields")
        self.profile_edges = _edges(profile["edges"], "profile")
        self.nb = len(self.profile_edges) - 1
        bin_dtype = np.int16 if self.nb <= np.iinfo(np.int16).max else np.int32
        bins = np.searchsorted(self.profile_edges, energy, side="right").astype(bin_dtype) - 1
        bins[energy == self.profile_edges[-1]] = self.nb - 1
        selected = mask & (ci >= 0)
        _require(np.all(np.isfinite(energy[selected])) and np.all(np.isfinite(residual[selected])), "Nonfinite profile fields")
        self.profile_outside_events = int(np.count_nonzero(selected & ((bins < 0) | (bins >= self.nb))))
        selected &= (bins >= 0) & (bins < self.nb)
        rows = np.flatnonzero(selected).astype(rank_dtype)
        group_dtype = np.int16 if self.nc * self.nb <= np.iinfo(np.int16).max else np.int32
        groups = ci[rows].astype(group_dtype) * self.nb + bins[rows]
        order = np.lexsort((residual[rows], groups))
        sorted_rows = rows[order]
        self.profile_rows = sorted_rows
        self.profile_residual = np.asarray(residual[sorted_rows], dtype=np.float64)
        self.profile_rank = np.full(self.n, -1, dtype=rank_dtype)
        self.profile_rank[sorted_rows] = np.arange(len(sorted_rows), dtype=rank_dtype)
        self.boundaries = np.searchsorted(groups[order], np.arange(self.nc * self.nb + 1), side="left")
        del order, groups, rows, bins, ci, selected
        self.profile_cdf = np.empty(len(sorted_rows), dtype=np.float64)
        ng = self.nc * self.nb
        self.profile_base = {k: np.zeros(ng) for k in ("sumw", "sumw2", "n_eff")}
        self.profile_base.update({k: np.full(ng, np.nan) for k in ("q16", "q84", "r68")})
        self.profile_base["events"] = np.diff(self.boundaries)
        self.profile_base["positive_weight_events"] = np.zeros(ng, dtype=np.int64)
        for g, (start, stop) in enumerate(zip(self.boundaries[:-1], self.boundaries[1:])):
            w = self.base[self.profile_rows[start:stop]]
            cdf = np.cumsum(w, dtype=np.float64)
            self.profile_cdf[start:stop] = cdf
            if not len(w):
                continue
            sw, sw2 = cdf[-1], np.dot(w, w)
            self.profile_base["sumw"][g], self.profile_base["sumw2"][g] = sw, sw2
            self.profile_base["positive_weight_events"][g] = np.count_nonzero(w > 0)
            if sw > 0:
                q16, q84 = sparse_inverted_cdf(self.profile_residual[start:stop], cdf,
                                               np.array([], dtype=np.int64), np.array([]))
                self.profile_base["q16"][g], self.profile_base["q84"][g] = q16, q84
                self.profile_base["r68"][g] = (q84 - q16) / 2
                self.profile_base["n_eff"][g] = sw * sw / sw2

    def aggregate(self, rows, responses, negative_policy="preserve"):
        """Aggregate one actual dial, keeping its original point order.

        preserve: signed rates; any negative final event weight makes that
        profile bin undefined. clip_to_zero: final weights below zero are set
        to zero explicitly. Nonfinite responses always raise in both modes.

        In the v2 app, make_policy has already applied the selected handling
        and this method is called with preserve for BOTH views. Thus this
        low-level negative_policy argument is not the viewer's Weight policy.
        For a visible histogram bin k and dial point a, the saved rate is
          H_k(a) = sum_{events i in bin k} b_i * r_i(a).
        The corresponding MC second moment is sum_i (b_i*r_i(a))**2, with
        explicit underflow/overflow slots. Counts are unchanged event counts,
        not weighted yields. Unit-area normalization is a later display step.

        Rows absent from the sparse input have r_i(a)=1 and remain in every
        spectrum. Adding delta=b_i*(r_i(a)-1) to the common baseline therefore
        gives the same rate as summing all final weights directly; sparsity
        neither drops events nor divides responses by their stored CV.
        """
        rows, response = np.asarray(rows), np.asarray(responses)
        _require(rows.ndim == 1 and rows.dtype.kind in "iu", "Sparse event rows must be integers")
        _require(not len(rows) or (np.min(rows) >= 0 and np.max(rows) < self.n), "Sparse event row out of range")
        _require(len(np.unique(rows)) == len(rows), "Sparse event rows must be unique")
        _require(response.ndim == 2 and response.shape[0] == len(rows) and response.shape[1] > 0
                 and response.dtype.kind in "fiu", "Sparse response matrix has wrong shape/type")
        _require(negative_policy in ("preserve", "clip_to_zero"), "Unknown negative-weight policy")
        # Validate columnwise so an n_active x n_values Boolean copy is not needed.
        for j in range(response.shape[1]):
            _require(np.all(np.isfinite(response[:, j])), f"Nonfinite sparse response in column {j}")
        nv, ng = response.shape[1], self.nc * self.nb
        hist = {}
        for name, spec in self.histograms.items():
            hist[name] = {"edges": spec["edges"].copy(), "counts_with_flow": spec["counts"].copy(),
                "sumw_with_flow": np.repeat(spec["sumw"][:, None, :], nv, axis=1),
                "sumw2_with_flow": np.repeat(spec["sumw2"][:, None, :], nv, axis=1)}
        profile = {k: np.repeat(self.profile_base[k].reshape(self.nc, 1, self.nb), nv, axis=1)
                   for k in ("sumw", "sumw2", "n_eff", "q16", "q84", "r68", "positive_weight_events")}
        profile["negative_weight_events"] = np.zeros((self.nc, nv, self.nb), dtype=np.int64)
        profile["events"] = self.profile_base["events"].reshape(self.nc, self.nb).copy()
        profile["edges"] = self.profile_edges.copy()
        ranks = self.profile_rank[rows]
        active_positions = np.flatnonzero(ranks >= 0)
        active_positions = active_positions[np.argsort(ranks[active_positions], kind="stable")]
        sorted_ranks = ranks[active_positions]
        divisions = np.searchsorted(sorted_ranks, self.boundaries, side="left")
        base = self.base[rows]
        audit = {"negative_policy": negative_policy, "sparse_rows": len(rows), "values": [],
            "category_codes": list(self.category_codes), "profile_outside_events": self.profile_outside_events,
            "dense_profile_fallbacks_for_cancellation": 0,
            "profile_definition": "Event-level sparse baseline-plus-delta weighted inverted CDF; float64 prefix sums; no residual histogram approximation"}
        for j in range(nv):
            raw = np.asarray(response[:, j], dtype=np.float64)
            r = np.maximum(raw, 0) if negative_policy == "clip_to_zero" else raw
            final_weight = base * r
            _require(np.all(np.isfinite(final_weight)) and np.all(np.isfinite(final_weight * final_weight)),
                     f"Combined sparse weights/squares overflow in column {j}")
            delta, delta2 = base * (r - 1), final_weight * final_weight - base * base
            for name, spec in self.histograms.items():
                group = spec["group"][rows]
                keep = group >= 0
                size = self.nc * spec["width"]
                shape = (self.nc, spec["width"])
                hist[name]["sumw_with_flow"][:, j, :] += np.bincount(group[keep], weights=delta[keep], minlength=size).reshape(shape)
                hist[name]["sumw2_with_flow"][:, j, :] += np.bincount(group[keep], weights=delta2[keep], minlength=size).reshape(shape)
                # Exact empty final-weight bins should not acquire tiny signed
                # roundoff remnants from subtracting a complete baseline sum.
                removed = keep & (base > 0) & (final_weight == 0)
                remaining = spec["positive_counts"] - np.bincount(group[removed], minlength=size).reshape(shape)
                empty = remaining == 0
                hist[name]["sumw_with_flow"][:, j, :][empty] = 0
                hist[name]["sumw2_with_flow"][:, j, :][empty] = 0
                # When every positive baseline event in a bin is in the sparse
                # response, sum its final weights directly. This also protects
                # very small nonzero responses from baseline-minus-baseline loss.
                covered = keep & (base > 0)
                complete = (spec["positive_counts"] > 0) & (spec["positive_counts"] ==
                    np.bincount(group[covered], minlength=size).reshape(shape))
                if np.any(complete):
                    final_sum = np.bincount(group[keep], weights=final_weight[keep], minlength=size).reshape(shape)
                    final_sum2 = np.bincount(group[keep], weights=final_weight[keep] ** 2, minlength=size).reshape(shape)
                    hist[name]["sumw_with_flow"][:, j, :][complete] = final_sum[complete]
                    hist[name]["sumw2_with_flow"][:, j, :][complete] = final_sum2[complete]
            for g in np.flatnonzero(np.diff(divisions)):
                icat, ibin = divmod(int(g), self.nb)
                slot = (icat, j, ibin)
                start, stop = self.boundaries[g:g + 2]
                pos = active_positions[divisions[g]:divisions[g + 1]]
                local_ranks = sorted_ranks[divisions[g]:divisions[g + 1]] - start
                dw = delta[pos]
                sw = self.profile_base["sumw"][g] + np.sum(dw)
                sw2 = self.profile_base["sumw2"][g] + np.sum(delta2[pos])
                negative = int(np.count_nonzero(final_weight[pos] < 0))
                positive = int(self.profile_base["positive_weight_events"][g]
                               - np.count_nonzero((base[pos] > 0) & (final_weight[pos] <= 0)))
                dense_weights = None
                tolerance = 64 * np.finfo(np.float64).eps
                if not negative and positive > 0 and (
                        sw <= tolerance * self.profile_base["sumw"][g]
                        or sw2 <= tolerance * self.profile_base["sumw2"][g]):
                    dense_weights = self.base[self.profile_rows[start:stop]].copy()
                    dense_weights[local_ranks] = final_weight[pos]
                    sw, sw2 = float(np.sum(dense_weights)), float(np.dot(dense_weights, dense_weights))
                    audit["dense_profile_fallbacks_for_cancellation"] += 1
                profile["sumw"][slot], profile["sumw2"][slot] = sw, sw2
                profile["negative_weight_events"][slot] = negative
                profile["positive_weight_events"][slot] = positive
                profile["n_eff"][slot] = sw * sw / sw2 if sw2 > 0 and not negative else 0
                if negative or positive == 0 or sw <= 0:
                    # A signed histogram is algebraically meaningful, but a
                    # probability CDF must be nondecreasing. One negative final
                    # event weight invalidates Q16/Q84 for this energy group,
                    # even if its total weight remains positive. No absolute
                    # value or hidden positive-only subset is substituted.
                    if positive == 0 and not negative:
                        for key in ("sumw", "sumw2", "n_eff"):
                            profile[key][slot] = 0
                    for key in ("q16", "q84", "r68"):
                        profile[key][slot] = np.nan
                    continue
                if dense_weights is None:
                    q16, q84 = sparse_inverted_cdf(self.profile_residual[start:stop], self.profile_cdf[start:stop], local_ranks, dw)
                else:
                    q16, q84 = inverted_cdf_sorted(self.profile_residual[start:stop], dense_weights)
                profile["q16"][slot], profile["q84"][slot] = q16, q84
                profile["r68"][slot] = (q84 - q16) / 2
            audit["values"].append({"column": j,
                "raw_negative_weight_events": int(np.count_nonzero((raw < 0) & (base > 0))),
                "raw_zero_weight_events_in_sparse_rows": int(np.count_nonzero((raw == 0) & (base > 0))),
                "undefined_profile_bins": int(np.count_nonzero(~np.isfinite(profile["r68"][:, j, :]))),
                "negative_weight_profile_bins": int(np.count_nonzero(profile["negative_weight_events"][:, j, :] > 0))})
        return {"histograms": hist, "profile": profile, "audit": audit}
