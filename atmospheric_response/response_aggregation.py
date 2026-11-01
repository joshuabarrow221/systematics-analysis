"""Source-independent aggregation of explicitly aligned systematic responses.

No files are opened here. Supply the actual parameter grid/CV separately to the
canvas manifest. Inputs are event-aligned response multipliers; this module does
not infer their normalization, repair nonfinite responses, or invent knots.

For large studies, store responses value-major on local disk and pass a transpose
view (events, values) of one parameter/bounded column block. One response column
is multiplied into the baseline at a time. Histogram bin assignments and sorted
angular residuals are prepared once and reused across every parameter.
"""

from __future__ import annotations

import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _edges(value, label):
    value = np.asarray(value, dtype=np.float64)
    _require(value.ndim == 1 and len(value) >= 2 and np.all(np.isfinite(value))
             and np.all(np.diff(value) > 0), f"{label}: invalid edges")
    return value


def inverted_cdf_sorted(sorted_values, sorted_weights, quantiles=(0.16, 0.84)):
    """NumPy weighted inverted-CDF definition, with sorting already performed.

    Matches NumPy's float64 cumulative-sum normalization/searchsorted convention.
    Sorting residuals is reusable because weight-only dials never change them.
    Negative, nonfinite, and all-zero weights are not meaningful quantile inputs.
    """
    x = np.asarray(sorted_values)
    w = np.asarray(sorted_weights, dtype=np.float64)
    q = np.asarray(quantiles, dtype=np.float64)
    _require(x.ndim == 1 and w.shape == x.shape and len(x), "Quantile arrays must be nonempty and aligned")
    _require(np.all(np.isfinite(x)) and np.all(np.isfinite(w)) and np.all(w >= 0),
             "Weighted quantiles require finite values and nonnegative finite weights")
    _require(q.ndim == 1 and np.all(np.isfinite(q)) and np.all((q >= 0) & (q <= 1)), "Invalid quantiles")
    _require(np.all(x[1:] >= x[:-1]), "Residual array must already be sorted")
    cdf = np.cumsum(w, dtype=np.float64)
    _require(np.isfinite(cdf[-1]) and cdf[-1] > 0, "Quantile weights must have a finite positive sum")
    cdf /= cdf[-1]
    cdf[cdf == 0] = -1  # NumPy's treatment of leading zero weights at q=0.
    index = np.minimum(np.searchsorted(cdf, q, side="left"), len(x) - 1)
    return x[index]


class ResponseStudy:
    """Precompute masks/binning, then aggregate one parameter or value block.

    category: one code per event; category_codes gives output-axis order.
    base_weight: finite, nonnegative, already assembled nominal normalization.
    histograms: {name: {'values': N-vector, 'mask': N-bool, 'edges': bin edges}}.
    profile: {'energy': N-vector, 'residual': N-vector, 'mask': N-bool,
              'edges': true-energy bin edges} for the signed angular R68 profile.

    aggregate(response) accepts shape (N_events, N_values), including np.memmap
    views. Returned histogram/profile arrays have axes (category, value, bin).
    The caller must keep the supplied value order attached to its actual grid.
    """

    def __init__(self, category, base_weight, histograms, profile, category_codes=(0, 1, 2, 3, 4)):
        self.category = np.asarray(category)
        self.base_weight = np.asarray(base_weight, dtype=np.float64)
        self.n = len(self.category)
        _require(self.category.ndim == 1 and self.base_weight.shape == (self.n,), "Event arrays are not aligned")
        _require(np.all(np.isfinite(self.base_weight) & (self.base_weight >= 0)), "Invalid nominal weights")
        self.category_codes = tuple(category_codes)
        _require(len(set(self.category_codes)) == len(self.category_codes) and self.category_codes,
                 "category_codes must be nonempty and unique")
        self.nc = len(self.category_codes)
        self.category_index = np.full(self.n, -1, dtype=np.int32)
        for index, code in enumerate(self.category_codes):
            self.category_index[self.category == code] = index
        self.active = np.zeros(self.n, dtype=bool)
        self.histograms = {}
        for name, spec in histograms.items():
            values = np.asarray(spec["values"], dtype=np.float64)
            mask = np.asarray(spec["mask"])
            edges = _edges(spec["edges"], name)
            _require(values.shape == (self.n,) and mask.shape == (self.n,) and mask.dtype == bool,
                     f"{name}: values/mask must align with events; masks must be Boolean")
            selected = mask & (self.category_index >= 0)
            _require(np.all(np.isfinite(values[selected])), f"{name}: selected nonfinite observable; define an explicit finite mask")
            rows = np.flatnonzero(selected)
            # Slots: 0 underflow; 1..nbin visible bins; nbin+1 overflow.
            slots = np.searchsorted(edges, values[rows], side="right")
            slots[values[rows] == edges[-1]] = len(edges) - 1
            width = len(edges) + 1
            group = self.category_index[rows] * width + slots
            self.histograms[name] = {"rows": rows, "group": group, "width": width, "edges": edges,
                "counts_with_flow": np.bincount(group, minlength=self.nc * width).reshape(self.nc, width)}
            self.active |= selected

        energy = np.asarray(profile["energy"], dtype=np.float64)
        residual = np.asarray(profile["residual"], dtype=np.float64)
        mask = np.asarray(profile["mask"])
        self.profile_edges = _edges(profile["edges"], "angular profile")
        _require(energy.shape == residual.shape == mask.shape == (self.n,) and mask.dtype == bool,
                 "Profile fields/mask must align with events; mask must be Boolean")
        selected = mask & (self.category_index >= 0)
        _require(np.all(np.isfinite(energy[selected])) and np.all(np.isfinite(residual[selected])),
                 "Selected profile values must be finite")
        self.profile_bins = len(self.profile_edges) - 1
        ibin = np.searchsorted(self.profile_edges, energy, side="right") - 1
        ibin[energy == self.profile_edges[-1]] = self.profile_bins - 1
        self.profile_outside = selected & ((ibin < 0) | (ibin >= self.profile_bins))
        selected &= (ibin >= 0) & (ibin < self.profile_bins)
        self.active |= selected
        rows = np.flatnonzero(selected)
        group = self.category_index[rows] * self.profile_bins + ibin[rows]
        order = np.lexsort((residual[rows], group))
        self.profile_rows = rows[order]
        self.profile_residual = residual[self.profile_rows]
        sorted_group = group[order]
        boundaries = np.searchsorted(sorted_group, np.arange(self.nc * self.profile_bins + 1), side="left")
        self.profile_groups = [(int(boundaries[g]), int(boundaries[g + 1]))
                               for g in range(self.nc * self.profile_bins)]
        self.profile_counts = np.diff(boundaries).reshape(self.nc, self.profile_bins)

    def aggregate(self, responses):
        """Preserve signed rate weights; mark negative-weight profile bins undefined.

        Nonfinite responses on any active event raise before an output for that
        column is returned. Do not cure them by replacing with one or clipping.
        A source adapter may apply a documented production policy explicitly,
        recording raw diagnostics separately before calling this function.

        This dense reference engine has the same weight algebra as the sparse
        production path: H_k(a)=sum_{i in k} base_weight_i*response_i(a), with
        sumw2=sum_i(final_weight_i**2). It does not add flux, oscillations, CV
        factors or an exposure normalization; the caller already assembled the
        baseline. Histogram outputs are bin sums, not unit-area densities.
        The viewer's Weight policy is implemented in v2_aggregate.make_policy;
        this generic engine does not silently choose a scientific policy.
        """
        response = np.asarray(responses)
        _require(response.ndim == 2 and response.shape[0] == self.n and response.shape[1] > 0,
                 f"Expected responses ({self.n}, N_values); got {response.shape}")
        _require(response.dtype.kind in "fiu", "Responses must be real numeric arrays")
        nv = response.shape[1]
        hist = {}
        for name, spec in self.histograms.items():
            shape = (self.nc, nv, spec["width"])
            hist[name] = {"sumw_with_flow": np.empty(shape), "sumw2_with_flow": np.empty(shape),
                          "counts_with_flow": spec["counts_with_flow"].copy(), "edges": spec["edges"].copy()}
        shape = (self.nc, nv, self.profile_bins)
        profile = {key: np.full(shape, np.nan) for key in ("q16", "q84", "r68")}
        profile.update({key: np.zeros(shape, dtype=np.float64) for key in ("sumw", "sumw2", "n_eff")})
        profile.update({key: np.zeros(shape, dtype=np.int64) for key in ("positive_weight_events", "negative_weight_events")})
        profile["events"] = self.profile_counts.copy()
        profile["edges"] = self.profile_edges.copy()
        audit = {"values": [], "category_codes": list(self.category_codes),
                 "profile_definition": "Exact sorted weighted inverted-CDF q16/q84, same float64 cumsum normalization as NumPy",
                 "profile_outside_events": int(np.count_nonzero(self.profile_outside))}
        for ivalue in range(nv):
            column = response[:, ivalue]
            active_finite = np.isfinite(column[self.active])
            _require(np.all(active_finite),
                     f"Response column {ivalue}: {np.count_nonzero(~active_finite)} nonfinite active-event values")
            with np.errstate(over="ignore", invalid="ignore"):
                weight = self.base_weight * column
            _require(np.all(np.isfinite(weight[self.active])), f"Column {ivalue}: nonfinite combined weights")
            for name, spec in self.histograms.items():
                w = weight[spec["rows"]]
                shape2 = (self.nc, spec["width"])
                hist[name]["sumw_with_flow"][:, ivalue, :] = np.bincount(spec["group"], weights=w,
                                                          minlength=np.prod(shape2)).reshape(shape2)
                hist[name]["sumw2_with_flow"][:, ivalue, :] = np.bincount(spec["group"], weights=w * w,
                                                          minlength=np.prod(shape2)).reshape(shape2)
            sorted_weight = weight[self.profile_rows]
            for group, (start, stop) in enumerate(self.profile_groups):
                icat, ibin = divmod(group, self.profile_bins)
                w = sorted_weight[start:stop]
                slot = (icat, ivalue, ibin)
                sumw, sumw2 = float(np.sum(w)), float(np.dot(w, w))
                profile["sumw"][slot], profile["sumw2"][slot] = sumw, sumw2
                profile["positive_weight_events"][slot] = int(np.count_nonzero(w > 0))
                nnegative = int(np.count_nonzero(w < 0))
                profile["negative_weight_events"][slot] = nnegative
                if sumw2 > 0 and nnegative == 0:
                    profile["n_eff"][slot] = sumw * sumw / sumw2
                if start == stop or sumw <= 0 or nnegative:
                    continue
                q16, q84 = inverted_cdf_sorted(self.profile_residual[start:stop], w)
                profile["q16"][slot], profile["q84"][slot] = q16, q84
                profile["r68"][slot] = (q84 - q16) / 2
            audit["values"].append({"column": ivalue,
                "active_events": int(np.count_nonzero(self.active)),
                "negative_weight_events": int(np.count_nonzero(weight[self.active] < 0)),
                "zero_weight_events": int(np.count_nonzero(weight[self.active] == 0)),
                "undefined_profile_bins": int(np.count_nonzero(~np.isfinite(profile["r68"][:, ivalue, :]))),
                "negative_weight_profile_bins": int(np.count_nonzero(profile["negative_weight_events"][:, ivalue, :] > 0))})
        return {"histograms": hist, "profile": profile, "audit": audit}


def from_cached_fields(cache, energy_edges=None, cosine_edges=None, profile_edges=None):
    """Convenience adapter for the existing cache schema; no disk or ROOT access."""
    from .build_nominal import prepare_events

    prep = prepare_events(cache)
    energy_edges = np.geomspace(0.001, 10, 201) if energy_edges is None else energy_edges
    cosine_edges = np.linspace(-1, 1, 201) if cosine_edges is None else cosine_edges
    profile_edges = np.logspace(-1, 1, 51) if profile_edges is None else profile_edges
    hist = {name: {"values": value, "mask": prep["masks"][name],
                   "edges": cosine_edges if "coszenith" in name else energy_edges}
            for name, value in prep["values"].items() if name != "angular_resolution"}
    profile = {"energy": cache["energy"], "residual": prep["values"]["angular_resolution"],
               "mask": prep["masks"]["angular_resolution"], "edges": profile_edges}
    return ResponseStudy(cache["category"], prep["weight"], hist, profile), prep


class StreamingHistograms:
    """Bounded-memory histogram accumulation while source files are read once.

    histogram_edges maps observable IDs to edges. update() receives per-chunk
    {observable: {'values': vector, 'mask': Boolean vector}} and a response matrix
    (chunk events, all stored points). Output slots include explicit underflow
    and overflow. Counts have no value axis; weighting never changes masks.
    """

    def __init__(self, histogram_edges, n_values, category_codes=(0, 1, 2, 3, 4)):
        _require(isinstance(n_values, int) and n_values > 0, "n_values must be positive")
        self.n_values = n_values
        self.category_codes = tuple(category_codes)
        _require(len(set(self.category_codes)) == len(self.category_codes) and self.category_codes,
                 "category_codes must be unique")
        self.nc = len(self.category_codes)
        self.outputs = {}
        for name, edge in histogram_edges.items():
            edge = _edges(edge, name)
            width = len(edge) + 1
            self.outputs[name] = {"edges": edge, "sumw_with_flow": np.zeros((self.nc, n_values, width)),
                "sumw2_with_flow": np.zeros((self.nc, n_values, width)),
                "counts_with_flow": np.zeros((self.nc, width), dtype=np.int64)}
        self.chunks = 0
        self.events = 0

    def update(self, category, base_weight, observables, responses, block_size=32):
        category = np.asarray(category)
        base_weight = np.asarray(base_weight, dtype=np.float64)
        response = np.asarray(responses)
        n = len(category)
        _require(category.shape == base_weight.shape == (n,), "Chunk event fields are not aligned")
        _require(response.shape == (n, self.n_values) and response.dtype.kind in "fiu", "Chunk response shape/type mismatch")
        _require(isinstance(block_size, int) and block_size > 0, "block_size must be positive")
        _require(np.all(np.isfinite(base_weight) & (base_weight >= 0)), "Invalid chunk baseline weights")
        _require(set(observables) == set(self.outputs), "Observable IDs differ from accumulator")
        ci = np.full(n, -1, dtype=np.int32)
        for i, code in enumerate(self.category_codes):
            ci[category == code] = i
        prepared = {}
        active = np.zeros(n, dtype=bool)
        for name, spec in observables.items():
            values = np.asarray(spec["values"])
            mask = np.asarray(spec["mask"])
            _require(values.shape == mask.shape == (n,) and mask.dtype == bool, f"{name}: invalid chunk observable/mask")
            selected = mask & (ci >= 0)
            _require(np.all(np.isfinite(values[selected])), f"{name}: nonfinite selected values")
            rows = np.flatnonzero(selected)
            output = self.outputs[name]
            edges = output["edges"]
            width = len(edges) + 1
            slot = np.searchsorted(edges, values[rows], side="right")
            slot[values[rows] == edges[-1]] = len(edges) - 1
            group = ci[rows] * width + slot
            order = np.argsort(group, kind="stable")
            sorted_group = group[order]
            starts = np.r_[0, np.flatnonzero(np.diff(sorted_group)) + 1] if len(rows) else np.array([], dtype=int)
            groups = sorted_group[starts] if len(rows) else np.array([], dtype=int)
            prepared[name] = {"rows": rows[order], "starts": starts, "groups": groups, "width": width,
                              "counts": np.bincount(group, minlength=self.nc * width).reshape(self.nc, width)}
            active |= selected
        # Preflight every column before mutating accumulated counts/sums.
        for start in range(0, self.n_values, block_size):
            block = response[:, start:start + block_size]
            _require(np.all(np.isfinite(block[active])), f"Nonfinite active responses in columns {start}:{start + block_size}")
        for start in range(0, self.n_values, block_size):
            stop = min(start + block_size, self.n_values)
            weighted = np.asarray(response[:, start:stop], dtype=np.float64) * base_weight[:, None]
            _require(np.all(np.isfinite(weighted[active])), "Combined response weights overflowed")
            for name, pre in prepared.items():
                if not len(pre["rows"]):
                    continue
                selected_weights = weighted[pre["rows"]]
                sums = np.add.reduceat(selected_weights, pre["starts"], axis=0)
                sums2 = np.add.reduceat(selected_weights * selected_weights, pre["starts"], axis=0)
                icat, ibin = np.divmod(pre["groups"], pre["width"])
                self.outputs[name]["sumw_with_flow"][icat, start:stop, ibin] += sums
                self.outputs[name]["sumw2_with_flow"][icat, start:stop, ibin] += sums2
        for name, pre in prepared.items():
            self.outputs[name]["counts_with_flow"] += pre["counts"]
        self.events += n
        self.chunks += 1


def profile_group(residual, base_weight, responses, block_size=32):
    """Exact weighted R68 for one category/energy bin, all actual response points.

    Intended for a bounded local spool group after the one remote streaming pass.
    responses may be a local memmap (rows in this bin, all response points).
    Residual sorting occurs once. Only block_size columns are converted to double
    and cumulatively summed together. Negative-weight points retain sumw/sumw² but
    have undefined quantiles; no clipping or fabricated values are introduced.
    """
    x = np.asarray(residual, dtype=np.float64)
    base = np.asarray(base_weight, dtype=np.float64)
    response = np.asarray(responses)
    _require(x.ndim == 1 and base.shape == x.shape and response.ndim == 2 and response.shape[0] == len(x),
             "Profile group arrays are not aligned")
    _require(response.dtype.kind in "fiu" and isinstance(block_size, int) and block_size > 0, "Invalid response/block type")
    _require(np.all(np.isfinite(x)) and np.all(np.isfinite(base) & (base >= 0)), "Invalid profile group baseline")
    nv = response.shape[1]
    result = {key: np.full(nv, np.nan) for key in ("q16", "q84", "r68")}
    result.update({key: np.zeros(nv) for key in ("sumw", "sumw2", "n_eff")})
    result.update({key: np.zeros(nv, dtype=np.int64) for key in ("positive_weight_events", "negative_weight_events")})
    result["events"] = len(x)
    if not len(x):
        return result
    order = np.argsort(x, kind="stable")
    sorted_x, sorted_base = x[order], base[order]
    for start in range(0, nv, block_size):
        stop = min(start + block_size, nv)
        raw = response[order, start:stop]
        _require(np.all(np.isfinite(raw)), f"Nonfinite profile responses in columns {start}:{stop}")
        weights = np.asarray(raw, dtype=np.float64) * sorted_base[:, None]
        _require(np.all(np.isfinite(weights)), "Combined profile weights overflowed")
        sumw, sumw2 = np.sum(weights, axis=0), np.einsum("ij,ij->j", weights, weights)
        negative = np.count_nonzero(weights < 0, axis=0)
        result["sumw"][start:stop], result["sumw2"][start:stop] = sumw, sumw2
        result["negative_weight_events"][start:stop] = negative
        result["positive_weight_events"][start:stop] = np.count_nonzero(weights > 0, axis=0)
        valid = (negative == 0) & (sumw > 0) & np.isfinite(sumw) & np.isfinite(sumw2)
        for local in np.flatnonzero(valid):
            target = start + local
            q16, q84 = inverted_cdf_sorted(sorted_x, weights[:, local])
            result["q16"][target], result["q84"][target] = q16, q84
            result["r68"][target] = (q84 - q16) / 2
            result["n_eff"][target] = sumw[local] ** 2 / sumw2[local] if sumw2[local] > 0 else 0
    return result
