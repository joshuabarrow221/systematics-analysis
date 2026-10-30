"""Sparse paired-weight moments for zenith-shape diagnostics.

The caller has already joined events, selected a direction/containment/binning,
and applied the desired numerical policy. This module makes no physics choice
and reads no files. It processes one response column at a time, so it never
materializes an all-event by all-dial dense weight matrix.
"""
from __future__ import annotations

import numpy as np


def _inputs(group, rows, responses, reference_index, group_shape):
    """Validate positional alignment and retain only observable-eligible rows."""
    group, rows, responses = np.asarray(group), np.asarray(rows), np.asarray(responses)
    if group.ndim != 1 or group.dtype.kind not in "iu":
        raise ValueError("group must be an integer event vector; -1 means ineligible")
    if rows.ndim != 1 or rows.dtype.kind not in "iu":
        raise ValueError("rows must be a unique integer sparse-event vector")
    if responses.ndim != 2 or responses.shape[0] != len(rows) or responses.dtype.kind not in "fiu":
        raise ValueError("responses must be an aligned event-by-point numeric matrix")
    if len(group_shape) != 2 or any(not isinstance(x, (int, np.integer)) or x < 1 for x in group_shape):
        raise ValueError("group_shape must be (number_of_categories, bins_including_any_flows)")
    nc, width = map(int, group_shape)
    nv = responses.shape[1]
    if nv < 1:
        raise ValueError("At least one stored response point is required")
    if reference_index is not None and (not isinstance(reference_index, (int, np.integer)) or not 0 <= reference_index < nv):
        raise ValueError("reference_index is a response-column position, or None for unity")
    if len(rows):
        if rows.min() < 0 or rows.max() >= len(group):
            raise ValueError("Sparse event index outside the event vector")
        # Actual producer rows are sorted. Do not sort/copy millions of indices
        # again when one linear check can establish uniqueness.
        if not np.all(rows[1:] > rows[:-1]) and len(np.unique(rows)) != len(rows):
            raise ValueError("Sparse event indices must be unique")
    selected_group = group[rows]
    if np.any(selected_group < -1) or np.any(selected_group >= nc * width):
        raise ValueError("Sparse-row group code lies outside declared group_shape")
    valid = selected_group >= 0
    all_valid = bool(np.all(valid))
    codes = np.asarray(selected_group if all_valid else selected_group[valid], dtype=np.intp)
    return rows, responses, valid, all_valid, codes, (nc, nv, width)


def get_difference_moments(group, rows, base_weight, responses,
                           reference_index=None, group_shape=(5, 202)):
    """Return direct paired moments, each (category,point,bin).

    group: length-N integer vector encoding category*width+bin; -1 is masked.
    rows: unique source positions of the sparse response rows.
    base_weight: length-N finite nonnegative external nominal weights.
    responses: shape(len(rows),nvalues), AFTER the requested numerical policy.
    reference_index: actual stored CV/reference column, or None for response1.

    Omitted sparse rows have response exactly1 at every point AND at the
    reference, so their paired difference is zero but their cross product is
    base_weight**2. Accumulate that untouched contribution directly; computing
    C=B2+sum(b*(a-b)) loses all precision when a is much smaller than b.
    Returned keys match paired_shape_flatness moment names:
      sumw2_difference = sum((w_var-w_ref)^2)
      sumw_ref_difference = sum(w_ref*(w_var-w_ref))
      sumw_cross = sum(w_var*w_ref), INCLUDING omitted unity rows.
    The reference may be nonunity; it is never silently replaced by baseline.
    """
    rows, responses, valid, all_valid, codes, shape = _inputs(
        group, rows, responses, reference_index, group_shape)
    base = np.asarray(base_weight)
    if base.shape != np.asarray(group).shape or base.dtype.kind not in "fiu":
        raise ValueError("base_weight must align with the complete event vector")
    wb = np.asarray(base[rows] if all_valid else base[rows][valid], dtype=np.float64)
    if not np.all(np.isfinite(wb) & (wb >= 0)):
        raise ValueError("Selected external weights must be finite and nonnegative")
    nc, nv, width = shape
    group = np.asarray(group)
    if len(group) and (group.min() < -1 or group.max() >= nc * width):
        raise ValueError("Event group code lies outside declared group_shape")
    out_delta2, out_ref_delta, out_cross = np.zeros(shape), np.zeros(shape), np.zeros(shape)
    untouched = group >= 0
    untouched[rows] = False
    untouched_cross = np.zeros(nc * width)
    # Chunk only the scratch arrays, not the arithmetic: each untouched event
    # contributes once. Peak temporary storage is bounded even for large CAFs.
    for start in range(0, len(group), 1_000_000):
        stop = min(start + 1_000_000, len(group))
        mask = untouched[start:stop]
        if not np.any(mask):
            continue
        w = np.asarray(base[start:stop][mask], dtype=np.float64)
        if not np.all(np.isfinite(w) & (w >= 0)):
            raise ValueError("Untouched external weights must be finite and nonnegative")
        untouched_cross += np.bincount(group[start:stop][mask], weights=w * w,
                                       minlength=nc * width)
    out_cross[:] = untouched_cross.reshape(nc, 1, width)
    del untouched
    if not len(codes):
        return {"sumw2_difference": out_delta2, "sumw_ref_difference": out_ref_delta,
                "sumw_cross": out_cross}

    def column(j):
        col = responses[:, j] if all_valid else responses[valid, j]
        if not np.all(np.isfinite(col)):
            raise ValueError("Difference moments require finite post-policy responses")
        return col

    ref_weight = wb if reference_index is None else wb * column(reference_index)
    for point in range(nv):
        varied_weight = ref_weight if point == reference_index else wb * column(point)
        out_cross[:, point, :] += np.bincount(codes, weights=varied_weight * ref_weight,
                                              minlength=nc * width).reshape(nc, width)
        if point == reference_index:
            # Exactly the same event weights: do not invent cancellation noise.
            continue
        # Match the multiplication and subtraction used for the actual weighted
        # spectra, instead of subtracting separately summed second moments.
        delta = varied_weight - ref_weight
        out_delta2[:, point, :] = np.bincount(codes, weights=delta * delta,
                                              minlength=nc * width).reshape(nc, width)
        out_ref_delta[:, point, :] = np.bincount(codes, weights=ref_weight * delta,
                                                 minlength=nc * width).reshape(nc, width)
    return {"sumw2_difference": out_delta2, "sumw_ref_difference": out_ref_delta,
            "sumw_cross": out_cross}


def get_response_quality_counts(group, rows, raw_responses,
                                reference_index=None, group_shape=(5, 202)):
    """Count per-bin raw invalid/sign unions for each varied/reference pair.

    This count helper allows raw NaN and +/-inf. With a strictly positive
    external baseline, a finite negative response means a negative event
    weight. The main study already excludes zero-baseline events; if a different
    caller retains them it must mask those rows before using these sign counts.

    Each event is counted once in the nonfinite union and once in the negative
    union, at most, regardless of whether both varied/reference entries fail.
    The two unions can overlap (e.g. a nonfinite reference and negative variant).
    Omitted sparse rows are finite positive unity and contribute zero.
    Counts describe RAW response quality. For a test of handled weights use
    handled invalid/sign counts, while keeping these raw flags visible.
    """
    _, responses, valid, all_valid, codes, shape = _inputs(
        group, rows, raw_responses, reference_index, group_shape)
    nc, nv, width = shape
    missing, negative = np.zeros(shape, dtype=np.int64), np.zeros(shape, dtype=np.int64)
    if not len(codes):
        return {"nonfinite_events": missing, "negative_events": negative}

    def column(j):
        return responses[:, j] if all_valid else responses[valid, j]

    if reference_index is None:
        ref_bad = np.zeros(len(codes), dtype=bool)
        ref_negative = np.zeros(len(codes), dtype=bool)
    else:
        ref = column(reference_index)
        ref_bad = ~np.isfinite(ref)
        ref_negative = np.isfinite(ref) & (ref < 0)
    for point in range(nv):
        col = column(point)
        finite = np.isfinite(col)
        bad = ~finite | ref_bad
        neg = (finite & (col < 0)) | ref_negative
        missing[:, point, :] = np.bincount(codes[bad], minlength=nc * width).reshape(nc, width)
        negative[:, point, :] = np.bincount(codes[neg], minlength=nc * width).reshape(nc, width)
    return {"nonfinite_events": missing, "negative_events": negative}
