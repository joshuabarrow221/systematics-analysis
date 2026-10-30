"""Paired-Monte-Carlo tests for proportional histogram shapes.

The two spectra MUST come from the same events, with identical observable
values and selections. A dial endpoint is deterministic; the statistical model
here describes finite Monte Carlo sampling, not random sampling of dial values.

For bin j supply A=sum(a), B=sum(b), A2=sum(a*a), B2=sum(b*b), C=sum(a*b),
where a and b are the varied and actual reference event weights. In particular,
b includes a stored nonunity CV response when that is the selected reference.
Direct difference moments D2=sum((a-b)**2), BD=sum(b*(a-b)) are recommended:
they avoid subtracting nearly equal second moments close to the CV.

The null hypothesis is that the MARGINAL binned shapes are proportional.
Honda flux, energy, interaction composition and selection correlations can
violate that null without an explicit incoming-angle dependence in a nuclear
cross section. Rejection is not evidence of a reconstruction or physics bug.

Only NumPy is required. SciPy's survival function is used when installed; an
integer-degree-of-freedom chi-square tail based on gamma recurrences is the
fallback. All returned structures are JSON-safe (undefined numbers are None).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

try:
    from scipy.special import gammaincc as _gammaincc
except ImportError:  # A small dependency-free fallback is sufficient for integer df.
    _gammaincc = None

_EPS = np.finfo(np.float64).eps


def _chi2_sf_fallback(statistic: float, ndof: int) -> float:
    """Upper chi-square tail from exact half/integer gamma recurrences.

    Q(a+1,x)=Q(a,x)+exp(-x)*x**a/Gamma(a+1), with Q(1,x)=exp(-x)
    and Q(1/2,x)=erfc(sqrt(x)). Log sums avoid overflow for large statistics.
    The log-erfc asymptotic is used only for x>=500, where ten terms are well
    inside double-precision accuracy. Very small probabilities can underflow
    to zero; the caller labels that representational limit explicitly.
    """
    x = statistic / 2.0
    if x == 0:
        return 1.0
    logx = math.log(x)
    if ndof % 2 == 0:
        logs = [-x + j * logx - math.lgamma(j + 1) for j in range(ndof // 2)]
    else:
        if x < 500:
            log_erfc = math.log(math.erfc(math.sqrt(x)))
        else:
            term = total = 1.0
            for j in range(1, 11):
                term *= -(2 * j - 1) / (2 * x)
                total += term
            log_erfc = -x - .5 * math.log(math.pi * x) + math.log(total)
        logs = [log_erfc]
        logs.extend(-x + (j + .5) * logx - math.lgamma(j + 1.5)
                    for j in range((ndof - 1) // 2))
    largest = max(logs)
    log_probability = largest + math.log(math.fsum(math.exp(y - largest) for y in logs))
    return min(1.0, math.exp(log_probability))


def chi2_survival(statistic: float, ndof: int) -> float:
    """Chi-square upper-tail probability; ndof must be a positive integer."""
    if not isinstance(ndof, (int, np.integer)) or ndof < 1:
        raise ValueError("ndof must be a positive integer")
    if not np.isfinite(statistic) or statistic < 0:
        raise ValueError("statistic must be finite and nonnegative")
    if _gammaincc is not None:
        return float(_gammaincc(ndof / 2, statistic / 2))
    return _chi2_sf_fallback(float(statistic), int(ndof))


def _json_numbers(array: np.ndarray) -> list[float | None]:
    return [float(x) if np.isfinite(x) else None for x in array]


def _nonnegative_roundoff(value: np.ndarray, scale: np.ndarray) -> tuple[np.ndarray, bool]:
    """Zero unresolved moment cancellation; reject materially negative variance."""
    tolerance = 128 * _EPS * np.maximum(scale, np.finfo(float).tiny)
    bad = np.any(value < -tolerance)
    return np.where(np.abs(value) <= tolerance, 0.0, np.maximum(value, 0.0)), bool(bad)


def paired_shape_flatness(
    sumw_var, sumw_ref, sumw2_var, sumw2_ref, sumw_cross, *,
    sumw2_difference=None, sumw_ref_difference=None,
    negative_events=0, nonfinite_events=0, min_effective_count: float = 20.0,
    return_covariance: bool = False,
) -> dict[str, Any]:
    """Compare a varied zenith shape to its paired reference, with covariance.

    Inputs are aligned one-dimensional VISIBLE-bin sums; all supplied bins
    define the normalization range. Pass observable-specific negative and
    nonfinite EVENT counts for either member of the pair, not just counts of
    negative bin sums. A raw signed spectrum can be displayed, but is excluded
    from this probability-shape diagnostic.

    The shape ratio is (A_j/sum(A))/(B_j/sum(B)). Zero reference bins are masked.
    With k=sum(A)/sum(B), D=A-k*B, p=B/sum(B), and
    S_j=sum_i_in_j (a_i-k*b_i)**2, the paired Poisson-MC delta covariance is

        V = (I-p*1.T) diag(S) (I-1*p.T).

    Its normalization null direction is respected. For positive S the Wald
    statistic is evaluated in O(number_of_bins), algebraically equivalent to
    D.T @ pinv(V) @ D. Singular cases use explicit covariance-support checks;
    an exactly/numerically proportional comparison has zero rank and no p.

    p_value is suppressed when an active bin has effective count below the
    configured threshold in either spectrum. That threshold is a screening
    rule, NOT a theorem guaranteeing Gaussian residuals: rare, heavy-tailed
    responses need event-level bootstrap checking. p-values are asymptotic,
    unadjusted for the many correlated dial comparisons, and never a test of
    fundamental rotational symmetry.
    """
    arrays = [np.asarray(x, dtype=np.float64) for x in
              (sumw_var, sumw_ref, sumw2_var, sumw2_ref, sumw_cross)]
    a, b, a2, b2, cross = arrays
    if a.ndim != 1 or not len(a) or any(x.shape != a.shape for x in arrays):
        raise ValueError("All five moment arrays must be aligned nonempty vectors")
    if not np.isfinite(min_effective_count) or min_effective_count < 0:
        raise ValueError("min_effective_count must be finite and nonnegative")
    if (sumw2_difference is None) != (sumw_ref_difference is None):
        raise ValueError("Supply both direct difference moments or neither")
    n = len(a)
    result = {
        "status": "unavailable", "statistic": None, "ndof": 0, "p_value": None,
        "ratio_shape": [None] * n, "ratio_shape_se": [None] * n,
        "ratio_rate": [None] * n, "ratio_rate_se": [None] * n,
        "rate_scale": None, "effective_bins": 0, "low_stat_bins": [],
        "min_effective_count": None, "effective_count_threshold": float(min_effective_count),
        "notes": [
            "Paired-event Poisson-MC delta-method test of proportional marginal zenith shapes.",
            "Energy/flux/selection correlations can change this shape without explicit nuclear incoming-angle dependence.",
            "Dial endpoints are deterministic; p-values describe finite MC sampling and are not multiplicity-adjusted.",
        ],
    }

    def unavailable(status, note):
        result["status"] = status
        result["notes"].append(note)
        if status in ("invalid_moments", "unavailable_moment_precision"):
            result["ratio_shape_se"] = [None] * n
            result["ratio_rate_se"] = [None] * n
        return result

    if np.any(np.asarray(nonfinite_events) > 0) or not all(np.all(np.isfinite(x)) for x in arrays):
        return unavailable("unavailable_nonfinite", "A missing/nonfinite response makes the displayed-range normalization unknown.")
    if np.any(np.asarray(negative_events) > 0) or np.any(a < 0) or np.any(b < 0):
        return unavailable("unavailable_signed_weights", "Negative event weights do not define the probability-shape test; raw spectra remain a separate diagnostic.")
    if np.any(a2 < 0) or np.any(b2 < 0):
        return unavailable("invalid_moments", "Negative sum of squared weights")
    if np.any((a == 0) & (a2 != 0)) or np.any((b == 0) & (b2 != 0)):
        return unavailable("invalid_moments", "Zero nonnegative sum must have zero squared-weight sum")
    if np.any(cross < 0) or np.any(cross * cross > a2 * b2 * (1 + 1e-10) + np.finfo(float).tiny):
        return unavailable("invalid_moments", "Paired cross moments violate nonnegative-weight covariance bounds")
    at, bt = float(np.sum(a)), float(np.sum(b))
    if at <= 0 or bt <= 0:
        return unavailable("unavailable_normalization", "Both displayed-range weighted totals must be positive")
    # Difference form preserves relative precision for small CV displacements.
    delta = a - b
    h = float(np.sum(delta)) / bt
    # A balanced variation can leave an O(epsilon) total-rate residue after
    # bin accumulation. Letting h² create extra variance directions in bins
    # whose paired differences are identically zero gives a spurious rank.
    if abs(h) <= 128 * _EPS:
        h = 0.0
    # Near unity the difference form is more precise. Far from unity use A/B:
    # 1+(A-B)/B can round to zero for a small but strictly positive A/B.
    k = 1.0 + h if abs(h) < .5 else at / bt
    h = k - 1.0
    result["rate_scale"] = k
    supported = b > 0
    ratio = np.full(n, np.nan)
    ratio[supported] = a[supported] / b[supported] / k
    result["ratio_shape"] = _json_numbers(ratio)
    result["ratio_rate"] = _json_numbers(np.divide(a, b, out=np.full(n, np.nan), where=supported))
    if sumw2_difference is None:
        d2, bad = _nonnegative_roundoff(a2 + b2 - 2 * cross, a2 + b2 + 2 * np.abs(cross))
        bd = cross - b2
        result["moment_method"] = "reconstructed_difference_moments"
        if bad:
            return unavailable("invalid_moments", "Difference second moment is materially negative")
    else:
        d2, bd = (np.asarray(x, dtype=float) for x in (sumw2_difference, sumw_ref_difference))
        if d2.shape != a.shape or bd.shape != a.shape:
            raise ValueError("Direct difference moments must have the same bin shape")
        if not np.all(np.isfinite(d2)) or not np.all(np.isfinite(bd)) or np.any(d2 < 0):
            return unavailable("invalid_moments", "Invalid direct difference moments")
        result["moment_method"] = "direct_difference_moments"
        if (np.any(np.abs(a2 - (b2 + 2 * bd + d2)) >
                   1e-9 * np.maximum(a2 + b2 + 2 * np.abs(bd) + d2, 1e-300))
                or np.any(np.abs(cross - (b2 + bd)) >
                          1e-9 * np.maximum(np.abs(cross) + b2 + np.abs(bd), 1e-300))):
            return unavailable("invalid_moments", "Direct moments disagree with A2/B2/cross; check that every moment uses the same event/reference pair")
    if np.any(bd * bd > b2 * d2 * (1 + 1e-9) + 128 * _EPS * np.maximum(a2 + b2, 1e-300)**2):
        return unavailable("invalid_moments", "Reference/difference moments violate covariance bounds")
    # Rescale the varied weights by the observed global rate k. This fixed
    # rescaling leaves all shape statistics unchanged and avoids tiny/huge
    # rate factors in the ratio derivatives. Choose the algebraic expression
    # with the smaller cancellation bound separately in each bin.
    an, a2n, cn = a / k, (a2 / k) / k, cross / k
    s_from_original = a2n + b2 - 2 * cn
    original_scale = a2n + b2 + 2 * np.abs(cn)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        s_from_difference = ((d2 - 2 * h * bd + h * h * b2) / k) / k
        difference_scale = ((d2 + 2 * abs(h) * np.abs(bd) + h * h * b2) / k) / k
    use_difference = np.isfinite(difference_scale) & (difference_scale < original_scale)
    s, bad = _nonnegative_roundoff(np.where(use_difference, s_from_difference, s_from_original),
                                 np.where(use_difference, difference_scale, original_scale))
    if bad:
        return unavailable("invalid_moments", "Paired normalized-residual variance is materially negative")
    bd_original = cn - b2
    bd_difference = (bd - h * b2) / k
    bd_scale_original = np.abs(cn) + b2
    bd_scale_difference = (np.abs(bd) + abs(h) * b2) / abs(k)
    bn = np.where(bd_scale_difference < bd_scale_original, bd_difference, bd_original)
    # A zero residual second moment mathematically forces its cross moment to
    # zero too. Remove the leftover roundoff after the variance cancellation.
    bn = np.where(s == 0, 0.0, bn)

    # Delta-method uncertainty of the ratio itself, including denominator and
    # total-normalization fluctuations, evaluated at the observed spectra.
    # In the rescaled (a/k-b,b) frame the observed total-rate factor is one.
    # Its reference-weight coefficient outside bin j is consequently zero.
    error = np.full(n, np.nan)
    rate_error = np.full(n, np.nan)
    jj = np.flatnonzero(supported)
    r, bj, dj = ratio[jj], b[jj], an[jj] - b[jj]
    u0, u1 = -r / bt, 1 / bj
    v1 = -dj / (bj * bj)
    inside = (u0 + u1)**2 * s[jj] + v1**2 * b2[jj] + 2 * (u0 + u1) * v1 * bn[jj]
    outside = u0**2 * (s.sum() - s[jj])
    scale = ((u0 + u1)**2 * s[jj] + v1**2 * b2[jj]
             + 2 * np.abs((u0 + u1) * v1 * bn[jj]) + outside)
    variance, bad = _nonnegative_roundoff(inside + outside, scale)
    if bad:
        return unavailable("invalid_moments", "Shape-ratio delta variance is materially negative")
    error[jj] = np.sqrt(variance)
    local_h = dj / bj
    vr, bad = _nonnegative_roundoff(s[jj] - 2 * local_h * bn[jj] + local_h**2 * b2[jj],
                                   s[jj] + 2 * np.abs(local_h * bn[jj]) + local_h**2 * b2[jj])
    if bad:
        return unavailable("invalid_moments", "Rate-ratio delta variance is materially negative")
    rate_error[jj] = abs(k) * np.sqrt(vr) / bj
    result["ratio_shape_se"] = _json_numbers(error)
    result["ratio_rate_se"] = _json_numbers(rate_error)
    if np.any((b == 0) & (a > 0)):
        return unavailable("unavailable_reference_support", "The variation populates a zero-reference bin; ratios and regular chi-square calibration cannot cover that support change.")
    active = supported
    indices = np.flatnonzero(active)
    p = b[active] / bt
    d = an[active] - b[active]
    # Mathematical sum(D)=0; remove only its floating-point sum residual.
    d -= p * np.sum(d)
    ss = s[active]
    result["effective_bins"] = len(indices)
    result["test_bins"] = indices.tolist()
    eff_a = np.divide(a[active]**2, a2[active], out=np.zeros(len(p)), where=a2[active] > 0)
    eff_b = np.divide(b[active]**2, b2[active], out=np.zeros(len(p)), where=b2[active] > 0)
    result["min_effective_count"] = float(np.minimum(eff_a, eff_b).min()) if len(p) else 0.0
    result["low_stat_bins"] = indices[np.minimum(eff_a, eff_b) < min_effective_count].tolist()
    if return_covariance:
        covariance = np.diag(ss) - np.outer(p, ss) - np.outer(ss, p) + ss.sum() * np.outer(p, p)
        result["difference_covariance"] = (covariance * k * k).tolist()
        result["difference"] = (d * k).tolist()
    positive = ss > 0
    tol_d = 1024 * _EPS * max(float(np.max(np.abs(an))), float(np.max(np.abs(b))), 1e-300)
    if not np.any(positive):
        if np.max(np.abs(d), initial=0) > tol_d:
            return unavailable("unavailable_moment_precision", "Shape displacement is nonzero but its variance vanished through moment precision; accumulate direct difference moments or inspect event weights.")
        result.update(status="proportional_within_precision", statistic=0.0, ndof=0)
        result["notes"].append("The paired shape contrast has zero variance/rank within numerical precision; no goodness-of-fit probability is assigned.")
        return result
    if np.all(positive):
        # Any u with M*u=D equals D+t*p. Minimize its independent-bin
        # Gaussian norm; this equals the Moore-Penrose quadratic form.
        z, direction = d / np.sqrt(ss), p / np.sqrt(ss)
        z -= direction * (np.dot(z, direction) / np.dot(direction, direction))
        statistic = float(np.dot(z, z))
        ndof = len(p) - 1
    else:
        # Zero-variance bins constrain the normalization nuisance t exactly.
        # Inconsistent constraints mean D lies outside covariance support.
        constrained_t = -d[~positive] / p[~positive]
        t = float(constrained_t[0])
        if np.any(np.abs(d[~positive] + t * p[~positive]) > tol_d):
            return unavailable("unavailable_covariance_support", "Observed contrast lies outside the supported covariance subspace; no regular chi-square p-value is justified.")
        z = (d[positive] + t * p[positive]) / np.sqrt(ss[positive])
        statistic = float(np.dot(z, z))
        ndof = int(positive.sum())
    result.update(statistic=statistic, ndof=int(ndof))
    if ndof < 1:
        return unavailable("unavailable_rank", "No nontrivial shape contrast remains")
    if result["low_stat_bins"]:
        return unavailable("low_statistics", "At least one active bin fails the configured effective-count screen; statistic is descriptive and p-value is withheld.")
    result["status"] = "ok_asymptotic"
    result["p_value"] = chi2_survival(statistic, ndof)
    if result["p_value"] == 0:
        result["notes"].append("The upper-tail probability underflows double precision; displayed zero means smaller than representable, not mathematically zero.")
    result["notes"].append("The effective-count screen does not guarantee Gaussian residuals; rare/heavy response tails require paired-event bootstrap checks.")
    return result


def moments_from_events(bin_index, varied_weight, reference_weight, n_bins: int):
    """Small reference accumulator; large production code can stream these sums.

    The same event row contributes both weights and the cross moment. No
    histogram joining or matching by bin contents is statistically equivalent.
    Rows outside [0,n_bins) are excluded consistently from all seven moments.
    """
    index = np.asarray(bin_index)
    a, b = np.asarray(varied_weight, float), np.asarray(reference_weight, float)
    if index.ndim != 1 or index.shape != a.shape or a.shape != b.shape or index.dtype.kind not in "iu":
        raise ValueError("Event bin indices and paired weights must be aligned vectors")
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    keep = (index >= 0) & (index < n_bins)
    ix, aa, bb = index[keep], a[keep], b[keep]
    nonfinite = int(np.count_nonzero(~np.isfinite(aa) | ~np.isfinite(bb)))
    negative = int(np.count_nonzero((aa < 0) | (bb < 0)))
    finite = np.isfinite(aa) & np.isfinite(bb)
    ix, aa, bb = ix[finite], aa[finite], bb[finite]
    delta = aa - bb
    result = {name: np.bincount(ix, weights=weight, minlength=n_bins) for name, weight in
              [("sumw_var", aa), ("sumw_ref", bb), ("sumw2_var", aa * aa),
               ("sumw2_ref", bb * bb), ("sumw_cross", aa * bb),
               ("sumw2_difference", delta * delta), ("sumw_ref_difference", bb * delta)]}
    result.update(negative_events=negative, nonfinite_events=nonfinite)
    return result


def energy_stratified_flatness(
    sumw_var, sumw_ref, sumw2_var, sumw2_ref, sumw_cross, *,
    sumw2_difference=None, sumw_ref_difference=None,
    negative_events=None, nonfinite_events=None, min_effective_count: float = 20.0,
) -> dict[str, Any]:
    """Optional conditional diagnostic from [true-energy-bin, zenith-bin] sums.

    Each disjoint energy stratum gets its OWN fitted normalization. Valid
    stratum Wald statistics and degrees of freedom add under the Poisson-MC
    model. No low-statistic stratum is silently omitted from a claimed global
    p-value. Completely empty strata are reported and have no contribution.

    ``standardized_ratio`` projects sum_e A_ez/k_e against sum_e B_ez: every
    energy stratum is first rescaled to its reference yield. This removes the
    supplied binned true-energy marginal difference. It does not remove
    residual within-bin energy, Q², interaction-class or acceptance effects.
    It is a descriptive ratio; this helper does not supply projected errors.

    Negative/nonfinite event counts, when present, are vectors of length n_E.
    The primary marginal diagnostic remains paired_shape_flatness; this
    optional cross-check answers a different, explicitly conditional question.
    """
    names = ("sumw_var", "sumw_ref", "sumw2_var", "sumw2_ref", "sumw_cross")
    values = [np.asarray(x, float) for x in
              (sumw_var, sumw_ref, sumw2_var, sumw2_ref, sumw_cross)]
    if values[0].ndim != 2 or any(x.shape != values[0].shape for x in values):
        raise ValueError("Energy-conditioned moment arrays must share [energy,angle] shape")
    ne, nz = values[0].shape
    if not ne or not nz:
        raise ValueError("Energy and angle dimensions must be nonempty")
    if (sumw2_difference is None) != (sumw_ref_difference is None):
        raise ValueError("Supply both direct difference moments or neither")
    extra = {}
    for name, value in (("sumw2_difference", sumw2_difference),
                        ("sumw_ref_difference", sumw_ref_difference)):
        if value is not None:
            extra[name] = np.asarray(value, float)
            if extra[name].shape != values[0].shape:
                raise ValueError("Direct difference moments must share [energy,angle] shape")
    neg = np.zeros(ne, int) if negative_events is None else np.asarray(negative_events)
    bad = np.zeros(ne, int) if nonfinite_events is None else np.asarray(nonfinite_events)
    if neg.shape != (ne,) or bad.shape != (ne,):
        raise ValueError("Invalid event counts must have one entry per energy stratum")
    result = {"status": "unavailable", "statistic": None, "ndof": 0, "p_value": None,
              "strata": [], "empty_strata": [], "problem_strata": [],
              "standardized_ratio": [None] * nz,
              "notes": ["Separate reference-yield normalization within each supplied true-energy bin.",
                        "This controls only the binned energy mixture, not all within-bin kinematic/acceptance correlations.",
                        "The summed test uses disjoint Poisson-MC strata and is an unadjusted asymptotic diagnostic."]}
    standardized = np.zeros(nz)
    total_statistic = 0.0
    total_ndof = 0
    for e in range(ne):
        if all(np.all(x[e] == 0) for x in values) and neg[e] == 0 and bad[e] == 0:
            result["empty_strata"].append(e)
            result["strata"].append({"status": "empty_stratum", "statistic": None, "ndof": 0, "p_value": None})
            continue
        kwargs = {key: value[e] for key, value in zip(names, values)}
        kwargs.update({key: value[e] for key, value in extra.items()})
        test = paired_shape_flatness(**kwargs, negative_events=neg[e], nonfinite_events=bad[e],
                                     min_effective_count=min_effective_count)
        result["strata"].append(test)
        if test["rate_scale"] is not None and test["rate_scale"] > 0:
            standardized += values[0][e] / test["rate_scale"]
        else:
            standardized[:] = np.nan
        if test["status"] not in ("ok_asymptotic", "proportional_within_precision"):
            result["problem_strata"].append(e)
        if test["statistic"] is not None:
            total_statistic += test["statistic"]
            total_ndof += test["ndof"]
    reference = np.sum(values[1], axis=0)
    ratio = np.divide(standardized, reference, out=np.full(nz, np.nan), where=reference > 0)
    result["standardized_ratio"] = _json_numbers(ratio)
    if result["problem_strata"]:
        result["status"] = "unavailable_strata"
        result["notes"].append("One or more nonempty strata fail the test requirements; no selected-subset global p-value is reported.")
        return result
    result.update(statistic=total_statistic, ndof=total_ndof)
    if total_ndof == 0:
        result["status"] = "proportional_within_precision" if len(result["empty_strata"]) < ne else "unavailable_normalization"
    else:
        result.update(status="ok_asymptotic", p_value=chi2_survival(total_statistic, total_ndof))
    return result
