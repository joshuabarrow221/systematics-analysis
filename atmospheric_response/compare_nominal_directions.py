#!/usr/bin/env python3
"""Compare old and hybrid directions on the SAME nominally weighted events.

Only named primitive arrays in local NPZ caches are read. No ROOT files or
cross-section response arrays are opened. The common-valid cohort intentionally
differs from either method-specific response atlas. It isolates reconstructed
value changes from the acceptance change caused by missing PFP directions.

Example (after installing this package):
  python -m atmospheric_response.compare_nominal_directions \
    --index /path/to/cache_v2/scan_index.json \
    --output-dir /path/to/comparison

The default is the bundled analysis_v2.yaml. Pass --config to change selection
or fine-bin definitions. This always compares both methods, independently of
the config's single direction_method value. Pilot runs require --limit-files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import time

import numpy as np

from . import v2_physics
from .v2_config import load_config, resolve_edges, SELECTIONS

FIELDS = ("E", "pdg", "iscc", "vtx", "true_momentum", "reco_heshw",
          "reco_lngtrk", "reco_calo", "n_particles", "enu_calo",
          "min_wall_cm", "wall_valid", "base_weight")

N_CATEGORIES = len(SELECTIONS)
PARTICLE_EDGES = np.array([-.5, .5, 1.5, 2.5, 3.5, 4.5, 5.5, 9.5, np.inf])
PARTICLE_LABELS = ["0", "1", "2", "3", "4", "5", "6–9", ">=10"]
CALO_DETAIL_EDGES = np.array([-np.inf, 0, .2, .5, 1, 1.3, 2, 5, 10, 100, np.inf])
CALO_DETAIL_LABELS = ["<0", "[0,0.2)", "[0.2,0.5)", "[0.5,1)", "[1,1.3)",
                      "[1.3,2)", "[2,5)", "[5,10)", "[10,100)", ">=100", "nonfinite"]


def encode_profile_groups(category, energy_bin, n_energy_bins):
    """Encode truth × energy groups without overflowing cached int8 categories.

    Widen before multiplying, including when custom bin counts exceed int16.
    Only selected, in-range events belong here; invalid indices must not be
    silently assigned to another truth channel by integer overflow or wrapping.
    """
    category, energy_bin = np.asarray(category), np.asarray(energy_bin)
    if (not isinstance(n_energy_bins, (int, np.integer)) or n_energy_bins < 1
            or n_energy_bins > np.iinfo(np.int64).max // N_CATEGORIES):
        raise ValueError("The energy-bin count must be a positive int64-safe integer")
    if (category.ndim != 1 or category.shape != energy_bin.shape
            or category.dtype.kind not in "iu" or energy_bin.dtype.kind not in "iu"):
        raise ValueError("Category and energy-bin indices must be matching integer vectors")
    if (np.any(category < 0) or np.any(category >= N_CATEGORIES)
            or np.any(energy_bin < 0) or np.any(energy_bin >= n_energy_bins)):
        raise ValueError("Profile group indices are outside the configured grid")
    return category.astype(np.int64)*n_energy_bins + energy_bin.astype(np.int64)


def weighted_profiles(group, weights, residuals, n_energy_bins):
    """Direct event-weighted q16/q84 for each truth/energy group and both methods.

    Sorting only group indices keeps the event-level residuals and weights in
    memory maps. Within each group NumPy uses an empirical inverted CDF, not a
    quantile of the plotted histogram. Counts permit independent closure checks.
    """
    group, weights, residuals = np.asarray(group), np.asarray(weights), np.asarray(residuals)
    if (group.ndim != 1 or weights.shape != group.shape or residuals.shape != (len(group), 2)
            or np.any(group < 0) or np.any(group >= N_CATEGORIES*n_energy_bins)):
        raise ValueError("Profile arrays do not match the category/energy grid")
    quantiles = np.full((2, N_CATEGORIES, n_energy_bins, 2), np.nan)
    neff = np.zeros((N_CATEGORIES, n_energy_bins))
    counts = np.zeros((N_CATEGORIES, n_energy_bins), dtype=np.int64)
    order = np.argsort(group, kind="stable")
    boundaries = np.searchsorted(group[order], np.arange(N_CATEGORIES*n_energy_bins+1))
    for category in range(N_CATEGORIES):
        for energy_bin in range(n_energy_bins):
            key = category*n_energy_bins + energy_bin
            rows = order[boundaries[key]:boundaries[key+1]]
            counts[category, energy_bin] = len(rows)
            if not len(rows):
                continue
            weight = weights[rows]
            neff[category, energy_bin] = weight.sum()**2/np.dot(weight, weight)
            for method in (0, 1):
                quantiles[method, category, energy_bin] = np.quantile(
                    residuals[rows, method], [.16, .84], weights=weight, method="inverted_cdf")
    if int(counts.sum()) != len(group):
        raise ValueError("Profile grouping lost events")
    return quantiles, neff, counts


def clean(value):
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def count_by_category(category, mask, weights=None):
    return np.bincount(category[mask], weights=None if weights is None else weights[mask], minlength=5)


def fill_hist(category, value, weights, mask, edges):
    return np.array([np.histogram(value[mask & (category == c)], edges,
                                  weights=weights[mask & (category == c)])[0]
                     for c in range(5)])


def fill_joint_hist(category, x, y, mask, xedges, yedges):
    """Unweighted diagnostics conditional on the chosen reconstructed angle."""
    return np.array([np.histogram2d(x[mask & (category == c)], y[mask & (category == c)],
                                    bins=(xedges, yedges))[0] for c in range(N_CATEGORIES)],
                    dtype=np.int64)


def describe_interior_peaks(edges, hist, counts, source_counts, source_sumw,
                            particle_counts, calo_counts):
    """Report observed local maxima near the user's approximate ±0.75 spikes.

    Search each side in 0.65 <= |cosZ center| <= 0.9, retaining the two adjacent
    bins and raw chosen-method counts. A maximum is descriptive, not a claim of
    statistical significance or a reconstruction-producer explanation.
    """
    centers = (edges[:-1] + edges[1:])/2
    peaks = []
    for category in range(N_CATEGORIES):
        for sign in (-1, 1):
            candidates = np.flatnonzero((sign*centers >= .65) & (sign*centers <= .9))
            if not len(candidates):
                continue
            index = candidates[np.argmax(hist[1, category, candidates])]
            if hist[1, category, index] <= 0:
                continue
            peaks.append({
                "truth_id": SELECTIONS[category]["id"], "sign": sign,
                "bin_index": int(index), "center": centers[index],
                "neighbor_bins": [{"low": edges[b], "high": edges[b+1],
                    "old_count": counts[0, category, b], "hybrid_count": counts[1, category, b],
                    "old_sumw": hist[0, category, b], "hybrid_sumw": hist[1, category, b],
                    "hybrid_chosen_method_counts": source_counts[:, category, b],
                    "hybrid_chosen_method_sumw": source_sumw[:, category, b]}
                    for b in range(max(index-1, 0), min(index+2, len(centers)))],
                "n_particles_counts_in_peak_bin": particle_counts[category, index],
                "enu_calo_counts_in_peak_bin": calo_counts[category, index],
            })
    return peaks


def calculate(args):
    started = time.monotonic()
    cfg = load_config(args.config)
    index = json.loads(args.index.read_text())
    files = sorted(index["files"], key=lambda item: item["cache"])
    expected = index["total_production_files"]
    if args.limit_files is not None:
        if args.limit_files < 1:
            raise ValueError("--limit-files must be positive")
        files = files[:args.limit_files]
    elif len(files) != expected or not index.get("complete"):
        raise ValueError("Full comparison requires a completed cache index")
    cos_edges, _ = resolve_edges(cfg, "fine", "all", SELECTIONS[0], "reco_coszenith")
    energy_edges, _ = resolve_edges(cfg, "fine", "all", SELECTIONS[0], "angular_resolution")
    calo_edges = np.linspace(0, 2.6, 53)
    nbin = len(energy_edges)-1
    # An empty completed cache is still a useful diagnostic: retain empty
    # plots/counts without asking NumPy to create a zero-length memory map.
    maximum = max(1, sum(record["counts"]["selected"] for record in files))
    hist = np.zeros((2, 5, len(cos_edges)-1))
    hist_counts = np.zeros_like(hist, dtype=np.int64)
    source_sumw = np.zeros((3, 5, len(cos_edges)-1))
    source_counts = np.zeros_like(source_sumw, dtype=np.int64)
    particle_detail = np.zeros((5, len(cos_edges)-1, len(PARTICLE_LABELS)), dtype=np.int64)
    calo_detail = np.zeros((5, len(cos_edges)-1, len(CALO_DETAIL_LABELS)), dtype=np.int64)
    calo_counts = np.zeros((2, 5, len(calo_edges)-1), dtype=np.int64)
    calo_sumw = np.zeros_like(calo_counts, dtype=float)
    count_names = ("before_direction", "invalid_true_direction", "hybrid_valid", "old_valid",
                   "both_valid", "hybrid_only", "old_only", "neither_valid",
                   "old_finite_nonzero_nonunit", "old_zero_or_nonfinite", "common_profile",
                   "common_calo_nonfinite", "hybrid_endpoint_all", "hybrid_endpoint_common",
                   "old_endpoint_common")
    counts = {key: np.zeros(5, dtype=np.int64) for key in count_names}
    sums = {key: np.zeros(5) for key in ("hybrid_valid", "old_valid", "both_valid",
             "hybrid_only", "old_only", "hybrid_endpoint_common", "old_endpoint_common")}
    chosen = np.zeros((5, 3), dtype=np.int64)
    bounds = np.array([cfg["selection"]["fiducial_bounds_cm"][key] for key in "xyz"])
    audit = {"complete_production": len(files) == expected, "source_files": len(files),
             "expected_source_files": expected, "config_sha256": cfg["_sha256"],
             "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "physics_sha256": hashlib.sha256(Path(v2_physics.__file__).read_bytes()).hexdigest(),
             "scope": "Nominal common-valid comparison; local primitive NPZ arrays only; no ROOT or response arrays",
             "weight": "Positive finite stored xsec times matching-flavour Honda flux; no oscillation or extra CV factor",
             "methods": {"old": "heshw for electron CC/NC; lngtrk for muon CC; no switch; |norm-1|<=1e-4",
                         "hybrid": "NC always hits; CC hits for nParticles<2 or !(Enu.calo<1.3), otherwise flavour PFP direction"},
             "selection": cfg["selection"], "category_ids": [s["id"] for s in SELECTIONS],
             "hybrid_switch_energy_GeV": cfg["physics"]["hybrid_switch_energy_gev"],
             "hybrid_method_order": ["heshw", "lngtrk", "hits"],
             "interior_peak_search_abs_coszenith": [.65, .9],
             "interior_peak_n_particles_labels": PARTICLE_LABELS,
             "interior_peak_enu_calo_labels_GeV": CALO_DETAIL_LABELS,
             "coszenith_edges": cos_edges, "profile_true_energy_edges_GeV": energy_edges,
             "profile_centers_GeV": np.where(energy_edges[:-1] > 0,
                np.sqrt(np.maximum(energy_edges[:-1]*energy_edges[1:], 0)),
                (energy_edges[:-1]+energy_edges[1:])/2),
             "switch_diagnostic_calo_edges_GeV": calo_edges, "files": []}
    with tempfile.TemporaryDirectory(prefix="atmo-common-direction-") as scratch:
        scratch = Path(scratch)
        max_group = N_CATEGORIES*nbin-1
        group_dtype = "i2" if max_group <= np.iinfo(np.int16).max else "i4" if max_group <= np.iinfo(np.int32).max else "i8"
        group = np.lib.format.open_memmap(scratch/"group.npy", mode="w+", dtype=group_dtype, shape=(maximum,))
        wprofile = np.lib.format.open_memmap(scratch/"weight.npy", mode="w+", dtype="f8", shape=(maximum,))
        residual = np.lib.format.open_memmap(scratch/"residual.npy", mode="w+", dtype="f8", shape=(maximum, 2))
        offset = 0
        for part, record in enumerate(files):
            path = Path(record["cache"])
            if not path.is_absolute():
                path = args.index.parent/path
            if path.suffix != ".npz" or path.stat().st_size != record["cache_bytes"]:
                raise ValueError("Expected an unchanged local NPZ cache")
            with np.load(path, allow_pickle=False) as archive:
                raw = {key: archive[key] for key in FIELDS}
            cat = v2_physics.truth_categories(raw["pdg"], raw["iscc"])
            wall = v2_physics.classify_containment(raw["min_wall_cm"], raw["wall_valid"],
                cfg["physics"]["containment_threshold_cm"], cfg["physics"]["containment_max_valid_cm"])
            true, true_valid = v2_physics.unit_direction(raw["true_momentum"])
            old, old_valid, _ = v2_physics.choose_direction(cat, raw["n_particles"], raw["enu_calo"],
                raw["reco_heshw"], raw["reco_lngtrk"], raw["reco_calo"], method="all_particles")
            new, new_valid, method = v2_physics.choose_direction(cat, raw["n_particles"], raw["enu_calo"],
                raw["reco_heshw"], raw["reco_lngtrk"], raw["reco_calo"], method="hybrid",
                switch_energy_gev=cfg["physics"]["hybrid_switch_energy_gev"])
            old_vector = np.where(np.isin(cat, [1, 4])[:, None], raw["reco_lngtrk"], raw["reco_heshw"])
            old_norm = np.linalg.norm(old_vector, axis=1)
            finite_positive = np.isfinite(old_norm) & (old_norm > 0)
            # Literal historical validity, independent of package defaults.
            strict_old = finite_positive & np.isclose(old_norm, 1, rtol=0, atol=1e-4)
            old_valid &= strict_old
            E, w = raw["E"], raw["base_weight"]
            if not np.all(np.isfinite(w) & (w > 0)):
                raise ValueError("Nominal cache weights must be finite and positive")
            base = (cat >= 0) & np.isfinite(E) & (E >= cfg["selection"]["true_energy_min_GeV"])
            emax = cfg["selection"].get("true_energy_max_GeV")
            if emax is not None:
                base &= E <= emax
            base &= np.all((raw["vtx"] > bounds[:, 0]) & (raw["vtx"] < bounds[:, 1]), axis=1)
            if cfg["selection"].get("require_valid_wall", True):
                base &= wall >= 0
            eligible = base & true_valid
            old_keep, new_keep = eligible & old_valid, eligible & new_valid
            common = old_keep & new_keep
            cos_old, cos_new = -old[:, 1], -new[:, 1]
            ep_old = np.isclose(np.abs(cos_old), 1, atol=1e-8, rtol=0)
            ep_new = np.isclose(np.abs(cos_new), 1, atol=1e-8, rtol=0)
            energy_bin = np.searchsorted(energy_edges, E, side="right")-1
            energy_bin[E == energy_edges[-1]] = nbin-1
            profile = common & (energy_bin >= 0) & (energy_bin < nbin)
            masks = {"before_direction": base, "invalid_true_direction": base & ~true_valid,
                "hybrid_valid": new_keep, "old_valid": old_keep, "both_valid": common,
                "hybrid_only": new_keep & ~old_keep, "old_only": old_keep & ~new_keep,
                "neither_valid": eligible & ~old_valid & ~new_valid,
                "old_finite_nonzero_nonunit": eligible & finite_positive & ~strict_old,
                "old_zero_or_nonfinite": eligible & ~finite_positive, "common_profile": profile,
                "common_calo_nonfinite": common & ~np.isfinite(raw["enu_calo"]),
                "hybrid_endpoint_all": new_keep & ep_new,
                "hybrid_endpoint_common": common & ep_new, "old_endpoint_common": common & ep_old}
            for key, mask in masks.items():
                counts[key] += count_by_category(cat, mask)
                if key in sums:
                    sums[key] += count_by_category(cat, mask, w)
            hist[0] += fill_hist(cat, cos_old, w, common, cos_edges)
            hist[1] += fill_hist(cat, cos_new, w, common, cos_edges)
            unit_weights = np.ones(len(w), dtype=np.int64)
            hist_counts[0] += fill_hist(cat, cos_old, unit_weights, common, cos_edges)
            hist_counts[1] += fill_hist(cat, cos_new, unit_weights, common, cos_edges)
            for source in range(3):
                source_mask = common & (method == source)
                source_sumw[source] += fill_hist(cat, cos_new, w, source_mask, cos_edges)
                source_counts[source] += fill_hist(cat, cos_new, unit_weights, source_mask, cos_edges)
            particle_detail += fill_joint_hist(cat, cos_new, raw["n_particles"], common,
                                               cos_edges, PARTICLE_EDGES)
            finite_calo = np.isfinite(raw["enu_calo"])
            calo_detail[..., :-1] += fill_joint_hist(cat, cos_new, raw["enu_calo"], common & finite_calo,
                                                    cos_edges, CALO_DETAIL_EDGES)
            calo_detail[..., -1] += fill_hist(cat, cos_new, unit_weights, common & ~finite_calo, cos_edges)
            for is_hits, mask in enumerate((common, common & (method == 2))):
                calo_sumw[is_hits] += fill_hist(cat, raw["enu_calo"], w, mask, calo_edges)
                calo_counts[is_hits] += fill_hist(cat, raw["enu_calo"], np.ones(len(w), dtype=np.int64), mask, calo_edges)
            for c in range(5):
                chosen[c] += np.bincount(method[common & (cat == c)], minlength=3)
            n = int(profile.sum())
            group[offset:offset+n] = encode_profile_groups(cat[profile], energy_bin[profile], nbin)
            wprofile[offset:offset+n] = w[profile]
            theta_true = np.arccos(np.clip(true[profile, 1], -1, 1))
            residual[offset:offset+n, 0] = np.degrees(np.arccos(np.clip(old[profile, 1], -1, 1))-theta_true)
            residual[offset:offset+n, 1] = np.degrees(np.arccos(np.clip(new[profile, 1], -1, 1))-theta_true)
            offset += n
            audit["files"].append({"cache": str(path), "cache_sha256": record["cache_sha256"],
                                   "common_events": int(common.sum()), "profile_events": n})
            print(f"CACHE {part+1}/{len(files)} common={int(common.sum()):,}", flush=True)
        quantiles, neff, pcounts = weighted_profiles(group[:offset], wprofile[:offset], residual[:offset], nbin)
        for array in (group, wprofile, residual):
            array._mmap.close()
    width = (quantiles[..., 1]-quantiles[..., 0])/2
    np.testing.assert_allclose(hist.sum(axis=2), np.broadcast_to(sums["both_valid"], (2, 5)), rtol=1e-11)
    np.testing.assert_array_equal(counts["hybrid_valid"], counts["both_valid"]+counts["hybrid_only"])
    np.testing.assert_array_equal(counts["old_valid"], counts["both_valid"]+counts["old_only"])
    np.testing.assert_array_equal(pcounts.sum(axis=1), counts["common_profile"])
    np.testing.assert_array_equal(hist_counts.sum(axis=2), np.broadcast_to(counts["both_valid"], (2, 5)))
    np.testing.assert_array_equal(source_counts.sum(axis=0), hist_counts[1])
    np.testing.assert_array_equal(particle_detail.sum(axis=2), hist_counts[1])
    np.testing.assert_array_equal(calo_detail.sum(axis=2), hist_counts[1])
    np.testing.assert_allclose(source_sumw.sum(axis=0), hist[1], rtol=1e-11, atol=1e-13)
    audit.update(counts_by_truth=counts, nominal_sumw_by_truth=sums,
        interior_peak_bins=describe_interior_peaks(cos_edges, hist, hist_counts,
            source_counts, source_sumw, particle_detail, calo_detail),
        chosen_hybrid_methods_on_common_by_truth=chosen, profile_counts=pcounts,
        profile_n_eff=neff, profile_q16_q84_deg=quantiles, profile_half_width_deg=width,
        calo_bin_common_counts=calo_counts[0], calo_bin_hits_counts=calo_counts[1],
        calo_bin_common_sumw=calo_sumw[0], calo_bin_hits_sumw=calo_sumw[1],
        common_spectrum_sumw=hist.sum(axis=2), elapsed_seconds=time.monotonic()-started,
        interpretation=["Both curves use exactly the same selected events and nominal weights.",
            "Hybrid and old method-specific acceptances differ; their counts are audited separately.",
            "The switch threshold is Enu.calo, not true neutrino energy.",
            "NC uses hits at all energies in hybrid; it has no energy-dependent method switch.",
            "Endpoint pile-ups are observations of cached reconstructed vectors, not a proved producer-level cause.",
            "No cross-section dial response or oscillation probability is applied in this diagnostic."])
    return audit, dict(cos_edges=cos_edges, energy_edges=energy_edges, calo_edges=calo_edges,
        cos_sumw=hist, cos_counts=hist_counts, hybrid_source_counts=source_counts,
        hybrid_source_sumw=source_sumw, hybrid_particle_counts_by_cosbin=particle_detail,
        hybrid_calo_counts_by_cosbin=calo_detail,
        q16_q84=quantiles, width68=width, profile_n_eff=neff,
        profile_counts=pcounts, calo_sumw=calo_sumw, calo_counts=calo_counts)


def plot(output, audit, arrays):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    labels = [r"$\nu_e$ CC", r"$\nu_\mu$ CC", "NC (both signs)", r"$\bar\nu_e$ CC", r"$\bar\nu_\mu$ CC"]
    colors, names = ["#2459a6", "#c54e17"], ["Old: PFP direction at all energies", "Pierre hybrid (NC always hits)"]
    title = "Common-valid nominal comparison" if audit["complete_production"] else "PILOT common-valid nominal comparison"
    with PdfPages(output/"common_valid_nominal.pdf") as pdf:
        fig, axes = plt.subplots(5, 2, figsize=(14, 18), layout="constrained")
        center = audit["profile_centers_GeV"]
        for c in range(5):
            for m in (0, 1):
                axes[c, 0].stairs(arrays["cos_sumw"][m, c], arrays["cos_edges"], color=colors[m], label=names[m], linewidth=1.1)
                axes[c, 1].plot(center, arrays["width68"][m, c], color=colors[m], label=names[m], linewidth=1.4)
            n = audit["counts_by_truth"]["both_valid"][c]
            axes[c, 0].set(title=f"{labels[c]} — {n:,} common events", ylabel="Honda × stored-xsec sum / bin", xlabel=r"Reconstructed $\cos\theta_z$")
            axes[c, 1].set(title=f"{labels[c]} — signed angular 68% half-width", ylabel=r"$(Q_{84}-Q_{16})/2$ [deg]", xlabel="True neutrino energy [GeV]")
            axes[c, 1].set_xscale("log" if arrays["energy_edges"][0] > 0 else "linear")
            axes[c, 0].set_xlim(-1, 1)
            axes[c, 1].set_xlim(arrays["energy_edges"][0], arrays["energy_edges"][-1])
            for ax in axes[c]:
                ax.grid(alpha=.18)
        # These are observed peak-bin positions, not an assumed detector effect.
        for peak in audit["interior_peak_bins"]:
            if peak["truth_id"] != "nc":
                continue
            value = arrays["cos_sumw"][1, 2, peak["bin_index"]]
            axes[2, 0].annotate(f'{peak["center"]:+.3f}', (peak["center"], value),
                                xytext=(0, 7), textcoords="offset points", ha="center",
                                color=colors[1], fontsize=8)
        axes[2, 0].set_ylim(top=axes[2, 0].get_ylim()[1]*1.12)
        axes[0, 0].legend(fontsize=8, loc="upper left")
        axes[0, 1].legend(fontsize=8)
        fig.suptitle(title+"\nSame events and weights for both methods; current FV/containment cuts; no oscillation or dial variation", fontsize=14)
        pdf.savefig(fig)
        fig.savefig(output/"common_valid_nominal.png", dpi=130)
        plt.close(fig)
        fig, axes = plt.subplots(2, 1, figsize=(11, 10), layout="constrained")
        palette = ["#2459a6", "#c54e17", "#555555", "#189978", "#8a4d9f"]
        for c in range(5):
            denom = arrays["calo_sumw"][0, c]
            fraction = np.divide(arrays["calo_sumw"][1, c], denom, out=np.full_like(denom, np.nan), where=denom>0)
            axes[0].stairs(fraction, arrays["calo_edges"], label=labels[c], color=palette[c])
        switch = audit["hybrid_switch_energy_GeV"]
        axes[0].axvline(switch, color="black", linestyle="--", label=f"Enu.calo = {switch:g} GeV")
        axes[0].set(xlabel="Enu.calo [GeV] — switching variable", ylabel="Honda-weighted fraction using hits", ylim=(-.03, 1.06), xlim=(0, 2.6))
        axes[0].legend(fontsize=9, ncols=2)
        x = np.arange(5)
        for m, key in enumerate(("old_endpoint_common", "hybrid_endpoint_common")):
            num = audit["nominal_sumw_by_truth"][key]
            den = audit["nominal_sumw_by_truth"]["both_valid"]
            frac = np.divide(num, den, out=np.zeros_like(num), where=den>0)*100
            axes[1].bar(x+(m-.5)*.34, frac, width=.34, color=colors[m], label=names[m])
        axes[1].set_xticks(x, labels)
        axes[1].set(ylabel="Weighted fraction at |cos(zenith)| ≈ 1 [%]", title="Endpoint accumulation on the same events (absolute tolerance 10⁻⁸)")
        axes[1].legend(fontsize=9)
        for ax in axes:
            ax.grid(axis="y", alpha=.18)
        fig.suptitle(title+"\nNC always uses hits; other channels switch by particle count and Enu.calo", fontsize=13)
        pdf.savefig(fig)
        fig.savefig(output/"switch_and_endpoints.png", dpi=130)
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True, help="Local v2 scan_index.json")
    parser.add_argument("--config", type=Path, default=Path(__file__).parent/"configs/analysis_v2.yaml",
                        help="Analysis YAML (default: bundled analysis_v2.yaml)")
    parser.add_argument("--output-dir", "--output", dest="output_dir", type=Path, required=True,
                        help="Destination for PDF, PNG, aggregate NPZ and audit JSON")
    parser.add_argument("--limit-files", type=int, help="Explicitly labelled pilot; default: full complete index")
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit, arrays = calculate(args)
    np.savez_compressed(args.output_dir/"common_valid_nominal.npz", **arrays)
    plot(args.output_dir, audit, arrays)
    (args.output_dir/"analysis_config.yaml").write_bytes(args.config.read_bytes())
    audit["status"] = "passed"
    (args.output_dir/"common_valid_nominal_audit.json").write_text(json.dumps(clean(audit), indent=2, allow_nan=False)+"\n")
    print(json.dumps({"status":"passed", "source_files":audit["source_files"],
        "common_events":int(audit["counts_by_truth"]["both_valid"].sum()),
        "seconds":audit["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
