#!/usr/bin/env python3
"""Independent dense-event comparison for a bounded one-file local pilot.

From the repository after installation, run::

    python tests/integration/validate_pilot.py --output PILOT_OUTPUT --report REPORT.json --dials 0 15 60 61 62 78 91 92

Only a completed one-file local cache with at most 200,000 events is accepted.
This is an integration check, not another ROOT extraction. It reads one local
NPZ cache, recomputes selected event weights, and calls NumPy's physical-edge
histogram directly. It never calls SparseResponseStudy, direct_bin_codes,
get_difference_moments, or materialize. Physics formulas themselves are covered
by the separate v2_physics tests and are intentionally reused here.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time

import numpy as np

from atmospheric_response.v2_config import load_config, OBSERVABLE_IDS
from atmospheric_response.v2_physics import derive_observables
from atmospheric_response.v2_statistics import moments_from_events, paired_shape_flatness


RAW_FIELDS = ["E", "q0", "inelasticity", "baseline_m", "pdg", "iscc",
              "true_momentum", "reco_heshw", "reco_lngtrk", "reco_calo", "vtx",
              "enu_calo", "enu_e_calo", "enu_e_had", "enu_lep_calo", "enu_mu_range",
              "enu_mu_had", "n_particles", "min_wall_cm", "wall_valid"]


def histogram(x, mask, weight, edges):
    """Invalidate exactly the raw bins touched by nonfinite weights."""
    good = mask & np.isfinite(weight)
    value = np.histogram(x[good], bins=edges, weights=weight[good])[0]
    missing = np.histogram(x[mask & ~np.isfinite(weight)], bins=edges)[0]
    value[missing > 0] = np.nan
    return value


def profile(energy, residual, mask, weight, edges):
    """Dense weighted inverted CDF, without a residual histogram."""
    output = np.full(len(edges) - 1, np.nan)
    code = np.searchsorted(edges, energy, side="right") - 1
    code[energy == edges[-1]] = len(edges) - 2
    for b in range(len(output)):
        rows = np.flatnonzero(mask & (code == b))
        w = weight[rows]
        if not len(rows) or np.any(~np.isfinite(w)) or np.any(w < 0) or np.sum(w) <= 0:
            continue
        rows = rows[w > 0]
        rows = rows[np.argsort(residual[rows], kind="stable")]
        cdf = np.cumsum(weight[rows], dtype=float)
        positions = np.searchsorted(cdf, np.array([.16, .84]) * cdf[-1], side="left")
        q16, q84 = residual[rows[positions]]
        output[b] = (q84 - q16) / 2
    return output


def compare(actual, expected, context, report, *, is_profile=False):
    actual, expected = np.asarray(actual, float), np.asarray(expected, float)
    if actual.shape != expected.shape or not np.array_equal(np.isfinite(actual), np.isfinite(expected)):
        raise AssertionError(f"Finite support or shape differs: {context}")
    finite = np.isfinite(expected)
    if not np.any(finite):
        return
    scale = max(float(np.max(np.abs(expected[finite]))), 1.)
    tolerance = 1e-9 if is_profile else scale * 5e-13
    if not np.allclose(actual[finite], expected[finite], atol=tolerance, rtol=3e-10):
        difference = float(np.max(np.abs(actual[finite] - expected[finite])))
        raise AssertionError(f"Dense comparison differs by {difference:g}: {context}")
    error = float(np.max(np.abs(actual[finite] - expected[finite])))
    key = "max_profile_abs_difference_degrees" if is_profile else "max_histogram_abs_difference"
    report[key] = max(report.get(key, 0.), error)


def validate(output, dial_slots):
    started = time.monotonic()
    output = Path(output)
    completion = json.loads((output / "completion.json").read_text())
    sources = json.loads((output / "source_order.json").read_text())
    if len(sources) != 1 or sources[0]["counts"]["selected"] > 200_000:
        raise ValueError("This validator intentionally reads only a one-file <=200k-event local pilot")
    if any(slot >= completion["dials"] for slot in dial_slots):
        raise ValueError("Requested dial was not yet completed by this pilot")
    cfg = load_config(output / "analysis_config.yaml")
    datasets = []
    for record in json.loads((output / "manifest_list.json").read_text()):
        path = Path(record["manifest"])
        manifest = json.loads(path.read_text())
        with np.load(path.parent / manifest["data_file"], allow_pickle=False) as archive:
            data = {key: archive[key] for key in archive.files}
        datasets.append((manifest, data))
    response_audit = json.loads((Path(json.loads((output / "manifest_list.json").read_text())[0]["manifest"]).parent
                                / "aggregation_audit.json").read_text())
    audit_by_slot = {entry["slot"]: entry["response_counts"] for entry in response_audit["dials"]}
    report = {"status": "running", "source": sources[0]["cache"], "source_files": 1,
              "dial_slots": list(dial_slots), "geometry": cfg["physics"]["geometry"],
              "config_sha256": cfg["_sha256"], "histogram_vectors": 0, "profile_vectors": 0,
              "baseline_vectors": 0, "flatness_comparisons": 0,
              "raw_response_quality_checks": 0, "raw_pair_quality_checks": 0,
              "fc_pc_histogram_closure_vectors": 0, "actual_reference_points": {},
              "scope": "Independent dense event aggregation on one bounded local cache; no ROOT read"}
    with np.load(sources[0]["cache"], allow_pickle=False) as archive:
        raw = {key: archive[key] for key in RAW_FIELDS}
        derived = derive_observables(raw, cfg["physics"])
        E, category, containment = raw["E"], derived["category"], derived["containment"]
        cut = cfg["selection"]
        selected = (category >= 0) & np.isfinite(E) & (E >= cut["true_energy_min_GeV"])
        if cut.get("true_energy_max_GeV") is not None:
            selected &= E <= cut["true_energy_max_GeV"]
        bounds = np.array([cut["fiducial_bounds_cm"][axis] for axis in "xyz"])
        selected &= np.all((raw["vtx"] > bounds[:, 0]) & (raw["vtx"] < bounds[:, 1]), axis=1)
        if cut.get("require_valid_wall", True):
            selected &= containment >= 0
        if cut.get("require_valid_direction", True):
            selected &= derived["masks"]["angular_resolution"]
        if int(selected.sum()) != completion["selected_events"]:
            raise AssertionError("Independent event selection count differs")
        report["selected_events"] = int(selected.sum())
        report["events_by_truth_and_containment_unknown_pc_fc"] = np.bincount(
            (3 * category + containment + 1)[selected], minlength=15).reshape(5, 3).tolist()
        selection_audit = json.loads((output / "selection_audit.json").read_text())
        if report["events_by_truth_and_containment_unknown_pc_fc"] != selection_audit["events_by_truth_and_containment_unknown_pc_fc"]:
            raise AssertionError("Independent FC/PC and truth-category counts differ")
        base = archive["base_weight"]
        for slot in dial_slots:
            identifier = f"d{slot:03d}"
            rows = archive[identifier + "_rows"]
            sparse = archive[identifier + "_responses"]
            responses = np.ones((len(E), sparse.shape[1]), dtype=np.float32)
            responses[rows] = sparse
            for point in range(sparse.shape[1]):
                x = responses[:, point]
                finite = np.isfinite(x)
                masks = {"nonfinite": ~finite, "negative_finite": finite & (x < 0),
                         "zero_finite": finite & (x == 0), "finite_nonunity": finite & (x != 1),
                         "below_minus9999": finite & (x < -9999), "above_9999": finite & (x > 9999)}
                for name, flag in masks.items():
                    mask = selected & flag
                    expected_count = {"total": int(mask.sum()),
                        "by_category": np.bincount(category[mask], minlength=5).tolist(),
                        "by_truth_and_containment": np.bincount((3 * category + containment + 1)[mask], minlength=15).tolist()}
                    if expected_count != audit_by_slot[slot][point][name]:
                        raise AssertionError(f"Raw quality audit differs: {identifier}/{point}/{name}")
                    report["raw_response_quality_checks"] += 1
            for manifest, data in datasets:
                policy = manifest["dimensions"]["policy"]
                state = manifest["dimensions"]["containment"]
                dial = next(d for d in manifest["dials"] if d["id"] == identifier)
                points = {item["id"]: i for i, item in enumerate(dial["values"])}
                ref = points.get(dial["cv"]["value_id"])
                report["actual_reference_points"][identifier] = {"index": ref, "label": dial["cv"]["label"]}
                response = responses.copy()
                if policy == "fitter_policy":
                    rule = cfg["weight_policy"]
                    response[~np.isfinite(response)] = rule["nonfinite_knot_replacement"]
                    np.clip(response, rule["knot_min"], rule["knot_max"], out=response)
                    np.maximum(response, 0, out=response)
                weights = base[:, None] * response
                reference = base if ref is None else weights[:, ref]
                raw_reference = base if ref is None else base * responses[:, ref]
                for curve in manifest["curves"]:
                    if curve["dial_id"] != identifier or curve.get("available") is False:
                        continue
                    c = next(i for i, s in enumerate(manifest["selections"]) if s["id"] == curve["selection_id"])
                    oid = curve["observable_id"]
                    mask = selected & (category == c) & derived["masks"][oid]
                    if state != "all":
                        mask &= containment == (1 if state == "fc" else 0)
                    edges = data[curve["edges_key"]]
                    actual = data[curve["values_key"]]
                    x = derived["values"][oid]
                    is_profile = oid == "angular_resolution"
                    for point in range(weights.shape[1]):
                        w = weights[:, point]
                        expected = (profile(E, x, mask, w, edges) if is_profile
                                    else histogram(x, mask, w, edges))
                        context = (identifier, policy, state, manifest["dimensions"]["binning"], c, oid, point)
                        compare(actual[point], expected, context, report, is_profile=is_profile)
                        report["profile_vectors" if is_profile else "histogram_vectors"] += 1
                        if oid == "true_coszenith":
                            codes = np.searchsorted(edges, x, side="right") - 1
                            codes[x == edges[-1]] = len(edges) - 2
                            codes[~mask] = -1
                            moments = moments_from_events(codes, w, reference, len(edges) - 1)
                            expected_test = paired_shape_flatness(**moments,
                                min_effective_count=cfg["statistics"]["minimum_effective_count"])
                            actual_test = curve["flatness"]["values"][point]
                            valid = (codes >= 0) & (codes < len(edges) - 1)
                            raw_var = base * responses[:, point]
                            bad = (~np.isfinite(raw_var) | ~np.isfinite(raw_reference)) & valid
                            neg = ((np.isfinite(raw_var) & (raw_var < 0))
                                   | (np.isfinite(raw_reference) & (raw_reference < 0))) & valid
                            if actual_test["raw_nonfinite_pair_events"] != int(bad.sum()) or actual_test["raw_negative_pair_events"] != int(neg.sum()):
                                raise AssertionError(f"Raw paired quality differs: {context}")
                            report["raw_pair_quality_checks"] += 2
                            # The raw production path invalidates touched histogram bins;
                            # the direct event reference signals the same failure by counts.
                            if actual_test["status"] != expected_test["status"]:
                                raise AssertionError(f"Flatness status differs: {context}: {actual_test['status']}/{expected_test['status']}")
                            for field in ("statistic", "ndof", "p_value"):
                                av, ev = actual_test[field], expected_test[field]
                                if (av is None) != (ev is None) or (ev is not None and not np.isclose(av, ev, rtol=1e-7, atol=1e-9)):
                                    raise AssertionError(f"Flatness {field} differs: {context}: {av}/{ev}")
                            if "ratio_shape_se" in actual_test:
                                compare(actual_test["ratio_shape_se"], expected_test["ratio_shape_se"],
                                        context + ("shape standard errors",), report)
                            report["flatness_comparisons"] += 1
                    if curve.get("cv_key"):
                        expected = (profile(E, x, mask, base, edges) if is_profile
                                    else histogram(x, mask, base, edges))
                        compare(data[curve["cv_key"]], expected, (identifier, state, c, oid, "unity reference"),
                                report, is_profile=is_profile)
                        report["baseline_vectors"] += 1
            print(f"Validated {identifier} from dense event weights", flush=True)
    # Fixed fine bins permit direct ALL=FC+PC histogram closure. Quantile widths
    # are intentionally excluded: pooled quantiles never add across selections.
    if cfg["selection"].get("require_valid_wall", True):
        fine = {(m["dimensions"]["policy"], m["dimensions"]["containment"]): (m, d)
                for m, d in datasets if m["dimensions"]["binning"] == "fine"}
        for policy in ("fitter_policy", "raw"):
            all_m, all_d = fine[(policy, "all")]
            fc_m, fc_d = fine[(policy, "fc")]
            pc_m, pc_d = fine[(policy, "pc")]
            def curve_map(manifest):
                return {(c["selection_id"], c["dial_id"], c["observable_id"]): c
                        for c in manifest["curves"] if c.get("available") is not False}
            fc_curves, pc_curves = curve_map(fc_m), curve_map(pc_m)
            for key, curve in curve_map(all_m).items():
                if key[2] == "angular_resolution" or int(key[1][1:]) not in dial_slots:
                    continue
                fc, pc = fc_curves[key], pc_curves[key]
                expected = fc_d[fc["values_key"]] + pc_d[pc["values_key"]]
                compare(all_d[curve["values_key"]], expected, ("FC+PC closure", policy) + key, report)
                report["fc_pc_histogram_closure_vectors"] += len(expected)
    report.update(status="passed", elapsed_seconds=time.monotonic() - started,
                  dataset_count=len(datasets), observables=len(OBSERVABLE_IDS))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--dials", nargs="+", type=int, default=[0, 15, 60, 61, 62, 78, 91, 92])
    args = parser.parse_args()
    report = validate(args.output, args.dials)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
