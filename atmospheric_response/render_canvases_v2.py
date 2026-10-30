#!/usr/bin/env python3
"""Render version-2 aggregate manifests; never open event-level ROOT files.

The first canvas retains the seven original observables. The second holds
true/reconstructed neutrino energy, generator/reconstructed inelasticity and
geometric baseline-over-energy proxies. Labels come from the manifest, so a
proxy is not silently promoted to an exact truth observable.

Version 2 extends the original JSON/NPZ contract without changing array meanings:

* Each available curve contains a value-by-bin ``values_key`` and either an
  in-grid CV coordinate or a separately named ``cv_key``. Histogram arrays are
  weighted bin SUMS. No response or central value is inferred by this renderer.
* A curve may override ``edges_key``, ``centers_key`` and ``x_scale``. This lets
  electron/muon selections share an observable but use different Miles bins.
  A zero-edge energy bin must use linear or symlog x, never an invented epsilon.
* ``x_scale: symlog`` uses base 10, linscale 1 and ``x_linthresh`` (default 0.1 in
  that observable's units). This only changes display coordinates. Histogram
  normalization and CSV/data coordinates remain in physical units.
* Optional ``flatness.values`` contains one JSON-safe paired shape-test result
  per stored dial point. Missing, signed and low-statistic tests remain explicit;
  a missing p-value is never converted into p=1. Tests are supplied upstream.
* Unavailable observables require ``available:false`` and a reason; an NC
  charged-lepton or reconstructed-y panel must not be filled with fake zeros.

The true-zenith panel always has rate/reference AND normalized-shape/reference
ratios. The latter equals normalizing every curve to the same reference yield
before division. It does not imply that a cross-section dial must preserve the
marginal zenith shape under flux and event-selection correlations.

Example:
  python -m atmospheric_response.render_canvases_v2 outputs/manifest.json \
      --canvas kinematics --selections nue_cc --dials d031 --modes rate \
      --individual-formats --output-dir outputs/review
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np

# Keep the original requested order stable. New observables occupy a second
# canvas, which makes comparisons readable without shrinking thirteen panels.
ORIGINAL_ORDER = ["true_lepton_energy", "reco_lepton_energy", "true_hadronic_transfer",
                  "reco_hadronic_energy", "true_coszenith", "reco_coszenith", "angular_resolution"]
KINEMATICS_ORDER = ["true_neutrino_energy", "reco_neutrino_energy", "true_inelasticity",
                    "reco_inelasticity", "true_l_over_e", "reco_l_over_e"]


def canvas_groups(observables):
    first = [x for x in ORIGINAL_ORDER if x in observables]
    second = [x for x in KINEMATICS_ORDER if x in observables]
    second += [x for x in observables if x not in first and x not in second]
    if first:
        yield "original", "Lepton, hadron and angular observables", first
    if second:
        require(len(second) <= 7, "The second canvas supports at most seven panels")
        yield "kinematics", "Neutrino energy, inelasticity and L/E", second


def flatness_summary(flatness):
    """A short label only: the full per-point tests remain in JSON/CSV."""
    values = flatness.get("values", [])
    tests = [x for x in values if isinstance(x, dict) and numeric(x.get("p_value"))]
    if not tests:
        if values and all(x.get("status") == "proportional_within_precision" for x in values):
            return "Proportional within precision; no nondegenerate p-value"
        return "Shape-flatness test unavailable" if values else ""
    smallest, largest = min(x["p_value"] for x in tests), max(x["p_value"] for x in tests)
    low = f"{smallest:.2g}" if smallest > 0 else "below numerical range"
    high = f"{largest:.2g}" if largest > 0 else "below numerical range"
    interval = low if smallest == largest else f"{low}–{high}"
    return f"Paired shape p={interval}; {len(tests)} points (descriptive)"


def raw_quality_note(flatness, policy):
    """Describe paired raw flags in the selected visible true-zenith range.

    Each count already covers the variation/reference pair at one dial point.
    Taking a maximum avoids incorrectly summing the same events over points.
    These audit flags never alter the plotted arrays or the test results.
    """
    tests = flatness.get("values", [])
    nonfinite = max((int(test.get("raw_nonfinite_pair_events", 0)) for test in tests), default=0)
    negative = max((int(test.get("raw_negative_pair_events", 0)) for test in tests), default=0)
    if nonfinite == 0 and negative == 0:
        return ""
    note = (f"Selected true-zenith range: max per stored point is {nonfinite:,} nonfinite-response "
            f"and {negative:,} negative raw-weight events in the variation/reference pair; "
            "these are not sums over points. ")
    if policy == "fitter_policy":
        note += ("The numerical policy replaces nonfinite responses by 1 and sets nonpositive final "
                 "weights to 0. Handled flatness can reflect these replacements, not physical insensitivity.")
    else:
        note += ("Raw nonfinite contributions leave undefined bins; negative event weights make "
                 "probability quantiles undefined.")
    return note


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unique_map(items, name):
    require(isinstance(items, list) and items, f"{name} must be a nonempty list")
    result = {}
    for item in items:
        require(isinstance(item, dict), f"Every {name} item must be an object")
        ident = item.get("id")
        require(isinstance(ident, str) and ident, f"Every {name} item needs an id")
        require(ident not in result, f"Duplicate {name} id: {ident}")
        result[ident] = item
    return result


def numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def slug(value):
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip(".-")[:70] or "item"
    return f"{stem}-{hashlib.sha256(value.encode()).hexdigest()[:8]}"



def add_response_quality_notes(manifest_path, dials):
    """Attach audit-derived warnings without changing any response arrays."""
    audit_path = Path(manifest_path).parent / "aggregation_audit.json"
    if not audit_path.exists():
        return None
    raw_audit = audit_path.read_text(encoding="utf-8")
    audit = json.loads(raw_audit)
    for record in audit.get("dials", []):
        ident = f"d{record['slot']:03d}"
        if ident not in dials:
            continue
        dial = dials[ident]
        require(record["name"] == dial["label"], f"Audit name mismatch for {ident}")
        counts = record["response_counts"]
        require(len(counts) == len(dial["values"]), f"Audit point-count mismatch for {ident}")
        require([c["point_index"] for c in counts] == list(range(len(counts))),
                f"Audit point order mismatch for {ident}")
        missing = [int(c["nonfinite"]["total"]) for c in counts]
        if not any(missing):
            continue
        if len(set(missing)) == 1:
            scope = f"{missing[0]:,} events have nonfinite responses at each of {len(counts)} stored points"
        else:
            scope = f"{min(missing):,}–{max(missing):,} events per stored point have nonfinite responses"
        count_scope = audit.get("response_count_scope", "selected sample represented by this audit")
        warning = f"Raw-response audit, {count_scope}: {scope}."
        if all(int(c["finite_nonunity"]["total"]) == 0 for c in counts):
            warning += (" All finite responses are unity. Numerical-policy flatness follows from replacing missing responses by 1;"
                        " it is not evidence of physical insensitivity.")
        else:
            warning += " Finite variations also exist; the missing responses remain unknown."
        metadata = dial.setdefault("metadata", {})
        metadata["response_quality_notes"] = [warning]
        metadata["response_counts"] = counts
        metadata["notes"] = [warning] + [x for x in metadata.get("notes", []) if x != warning]
    return {"path": str(audit_path.resolve()), "sha256": hashlib.sha256(raw_audit.encode()).hexdigest()}


def load_inputs(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    raw_manifest = manifest_path.read_text(encoding="utf-8")
    doc = json.loads(raw_manifest)
    require(doc.get("schema_version") in (1, 2), "Expected schema_version=1 or 2")
    require(isinstance(doc.get("allow_missing_histogram_bins", False), bool),
            "allow_missing_histogram_bins must be Boolean")
    allow_missing_histograms = doc.get("allow_missing_histogram_bins", False)
    for field in ("title", "weight_description"):
        require(field not in doc or isinstance(doc[field], str), f"{field} must be a string")
    if "output_prefix" in doc:
        require(isinstance(doc["output_prefix"], str) and re.fullmatch(r"[A-Za-z0-9_-]+", doc["output_prefix"]),
                "output_prefix must contain only ASCII letters, numbers, underscores or hyphens")
    require(isinstance(doc.get("data_file"), str), "data_file must name an NPZ file")
    data_path = (manifest_path.parent / doc["data_file"]).resolve()
    require(data_path.suffix.lower() == ".npz", "data_file must be NPZ, not ROOT")
    selections = unique_map(doc.get("selections"), "selections")
    observables = unique_map(doc.get("observables"), "observables")
    require(1 <= len(observables) <= 14, "Expected 1–14 explicitly defined observables")
    dials = unique_map(doc.get("dials"), "dials")
    response_audit_source = add_response_quality_notes(manifest_path, dials)
    for name, mapping in (("selection", selections), ("dial", dials)):
        for ident, item in mapping.items():
            require(isinstance(item.get("label"), str), f"{name} {ident}: label is required")
            require("description" not in item or isinstance(item["description"], str),
                    f"{name} {ident}: description must be a string")
    for ident, obs in observables.items():
        for field in ("title", "x_label", "y_label", "edges_key"):
            require(isinstance(obs.get(field), str), f"Observable {ident}: {field} is required")
        require(obs.get("kind") in ("histogram", "profile"), f"Observable {ident}: invalid kind")
        require(obs.get("x_scale", "linear") in ("linear", "log", "symlog"), f"Observable {ident}: invalid x_scale")
        if obs.get("x_scale") == "symlog":
            require(numeric(obs.get("x_linthresh", .1)) and obs.get("x_linthresh", .1) > 0,
                    f"Observable {ident}: symlog requires positive x_linthresh")
        require(obs.get("y_scale", "linear") in ("linear", "log"), f"Observable {ident}: invalid y_scale")
        require("shape_y_label" not in obs or isinstance(obs["shape_y_label"], str),
                f"Observable {ident}: shape_y_label must be a string")
        if "ratio_limits" in obs:
            limits = obs["ratio_limits"]
            require(isinstance(limits, list) and len(limits) == 2 and all(numeric(x) for x in limits)
                    and limits[0] < limits[1], f"{ident}: invalid ratio_limits")
    for ident, dial in dials.items():
        require(isinstance(dial.get("units"), str), f"Dial {ident}: explicit units are required (empty is allowed)")
        require(isinstance(dial.get("reference_is_cv", True), bool), f"Dial {ident}: reference_is_cv must be Boolean")
        values = unique_map(dial.get("values"), f"dial {ident} values")
        for vid, value in values.items():
            require(isinstance(value.get("label"), str), f"Dial {ident}, value {vid}: label is required")
            require("value" in value, f"Dial {ident}, value {vid}: physical value/choice is required")
            if "offset" in value:
                require(numeric(value["offset"]), f"Dial {ident}, value {vid}: offset must be finite numeric")
        cv = dial.get("cv")
        require(isinstance(cv, dict) and "value_id" in cv and isinstance(cv.get("label"), str),
                f"Dial {ident}: cv needs explicit value_id (or null) and label")
        require(cv["value_id"] is None or cv["value_id"] in values, f"Dial {ident}: unknown CV value_id")
        if cv["value_id"] is None:
            require("value" in cv, f"Dial {ident}: separate CV needs its physical value/choice")
        elif "value" in cv:
            require(cv["value"] == values[cv["value_id"]]["value"], f"Dial {ident}: inconsistent CV value")

    curves = {}
    available_count = 0
    # Load only explicitly referenced arrays. Pickled/object arrays are forbidden.
    with np.load(data_path, allow_pickle=False) as archive:
        def array(key, context):
            require(isinstance(key, str) and key in archive, f"{context}: missing NPZ array {key!r}")
            value = np.asarray(archive[key])
            require(value.dtype.kind in "fiu", f"{context}: array {key} must be real numeric")
            return value.astype(np.float64, copy=True)

        for oid, obs in observables.items():
            edges = array(obs["edges_key"], oid)
            require(edges.ndim == 1 and len(edges) >= 2 and np.all(np.isfinite(edges))
                    and np.all(np.diff(edges) > 0), f"{oid}: edges must be finite and strictly increasing")
            if obs.get("x_scale") == "log":
                require(np.all(edges > 0), f"{oid}: log x requires positive edges")
            centers = array(obs["centers_key"], oid) if "centers_key" in obs else (edges[:-1] + edges[1:]) / 2
            require(centers.shape == (len(edges) - 1,) and np.all(np.isfinite(centers)), f"{oid}: invalid centers")
            require(np.all((centers >= edges[:-1]) & (centers <= edges[1:])), f"{oid}: centers outside bins")
            obs["_edges"], obs["_centers"] = edges, centers

        records = doc.get("curves")
        require(isinstance(records, list), "curves must be a list")
        for record in records:
            require(isinstance(record, dict), "Every curve record must be an object")
            key = tuple(record.get(k) for k in ("selection_id", "dial_id", "observable_id"))
            sid, did, oid = key
            require(sid in selections and did in dials and oid in observables, f"Unknown curve coordinates: {key}")
            require(key not in curves, f"Duplicate curve record: {key}")
            if record.get("available", True) is False:
                require(isinstance(record.get("reason"), str) and record["reason"], f"{key}: unavailable record needs reason")
                curves[key] = {"available": False, "reason": record["reason"]}
                continue
            dial, obs = dials[did], observables[oid]
            matrix = array(record.get("values_key"), str(key))
            edges = array(record["edges_key"], str(key)) if "edges_key" in record else obs["_edges"]
            require(edges.ndim == 1 and len(edges) >= 2 and np.all(np.isfinite(edges))
                    and np.all(np.diff(edges) > 0), f"{key}: invalid per-selection bin edges")
            if "centers_key" in record:
                centers = array(record["centers_key"], str(key))
            elif "edges_key" not in record:
                centers = obs["_centers"]
            else:
                centers = (edges[:-1]+edges[1:])/2
            require(centers.shape == (len(edges)-1,) and np.all(np.isfinite(centers))
                    and np.all((centers >= edges[:-1]) & (centers <= edges[1:])), f"{key}: invalid centers")
            xscale = record.get("x_scale", obs.get("x_scale", "linear"))
            require(xscale in ("linear", "log", "symlog"), f"{key}: invalid x_scale")
            require(xscale != "log" or np.all(edges > 0), f"{key}: log x requires positive edges; zero-edge bins need linear or symlog x")
            linthresh = record.get("x_linthresh", obs.get("x_linthresh", .1))
            require(numeric(linthresh) and linthresh > 0, f"{key}: x_linthresh must be positive")
            nbins = len(edges) - 1
            require(matrix.shape == (len(dial["values"]), nbins),
                    f"{key}: values shape {matrix.shape}, expected ({len(dial['values'])}, {nbins}); do not transpose silently")
            require(not np.any(np.isinf(matrix)), f"{key}: infinite curve values")
            if obs["kind"] == "histogram" and not allow_missing_histograms:
                require(np.all(np.isfinite(matrix)), f"{key}: histogram values must be finite")
            cv_id = dial["cv"]["value_id"]
            if cv_id is None:
                cv = array(record.get("cv_key"), f"{key} separate CV")
                require(cv.shape == (nbins,), f"{key}: CV shape must be ({nbins},)")
            else:
                require("cv_key" not in record, f"{key}: cv_key conflicts with an in-grid CV")
                cv_index = next(i for i, value in enumerate(dial["values"]) if value["id"] == cv_id)
                cv = matrix[cv_index].copy()
            require(not np.any(np.isinf(cv)), f"{key}: infinite CV values")
            if obs["kind"] == "histogram" and not allow_missing_histograms:
                require(np.all(np.isfinite(cv)), f"{key}: histogram CV must be finite")
            if obs.get("y_scale") == "log":
                require(not np.any(matrix < 0) and not np.any(cv < 0),
                        f"{key}: log y would hide negative values; choose linear y_scale")
            curves[key] = {"available": True, "values": matrix, "cv": cv,
                           "edges": edges, "centers": centers, "x_scale": xscale, "x_linthresh": linthresh}
            if "flatness" in record:
                require(isinstance(record["flatness"], dict), f"{key}: flatness must be a JSON object")
                flatness = record["flatness"]
                require(isinstance(flatness.get("values"), list)
                        and len(flatness["values"]) == len(dial["values"]),
                        f"{key}: flatness tests must follow every actual stored point")
                tests = flatness["values"] + ([flatness["reference"]] if "reference" in flatness else [])
                for test in tests:
                    require(isinstance(test, dict) and isinstance(test.get("status"), str),
                            f"{key}: each flatness test requires an explicit status")
                    require(test.get("p_value") is None or numeric(test["p_value"])
                            and 0 <= test["p_value"] <= 1, f"{key}: invalid flatness p-value")
                    for field in ("ratio_shape", "ratio_shape_se"):
                        if field in test:
                            require(isinstance(test[field], list) and len(test[field]) == nbins
                                    and all(x is None or numeric(x) for x in test[field]),
                                    f"{key}: {field} must match this curve's bins")
                curves[key]["flatness"] = flatness
            for field in ("note", "missing_reason"):
                if field in record:
                    require(isinstance(record[field], str), f"{key}: {field} must be text")
                    curves[key][field] = record[field]
            available_count += 1
    for sid in selections:
        for did in dials:
            for oid in observables:
                require((sid, did, oid) in curves,
                        f"Missing record {(sid, did, oid)}; use available=false with a reason for unavailable observables")
    require(available_count > 0, "No actual response curves supplied; no output will be generated")
    return doc, selections, observables, dials, curves, {
        "manifest": str(manifest_path), "data_file": str(data_path),
        "manifest_sha256": hashlib.sha256(raw_manifest.encode()).hexdigest(),
        "available_records": available_count, "response_audit_source": response_audit_source,
    }


def normalized(values, edges, kind, mode):
    """Histogram inputs are bin SUMS; shape output is density of unit area."""
    result = np.asarray(values, dtype=float).copy()
    if mode != "shape" or kind == "profile":
        return result, None
    if not np.all(np.isfinite(result)):
        return np.full_like(result, np.nan), "Unit-area shape unavailable: missing bins make the integral unknown"
    total = float(np.sum(result))
    if not math.isfinite(total) or total <= 0:
        return np.full_like(result, np.nan), "Unit-area shape unavailable: nonpositive or nonfinite total"
    return result / total / np.diff(edges), None


def ratio(values, cv):
    valid = np.isfinite(values) & np.isfinite(cv) & (cv != 0)
    result = np.full(np.shape(cv), np.nan, dtype=float)
    np.divide(values, cv, out=result, where=valid)
    return result


def line_styles(dial, color_map):
    cv = dial["cv"]
    cv_value = cv.get("value")
    if cv["value_id"] is not None:
        cv_value = next(v["value"] for v in dial["values"] if v["id"] == cv["value_id"])
    magnitudes = []
    styles = []
    for value in dial["values"]:
        if value["id"] == cv["value_id"]:
            styles.append({"color": "black", "linestyle": "-", "linewidth": 2.1, "zorder": 5})
            continue
        offset = value.get("offset")
        if offset is None and numeric(value["value"]) and numeric(cv_value):
            offset = value["value"] - cv_value
        # Pair equal physical offsets from CV; never interpret them as sigma.
        magnitude = abs(offset) if offset is not None else None
        idx = next((i for i, m in enumerate(magnitudes)
                    if magnitude is not None and m is not None
                    and math.isclose(m, magnitude, rel_tol=1e-9, abs_tol=1e-12)), None)
        if idx is None:
            idx = len(magnitudes)
            magnitudes.append(magnitude)
        linestyle = "--" if offset is not None and offset < 0 else ":" if offset is not None and offset > 0 else "-"
        styles.append({"color": color_map((0.1 + idx * 0.618033988749895) % 1),
                       "linestyle": linestyle, "linewidth": 1.1, "zorder": 2})
    return styles


def draw_curve(ax, obs, values, style):
    if obs["kind"] == "histogram":
        # A zero baseline would add artificial end drops, especially in ratios.
        ax.stairs(values, obs["_edges"], baseline=None, **style)
    else:
        ax.plot(obs["_centers"], values, marker=".", markersize=3.3, **style)


def make_page(doc, selection, dial, observables, curves, mode, plt, Line2D, canvas_label=""):
    """Render one at-most-seven-panel page, leaving the eighth cell for notes."""
    import textwrap

    fig = plt.figure(figsize=(18, 20), layout="constrained")
    grid = fig.add_gridspec(4, 2, hspace=0.09, wspace=0.07)
    styles = line_styles(dial, plt.get_cmap("turbo"))
    notes = []
    reference_term = "CV" if dial.get("reference_is_cv", True) else "reference"
    mode_label = "Rate / binned statistic" if mode == "rate" else "Unit-area distributions / unchanged binned statistics"
    fig.suptitle(f"{doc.get('title', 'Cross-section dial responses')}\n"
                 f"{selection['label']} · {dial['label']}\n{mode_label} · {canvas_label}", fontsize=14)
    for panel, (oid, obs) in enumerate(observables.items()):
        key = (selection["id"], dial["id"], oid)
        record = curves[key]
        obs = {**obs, "_edges": record.get("edges", obs["_edges"]),
               "_centers": record.get("centers", obs["_centers"]),
               "x_scale": record.get("x_scale", obs.get("x_scale", "linear")),
               "x_linthresh": record.get("x_linthresh", obs.get("x_linthresh", .1))}
        cell = grid[panel // 2, panel % 2]
        if not record["available"]:
            ax = fig.add_subplot(cell)
            ax.axis("off")
            ax.set_title(obs["title"], loc="left", fontsize=12)
            ax.text(0.03, 0.55, "Unavailable\n" + textwrap.fill(record["reason"], 72), transform=ax.transAxes, va="center", fontsize=10)
            continue
        triple = oid == "true_coszenith"
        sub = cell.subgridspec(3 if triple else 2, 1, height_ratios=[2.5, 1.0, 1.0] if triple else [3.0, 1.0], hspace=0.035)
        ax = fig.add_subplot(sub[0])
        rx = fig.add_subplot(sub[1], sharex=ax)
        sx = fig.add_subplot(sub[2], sharex=ax) if triple else None
        shape_cv, _ = normalized(record["cv"], obs["_edges"], obs["kind"], "shape")
        cv, warning = normalized(record["cv"], obs["_edges"], obs["kind"], mode)
        plotted_arrays = [cv]
        unavailable_labels = []
        if warning:
            notes.append({"selection": selection["id"], "dial": dial["id"], "observable": oid, "value": "CV", "warning": warning})
            unavailable_labels.append(dial["cv"]["label"])
        for index, (value, style) in enumerate(zip(dial["values"], styles)):
            if value["id"] == dial["cv"]["value_id"]:
                continue
            shifted, warning = normalized(record["values"][index], obs["_edges"], obs["kind"], mode)
            plotted_arrays.append(shifted)
            if warning:
                notes.append({"selection": selection["id"], "dial": dial["id"], "observable": oid, "value": value["id"], "warning": warning})
                unavailable_labels.append(value["label"])
            draw_curve(ax, obs, shifted, style)
            draw_curve(rx, obs, ratio(record["values"][index], record["cv"]) if triple else ratio(shifted, cv), style)
            if triple:
                shape_shifted, _ = normalized(record["values"][index], obs["_edges"], obs["kind"], "shape")
                draw_curve(sx, obs, ratio(shape_shifted, shape_cv), style)
        cvstyle = {"color": "black", "linestyle": "-", "linewidth": 2.1, "zorder": 5}
        draw_curve(ax, obs, cv, cvstyle)
        draw_curve(rx, obs, ratio(record["cv"], record["cv"]) if triple else ratio(cv, cv), {**cvstyle, "linewidth": 1.0})
        if triple:
            draw_curve(sx, obs, ratio(shape_cv, shape_cv), {**cvstyle, "linewidth": 1.0})
        ax.set_title(obs["title"], loc="left", fontsize=12)
        if mode == "shape" and obs["kind"] == "histogram":
            ax.set_ylabel(obs.get("shape_y_label", "Unit-area density"), fontsize=10)
        else:
            ax.set_ylabel(obs["y_label"], fontsize=10)
        if triple:
            ax.set_ylabel("Unit-area density" if mode == "shape" else "Weighted sum / bin", fontsize=10)
        if mode == "shape" and obs["kind"] == "profile":
            ax.text(0.98, 0.96, "Statistic: no area normalization", transform=ax.transAxes,
                    ha="right", va="top", fontsize=8, color="#444444")
        if unavailable_labels:
            visible_note = "Shape unavailable: " + ", ".join(unavailable_labels[:3])
            if len(unavailable_labels) > 3:
                visible_note += f"; {len(unavailable_labels)-3} additional curves (see audit)"
            ax.text(0.02, 0.96, textwrap.fill(visible_note, 75), transform=ax.transAxes,
                    ha="left", va="top", fontsize=8, color="#8c2d04")
        missing_bins = int(np.count_nonzero(~np.isfinite(record["values"])))
        if missing_bins:
            missing_note = record.get("missing_reason", "Undefined values are shown as gaps")
            ax.text(0.98, 0.04, textwrap.fill(missing_note, 82), transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=7, color="#8c2d04")
            notes.append({"selection": selection["id"], "dial": dial["id"], "observable": oid,
                          "missing_value_bins": missing_bins, "warning": missing_note})
        if obs.get("y_scale") == "log" and (np.any(record["values"] == 0) or np.any(record["cv"] == 0)):
            ax.text(0.98, 0.05, "Zero bins are not visible on log y", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=8, color="#444444")
        if obs.get("x_scale") == "symlog":
            ax.set_xscale("symlog", linthresh=obs["x_linthresh"], linscale=1, base=10)
            ticks = [x for x in (-100,-10,-5,-2,-1,-.5,-.2,-.1,0,.1,.2,.5,1,2,5,10,100)
                     if obs["_edges"][0] <= x <= obs["_edges"][-1]]
            ax.set_xticks(ticks, labels=[f"{x:g}" for x in ticks])
            ax.text(.02,.95,f"Symlog x: linear within ±{obs['x_linthresh']:g}",
                    transform=ax.transAxes, va="top", fontsize=7, color="#555555")
        else:
            ax.set_xscale(obs.get("x_scale", "linear"))
        yscale = obs.get("y_scale", "linear")
        if yscale == "log" and not any(np.any(np.isfinite(a) & (a > 0)) for a in plotted_arrays):
            yscale = "linear"
            message = "No positive values: displayed with linear y"
            ax.text(0.98, 0.13, message, transform=ax.transAxes, ha="right", fontsize=8)
            notes.append({"selection": selection["id"], "dial": dial["id"], "observable": oid, "warning": message})
        ax.set_yscale(yscale)
        ax.tick_params(labelbottom=False)
        short_reference = "CV" if dial.get("reference_is_cv", True) else "ref."
        rx.set_ylabel(("Rate / " if triple else "/ ") + short_reference, fontsize=9)
        bottom = sx if triple else rx
        bottom.set_xlabel(obs["x_label"], fontsize=10)
        if triple:
            rx.tick_params(labelbottom=False)
            sx.set_ylabel("Shape / " + short_reference, fontsize=9)
            sx.axhline(1, color="0.7", linewidth=0.6, zorder=0)
            summary = flatness_summary(record.get("flatness", {}))
            if summary:
                ax.text(.02,.95,summary, transform=ax.transAxes, va="top", fontsize=7, color="#555555")
        rx.axhline(1, color="0.7", linewidth=0.6, zorder=0)
        if "ratio_limits" in obs:
            rx.set_ylim(obs["ratio_limits"])
        for axis in ((ax, rx, sx) if triple else (ax, rx)):
            axis.grid(True, alpha=0.2, linewidth=0.5)
            axis.tick_params(labelsize=9)
            axis.set_xlim(obs["_edges"][0], obs["_edges"][-1])
        n_masked = int(np.count_nonzero(~np.isfinite(cv) | (cv == 0)))
        if n_masked:
            rx.text(0.99, 0.07, f"{n_masked} {reference_term}-zero/invalid bins masked", transform=rx.transAxes, ha="right", fontsize=7)
    legend_ax = fig.add_subplot(grid[3, 1])
    legend_ax.axis("off")
    handles = [Line2D([], [], color="black", linewidth=2.1, label=dial["cv"]["label"])]
    handles += [Line2D([], [], **style, label=value["label"])
                for value, style in zip(dial["values"], styles) if value["id"] != dial["cv"]["value_id"]]
    legend_ax.legend(handles=handles, loc="upper left", ncols=max(1, math.ceil(len(handles) / 13)),
                     fontsize=8 if len(handles) > 18 else 9, frameon=False,
                     title=f"{dial['label']}" + (f" [{dial['units']}]" if dial["units"] else ""), title_fontsize=10)
    if doc.get("nominal_only", False):
        metadata = ["Nominal baseline only. No systematic response curves are available.",
                    "Black ratios are CV/CV where defined; they do not show dial sensitivity."]
    else:
        style_note = ("Scenario curves are solid; colors distinguish configured alternatives."
                      if dial.get("metadata", {}).get("kind") == "coupled_categorical_scenarios"
                      else "Paired offsets share a color; lower values dashed, higher values dotted.")
        metadata = [f"{reference_term.capitalize()} is black. {style_note}",
                    f"Ratios mask zero/invalid {reference_term} bins. All supplied variations are drawn when defined."]
    if mode == "shape":
        metadata.append("Each histogram is divided by its own sum and bin width; profiles are not normalized.")
        if doc.get("allow_missing_histogram_bins", False):
            metadata.append("Signed densities may be negative; they are not probability densities. Missing histogram bins make a unit-area curve unavailable.")
    selected_zenith = curves.get((selection["id"], dial["id"], "true_coszenith"), {})
    policy = doc.get("dimensions", {}).get("policy", "raw" if doc.get("allow_missing_histogram_bins") else "fitter_policy")
    quality_note = raw_quality_note(selected_zenith.get("flatness", {}), policy)
    if quality_note:
        metadata.append(quality_note)
        notes.append({"selection": selection["id"], "dial": dial["id"],
                      "observable": "true_coszenith", "warning": quality_note})
    metadata.extend(dial.get("metadata", {}).get("response_quality_notes", []))
    if selection.get("description"):
        metadata.append(selection["description"])
    geometry = doc.get("provenance", {}).get("geometry", {})
    if any("l_over_e" in oid for oid in observables) and geometry:
        metadata.append(
            f"L/E geometry proxy: Earth radius {geometry['earth_radius_km']:g} km, "
            f"fixed production height {geometry['production_height_km']:g} km, "
            f"detector depth {geometry['detector_depth_km']:g} km; no production-height average.")
    if doc.get("weight_description"):
        metadata.append(doc["weight_description"])
    legend_ax.text(0.03, 0.02, "\n\n".join(textwrap.fill(t, 80) for t in metadata),
                   transform=legend_ax.transAxes, va="bottom", fontsize=8)
    return fig, notes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path,
                        help="Version-2 aggregate JSON manifest (legacy version 1 is also readable)")
    parser.add_argument("--output-dir", type=Path, help="Default: manifest directory / canvases")
    parser.add_argument("--modes", nargs="+", choices=("rate", "shape"), default=["rate", "shape"])
    parser.add_argument("--individual-formats", nargs="*", choices=("png", "pdf"), default=["png", "pdf"],
                        help="Use the flag with no values to produce only multipage books")
    parser.add_argument("--book-group", choices=("all", "selection", "dial"), default="all")
    parser.add_argument("--selections", nargs="+", help="Render only these selection IDs")
    parser.add_argument("--dials", nargs="+", help="Render only these dial IDs")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--canvas", choices=("all", "original", "kinematics"), default="all")
    parser.add_argument("--validate-only", action="store_true", help="Validate inputs without creating figures/files")
    args = parser.parse_args(argv)
    require(args.dpi > 0, "dpi must be positive")
    doc, selections, observables, dials, curves, audit = load_inputs(args.manifest)
    for ids, available, what in ((args.selections, selections, "selection"), (args.dials, dials, "dial")):
        if ids is not None:
            require(set(ids) <= set(available), f"Unknown {what} IDs: {set(ids)-set(available)}")
    if args.selections:
        selections = {k:v for k,v in selections.items() if k in args.selections}
    if args.dials:
        dials = {k:v for k,v in dials.items() if k in args.dials}
    if args.validate_only:
        print(json.dumps({**audit, "status": "valid", "selections": len(selections), "dials": len(dials)}, indent=2))
        return 0

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "savefig.facecolor": "white"})
    output_dir = (args.output_dir or args.manifest.resolve().parent / "canvases").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    audit.update({"modes": list(dict.fromkeys(args.modes)), "book_group": args.book_group,
                  "pages": [], "warnings": []})
    pairs = [(s, d) for s in selections.values() for d in dials.values()]
    for mode in audit["modes"]:
        groups = {}
        for selection, dial in pairs:
            group = "all" if args.book_group == "all" else selection["id"] if args.book_group == "selection" else dial["id"]
            groups.setdefault(group, []).append((selection, dial))
        for group, members in groups.items():
            suffix = "" if args.book_group == "all" else "_"+slug(group)
            book_path = output_dir / f"{doc.get('output_prefix', 'all_selections_all_dials')}{suffix}_{mode}.pdf"
            with PdfPages(book_path, metadata={"Title": doc.get("title", "Cross-section dial responses"), "Subject": "Actual supplied responses only"}) as book:
              for selection, dial in members:
                directory = output_dir / mode / slug(selection["id"])
                if args.individual_formats:
                    directory.mkdir(parents=True, exist_ok=True)
                for canvas_id, canvas_label, keys in canvas_groups(observables):
                    if args.canvas != "all" and args.canvas != canvas_id:
                        continue
                    subset = {key: observables[key] for key in keys}
                    fig, warnings = make_page(doc, selection, dial, subset, curves, mode, plt, Line2D, canvas_label)
                    outputs = []
                    try:
                        book.savefig(fig, bbox_inches="tight")
                        for extension in dict.fromkeys(args.individual_formats):
                            path = directory / f"{slug(dial['id'])}_{canvas_id}.{extension}"
                            fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
                            outputs.append(str(path))
                    finally:
                        plt.close(fig)
                    audit["pages"].append({"selection": selection["id"], "dial": dial["id"], "mode": mode,
                                           "canvas": canvas_id, "files": outputs, "multipage_pdf": str(book_path)})
                    audit["warnings"].extend(warnings)
                    print(f"Rendered {mode}: {selection['id']} / {dial['id']} / {canvas_id}", flush=True)
    summary = output_dir / "render_summary.json"
    summary.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(f"Rendered {len(audit['pages'])} pages; audit: {summary}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, OSError) as exc:
        raise SystemExit(f"Input/render error: {exc}") from exc
