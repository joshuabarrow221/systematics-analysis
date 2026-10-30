#!/usr/bin/env python3
"""Bounded, resumable raw-observable and sparse-response cache builder.

Only explicitly listed CAF/weight branches are read. No GENIE records, mmap,
remote writes, or full-file copies. The broad cache preserves later choices of
fiducial volume, containment, angle estimator, energy range and observable mask.
Per-file byte caps and chunk size bound network I/O and working arrays.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import resource
import time

import awkward as ak
import numpy as np
import uproot

from .v2_physics import reduce_particles, truth_categories

HERE = Path(__file__).resolve().parent
MC, RC, HD = "rec.mc.nu.", "rec.common.ixn.pandora.", "rec.meta.fd_hd."
SP, PP = MC + "xsec_systs.", RC + "part.pandora."
SCHEMA_VERSION = 2
SCHEMA_NAME = "broad_raw_observables_sparse_responses_v2"
TRUTH_SCALARS = ["E", "q0", "inelasticity", "baseline", "pdg", "iscc", "genweight", "xsec_cvwgt"]
ENERGY_FIELDS = ["calo", "e_calo", "e_had", "lep_calo", "mu_range", "mu_had"]
CVN_FIELDS = ["nue", "numu", "nc"]
VECTOR_BRANCHES = {"true_momentum": MC + "momentum.",
                   "reco_heshw": RC + "dir.heshw.",
                   "reco_lngtrk": RC + "dir.lngtrk.",
                   "reco_calo": RC + "dir.calo.", "vtx": RC + "vtx."}
RAW_FIELDS = (["E", "q0", "inelasticity", "baseline_m", "pdg", "iscc", "genweight", "xsec_cvwgt"]
              + list(VECTOR_BRANCHES) + ["enu_" + k for k in ENERGY_FIELDS]
              + ["cvn_" + k for k in CVN_FIELDS]
              + ["n_particles", "min_wall_cm", "wall_valid", "invalid_wall_count"])
BASE_FIELDS = RAW_FIELDS + ["category", "energy", "base_weight", "xsec", "flux_nue", "flux_numu",
                            "source_entry", "run", "subrun", "event"]


class BoundedReader:
    """Seekable file-object source avoids uproot's mmap on the SSHFS mount."""
    def __init__(self, path, cap=256 * 2**20):
        self.fp = open(path, "rb")
        self.n = 0
        self.cap = cap

    def read(self, n=-1):
        if n < 0 or n > 16 * 2**20 or self.n + n > self.cap:
            raise RuntimeError(f"Bounded read limit: request={n}, read={self.n}, cap={self.cap}")
        data = self.fp.read(n)
        self.n += len(data)
        return data

    def seek(self, *args):
        return self.fp.seek(*args)

    def tell(self):
        return self.fp.tell()

    def close(self):
        self.fp.close()

    @property
    def closed(self):
        return self.fp.closed


def atomic_json(path, value):
    """Publish complete JSON on the cache volume, never partially-written JSON."""
    path = Path(path)
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def digest_file(path):
    hasher = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def inventory_source_directory(inventory, inventory_path):
    """Resolve relative source directories beside their inventory, not the CWD."""
    directory = inventory.get("directory")
    if not isinstance(directory, str) or not directory.strip():
        raise ValueError("Inventory directory must be a nonempty path string")
    source = Path(directory).expanduser()
    if not source.is_absolute():
        source = Path(inventory_path).resolve().parent / source
    return source.resolve()


def validate_resume_metadata(index, expected_hashes, code_hash):
    """Refuse to reuse an index after either source manifest has changed.

    Even an all-cached run must make these comparisons: it does not reopen ROOT
    headers, so shape-compatible but relabelled dial metadata could otherwise
    silently reuse responses from the wrong named slots.
    """
    if index.get("schema_name") != SCHEMA_NAME or index.get("extractor_and_physics_sha256") != code_hash:
        raise RuntimeError("Extractor/schema changed: use a new cache directory or review cached provenance")
    for field, expected in expected_hashes.items():
        if index.get(field) != expected:
            raise RuntimeError(f"Resume metadata mismatch: {field}; use a new cache directory after reviewing the change")


def validate_cached_record(record, source, cache, code_hash, expected_hashes):
    """Check identities, fingerprints and full local-cache SHA256 before reuse.

    A cache with the original size can still be corrupt. Never repair, replace
    or accept it silently; the caller must inspect the discrepancy. These
    checks read local cache bytes and remote file metadata only, not ROOT data.
    The input-manifest hashes also protect orphan sidecar recovery when an
    interrupted process did not publish its index.
    """
    validate_resume_metadata(record, expected_hashes, code_hash)
    source, cache = Path(source), Path(cache)
    if Path(record.get("source", "")).resolve() != source.resolve():
        raise RuntimeError("Cached source identity does not match the inventory")
    if Path(record.get("cache", "")).resolve() != cache.resolve():
        raise RuntimeError("Cached output identity does not match the requested part")
    stat = source.stat()
    if (stat.st_size, stat.st_mtime_ns) != (record["size"], record["mtime_ns"]):
        raise RuntimeError("Cached source fingerprint changed")
    if not cache.exists() or cache.stat().st_size != record["cache_bytes"]:
        raise RuntimeError("Completed cache missing or size changed")
    if digest_file(cache) != record.get("cache_sha256"):
        raise RuntimeError("Completed cache SHA256 mismatch; refusing to reuse corrupted or unverified data")


def header(f):
    prefix = "global/wgts/wgts.xsec_params/wgts.xsec_params."
    tree = f["globalTree"]
    data = {key: ak.to_list(tree[prefix + key].array(library="ak")[0])
            for key in ["name", "id", "vals"]}
    return [dict(index=i, name=name, id=data["id"][i], values=data["vals"][i])
            for i, name in enumerate(data["name"])]


def expand_responses(lengths, indices, raw, variation_counts):
    """Validate ragged offsets; expand missing responses as exact unit weights."""
    lengths, indices = np.asarray(lengths), np.asarray(indices)
    variation_counts = np.asarray(variation_counts, dtype=np.int64)
    n, ndial = lengths.shape
    if indices.shape != (n, ndial) or ndial != len(variation_counts):
        raise RuntimeError("Unexpected response-offset shape")
    if not np.all((lengths == variation_counts[None, :]) | (lengths == 0)):
        raise RuntimeError("Stored variation count differs from header")
    expected = np.c_[np.zeros(n, dtype=np.int64), np.cumsum(lengths, axis=1)[:, :-1]]
    if not np.array_equal(indices, expected):
        raise RuntimeError("Noncontiguous response offsets")
    total = ak.to_numpy(ak.num(raw))
    if not np.array_equal(total, lengths.sum(axis=1)):
        raise RuntimeError("Response length/offset closure failed")
    offsets = np.r_[0, np.cumsum(variation_counts)]
    npoints = int(offsets[-1])
    if np.all(total == npoints):
        return ak.to_numpy(ak.to_regular(raw, axis=1)).astype(np.float32, copy=False)
    dense = np.ones((n, npoints), dtype=np.float32)
    flat = ak.to_numpy(ak.flatten(raw))
    rowstarts = np.r_[0, np.cumsum(total[:-1])]
    for j, nv in enumerate(variation_counts):
        present = lengths[:, j] > 0
        positions = rowstarts[present, None] + indices[present, j, None] + np.arange(nv)[None, :]
        dense[present, offsets[j]:offsets[j+1]] = flat[positions]
    return dense


def empty_field(key):
    if key in VECTOR_BRANCHES:
        return np.empty((0, 3), dtype=np.float64)
    dtype = np.bool_ if key == "wall_valid" else np.float64
    if key in ["category", "pdg", "iscc", "n_particles", "invalid_wall_count", "source_entry", "run", "subrun", "event"]:
        dtype = np.int64
    return np.empty(0, dtype=dtype)


def scan_file(path, out, expected, chunk_size=25000, read_cap_mib=256, input_hashes=None):
    before, started = path.stat(), time.monotonic()
    reader = BoundedReader(path, cap=read_cap_mib * 2**20)
    variation_counts = np.array([len(d["values"]) for d in expected], dtype=np.int64)
    offsets = np.r_[0, np.cumsum(variation_counts)]
    ndial = len(variation_counts)
    primitive = [MC + key for key in TRUTH_SCALARS]
    primitive += [prefix + axis for prefix in VECTOR_BRANCHES.values() for axis in "xyz"]
    primitive += [RC + "Enu." + key for key in ENERGY_FIELDS]
    primitive += [RC + "nuhyp.cvn." + key for key in CVN_FIELDS]
    ids = [HD + key for key in ["run", "subrun", "event"]]
    particle_fields = [PP + key for key in [".length", ".idx", "origRecoObjType", "walldist"]]
    response_fields = [SP + key for key in [".length", ".idx", "weights..length", "weights..idx", "weights"]]
    weight_fields = ["xsec", "flux_nue", "flux_numu", "run", "genie_weight"]
    branches = primitive + ids + particle_fields + response_fields
    counts = dict(entries=0, one_truth_one_reco=0, selected_before_zero_weight=0,
        zero_weight_selected=0, selected=0, xsec_cvwgt_nonfinite=0,
        negative_response_cells=0, nonfinite_response_cells=0, implicit_unit_slots=0,
        nominal_run_mismatches=0, nominal_genweight_checked=0, nominal_genweight_mismatches=0,
        true_baseline_nonfinite=0, true_y_nonfinite=0, particles_with_invalid_walls=0,
        unknown_wall_minimum=0, E_above_10_GeV=0, E_above_100_GeV=0)
    parts = {key: [] for key in BASE_FIELDS}
    sparse = [([], []) for _ in range(ndial)]
    energy_min, energy_max = np.inf, -np.inf
    try:
        with uproot.open(reader, object_cache=None, array_cache=None, num_workers=1) as root:
            if header(root) != expected:
                raise RuntimeError("Global dial metadata differs: " + path.name)
            tree, weights = root["cafTree"], root["weights"]
            if tree.num_entries != weights.num_entries:
                raise RuntimeError("CAF/nominal entry-count mismatch")
            estimate = sum(tree[k].compressed_bytes for k in branches)
            estimate += sum(weights[k].compressed_bytes for k in weight_fields)
            for start in range(0, tree.num_entries, chunk_size):
                stop = min(start + chunk_size, tree.num_entries)
                a = tree.arrays(branches, entry_start=start, entry_stop=stop, library="ak", how=dict)
                w = weights.arrays(weight_fields, entry_start=start, entry_stop=stop, library="np")
                for key in ["xsec", "flux_nue", "flux_numu"]:
                    if not np.all(np.isfinite(w[key]) & (w[key] >= 0)):
                        raise RuntimeError("Invalid nominal normalization field: " + key)

                def first(key):
                    return ak.to_numpy(ak.fill_none(ak.values_astype(ak.firsts(a[key]), np.float64), np.nan))

                nt = ak.to_numpy(ak.num(a[MC + "E"]))
                nr = ak.to_numpy(ak.num(a[RC + "dir.heshw.x"]))
                for key in primitive:
                    if not np.array_equal(ak.to_numpy(ak.num(a[key])), nt if key.startswith(MC) else nr):
                        raise RuntimeError("Inconsistent primitive array lengths: " + key)
                E, pdg, cc = first(MC + "E"), first(MC + "pdg"), first(MC + "iscc")
                genweight = first(MC + "genweight")
                checked = (nt == 1) & np.isfinite(genweight)
                mismatch = checked & ~np.isclose(genweight, w["genie_weight"], rtol=2e-6, atol=0)
                counts["nominal_genweight_checked"] += int(checked.sum())
                counts["nominal_genweight_mismatches"] += int(mismatch.sum())
                if mismatch.any():
                    raise RuntimeError("Nominal/CAF GENIE-weight row-alignment check failed")
                counts["nominal_run_mismatches"] += int(np.count_nonzero(w["run"] != ak.to_numpy(a[HD + "run"])))
                category = truth_categories(pdg, cc)
                one = (nt == 1) & (nr == 1)
                mask = one & (category >= 0) & np.isfinite(E) & (E > 0)
                flux = np.where(np.abs(pdg) == 12, w["flux_nue"],
                                np.where(np.abs(pdg) == 14, w["flux_numu"], 0.0))
                base = w["xsec"] * flux
                if not np.all(np.isfinite(base) & (base >= 0)):
                    raise RuntimeError("Invalid nominal baseline product")
                counts["entries"] += stop-start
                counts["one_truth_one_reco"] += int(one.sum())
                counts["selected_before_zero_weight"] += int(mask.sum())
                counts["zero_weight_selected"] += int((mask & (base == 0)).sum())
                mask &= base > 0
                n, row_offset = int(mask.sum()), counts["selected"]
                counts["selected"] += n
                if n == 0:
                    continue

                # All selected rows have exactly one interaction. Check the
                # corresponding particle counts and offsets before flattening.
                part_lengths = first(PP + ".length")[mask]
                part_indices = first(PP + ".idx")[mask]
                if not np.all(part_indices == 0):
                    raise RuntimeError("Selected one-interaction particle offset is not zero")
                particle_types, particle_walls = a[PP + "origRecoObjType"][mask], a[PP + "walldist"][mask]
                if not np.array_equal(ak.to_numpy(ak.num(particle_types)), part_lengths):
                    raise RuntimeError("Particle type lengths do not close")
                if not np.array_equal(ak.to_numpy(ak.num(particle_walls)), part_lengths):
                    raise RuntimeError("Particle wall lengths do not close")
                reduced = reduce_particles(part_lengths, ak.to_numpy(ak.flatten(particle_types)),
                                           ak.to_numpy(ak.flatten(particle_walls)))
                cols = {(key if key != "baseline" else "baseline_m"): first(MC + key)[mask]
                        for key in TRUTH_SCALARS}
                cols.update({key: np.column_stack([first(prefix + axis)[mask] for axis in "xyz"])
                             for key, prefix in VECTOR_BRANCHES.items()})
                cols.update({"enu_" + key: first(RC + "Enu." + key)[mask] for key in ENERGY_FIELDS})
                cols.update({"cvn_" + key: first(RC + "nuhyp.cvn." + key)[mask] for key in CVN_FIELDS})
                cols.update({"n_particles": reduced["n_particles"], "min_wall_cm": reduced["min_wall_cm"],
                    "wall_valid": reduced["wall_valid"], "invalid_wall_count": reduced["invalid_wall_particles"],
                    "category": category[mask], "energy": E[mask], "base_weight": base[mask],
                    "source_entry": np.arange(start, stop, dtype=np.int64)[mask]})
                cols.update({key: ak.to_numpy(a[HD + key])[mask] for key in ["run", "subrun", "event"]})
                cols.update({key: w[key][mask] for key in ["xsec", "flux_nue", "flux_numu"]})
                cols["pdg"] = cols["pdg"].astype(np.int16)
                cols["iscc"] = cols["iscc"].astype(np.int8)
                for key, value in cols.items():
                    parts[key].append(value)
                counts["xsec_cvwgt_nonfinite"] += int((~np.isfinite(cols["xsec_cvwgt"])).sum())
                counts["true_baseline_nonfinite"] += int((~np.isfinite(cols["baseline_m"])).sum())
                counts["true_y_nonfinite"] += int((~np.isfinite(cols["inelasticity"])).sum())
                counts["particles_with_invalid_walls"] += int(cols["invalid_wall_count"].sum())
                counts["unknown_wall_minimum"] += int((~cols["wall_valid"]).sum())
                counts["E_above_10_GeV"] += int((cols["E"] > 10).sum())
                counts["E_above_100_GeV"] += int((cols["E"] > 100).sum())
                energy_min, energy_max = min(energy_min, cols["E"].min()), max(energy_max, cols["E"].max())

                parent_len, parent_idx = first(SP + ".length")[mask], first(SP + ".idx")[mask]
                if not np.all((parent_len == ndial) & (parent_idx == 0)):
                    raise RuntimeError("Unexpected dial-parent layout")
                lens = ak.to_numpy(ak.to_regular(a[SP + "weights..length"][mask], axis=1))
                idxs = ak.to_numpy(ak.to_regular(a[SP + "weights..idx"][mask], axis=1))
                if lens.shape != (n, ndial):
                    raise RuntimeError("Unexpected response count-array shape")
                dense = expand_responses(lens, idxs, a[SP + "weights"][mask], variation_counts)
                counts["implicit_unit_slots"] += int((lens == 0).sum())
                counts["negative_response_cells"] += int((dense < 0).sum())
                counts["nonfinite_response_cells"] += int((~np.isfinite(dense)).sum())
                for j in range(ndial):
                    response = dense[:, offsets[j]:offsets[j+1]]
                    affected = np.any(response != 1.0, axis=1)
                    if affected.any():
                        sparse[j][0].append((np.flatnonzero(affected) + row_offset).astype(np.int32))
                        sparse[j][1].append(response[affected].copy())
                del dense, a
    finally:
        if not reader.closed:
            reader.close()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError("Remote source changed during scan")
    arrays = {key: np.concatenate(value, axis=0) if value else empty_field(key)
              for key, value in parts.items()}
    for j, (rows, response) in enumerate(sparse):
        arrays[f"d{j:03d}_rows"] = np.concatenate(rows) if rows else np.empty(0, dtype=np.int32)
        arrays[f"d{j:03d}_responses"] = (np.concatenate(response, axis=0) if response
            else np.empty((0, variation_counts[j]), dtype=np.float32))
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, out)
    record = dict(schema_version=SCHEMA_VERSION, schema_name=SCHEMA_NAME,
        extractor_and_physics_sha256=hashlib.sha256(Path(__file__).read_bytes() + (HERE / "v2_physics.py").read_bytes()).hexdigest(),
        source=str(path), size=before.st_size, mtime_ns=before.st_mtime_ns,
        cache=str(out), cache_bytes=out.stat().st_size, cache_sha256=digest_file(out), counts=counts,
        energy_min_GeV=float(energy_min) if counts["selected"] else None,
        energy_max_GeV=float(energy_max) if counts["selected"] else None,
        selected_compressed_branch_bytes=estimate, logical_bytes_read=reader.n,
        read_cap_bytes=reader.cap, seconds=time.monotonic()-started,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        arrays={key: dict(dtype=str(value.dtype), shape=list(value.shape)) for key, value in arrays.items()},
        nominal_genweight_alignment_passed=True, dial_metadata_exact_match=True,
        particle_offsets_closed=True, response_offsets_closed=True)
    record.update(input_hashes or {})
    atomic_json(out.with_suffix(".json"), record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit-files", type=int)
    parser.add_argument("--chunk-size", type=int, default=25000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--read-cap-mib", type=int, default=256)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True,
                        help="JSON with directory and entries [{name: file.root}, ...]")
    parser.add_argument("--dial-metadata", type=Path,
                        default=HERE / "metadata/actual_dial_metadata.json")
    args = parser.parse_args()
    if not 1 <= args.workers <= 3 or not 1 <= args.chunk_size <= 25000:
        raise ValueError("Use 1-3 workers and chunk sizes <=25000")
    if not 1 <= args.read_cap_mib <= 256:
        raise ValueError("Per-file logical read cap must be <=256 MiB")
    if args.limit_files is not None and args.limit_files < 1:
        raise ValueError("--limit-files must be positive")
    inventory = json.loads(args.inventory.read_text())
    source_directory = inventory_source_directory(inventory, args.inventory)
    dials = json.loads(args.dial_metadata.read_text())
    files = sorted([row for row in inventory["entries"] if row["name"].endswith(".root")], key=lambda row: row["name"])
    if not files or len({row['name'] for row in files}) != len(files):
        raise ValueError("Inventory must contain distinct ROOT filenames")
    if any(Path(row['name']).name != row['name'] for row in files):
        raise ValueError("Inventory entries must be filenames relative to directory")
    if len(dials) != 101 or sum(len(d['values']) for d in dials) != 623:
        raise ValueError("This reviewed production requires 101 slots and 623 points")
    args.cache_dir = args.cache_dir.resolve()
    target = files[:args.limit_files] if args.limit_files is not None else files
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    index_path = args.cache_dir / "scan_index.json"
    code_hash = hashlib.sha256(Path(__file__).read_bytes() + (HERE / "v2_physics.py").read_bytes()).hexdigest()
    input_hashes = {"inventory_sha256": digest_file(args.inventory),
                    "dial_metadata_sha256": digest_file(args.dial_metadata)}
    if index_path.exists():
        index = json.loads(index_path.read_text())
        validate_resume_metadata(index, input_hashes, code_hash)
        if index.get("production_directory") != str(source_directory):
            raise RuntimeError("Inventory resolves to a different source directory; refusing to reuse the cache")
    else:
        index = dict(schema_version=SCHEMA_VERSION, schema_name=SCHEMA_NAME,
            extractor_and_physics_sha256=code_hash, production_directory=str(source_directory),
            **input_hashes,
            dial_count=len(dials), variation_count=sum(len(d["values"]) for d in dials),
            raw_fields=RAW_FIELDS, base_fields=BASE_FIELDS,
            selection="One truth and one Pandora interaction; five truth channels; finite E>0; positive finite xsec*matching-flavour flux. No FV, containment, reconstructed-direction, or upper-energy cut.",
            weight="xsec * matching-flavor flux; tau-generated zero flux excluded; no oscillation or additional GENIE/CV product",
            nominal_alignment="Same-file equal-length trees; per-row CAF genweight versus weights.genie_weight (rtol2e-6). weights.run is historical zero/unusable: mismatches diagnostic only.",
            response_policy="Raw responses retained exactly; omitted sparse rows mean unit response. Malformed/negative/nonfinite responses are audited, not modified here.",
            total_production_files=len(files), files=[])
    # Sidecars allow recovery after a worker completed but index publication did
    # not. Original source names and fingerprints prevent accidental row reuse.
    completed = {record["source"]: record for record in index["files"]}
    pending = []
    for i, row in enumerate(target):
        path = (source_directory / row["name"]).resolve()
        out = args.cache_dir / f"part_{i:03d}.npz"
        prior = completed.get(str(path))
        if prior is None and out.with_suffix(".json").exists():
            prior = json.loads(out.with_suffix(".json").read_text())
            if prior.get("source") != str(path) or prior.get("schema_name") != SCHEMA_NAME:
                raise RuntimeError("Sidecar source/schema does not match inventory")
        if prior is not None:
            validate_cached_record(prior, path, out, code_hash, input_hashes)
            completed[str(path)] = prior
            continue
        pending.append((i, path, out))
    index["files"] = sorted(completed.values(), key=lambda record: record["cache"])
    atomic_json(index_path, index)

    def job(item):
        i, path, out = item
        print("START", i+1, len(files), path.name, flush=True)
        return i, scan_file(path, out, dials, args.chunk_size, args.read_cap_mib, input_hashes)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(job, item): item for item in pending}
        for future in as_completed(futures):
            try:
                i, record = future.result()
            except BaseException:
                for waiting in futures:
                    waiting.cancel()
                raise
            index["files"].append(record)
            index["files"].sort(key=lambda value: value["cache"])
            atomic_json(index_path, index)
            print("DONE", json.dumps(dict(file=i+1, completed=len(index["files"]),
                entries=record["counts"]["entries"], selected=record["counts"]["selected"],
                seconds=round(record["seconds"], 2), read_MiB=round(record["logical_bytes_read"]/2**20, 2),
                cache_MiB=round(record["cache_bytes"]/2**20, 2), negative=record["counts"]["negative_response_cells"],
                nonfinite=record["counts"]["nonfinite_response_cells"])), flush=True)
    index["complete"] = len(index["files"]) == len(files)
    atomic_json(index_path, index)
    print("COMPLETE", len(index["files"]), "of", len(files), flush=True)


if __name__ == "__main__":
    main()
