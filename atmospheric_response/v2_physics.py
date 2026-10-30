"""Pure, chunk-friendly atmospheric observables; no file or network I/O.

The five categories remain *truth* channels, not CVN fitter selections.  The
estimator conventions are pinned in docs/source-audit.txt in the repository.
All energies are GeV, particle wall distances cm, and baselines km after the
explicit conversion from the CAF's stored metres.  Invalid observables have
their own masks: a failed direction must not remove a valid true-energy event.
"""
from __future__ import annotations

from collections.abc import Mapping
import numpy as np

HIT_COLLECTION = 3
METHOD_NAMES = ("heshw", "lngtrk", "calo")
OBSERVABLE_IDS = [
    "true_lepton_energy", "reco_lepton_energy", "true_hadronic_transfer",
    "reco_hadronic_energy", "true_coszenith", "reco_coszenith",
    "angular_resolution", "true_neutrino_energy", "reco_neutrino_energy",
    "true_inelasticity", "reco_inelasticity", "true_l_over_e", "reco_l_over_e",
]
DEFAULT_CONFIG = {
    "direction_method": "hybrid",
    "hybrid_switch_energy_gev": 1.3,
    "containment_threshold_cm": 10.0,
    "containment_max_valid_cm": 10000.0,
    "energy_estimator": "mach3",
    "true_baseline_mode": "stored",
    "geometry": {"earth_radius_km": 6371.0, "production_height_km": 15.0,
                 "detector_depth_km": 0.0},
}


def reduce_particles(lengths, orig_type_flat, wall_flat):
    """Reduce one interaction per event from concatenated particle arrays.

    ``lengths`` includes the unassociated-hit object; exclude only type 3 from
    both the particle count and wall minimum, as in Pierre's supplied formula.
    Match the fitter's loop initialized to 1e8: NaN/+inf do not update a finite
    minimum; -inf and negative distances do update it and invalidate the event.
    An event with no usable minimum is unknown.  Count nonfinite/negative walls
    separately for auditing: missing walls on one particle do not discard
    a valid minimum measured on another particle.
    """
    lengths = np.asarray(lengths)
    types = np.asarray(orig_type_flat)
    walls = np.asarray(wall_flat, dtype=np.float64)
    if lengths.ndim != 1 or np.any(~np.isfinite(lengths)) or np.any(lengths < 0):
        raise ValueError("Particle lengths must be a finite nonnegative vector")
    if np.any(lengths != np.floor(lengths)):
        raise ValueError("Particle lengths must be integers")
    lengths = lengths.astype(np.int64, copy=False)
    if types.ndim != 1 or walls.ndim != 1 or types.size != walls.size:
        raise ValueError("Particle types and walls must be aligned flat arrays")
    if int(lengths.sum()) != types.size:
        raise ValueError("Sum of particle lengths does not match flat arrays")
    n = lengths.size
    rows = np.repeat(np.arange(n), lengths)
    selected = types != HIT_COLLECTION
    counts = np.bincount(rows[selected], minlength=n).astype(np.int32)
    finite = np.isfinite(walls)
    min_wall = np.full(n, 1e8)
    # Comparison with NaN is false, exactly as in the source's if(wall<min).
    updates = selected & ~np.isnan(walls) & (walls < 1e8)
    np.minimum.at(min_wall, rows[updates], walls[updates])
    bad = selected & (~finite | (walls < 0))
    bad_counts = np.bincount(rows[bad], minlength=n).astype(np.int32)
    wall_valid = (counts > 0) & np.isfinite(min_wall) & (min_wall >= 0) & (min_wall <= 10000)
    return {"n_particles": counts, "min_wall_cm": min_wall,
            "wall_valid": wall_valid, "invalid_wall_particles": bad_counts}


def classify_containment(min_wall_cm, wall_valid=None, threshold_cm=10.0,
                         max_valid_cm=10000.0):
    """Return -1 unknown, 0 partially contained, 1 fully contained.

    FC is strictly greater than 10 cm by default: equality belongs to PC.
    The 10000 cm bound reproduces the fitter's guard against absent minima.
    """
    if not (np.isfinite(threshold_cm) and 0 <= threshold_cm < max_valid_cm):
        raise ValueError("Containment threshold must lie in [0, max_valid_cm)")
    wall = np.asarray(min_wall_cm, dtype=np.float64)
    valid = np.isfinite(wall) & (wall >= 0) & (wall <= max_valid_cm)
    if wall_valid is not None:
        provided = np.asarray(wall_valid, dtype=bool)
        if provided.shape != wall.shape:
            raise ValueError("wall_valid and min_wall_cm have different shapes")
        valid &= provided
    return np.where(valid, (wall > threshold_cm).astype(np.int8), -1).astype(np.int8)


def unit_direction(vector):
    """Normalize finite nonzero vectors, including valid horizontal directions.

    CAF directions normally have unit length.  No arbitrary closeness-to-one
    requirement is imposed: Pierre explicitly normalizes all three components.
    This follows the supplied formula without rejecting a horizontal y=0.
    """
    vector = np.asarray(vector, dtype=np.float64)
    if vector.ndim != 2 or vector.shape[1] != 3:
        raise ValueError("A direction must have shape (events, 3)")
    norm = np.linalg.norm(vector, axis=1)
    valid = np.all(np.isfinite(vector), axis=1) & np.isfinite(norm) & (norm > 0)
    unit = np.full_like(vector, np.nan)
    np.divide(vector, norm[:, None], out=unit, where=valid[:, None])
    return unit, valid


def truth_categories(pdg, iscc):
    """0 nueCC, 1 numuCC, 2 all NC, 3 nuebarCC, 4 numubarCC; -1 other."""
    pdg, iscc = np.asarray(pdg), np.asarray(iscc)
    if pdg.shape != iscc.shape:
        raise ValueError("pdg/iscc arrays are not aligned")
    cat = np.full(pdg.shape, -1, dtype=np.int8)
    cat[(iscc == 0) & np.isin(np.abs(pdg), [12, 14, 16])] = 2
    for code, flavour in enumerate([12, 14, None, -12, -14]):
        if flavour is not None:
            cat[(iscc == 1) & (pdg == flavour)] = code
    return cat


def choose_direction(category, n_particles, enu_calo, reco_heshw, reco_lngtrk,
                     reco_calo, method="hybrid", switch_energy_gev=1.3):
    """Select and normalize directions, returning unit vectors, mask, method.

    Hybrid uses hits for NC, fewer than two non-hit particles, or !(Ecalo<1.3).
    Thus NaN energy and equality at 1.3 both use hits.  Otherwise truth e-like
    channels use heshw and truth mu-like channels lngtrk.  The all_particles
    comparison deliberately keeps the previous heshw convention for NC.
    """
    category = np.asarray(category)
    mu = np.isin(category, [1, 4])
    selected = np.where(mu, 1, 0).astype(np.int8)
    if method == "hybrid":
        hits = ((category == 2) | (np.asarray(n_particles) < 2)
                | ~(np.asarray(enu_calo) < switch_energy_gev))
        selected[hits] = 2
    elif method == "hits":
        selected[:] = 2
    elif method != "all_particles":
        raise ValueError(f"Unknown direction_method: {method}")
    vectors = np.stack([reco_heshw, reco_lngtrk, reco_calo], axis=1)
    chosen = vectors[np.arange(category.size), selected]
    unit, valid = unit_direction(chosen)
    valid &= category >= 0
    selected[category < 0] = -1
    return unit, valid, selected


def geometric_baseline_km(coszenith, earth_radius_km=6371.0,
                          production_height_km=15.0, detector_depth_km=0.0):
    """Straight path from a fixed-height spherical production shell.

    cosZ=+1 is down-going.  L=sqrt(r^2*c^2+(R+h)^2-r^2)-r*c,
    r=R-depth.  The down-going expression is rationalized for stability.
    This is a configurable geometric proxy, not production-height averaging
    or a recovered event production vertex.  All parameters and output are km.
    """
    R, h, depth = map(float, (earth_radius_km, production_height_km,
                             detector_depth_km))
    if not (np.isfinite([R, h, depth]).all() and R > 0 and h >= 0
            and 0 <= depth < R):
        raise ValueError("Require finite R>0, height>=0 and 0<=depth<R")
    c = np.asarray(coszenith, dtype=np.float64)
    valid = np.isfinite(c) & (np.abs(c) <= 1 + 1e-12)
    clipped = np.clip(c, -1, 1)
    r = R - depth
    shell_delta = (h + depth) * (2 * R + h - depth)
    term = np.sqrt(r * r * clipped * clipped + shell_delta)
    result = term - r * clipped
    down = valid & (clipped >= 0) & ((term + r * clipped) > 0)
    np.divide(shell_delta, term + r * clipped, out=result, where=down)
    return np.where(valid, result, np.nan)


def derive_observables(raw: Mapping, config: Mapping | None = None):
    """Build aligned values/masks from one-MC, one-Pandora interaction rows.

    The caller applies its fiducial, true-energy and positive nominal-weight
    selection.  This function never filters rows and never weights events.
    Required raw scalars: E,q0,inelasticity,baseline_m,pdg,iscc,enu_calo,
    enu_e_calo,enu_e_had,enu_lep_calo,enu_mu_range,enu_mu_had,n_particles,
    min_wall_cm.  Four (N,3) arrays: true_momentum,reco_heshw,reco_lngtrk,
    reco_calo.  Optional wall_valid carries particle-minimum validity.

    energy_estimator='mach3' follows checked Pierre/Dan/Miles branches: e-like
    e_calo/e_had; mu-like lep_calo/e_had.  'matched_range' uses mu_range/mu_had
    and the earlier conservative had>0,total>=had decomposition mask.  Explicit
    energy_branches={'electron':{'total':key,'had':key},'muon':{...}} overrides
    either pairing.  NC uses Enu.calo as a visible-energy proxy; no reco y.

    true_baseline_mode='stored' uses CAF metres/1000 and leaves missing L/E
    unavailable.  'geometric' explicitly uses true-angle fixed-height geometry;
    its output must be labelled as a proxy, not an actual stored true baseline.
    """
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(config or {})
    geometry = dict(DEFAULT_CONFIG["geometry"])
    geometry.update(cfg.get("geometry", {}))
    n = len(np.asarray(raw["E"]))

    def scalar(key):
        value = np.asarray(raw[key], dtype=np.float64)
        if value.shape != (n,):
            raise ValueError(f"{key} must have shape ({n},), got {value.shape}")
        return value

    def vector(key):
        value = np.asarray(raw[key], dtype=np.float64)
        if value.shape != (n, 3):
            raise ValueError(f"{key} must have shape ({n},3), got {value.shape}")
        return value

    E, q0 = scalar("E"), scalar("q0")
    category = truth_categories(scalar("pdg"), scalar("iscc"))
    common = category >= 0
    mu, cc, nc = np.isin(category, [1, 4]), np.isin(category, [0, 1, 3, 4]), category == 2
    n_particles, min_wall = scalar("n_particles"), scalar("min_wall_cm")
    wall_valid = np.asarray(raw.get("wall_valid", np.isfinite(min_wall)), dtype=bool)
    contained = classify_containment(min_wall, wall_valid,
        cfg["containment_threshold_cm"], cfg["containment_max_valid_cm"])
    true_unit, true_valid = unit_direction(vector("true_momentum"))
    reco_unit, reco_valid, method = choose_direction(category, n_particles,
        scalar("enu_calo"), vector("reco_heshw"), vector("reco_lngtrk"),
        vector("reco_calo"), cfg["direction_method"], cfg["hybrid_switch_energy_gev"])
    true_cos, reco_cos = -true_unit[:, 1], -reco_unit[:, 1]
    residual = np.rad2deg(np.arccos(np.clip(reco_unit[:, 1], -1, 1))
                         - np.arccos(np.clip(true_unit[:, 1], -1, 1)))

    estimator = cfg["energy_estimator"]
    if estimator not in ("mach3", "matched_range"):
        raise ValueError(f"Unknown energy_estimator: {estimator}")
    branches = {"electron": {"total": "enu_e_calo", "had": "enu_e_had"},
                "muon": {"total": "enu_lep_calo", "had": "enu_e_had"}}
    if estimator == "matched_range":
        branches["muon"] = {"total": "enu_mu_range", "had": "enu_mu_had"}
    for key, values in cfg.get("energy_branches", {}).items():
        branches[key].update(values)
    total = np.where(mu, scalar(branches["muon"]["total"]),
                     scalar(branches["electron"]["total"]))
    had = np.where(mu, scalar(branches["muon"]["had"]),
                   scalar(branches["electron"]["had"]))
    decomposition = np.isfinite(total) & np.isfinite(had) & np.isfinite(total - had)
    if estimator == "matched_range":
        decomposition &= (had > 0) & (total >= had)
    reco_energy = np.where(nc, scalar("enu_calo"), total)
    reco_had = np.where(nc, scalar("enu_calo"), had)
    reco_y = np.full(n, np.nan)
    # Exactly the fitter's finite nonpositive-total convention in mach3 mode.
    # Out-of-range finite y is retained for flow audits, never clipped to [0,1].
    if estimator == "mach3":
        reco_y[cc & decomposition & (total <= 0)] = 0.0
    np.divide(had, total, out=reco_y, where=cc & decomposition & (total > 0))
    q0_over_E = np.full(n, np.nan)
    np.divide(q0, E, out=q0_over_E, where=np.isfinite(E) & (E > 0))

    stored_baseline = scalar("baseline_m") / 1000.0
    geometric_true = geometric_baseline_km(true_cos, **geometry)
    geometric_reco = geometric_baseline_km(reco_cos, **geometry)
    baseline_mode = cfg["true_baseline_mode"]
    if baseline_mode not in ("stored", "geometric"):
        raise ValueError(f"Unknown true_baseline_mode: {baseline_mode}")
    true_baseline = stored_baseline if baseline_mode == "stored" else geometric_true
    true_le, reco_le = np.full(n, np.nan), np.full(n, np.nan)
    np.divide(true_baseline, E, out=true_le,
              where=np.isfinite(true_baseline) & (true_baseline > 0)
              & np.isfinite(E) & (E > 0))
    np.divide(geometric_reco, reco_energy, out=reco_le,
              where=(geometric_reco > 0) & np.isfinite(reco_energy) & (reco_energy > 0))

    values = {
        "true_lepton_energy": E - q0, "reco_lepton_energy": total - had,
        "true_hadronic_transfer": q0, "reco_hadronic_energy": reco_had,
        "true_coszenith": true_cos, "reco_coszenith": reco_cos,
        "angular_resolution": residual, "true_neutrino_energy": E,
        "reco_neutrino_energy": reco_energy, "true_inelasticity": scalar("inelasticity"),
        "reco_inelasticity": reco_y, "true_l_over_e": true_le,
        "reco_l_over_e": reco_le,
    }
    masks = {key: common & np.isfinite(value) for key, value in values.items()}
    masks["true_lepton_energy"] &= cc
    masks["reco_lepton_energy"] &= cc & decomposition
    masks["reco_hadronic_energy"] &= nc | (cc & decomposition)
    masks["true_coszenith"] &= true_valid
    masks["reco_coszenith"] &= reco_valid
    masks["angular_resolution"] &= true_valid & reco_valid & np.isfinite(E) & (E > 0)
    masks["reco_inelasticity"] &= cc & decomposition
    masks["reco_l_over_e"] &= reco_valid
    return {
        "category": category, "containment": contained, "n_particles": n_particles,
        "min_wall_cm": min_wall, "wall_valid": wall_valid, "chosen_method": method,
        "values": values, "masks": masks, "common": common,
        "observable_values": np.column_stack([values[k] for k in OBSERVABLE_IDS]),
        "observable_mask": np.column_stack([masks[k] for k in OBSERVABLE_IDS]),
        "signed_residual_deg": residual, "chosen_total": reco_energy,
        "chosen_had": reco_had, "energy_proxy": decomposition,
        "true_y_q0_over_E": q0_over_E,
        "true_y_minus_q0_over_E": values["true_inelasticity"] - q0_over_E,
        "true_baseline_stored_km": stored_baseline,
        "true_baseline_geometric_km": geometric_true,
        "reco_baseline_geometric_km": geometric_reco,
        "metadata": {"direction_method": cfg["direction_method"],
                     "energy_estimator": estimator, "energy_branches": branches,
                     "true_baseline_mode": baseline_mode, "geometry": geometry,
                     "categories_are_truth_channels": True,
                     "nc_reco_energy_is_visible_proxy": True},
    }
