"""Validated YAML configuration and observable/binning definitions.

The YAML file owns numerical cuts and every edge. Changing binning requires
local reaggregation, not another ROOT scan. Never silently sort user edges.
"""
from __future__ import annotations
import copy
import hashlib
from pathlib import Path
import numpy as np
import yaml

OBSERVABLE_IDS = [
    'true_lepton_energy', 'reco_lepton_energy', 'true_hadronic_transfer',
    'reco_hadronic_energy', 'true_coszenith', 'reco_coszenith',
    'angular_resolution', 'true_neutrino_energy', 'reco_neutrino_energy',
    'true_inelasticity', 'reco_inelasticity', 'true_l_over_e', 'reco_l_over_e',
]
SELECTIONS = [
    {'id':'nue_cc','label':'νe CC','family':'nue'},
    {'id':'numu_cc','label':'νμ CC','family':'numu'},
    {'id':'nc','label':'NC (both signs)','family':'nc'},
    {'id':'nuebar_cc','label':'ν̄e CC','family':'nue'},
    {'id':'numubar_cc','label':'ν̄μ CC','family':'numu'},
]


def edges_from_spec(spec):
    """Accept explicit edges or a reproducible linear/logarithmic generator."""
    if isinstance(spec, (list, tuple)):
        edges = np.asarray(spec, dtype=float)
    elif isinstance(spec, dict):
        if 'edges' in spec:
            edges = np.asarray(spec['edges'], dtype=float)
        else:
            bins = spec.get('bins')
            if not isinstance(bins, int) or isinstance(bins, bool) or bins < 1:
                raise ValueError('bins must be a positive integer')
            lo, hi = float(spec['min']), float(spec['max'])
            spacing = spec.get('spacing', 'linear')
            if spacing == 'log':
                if lo <= 0:
                    raise ValueError('Logarithmic bin edges must start above zero')
                edges = np.geomspace(lo, hi, bins+1)
            elif spacing == 'linear':
                edges = np.linspace(lo, hi, bins+1)
            else:
                raise ValueError(f'Unknown bin spacing {spacing!r}')
    else:
        raise ValueError('A bin specification is an edge list or mapping')
    if (edges.ndim != 1 or len(edges) < 2 or not np.isfinite(edges).all()
            or not np.all(np.diff(edges) > 0)):
        raise ValueError('Edges must be finite, strictly increasing and one-dimensional')
    return edges


def resolved_profile(cfg, name):
    """Resolve recursive inheritance without mutating the source YAML mapping.

    Axis overrides replace a complete axis specification. Sample overrides
    replace individual fields, so a child can alter FC_nue/RecoENu and retain
    its inherited CosZ/RecoY and all other supplied samples. Missing parents
    and cycles are configuration errors, never silent fallback to fine bins.
    """
    profiles = cfg.get('binning_profiles')
    if not isinstance(profiles, dict):
        raise ValueError('Missing configuration mapping binning_profiles')

    def visit(current, ancestors):
        if not isinstance(current, str) or current not in profiles:
            raise ValueError(f'Unknown binning profile or parent {current!r}')
        if current in ancestors:
            raise ValueError('Cyclic binning inheritance: '+ ' -> '.join((*ancestors, current)))
        own = profiles[current]
        if not isinstance(own, dict):
            raise ValueError(f'{current}: profile must be a mapping')
        parent = own.get('inherits')
        inherited = {} if parent is None else visit(parent, (*ancestors, current))
        merged = {**copy.deepcopy(inherited), **copy.deepcopy(own)}
        for part in ('axes', 'samples'):
            if part in own and not isinstance(own[part], dict):
                raise ValueError(f'{current}: {part} must be a mapping')
        merged['axes'] = {**copy.deepcopy(inherited.get('axes', {})), **copy.deepcopy(own.get('axes', {}))}
        if 'samples' in inherited or 'samples' in own:
            samples = copy.deepcopy(inherited.get('samples', {}))
            for sample, fields in own.get('samples', {}).items():
                if not isinstance(fields, dict):
                    raise ValueError(f'{current}/{sample}: sample fields must be a mapping')
                samples[sample] = {**samples.get(sample, {}), **copy.deepcopy(fields)}
            merged['samples'] = samples
        return merged

    return visit(name, ())


def profile_has_samples(cfg, name):
    """Whether this profile or any ancestor declares sample-specific grids."""
    return 'samples' in resolved_profile(cfg, name)


def _finite_number(value, label):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f'{label} must be a finite number, not a Boolean')
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f'{label} must be a finite number') from error
    if not np.isfinite(number):
        raise ValueError(f'{label} must be finite')
    return number


def load_config(path):
    path = Path(path)
    text = path.read_text()
    cfg = yaml.safe_load(text)
    if not isinstance(cfg, dict) or cfg.get('schema_version') != 2:
        raise ValueError('Expected an analysis YAML with schema_version: 2')
    for key in ('selection', 'physics', 'binning_profiles', 'statistics', 'weight_policy'):
        if not isinstance(cfg.get(key), dict):
            raise ValueError(f'Missing configuration mapping {key}')
    for name in cfg['binning_profiles']:
        profile = resolved_profile(cfg, name)
        for spec in profile.get('axes', {}).values():
            edges_from_spec(spec)
        for group in profile.get('samples', {}).values():
            for spec in group.values():
                edges_from_spec(spec)
    if cfg['physics'].get('nc_direction') != 'hits':
        raise ValueError('This study requires the user-selected NC hits-only hybrid rule')
    if cfg['physics'].get('direction_method') not in ('hybrid', 'hits', 'all_particles'):
        raise ValueError('direction_method must be hybrid, hits or all_particles')
    if cfg['physics'].get('true_baseline_mode') not in ('geometric', 'stored'):
        raise ValueError('true_baseline_mode must be geometric or stored')
    statistics = cfg['statistics']
    for key in ('conditional_energy_edges', 'conditional_coszenith_edges'):
        if key not in statistics:
            raise ValueError(f'Missing statistics/{key}')
        statistics[key] = edges_from_spec(statistics[key]).tolist()
    minimum = _finite_number(statistics.get('minimum_effective_count'), 'minimum_effective_count')
    if minimum < 0:
        raise ValueError('minimum_effective_count must be nonnegative')
    statistics['minimum_effective_count'] = minimum
    policy = cfg['weight_policy']
    for key in ('nonfinite_knot_replacement', 'knot_min', 'knot_max'):
        policy[key] = _finite_number(policy.get(key), f'weight_policy/{key}')
    if policy['knot_min'] > policy['knot_max']:
        raise ValueError('weight_policy requires knot_min <= knot_max')
    cfg['_path'] = str(path.resolve())
    cfg['_sha256'] = hashlib.sha256(text.encode()).hexdigest()
    return cfg


AXIS_FOR_OBSERVABLE = {
    **{k:'particle_energy' for k in OBSERVABLE_IDS[:4]},
    'true_coszenith':'coszenith', 'reco_coszenith':'coszenith',
    'angular_resolution':'profile_energy',
    'true_neutrino_energy':'neutrino_energy', 'reco_neutrino_energy':'neutrino_energy',
    'true_inelasticity':'inelasticity', 'reco_inelasticity':'inelasticity',
    'true_l_over_e':'l_over_e', 'reco_l_over_e':'l_over_e',
}


def resolve_edges(cfg, profile_name, containment, selection, observable):
    """Return exact edges and a note describing any explicit fallback.

    Miles supplied only FC/PC electron/muon samples. Their union is not a
    sample binning. Unsupported ALL/NC combinations remain unavailable.
    Unspecified observables (lepton/hadron and R68) inherit the fine axes.
    """
    profile = resolved_profile(cfg, profile_name)
    axis = AXIS_FOR_OBSERVABLE[observable]
    specs = profile.get('axes', {})
    if 'samples' in profile:
        key = f'{containment.upper()}_{selection["family"]}'
        if key not in profile['samples']:
            return None, f'{profile_name} has no supplied {key} binning; select fine or provide a YAML sample override.'
        sample = profile['samples'][key]
        field = {'coszenith':'CosZ', 'neutrino_energy':'RecoENu',
                 'inelasticity':'RecoY'}.get(axis)
        if field in sample:
            return edges_from_spec(sample[field]), (
                f'Exact configured {key}/{field} sample edges; applied to both true/reco counterparts.'
            )
    if axis not in specs:
        raise ValueError(f'{profile_name}: no edges for {observable}/{axis}')
    return edges_from_spec(specs[axis]), ('Configured axis' if 'samples' not in profile
        else 'Configured axis fallback for an observable without sample-specific edges')
