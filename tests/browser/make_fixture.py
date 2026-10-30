#!/usr/bin/env python3
"""Create small, explicitly SYNTHETIC browser-test inputs, never physics output.

Run from the repository root, after installing the package and its dependencies::

    python tests/browser/make_fixture.py /tmp/response-browser-test
    python -m atmospheric_response.build_interactive_v2 \
        --manifest-list /tmp/response-browser-test/collection.json \
        --output /tmp/response-browser-test/explorer.html
    python tests/browser/check_viewer.py /tmp/response-browser-test

The deliberately small test bins are NOT the production Miles binning. They
exercise zero lower edges, a wide final bin, selection-specific overrides,
unsupported combinations, two canvases, both policy choices, and two distinct
angle-reconstruction datasets. The latter are deliberately different so a stale
page/cache cannot pass the method-toggle regression. Histograms
other than true zenith contain arbitrary test counts. The true-zenith fixture
uses seeded random samples to exercise the actual shared-event test helper.
No ROOT file, remote mount, or analysis dataset is read.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np

# Permit a checkout without an editable install; this path is repository-local.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from atmospheric_response.render_canvases_v2 import ORIGINAL_ORDER, KINEMATICS_ORDER
from atmospheric_response.v2_statistics import moments_from_events, paired_shape_flatness


def make_fixture(output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2031)
    n_events = 6000
    zenith = rng.uniform(-1, 1, n_events)
    weights = rng.uniform(.5, 1.5, n_events)
    collection = []
    variants = [('fine', 'all'), ('fine', 'fc'), ('fine', 'pc'),
                ('miles', 'fc'), ('miles', 'pc')]
    for binning, containment in variants:
        for policy in ('fitter_policy', 'raw'):
            arrays, observables, curves = {}, [], []
            for oid in ORIGINAL_ORDER + KINEMATICS_ORDER:
                edges = (np.linspace(-1, 1, 11) if 'coszenith' in oid
                         else np.geomspace(.1, 10, 11))
                if 'inelasticity' in oid:
                    edges = np.linspace(0, 1, 11)
                arrays[oid + '_edges'] = edges
                linear = 'coszenith' in oid or 'inelasticity' in oid
                observables.append(dict(
                    id=oid, title=oid, x_label=oid, y_label='UI fixture bin sum',
                    kind='profile' if oid == 'angular_resolution' else 'histogram',
                    edges_key=oid + '_edges', x_scale='linear' if linear else 'log'))
            dial = dict(
                id='test', label='SYNTHETIC UI TEST (not analysis)', units='test coordinate',
                values=[dict(id='minus', label='-1', value=-1),
                        dict(id='zero', label='0', value=0),
                        dict(id='plus', label='+1', value=1)],
                cv=dict(value_id='zero', label='Test reference (0)'), reference_is_cv=True)
            for sid in ('nue_cc', 'numu_cc', 'nc'):
                for obs in observables:
                    oid = obs['id']
                    record = dict(selection_id=sid, dial_id='test', observable_id=oid)
                    unavailable = sid == 'nc' and (
                        binning == 'miles' or oid in (
                            'true_lepton_energy', 'reco_lepton_energy', 'reco_inelasticity'))
                    if unavailable:
                        record.update(available=False,
                                      reason='Explicitly unsupported UI-test combination')
                        curves.append(record)
                        continue
                    edges = arrays[oid + '_edges']
                    if binning == 'miles' and oid == 'reco_neutrino_energy':
                        edges = np.array([0, .4, .6, 1, 100] if sid == 'numu_cc'
                                         else [0, .2, .6, .8, 1.5, 100])
                        edge_key = sid + '_' + oid + '_edges'
                        arrays[edge_key] = edges
                        record.update(edges_key=edge_key, x_scale='symlog', x_linthresh=.1)
                    n_bins = len(edges) - 1
                    values = np.stack([np.arange(1, n_bins + 1) * factor
                                       for factor in (.8, 1, 1.2)])
                    if oid == 'true_coszenith':
                        indices = np.minimum(np.searchsorted(edges, zenith, side='right') - 1,
                                             n_bins - 1)
                        values, tests = [], []
                        for factor in (1 - .15 * zenith, np.ones(n_events), 1 + .15 * zenith):
                            moments = moments_from_events(indices, weights * factor,
                                                          weights, n_bins)
                            values.append(moments['sumw_var'])
                            test = paired_shape_flatness(**moments)
                            # One inclusive energy stratum has the same test.
                            # This tests optional table/CSV fields without
                            # pretending this toy has an energy distribution.
                            test['energy_conditioned'] = copy.deepcopy(test)
                            # Inject display-only audit flags into the toy;
                            # these do not assert a physics calculation.
                            test['raw_nonfinite_pair_events'] = 7 if not tests else 0
                            test['raw_negative_pair_events'] = 3 if len(tests) == 2 else 0
                            tests.append(test)
                        values = np.asarray(values)
                        record['flatness'] = dict(
                            values=tests, conditional_energy_edges_GeV=[.1, 100],
                            conditional_coszenith_edges=edges.tolist())
                    key = sid + '_' + oid
                    arrays[key] = values
                    record['values_key'] = key
                    if 'l_over_e' in oid:
                        record['note'] = 'SYNTHETIC UI TEST geometry caption: R=6371km, h=15km, depth=0km'
                    curves.append(record)
            ident = '_'.join((binning, containment, policy))
            target = output / ident
            target.mkdir(exist_ok=True)
            np.savez_compressed(target / 'curves.npz', **arrays)
            second_dial = copy.deepcopy(dial)
            second_dial.update(id='test_second', label='SECOND SYNTHETIC UI TEST (not analysis)')
            # A nonfirst parameter catches accidental dropdown resets when a
            # reconstruction change selects another aggregate dataset.
            curves += [{**record, 'dial_id': 'test_second'} for record in list(curves)]
            document = dict(
                schema_version=2, title='SYNTHETIC UI TEST — NOT ANALYSIS DATA',
                data_file='curves.npz', weight_description='Synthetic testing values only.',
                allow_missing_histogram_bins=policy == 'raw',
                dimensions=dict(policy=policy, binning=binning, containment=containment,
                                direction='hybrid'),
                provenance=dict(geometry=dict(earth_radius_km=6371, production_height_km=15,
                                              detector_depth_km=0),
                                selection=dict(require_valid_direction=True)),
                selections=[dict(id=sid, label=sid) for sid in ('nue_cc', 'numu_cc', 'nc')],
                dials=[dial, second_dial], observables=observables, curves=curves)
            (target / 'manifest.json').write_text(json.dumps(document, allow_nan=False),
                                                encoding='utf-8')
            collection.append(dict(id=ident, manifest=ident + '/manifest.json'))
    for entry in list(collection):
        source = output / entry['manifest']
        document = json.loads(source.read_text())
        with np.load(source.parent / 'curves.npz') as original:
            arrays = {key: value.copy() for key, value in original.items()}
        # These are arbitrary synthetic display distinctions, not a proposed
        # physics transformation. Every unrelated observable stays identical.
        for sid in ('nue_cc', 'numu_cc', 'nc'):
            for oid in ('reco_coszenith', 'reco_l_over_e'):
                key = sid + '_' + oid
                if key in arrays:
                    arrays[key] = np.roll(arrays[key], 1, axis=1)
            key = sid + '_angular_resolution'
            if key in arrays:
                arrays[key] *= .75
        document['dimensions']['direction'] = 'all_particles'
        document['provenance']['direction_acceptance'] = (
            'SYNTHETIC provenance acceptance note: selected method validity can change '
            'the sample; a common-valid comparison isolates direction values.')
        for selection in document['selections']:
            selection['description'] = 'Synthetic previous-method fixture; no analysis events.'
        ident = 'all_particles_' + entry['id']
        target = output / ident
        target.mkdir(exist_ok=True)
        np.savez_compressed(target / 'curves.npz', **arrays)
        (target / 'manifest.json').write_text(json.dumps(document, allow_nan=False), encoding='utf-8')
        collection.append(dict(id=ident, manifest=ident + '/manifest.json'))
    path = output / 'collection.json'
    path.write_text(json.dumps(collection, indent=2) + '\n', encoding='utf-8')
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path, help='Directory for SYNTHETIC test files')
    print(make_fixture(parser.parse_args().output.resolve()))
