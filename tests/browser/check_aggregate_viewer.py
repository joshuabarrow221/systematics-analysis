#!/usr/bin/env python3
"""Check a built real-data viewer against its aggregate files, never ROOT.

Usage::

    python tests/browser/check_aggregate_viewer.py \
        --viewer outputs/explorer.html --collection outputs/manifest_list.json \
        --output /tmp/response-browser-validation

This complements the synthetic UI regression test. It verifies the actual
default policy, both canvases, every supplied dimension combination, all truth
selections, NC exclusions, per-curve bin edges, and exported true-zenith ratios.
It does not independently validate the upstream event selection or weights.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import numpy as np
from playwright.sync_api import sync_playwright


def check(viewer: Path, collection: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    entries = json.loads(collection.read_text())
    if isinstance(entries, dict):
        entries = entries['datasets']
    manifests = {}
    for entry in entries:
        path = Path(entry['manifest'])
        if not path.is_absolute():
            path = collection.parent / path
        manifests[entry['id']] = (path, json.loads(path.read_text()))
    errors, requests, states = [], [], []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=True, args=['--no-sandbox', '--disable-dev-shm-usage'])
        page = browser.new_page(viewport={'width': 1600, 'height': 1100})
        page.set_default_timeout(60_000)
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.on('request', lambda request: requests.append(request.url))
        page.goto(viewer.as_uri())
        page.wait_for_function('window.__viewerReady===true')

        def change(key, value):
            if page.locator('#' + key).input_value() == value:
                return
            page.select_option('#' + key, value)
            page.wait_for_function('window.__viewerReady===true')

        def inspect(label):
            state = page.evaluate('window.__viewerState')
            states.append(dict(label=label, state=state))
            warnings = page.evaluate("Array.from(document.querySelectorAll('.chart')).flatMap(p=>"
                                     "p.data?(Plotly.validate(p.data,p.layout)||[]):[])")
            assert not warnings, warnings
            assert page.evaluate('window.__viewerError') is None
            return state

        default = inspect('default')
        assert default['dimensions']['policy'] == 'fitter_policy'
        assert default['canvas'] == 'original' and default['panels'] == 7
        assert 'Shape /' in page.locator('#plot-4').evaluate('p=>p.layout.yaxis3.title.text')
        path, doc = manifests[default['policy']]
        did, sid = default['dial'], default['selection']
        dial = next(d for d in doc['dials'] if d['id'] == did)
        record = next(c for c in doc['curves'] if
                      (c['selection_id'], c['dial_id'], c['observable_id']) ==
                      (sid, did, 'true_coszenith'))
        obs = next(o for o in doc['observables'] if o['id'] == 'true_coszenith')
        with np.load(path.parent / doc['data_file']) as arrays:
            values = arrays[record['values_key']]
            edges = arrays[record.get('edges_key', obs['edges_key'])]
            cv_index = next((i for i, value in enumerate(dial['values'])
                             if value['id'] == dial['cv']['value_id']), None)
            reference = values[cv_index] if cv_index is not None else arrays[record['cv_key']]
        with page.expect_download() as download:
            page.click('#download')
        download.value.save_as(output / 'original.csv')
        by_label = {value['label']: i for i, value in enumerate(dial['values'])}
        for row in csv.DictReader((output / 'original.csv').open()):
            if row['observable'] != 'true_coszenith':
                continue
            varied = reference if row['is_reference'] == 'true' else values[by_label[row['point_label']]]
            index = int(np.argmin(np.abs(edges[:-1] - float(row['bin_low']))))
            if reference[index] != 0 and np.isfinite(varied[index]):
                np.testing.assert_allclose(float(row['rate_ratio']), varied[index] / reference[index], rtol=1e-13)
                if varied.sum() > 0 and reference.sum() > 0:
                    np.testing.assert_allclose(float(row['shape_ratio']),
                        (varied[index] / varied.sum()) / (reference[index] / reference.sum()), rtol=1e-13)
        page.locator('summary').filter(has_text='True-zenith shape tests').click()
        assert 'Energy-stratified p' in page.locator('#notes').inner_text()
        page.screenshot(path=str(output / 'real_original.png'), full_page=True)
        change('canvas', 'kinematics')
        assert inspect('kinematics')['panels'] == 6
        headings = page.locator('#grid h2').all_text_contents()
        assert any('True generator y' in text for text in headings)
        assert sum('proxy' in text.lower() for text in headings) >= 2
        geometry = doc.get('provenance', {}).get('geometry', {})
        if 'production_height_km' in geometry:
            height = geometry['production_height_km']
            assert f'fixed production height {height:g} km' in page.locator('#notes').inner_text()
            for observable in ('true_l_over_e', 'reco_l_over_e'):
                assert f'h={height:g}km' in page.evaluate(
                    '(oid)=>state.page[oid].note', observable)
        page.screenshot(path=str(output / 'real_kinematics.png'), full_page=True)

        # Inspect each combination and all five truth selections. Data are
        # independently compared to NPZ edges, including the 0 and 100 endpoints.
        for dataset_id, (path, doc) in manifests.items():
            for key in ('binning', 'containment', 'direction', 'policy'):
                change(key, doc['dimensions'][key])
            assert page.evaluate('window.__viewerState.policy') == dataset_id
            for selection in doc['selections']:
                sid = selection['id']
                change('selection', sid)
                inspect(dataset_id + '/' + sid)
                record = next(c for c in doc['curves'] if
                              (c['selection_id'], c['dial_id'], c['observable_id']) ==
                              (sid, did, 'reco_neutrino_energy'))
                if record.get('available', True):
                    obs = next(o for o in doc['observables'] if o['id'] == 'reco_neutrino_energy')
                    with np.load(path.parent / doc['data_file']) as arrays:
                        expected = arrays[record.get('edges_key', obs['edges_key'])]
                    np.testing.assert_array_equal(
                        page.evaluate('state.page.reco_neutrino_energy.edges'), expected)
                    if expected[0] == 0:
                        assert page.evaluate('state.page.reco_neutrino_energy.x_scale') == 'symlog'
                        assert page.locator('#plot-1').evaluate('p=>p.layout.xaxis2.ticktext[0]') == '0'
                else:
                    assert record['reason'] in page.locator('#plot-1').inner_text()
                if sid == 'nc' and doc['dimensions']['binning'] == 'fine':
                    assert not page.evaluate('state.page.reco_inelasticity.available')
            if doc['dimensions']['binning'] == 'miles':
                assert 'all' not in page.locator('#containment').evaluate(
                    's=>Array.from(s.options).map(o=>o.value)')
            print(f'Checked {dataset_id}: {len(doc["selections"])} truth selections', flush=True)
        # Capture a useful coarse-binning panel with data, not unsupported NC.
        change('policy', 'fitter_policy')
        change('selection', 'numu_cc')
        page.screenshot(path=str(output / 'real_miles_pc_muon.png'), full_page=True)
        # A multi-method collection must retain its other controls while the
        # angular datasets change. Compare actual angular arrays, not just the
        # selector label, to catch a stale decoded page after asynchronous loads.
        method_checks = []
        methods = page.locator('#direction').evaluate('s=>Array.from(s.options).map(o=>o.value)')
        if len(methods) > 1:
            change('canvas', 'original')
            available_dials = page.locator('#dial').evaluate('s=>Array.from(s.options).map(o=>o.value)')
            selected_dial = 'd031' if 'd031' in available_dials else available_dials[-1]
            change('dial', selected_dial)
            change('mode', 'shape')
            before = page.evaluate('window.__viewerState')
            for method in methods + [before['dimensions']['direction']]:
                change('direction', method)
                current = inspect('direction_toggle/' + method)
                assert current['dimensions']['direction'] == method
                for key in ('binning', 'containment', 'policy'):
                    assert current['dimensions'][key] == before['dimensions'][key], key
                for key in ('selection', 'dial', 'mode', 'canvas'):
                    assert current[key] == before[key], key
                source, method_doc = manifests[current['policy']]
                assert page.locator('#direction-acceptance').is_visible()
                supplied_note = method_doc.get('provenance', {}).get('direction_acceptance')
                shown_note = page.locator('#direction-acceptance').inner_text()
                if supplied_note:
                    assert shown_note == supplied_note
                else:
                    assert 'event counts and truth spectra can therefore differ' in shown_note
                selection_description = next(s['description'] for s in method_doc['selections']
                    if s['id'] == current['selection'])
                assert selection_description in page.locator('#notes').inner_text()
                with np.load(source.parent / method_doc['data_file']) as arrays:
                    for oid in ('reco_coszenith', 'angular_resolution', 'reco_l_over_e'):
                        curve = next(c for c in method_doc['curves'] if
                            (c['selection_id'], c['dial_id'], c['observable_id']) ==
                            (current['selection'], current['dial'], oid))
                        if not curve.get('available', True):
                            assert not page.evaluate('(oid)=>state.page[oid].available', oid)
                            continue
                        expected = arrays[curve['values_key']]
                        actual = page.evaluate('(oid)=>state.page[oid].values', oid)
                        np.testing.assert_allclose(np.asarray(actual, dtype=float), expected,
                                                   rtol=0, atol=0, equal_nan=True)
                method_checks.append(dict(dataset=current['policy'], direction=method,
                                          angular_arrays_match_npz=True,
                                          acceptance_note=shown_note,
                                          selection_description=selection_description))
            last_state = page.evaluate('window.__viewerState')
            address = parse_qs(urlsplit(page.url).fragment)
            assert address['direction'] == [last_state['dimensions']['direction']]
            page.reload()
            page.wait_for_function('window.__viewerReady===true')
            assert page.evaluate('window.__viewerState') == last_state
            with page.expect_download() as download:
                page.click('#download')
            download.value.save_as(output / 'direction_toggle.csv')
            method_rows = list(csv.DictReader((output / 'direction_toggle.csv').open()))
            assert method_rows and all(row['direction'] == last_state['dimensions']['direction']
                                       and row['dial'] == selected_dial for row in method_rows)
            page.screenshot(path=str(output / 'real_direction_toggle.png'), full_page=True)
        cache = page.evaluate('({compressed:loadedPayloads.size,decoded:decoded.size})')
        assert cache['compressed'] <= 1 and cache['decoded'] <= 3
        external = [url for url in requests if url.startswith(('http:', 'https:'))]
        assert not errors, errors
        assert not external, external
        report = dict(viewer=str(viewer), collection=str(collection), errors=errors,
                      external_requests=external, cache=cache, states=states,
                      dataset_count=len(manifests), observable_count=len(doc['observables']),
                      dials_per_dataset=len(doc['dials']), csv_ratios_match_npz=True,
                      direction_toggle_checks=method_checks,
                      geometry=geometry,
                      limitation='Verifies aggregate presentation, not upstream physics or full-sample completeness')
        (output / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
        browser.close()
    print(f'PASS: real aggregate viewer, {len(manifests)} variants, all selections, CSV and NPZ edges')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--viewer', required=True, type=Path)
    parser.add_argument('--collection', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    check(args.viewer.resolve(), args.collection.resolve(), args.output.resolve())
