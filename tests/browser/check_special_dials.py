#!/usr/bin/env python3
"""Check difficult real dial grids and central values in a built explorer.

Use the same --viewer, --collection and --output arguments as
check_aggregate_viewer.py. The default IDs cover a two-point grid, an unstored
central knot, a physical CV of 1, and two Woods–Saxon physical-coordinate CVs.
Only aggregate JSON/NPZ files and the offline HTML are read.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from playwright.sync_api import sync_playwright


def check(viewer: Path, collection: Path, output: Path, dial_ids: list[str]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    entries = json.loads(collection.read_text())
    if isinstance(entries, dict):
        entries = entries['datasets']
    variants = {}
    for entry in entries:
        path = Path(entry['manifest'])
        if not path.is_absolute():
            path = collection.parent / path
        doc = json.loads(path.read_text())
        if doc['dimensions']['binning'] == 'fine' and doc['dimensions']['containment'] == 'all':
            variants[doc['dimensions']['policy']] = (path, doc)
    reports, errors, requests = [], [], []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True,
                                             args=['--no-sandbox', '--disable-dev-shm-usage'])
        page = browser.new_page(viewport={'width': 1600, 'height': 1100})
        page.set_default_timeout(60_000)
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.on('request', lambda request: requests.append(request.url))
        page.goto(viewer.as_uri())
        page.wait_for_function('window.__viewerReady===true')

        def change(key, value):
            if page.locator('#' + key).input_value() != value:
                page.select_option('#' + key, value)
                page.wait_for_function('window.__viewerReady===true')

        for policy in ('fitter_policy', 'raw'):
            path, doc = variants[policy]
            change('binning', 'fine')
            change('containment', 'all')
            change('policy', policy)
            change('selection', 'nue_cc')
            change('canvas', 'original')
            with np.load(path.parent / doc['data_file']) as arrays:
                for did in dial_ids:
                    dial = next(d for d in doc['dials'] if d['id'] == did)
                    change('dial', did)
                    state = page.evaluate('window.__viewerState')
                    assert state['dial'] == did and state['stored_points'] == len(dial['values'])
                    reference_index = next((i for i, value in enumerate(dial['values'])
                                            if value['id'] == dial['cv']['value_id']), None)
                    reference_value = (dial['cv']['value'] if reference_index is None
                                       else dial['values'][reference_index]['value'])
                    assert page.evaluate('referenceValue(state.dial)') == reference_value
                    record = next(c for c in doc['curves'] if
                                  (c['selection_id'], c['dial_id'], c['observable_id']) ==
                                  ('nue_cc', did, 'true_coszenith'))
                    values = arrays[record['values_key']]
                    expected = (arrays[record['cv_key']] if reference_index is None
                                else values[reference_index])
                    rendered = page.locator('#plot-4').evaluate(
                        "p=>p.data.find(t=>t.line.color==='#000000'&&t.yaxis==='y').y")
                    rendered = np.array([np.nan if x is None else x for x in rendered])
                    np.testing.assert_allclose(rendered, np.repeat(expected, 2), equal_nan=True)
                    styles = page.evaluate('state.dial.styles')
                    paired = {}
                    for point, style in zip(dial['values'], styles):
                        offset = point['value'] - reference_value
                        if offset:
                            magnitude = round(abs(offset), 10)
                            paired.setdefault(magnitude, []).append(style['color'])
                            assert style['dash'] == ('dash' if offset < 0 else 'dot')
                    assert all(len(set(colors)) == 1 for colors in paired.values())
                    statuses = page.evaluate('state.page.true_coszenith.flatness.values.map(t=>t.status)')
                    flags = page.evaluate('state.page.true_coszenith.flatness.values.map(t=>'
                                          '[t.raw_nonfinite_pair_events,t.raw_negative_pair_events])')
                    if any(nonfinite or negative for nonfinite, negative in flags):
                        assert 'max per stored point' in page.locator('#notes').inner_text()
                        assert ('not physical insensitivity' if policy == 'fitter_policy'
                                else 'probability quantiles undefined') in page.locator('#notes').inner_text()
                    reports.append(dict(policy=policy, dial=did, reference_value=reference_value,
                                        stored_points=state['stored_points'], shape_test_statuses=statuses,
                                        raw_pair_flags=flags))
                    if did in ('d062', 'd091'):
                        page.screenshot(path=str(output / f'{policy}_{did}.png'), full_page=True)
                    warnings = page.evaluate("Array.from(document.querySelectorAll('.chart')).flatMap(p=>"
                                             "p.data?(Plotly.validate(p.data,p.layout)||[]):[])")
                    assert not warnings, warnings
                    print(f'Checked {policy} / {did}: actual CV {reference_value}', flush=True)
        # Search must retain the full parameter inventory when cleared.
        page.fill('#search', 'WSReweight')
        page.wait_for_function("document.querySelector('#dial').options.length===2")
        page.wait_for_function('window.__viewerReady===true')
        page.fill('#search', '')
        page.wait_for_function('(n)=>document.querySelector("#dial").options.length===n',
                               arg=len(doc['dials']))
        page.wait_for_function('window.__viewerReady===true')
        external = [url for url in requests if url.startswith(('http:', 'https:'))]
        assert not errors and not external
        (output / 'special_dials_validation.json').write_text(json.dumps(
            dict(checks=reports, errors=errors, external_requests=external,
                 limitation='Presentation and reference/grid validation, not a full physics audit'),
            indent=2) + '\n')
        browser.close()
    print('PASS: special dial grids, numerical policies, physical CVs and search')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--viewer', required=True, type=Path)
    parser.add_argument('--collection', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--dials', nargs='+', default=['d061', 'd062', 'd078', 'd091', 'd092'])
    args = parser.parse_args()
    check(args.viewer.resolve(), args.collection.resolve(), args.output.resolve(), args.dials)
