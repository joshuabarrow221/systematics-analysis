#!/usr/bin/env python3
"""Browser regression check for the explicitly SYNTHETIC make_fixture.py output.

Requires the optional Playwright Python package and its Chromium installation::

    python -m pip install playwright
    python -m playwright install chromium
    python tests/browser/check_viewer.py /tmp/response-browser-test

Build explorer.html with the command in make_fixture.py first. This test opens
the local HTML by file://, records errors and network requests, checks plotted
ratios against NumPy, and saves screenshots, a CSV, a print PDF, and a JSON
validation report inside the fixture directory. No production data are read.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from matplotlib.scale import SymmetricalLogTransform
from playwright.sync_api import sync_playwright


def check_viewer(root: Path) -> None:
    errors, requests, checks = [], [], []
    # Ensure an accidental production directory is never treated as this toy.
    manifest = json.loads((root / 'fine_all_fitter_policy/manifest.json').read_text())
    assert manifest['title'] == 'SYNTHETIC UI TEST — NOT ANALYSIS DATA'
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=True, args=['--no-sandbox', '--disable-dev-shm-usage'])
        page = browser.new_page(viewport={'width': 1600, 'height': 1100})
        page.set_default_timeout(60_000)
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.on('request', lambda request: requests.append(request.url))
        page.goto((root / 'explorer.html').as_uri())
        page.wait_for_function('window.__viewerReady===true')

        def change(key, value):
            page.select_option('#' + key, value)
            page.wait_for_function('window.__viewerReady===true')
            page.wait_for_timeout(100)

        def snapshot(label):
            state = page.evaluate('window.__viewerState')
            checks.append(dict(label=label, state=state))
            return state

        assert snapshot('default')['dimensions'] == dict(
            policy='fitter_policy', binning='fine', containment='all', direction='hybrid')
        assert page.locator('#plot-4').evaluate(
            "p=>Object.keys(p.layout).filter(x=>x.startsWith('yaxis'))"
        ) == ['yaxis', 'yaxis2', 'yaxis3']
        page.locator('summary').filter(has_text='True-zenith shape tests').click()
        assert 'Asymptotic test' in page.locator('#notes').inner_text()
        assert 'Energy-stratified p' in page.locator('#notes').inner_text()
        assert 'max per stored point is 7 nonfinite-response and 3 negative raw-weight events' in page.locator('#notes').inner_text()
        assert 'not physical insensitivity' in page.locator('#notes').inner_text()
        page.screenshot(path=str(root / 'original.png'), full_page=True)
        with page.expect_download() as download:
            page.click('#download')
        download.value.save_as(root / 'original.csv')
        with np.load(root / 'fine_all_fitter_policy/curves.npz') as arrays:
            true_zenith = arrays['nue_cc_true_coszenith']
        reference = true_zenith[1]
        rows = list(csv.DictReader((root / 'original.csv').open()))
        for row in rows:
            if row['observable'] != 'true_coszenith':
                continue
            point = {'-1': 0, 'Test reference (0)': 1, '+1': 2}[row['point_label']]
            varied = true_zenith[point]
            index = int(round((float(row['bin_low']) + 1) * 5))
            assert np.isclose(float(row['rate_ratio']), varied[index] / reference[index], rtol=1e-14)
            assert np.isclose(float(row['shape_ratio']),
                              (varied[index] / varied.sum()) / (reference[index] / reference.sum()),
                              rtol=1e-14)
            assert row['energy_stratified_status']
            if row['is_reference'] == 'true':
                assert row['shape_test_status'] == 'proportional_within_precision'

        # A shape-only top spectrum must retain the primary RATE ratio in this
        # particular three-axis panel. Check the CSV uses the same convention.
        change('mode', 'shape')
        with page.expect_download() as download:
            page.click('#download')
        download.value.save_as(root / 'shape.csv')
        for row in csv.DictReader((root / 'shape.csv').open()):
            if row['observable'] == 'true_coszenith':
                assert row['display_ratio'] == row['rate_ratio']
        change('mode', 'rate')
        change('canvas', 'kinematics')
        assert snapshot('second_canvas')['panels'] == 6
        assert 'fixed production height 15 km' in page.locator('#notes').inner_text()
        assert page.locator('#grid .panel').count() == 8
        change('binning', 'miles')
        assert snapshot('miles_auto_fc')['dimensions']['containment'] == 'fc'
        assert 'all' not in page.locator('#containment').evaluate(
            's=>Array.from(s.options).map(o=>o.value)')
        axis = page.locator('#plot-1').evaluate(
            'p=>({range:p.layout.xaxis.range,type:p.layout.xaxis.type})')
        assert axis['type'] == 'linear'  # Symlog coordinates are transformed explicitly.
        np.testing.assert_allclose(axis['range'], [0, .1 * (1 / (1 - .1) + 3)])
        physical = np.array([-100, -1, -.1, -.05, 0, .05, .1, .2, 1, 100.])
        transformed = page.evaluate('(xs)=>xs.map(x=>symlogTransform(x,.1))', physical.tolist())
        np.testing.assert_allclose(transformed, SymmetricalLogTransform(10, .1, 1).transform(physical),
                                   rtol=1e-15, atol=1e-16)
        assert page.locator('#plot-1').evaluate('p=>p.layout.xaxis2.ticktext') == [
            '0', '0.1', '0.2', '0.5', '1', '2', '5', '10', '100']
        assert page.locator('#plot-1').evaluate('p=>p.data[0].customdata[0]') == [0, .2]
        change('selection', 'numu_cc')
        assert page.evaluate('state.page.reco_neutrino_energy.edges') == [0, .4, .6, 1, 100]
        page.screenshot(path=str(root / 'miles_kinematics.png'), full_page=True)
        change('selection', 'nc')
        snapshot('unsupported_nc')
        assert 'Explicitly unsupported' in page.locator('#grid').inner_text()
        assert page.evaluate("Array.from(document.querySelectorAll('.chart')).every(p=>!p.data)")
        for key, value in [('selection', 'nue_cc'), ('containment', 'pc'), ('policy', 'raw'),
                           ('binning', 'fine'), ('containment', 'all'),
                           ('policy', 'fitter_policy'), ('canvas', 'original')]:
            change(key, value)
        snapshot('round_trip')
        cache = page.evaluate('({compressed:loadedPayloads.size,decoded:decoded.size})')
        assert cache['compressed'] <= 1 and cache['decoded'] <= 3
        warnings = page.evaluate("Array.from(document.querySelectorAll('.chart')).flatMap(p=>"
                                 "p.data?(Plotly.validate(p.data,p.layout)||[]):[])")
        assert not warnings
        page.emulate_media(media='print')
        page.evaluate("window.dispatchEvent(new Event('beforeprint'))")
        page.wait_for_timeout(500)
        page.pdf(path=str(root / 'print.pdf'), prefer_css_page_size=True, print_background=True)
        page.emulate_media(media='screen')
        page.set_viewport_size({'width': 600, 'height': 1100})
        page.wait_for_timeout(300)
        page.screenshot(path=str(root / 'narrow.png'), full_page=True)
        external = [url for url in requests if url.startswith(('http:', 'https:'))]
        assert not errors
        assert not external
        report = dict(synthetic_fixture_only=True, checks=checks, errors=errors,
                      external_requests=external, plotly_warnings=warnings, cache=cache,
                      csv_shape_and_rate_ratios_match_numpy=True,
                      physical_symlog_matches_matplotlib=True)
        (root / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
        browser.close()
    print('PASS: synthetic offline browser fixture, CSV ratios, axes, dimensions and caches')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('fixture_dir', type=Path)
    check_viewer(parser.parse_args().fixture_dir.resolve())
