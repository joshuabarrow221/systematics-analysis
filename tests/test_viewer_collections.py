"""Combining method runs must preserve their input-relative paths and identity."""
import json
from pathlib import Path

import pytest

from atmospheric_response import build_interactive_v2
from atmospheric_response.build_interactive_v2 import collection_specs


def write_list(folder, entries, wrapped=False):
    folder.mkdir(parents=True, exist_ok=True)
    source = folder / 'manifest_list.json'
    source.write_text(json.dumps({'datasets': entries} if wrapped else entries))
    return source


def test_one_list_preserves_ids_and_resolves_beside_input(tmp_path):
    source = write_list(tmp_path / 'hybrid', [{'id': 'fine_fc', 'manifest': 'fine/manifest.json'}])
    assert collection_specs([source]) == [
        f'fine_fc={(source.parent / "fine/manifest.json").resolve()}']


def test_two_methods_namespace_only_collisions_and_keep_own_paths(tmp_path):
    first = write_list(tmp_path / 'hybrid', [
        {'id': 'fine_fc', 'manifest': 'result/manifest.json'},
        {'id': 'unique', 'manifest': 'single.json'}])
    absolute = tmp_path / 'already_absolute.json'
    second = write_list(tmp_path / 'previous', [
        {'id': 'fine_fc', 'manifest': 'result/manifest.json'},
        {'id': 'absolute', 'manifest': str(absolute)}], wrapped=True)
    assert collection_specs([first, second]) == [
        f'collection1__fine_fc={first.parent / "result/manifest.json"}',
        f'unique={first.parent / "single.json"}',
        f'collection2__fine_fc={second.parent / "result/manifest.json"}',
        f'absolute={absolute}',
    ]


def test_generated_prefix_cannot_overwrite_an_explicit_id(tmp_path):
    first = write_list(tmp_path / 'first', [{'id': 'same', 'manifest': 'one.json'}])
    second = write_list(tmp_path / 'second', [
        {'id': 'same', 'manifest': 'two.json'},
        {'id': 'collection1__same', 'manifest': 'three.json'}])
    names = [spec.split('=', 1)[0] for spec in collection_specs([first, second])]
    assert names == ['collection1__same__2', 'collection2__same', 'collection1__same']
    duplicate = write_list(tmp_path / 'bad', [
        {'id': 'same', 'manifest': 'one.json'}, {'id': 'same', 'manifest': 'two.json'}])
    with pytest.raises(ValueError, match='duplicate dataset ID'):
        collection_specs([duplicate])


def test_repeat_list_namespacing_does_not_allow_duplicate_physical_dimensions(tmp_path, monkeypatch):
    """A second name must not hide an accidentally repeated method run."""
    source = write_list(tmp_path / 'run', [{'id': 'fine', 'manifest': 'one.json'}])

    def small_dataset(name, manifest):
        assert manifest == source.parent / 'one.json'
        return {'id': name, 'dimensions': dict(binning='fine', containment='all',
                                              direction='hybrid', policy='fitter_policy')}, {
            'canvases': 0, 'stored_parameter_points': 0, 'compressed_page_bytes': 0}

    monkeypatch.setattr(build_interactive_v2, 'pack_dataset', small_dataset)
    monkeypatch.setattr('sys.argv', ['build_interactive_v2', '--manifest-list', str(source),
        '--manifest-list', str(source), '--output', str(tmp_path / 'duplicate.html'), '--embed'])
    with pytest.raises(ValueError, match='Duplicate dataset dimensions'):
        build_interactive_v2.main()
    assert not (tmp_path / 'duplicate.html').exists()
