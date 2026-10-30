#!/usr/bin/env python3
"""Validate completed v2 output using saved aggregate arrays only.

From the repository after installation, run::

    python tests/integration/validate_saved_aggregates.py --output OUTPUT --report REPORT.json --expected-production-height-km 15

This manual validator targets the current 101-dial, 623-point, 10-dataset
production with one shared R68 energy grid. It is not a general public CLI.
Use --allow-pilot only for a completed one-file 101-dial validation sample.
No ROOT file, response cache, or event-level memmap is opened. This checks the
published curves against retained sums/quantile diagnostics, exact requested
edges, actual CV metadata, raw validity flags and FC+PC histogram closure.
It complements, rather than replaces, the independent dense-event pilot test.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from atmospheric_response.v2_config import load_config, resolve_edges, profile_has_samples, SELECTIONS, OBSERVABLE_IDS


MILES_ENU = {
    'PC_nue': [0,.2,.4,.6,.8,1.05,1.4,2,3.25,100],
    'FC_numu': [0,.4,.6,.8,1,1.1,1.2,1.45,1.65,1.9,100],
    'FC_nue': [0,.2,.4,.6,.8,1,1.1,1.2,1.35,1.5,1.7,1.95,2.25,3.25,100],
    'PC_numu': [0,.4,.6,.8,1,1.15,1.3,1.45,1.6,1.8,2,2.2,2.45,2.7,3,3.35,3.75,4.25,4.85,6.85,100],
}
MILES_COS = [-1,-.8,-.6,-.4,-.2,0,.2,.4,.6,.8,1]
MILES_Y = [0,.1,.2,.3,.4,.5,1]


def require(condition, context):
    if not condition:
        raise AssertionError(context)


def read_json(path):
    return json.loads(Path(path).read_text())


def equal(actual, expected, context, *, exact=False):
    a, b = np.asarray(actual), np.asarray(expected)
    require(a.shape == b.shape, f'{context}: shape mismatch {a.shape}/{b.shape}')
    require(np.array_equal(np.isnan(a), np.isnan(b)), f'{context}: NaN support mismatch')
    require(not np.isinf(a).any() and not np.isinf(b).any(), f'{context}: infinity')
    if exact:
        require(np.array_equal(a, b, equal_nan=True), f'{context}: exact values differ')
    else:
        scale = max(float(np.max(np.abs(b[np.isfinite(b)]), initial=0)), 1.)
        require(np.allclose(a, b, rtol=3e-11, atol=scale*2e-13, equal_nan=True), f'{context}: values differ')


def groups(category, containment):
    return [3*category,3*category+1,3*category+2] if containment == 'all' else [3*category+(2 if containment == 'fc' else 1)]


def reference_index(record):
    return (record['reference_index'] if record['kind'] == 'coupled_categorical_scenarios'
            else record['cv_index'] if record['cv_is_stored'] else None)


def validate_profile(diag, context, policy):
    """A missing R68 is legitimate only for an undefined weighted quantile."""
    q16, q84, r68 = (diag[k] for k in ('q16','q84','r68'))
    sw, sw2 = diag['sumw'], diag['sumw2']
    positive, negative = diag['positive_weight_events'], diag['negative_weight_events']
    eligible = np.isfinite(sw) & (sw > 0) & (positive > 0) & (negative == 0)
    require(np.array_equal(np.isfinite(r68), eligible), f'{context}: R68 validity does not follow weights')
    require(np.array_equal(np.isfinite(q16), eligible) and np.array_equal(np.isfinite(q84), eligible),
            f'{context}: quantile validity differs')
    require(not any(np.isinf(x).any() for x in (sw,sw2,q16,q84,r68)), f'{context}: infinite profile value')
    require(np.all(q16[eligible] >= -180-1e-10) and np.all(q84[eligible] <= 180+1e-10),
            f'{context}: signed angular quantile outside physical support')
    require(np.all(q16[eligible] <= q84[eligible]), f'{context}: quantile ordering')
    require(np.all((r68[eligible] >= 0) & (r68[eligible] <= 180+1e-10)), f'{context}: R68 physical range')
    equal(r68[eligible], (q84[eligible]-q16[eligible])/2, context+' quantile half-width', exact=True)
    require(np.all(sw2[np.isfinite(sw2)] >= 0), f'{context}: negative sumw2')
    require(np.all(diag['events'] >= 0) and np.all(positive >= 0) and np.all(negative >= 0), f'{context}: negative event count')
    if policy == 'fitter_policy':
        require(np.isfinite(sw).all() and np.isfinite(sw2).all(), f'{context}: nonfinite handled weight sums')
        require(np.all(sw >= 0) and np.all(negative == 0), f'{context}: handled negative weights')
    return int(np.count_nonzero(~eligible))


def validate(output, cv_map_path, *, allow_pilot=False, expected_source_files=145,
             expected_production_height_km=None):
    started = time.monotonic()
    output = Path(output)
    complete = read_json(output/'completion.json')
    require(allow_pilot or complete['complete_production'], 'Output is not declared complete production')
    cfg = load_config(output/'analysis_config.yaml')
    if expected_production_height_km is not None:
        require(cfg['physics']['geometry']['production_height_km']==expected_production_height_km,
                'Saved production height does not match the requested geometry')
    metadata = read_json(cv_map_path)['records']
    require(len(metadata) == 101 and sum(len(r['values']) for r in metadata) == 623, 'Unexpected 101/623 metadata contract')
    require(complete['dials'] == 101 and complete['actual_points'] == 623, 'Incomplete dial/point count')
    require(complete['observables'] == 13, 'Observable count is not13')
    require(complete['config_sha256'] == cfg['_sha256'], 'Configuration hash mismatch')
    sources = read_json(output/'source_order.json')
    audit = read_json(output/'selection_audit.json')
    expected_sources = 1 if allow_pilot else expected_source_files
    require(len(sources) == complete['source_files'] == expected_sources, 'Source file count mismatch')
    require(len({r['source'] for r in sources}) == len(sources), 'Duplicate source identity')
    counts = np.asarray(audit['events_by_truth_and_containment_unknown_pc_fc'])
    require(counts.shape == (5,3) and counts.dtype.kind in 'iu' and np.all(counts >= 0), 'Invalid FC/PC event-count matrix')
    total = int(counts.sum())
    require(total == complete['selected_events'] == audit['selected_events'] == sum(r['selected'] for r in sources),
            'Selected event counts do not close')
    require(audit['cached_events'] == sum(r['counts']['selected'] for r in sources), 'Cached-event count mismatch')
    if cfg['selection'].get('require_valid_wall',True):
        require(np.all(counts[:,0] == 0), 'Unknown containment survived required-valid-wall selection')
    require(all(0 <= r['selected'] <= r['counts']['selected'] for r in sources), 'Invalid per-source selected count')
    sumw = np.asarray(audit['sumw_by_truth_and_containment_unknown_pc_fc'])
    require(sumw.shape == (5,3) and np.isfinite(sumw).all() and np.all(sumw >= 0), 'Invalid baseline-weight category sums')

    listed = read_json(output/'manifest_list.json')
    expected_dimensions = {(p,c,w) for p in cfg['binning_profiles']
        for c in (('fc','pc') if profile_has_samples(cfg,p) else ('all','fc','pc'))
        for w in ('fitter_policy','raw')}
    require(len(listed) == complete['datasets'] == len(expected_dimensions) == 10, 'Expected exactly10 configured datasets')
    paths = {item['id']:Path(item['manifest']) for item in listed}
    require(len(paths) == len(listed), 'Duplicate manifest identifier')
    profile_grids = set()
    for p,c,_ in expected_dimensions:
        for selection in SELECTIONS:
            edge,_ = resolve_edges(cfg,p,c,selection,'angular_resolution')
            if edge is not None:
                profile_grids.add(hashlib.sha256(edge.tobytes()).hexdigest()[:12])
    require(len(profile_grids) == 1, 'This saved-diagnostic validation currently expects one shared R68 energy grid')
    grid = profile_grids.pop()
    with np.load(output/'diagnostics'/f'baseline_{grid}_split.npz',allow_pickle=False) as archive:
        baseline = {k:archive[k] for k in archive.files}
    with np.load(output/'diagnostics'/f'baseline_{grid}_all.npz',allow_pickle=False) as archive:
        baseline_all_profile = {k[9:]:archive[k] for k in archive.files if k.startswith('profile__')}
    baseline_counts = baseline['hist__fine__true_neutrino_energy__counts_with_flow'].sum(axis=1)
    equal(baseline_counts, counts.ravel(), 'Baseline true-energy events including flow', exact=True)
    equal(baseline['hist__fine__true_neutrino_energy__sumw_with_flow'][:,0,:].sum(axis=1),
          sumw.ravel(), 'Baseline true-energy weight closure')

    report = {'status':'running','scope':'Saved aggregates only; no ROOT, response cache, or event-level data opened',
        'complete_production':complete['complete_production'],'source_files':len(sources),'selected_events':total,
        'dials':101,'actual_points':623,'datasets':10,'observables':13,'config_sha256':cfg['_sha256'],
        'geometry':cfg['physics']['geometry'],'direction_method':cfg['physics']['direction_method'],
        'energy_estimator':cfg['physics']['energy_estimator'],
        'events_by_truth_and_containment_unknown_pc_fc':counts.tolist(),
        'histogram_vectors':0,'profile_vectors':0,'baseline_reference_checks':0,
        'actual_cv_metadata_checks':0,'exact_curve_edge_checks':0,'fc_pc_histogram_closure_vectors':0,
        'raw_pair_missing_support_checks':0,'raw_audit_count_checks':0,
        'profile_diagnostic_checks':0,'curve_records':0,'unavailable_curve_records':0,
        'dataset_summaries':{},'flatness_status_counts':{}}
    status_counts = Counter()
    canonical_audit = None
    dimensions_seen = set()
    for identifier,path in paths.items():
        manifest = read_json(path)
        dim = manifest['dimensions']
        p,c,policy = dim['binning'],dim['containment'],dim['policy']
        require((p,c,policy) in expected_dimensions and (p,c,policy) not in dimensions_seen, 'Unexpected/duplicate dataset dimensions')
        dimensions_seen.add((p,c,policy))
        require(manifest['schema_version'] == 2, f'{identifier}: schema')
        require(manifest['provenance']['config_sha256'] == cfg['_sha256'], f'{identifier}: hash')
        require(manifest['provenance']['selected_events'] == total, f'{identifier}: selected events')
        require([s['id'] for s in manifest['selections']] == [s['id'] for s in SELECTIONS], f'{identifier}: selections')
        require([o['id'] for o in manifest['observables']] == OBSERVABLE_IDS, f'{identifier}: observables')
        require(len(manifest['dials']) == 101, f'{identifier}: dial count')
        for slot,(dial,record) in enumerate(zip(manifest['dials'],metadata)):
            require(dial['id'] == f'd{slot:03d}' and dial['metadata'] == record, f'{identifier}: actual metadata mismatch at{slot}')
            ref = reference_index(record)
            require(dial['cv']['value_id'] == (None if ref is None else f'v{ref:03d}'), f'{identifier}: actual CV index mismatch')
            require(dial['reference_is_cv'] == (ref is not None and record['kind'] != 'coupled_categorical_scenarios'),
                    f'{identifier}: categorical reference mislabeled CV')
            require([v['id'] for v in dial['values']] == [f'v{i:03d}' for i in range(len(record['values']))], f'{identifier}: value ordering')
            expected_values = ([f'scenario_{i}' for i in range(len(record['values']))]
                               if record['kind']=='coupled_categorical_scenarios' else record['values'])
            require([v['value'] for v in dial['values']] == expected_values, f'{identifier}: stored coordinates changed')
            report['actual_cv_metadata_checks'] += 1
        curves = {(x['dial_id'],x['selection_id'],x['observable_id']):x for x in manifest['curves']}
        require(len(curves) == len(manifest['curves']) == 101*5*13, f'{identifier}: missing/duplicate curve record')
        report['curve_records'] += len(curves)
        local_audit = read_json(path.parent/'aggregation_audit.json')['dials']
        if canonical_audit is None:
            canonical_audit = local_audit
            require(len(local_audit)==101, 'Response audit missing dial')
            for slot,record in enumerate(local_audit):
                require(record['slot']==slot and len(record['response_counts'])==len(metadata[slot]['values']), 'Response audit point count')
                for j,point in enumerate(record['response_counts']):
                    require(point['point_index']==j, 'Response audit point order')
                    for name,flag in point.items():
                        if name=='point_index':
                            continue
                        cat,combined = np.asarray(flag['by_category']),np.asarray(flag['by_truth_and_containment'])
                        require(cat.shape==(5,) and combined.shape==(15,), 'Response count shape')
                        require(np.all(cat>=0) and np.all(combined>=0) and np.all(combined<=counts.ravel()), 'Response count bounds')
                        require(int(cat.sum())==int(combined.sum())==flag['total'], 'Response count total closure')
                        equal(combined.reshape(5,3).sum(axis=1),cat,'Response count containment closure',exact=True)
                        report['raw_audit_count_checks'] += 1
        else:
            require(local_audit==canonical_audit, f'{identifier}: global response audit differs across datasets')
        summary = {'handled_histogram_bins':0,'raw_histogram_missing_bins':0,'raw_histogram_negative_bins':0,
                   'undefined_profile_bins':0,'unavailable_curves':0}
        with np.load(path.parent/manifest['data_file'],allow_pickle=False) as data:
            closure_fc = closure_pc = None
            if p=='fine' and c=='all' and np.all(counts[:,0]==0):
                closure_fc = np.load(output/f'fine_fc_{policy}'/'curves.npz',allow_pickle=False)
                closure_pc = np.load(output/f'fine_pc_{policy}'/'curves.npz',allow_pickle=False)
            for slot,record in enumerate(metadata):
                did, nv, ref = f'd{slot:03d}',len(record['values']),reference_index(record)
                with np.load(output/'diagnostics'/f'{did}_{policy}.npz',allow_pickle=False) as archive:
                    # Per-dial aggregate arrays only; memory stays independent of event count.
                    diagnostic = {k:archive[k] for k in archive.files if k.startswith(('hist__','profile__','raw_quality__'))}
                split_profile = {k[9:]:v for k,v in diagnostic.items() if k.startswith('profile__')}
                if c=='all':
                    with np.load(output/'diagnostics'/f'{did}_{policy}_{grid}_all_profile.npz',allow_pickle=False) as archive:
                        prof = {k:archive[k] for k in archive.files}
                else:
                    prof = split_profile
                validate_profile(prof,f'{identifier}/{did}',policy)
                report['profile_diagnostic_checks'] += 1
                for ci,selection in enumerate(SELECTIONS):
                    gs = groups(ci,c)
                    for oid in OBSERVABLE_IDS:
                        curve = curves[(did,selection['id'],oid)]
                        edge,_ = resolve_edges(cfg,p,c,selection,oid)
                        unavailable = edge is None or (ci==2 and oid in ('true_lepton_energy','reco_lepton_energy','reco_inelasticity'))
                        require((curve.get('available') is False)==unavailable, f'{identifier}/{did}/{ci}/{oid}: availability')
                        if unavailable:
                            require(bool(curve.get('reason')), 'Unavailable curve lacks reason')
                            report['unavailable_curve_records'] += 1
                            summary['unavailable_curves'] += 1
                            continue
                        context = f'{identifier}/{did}/{ci}/{oid}'
                        physical = data[curve['edges_key']]
                        equal(physical,edge,context+' configured edges',exact=True)
                        require(np.isfinite(physical).all() and np.all(np.diff(physical)>0),context+' nonmonotonic edges')
                        if 'coszenith' in oid:
                            require(physical[0]>=-1 and physical[-1]<=1,context+' cosZ support')
                        elif 'inelasticity' in oid:
                            require(physical[0]>=0 and physical[-1]<=1,context+' y support')
                        else:
                            require(physical[0]>=0,context+' negative energy or L/E bin')
                        if p=='miles' and ('coszenith' in oid or 'neutrino_energy' in oid or 'inelasticity' in oid):
                            sample=c.upper()+'_'+selection['family']
                            supplied=(MILES_COS if 'coszenith' in oid else MILES_ENU[sample] if 'neutrino_energy' in oid
                                      else [0,.1,.2,.3,1] if sample=='PC_numu' else MILES_Y)
                            equal(physical,supplied,context+' original Miles edges',exact=True)
                        report['exact_curve_edge_checks'] += 1
                        value = data[curve['values_key']]
                        require(value.shape==(nv,len(edge)-1),context+' value matrix shape')
                        require(not np.isinf(value).any(),context+' infinite curve value')
                        require(('cv_key' in curve)==(ref is None),context+' reference encoding')
                        if oid=='angular_resolution':
                            g = ci if c=='all' else gs[0]
                            expected = prof['r68'][g]
                            expected_base = (baseline_all_profile['r68'][ci,0] if c=='all'
                                             else baseline['profile__r68'][gs[0],0])
                            if not allow_pilot:
                                require('centers_key' in curve,context+' profile centers missing')
                            if 'centers_key' in curve:
                                centers=np.sqrt(edge[:-1]*edge[1:]) if np.all(edge>0) else (edge[:-1]+edge[1:])/2
                                equal(data[curve['centers_key']],centers,context+' energy centers',exact=True)
                            report['profile_vectors'] += nv
                            summary['undefined_profile_bins'] += int(np.isnan(value).sum())
                        else:
                            name=f'hist__{p}__{oid}__'
                            expected = diagnostic[name+'sumw_with_flow'][gs].sum(axis=0)[:,1:len(edge)]
                            second = diagnostic[name+'sumw2_with_flow'][gs].sum(axis=0)[:,1:len(edge)]
                            expected_base = baseline[name+'sumw_with_flow'][gs].sum(axis=0)[0,1:len(edge)]
                            equal(diagnostic[name+'counts_with_flow'],baseline[name+'counts_with_flow'],context+' fixed selected counts',exact=True)
                            require(np.array_equal(np.isnan(expected),np.isnan(second)),context+' first/second-moment missing support')
                            require(np.all(second[np.isfinite(second)]>=0),context+' negative histogram sumw2')
                            if policy=='fitter_policy':
                                require(np.isfinite(value).all(),context+' nonfinite handled histogram')
                                scale=max(float(np.max(np.abs(value),initial=0)),1.)
                                require(np.all(value>=-scale*2e-13),context+' negative handled histogram')
                                summary['handled_histogram_bins'] += value.size
                            else:
                                summary['raw_histogram_missing_bins'] += int(np.isnan(value).sum())
                                summary['raw_histogram_negative_bins'] += int((value<0).sum())
                                for j in range(nv):
                                    bad=np.asarray(canonical_audit[slot]['response_counts'][j]['nonfinite']['by_truth_and_containment'])[gs].sum()
                                    require(bad>0 or np.isfinite(value[j]).all(),context+' NaN without any raw invalid event')
                            if closure_fc is not None:
                                equal(value,closure_fc[curve['values_key']]+closure_pc[curve['values_key']],context+' ALL=FC+PC')
                                report['fc_pc_histogram_closure_vectors'] += nv
                            report['histogram_vectors'] += nv
                        equal(value,expected,context+' retained diagnostics',exact=True)
                        if ref is None:
                            equal(data[curve['cv_key']],expected_base,context+' baseline reference',exact=True)
                            report['baseline_reference_checks'] += 1
                        if oid=='true_coszenith':
                            tests=curve['flatness']['values']
                            require(len(tests)==nv,context+' flatness point count')
                            pair_bad=diagnostic[f'raw_quality__{p}__true_coszenith__nonfinite_events'][gs].sum(axis=0)[:,1:len(edge)]
                            pair_neg=diagnostic[f'raw_quality__{p}__true_coszenith__negative_events'][gs].sum(axis=0)[:,1:len(edge)]
                            reference=expected_base if ref is None else value[ref]
                            for j,test in enumerate(tests):
                                status_counts[test['status']] += 1
                                require(test['raw_nonfinite_pair_events']==int(pair_bad[j].sum()) and test['raw_negative_pair_events']==int(pair_neg[j].sum()),context+' paired quality counts')
                                if policy=='raw':
                                    require(np.array_equal(np.isnan(value[j])|np.isnan(reference),pair_bad[j]>0),context+' raw paired NaN support')
                                    if pair_bad[j].sum()>0:
                                        require(test['status']=='unavailable_nonfinite',context+' invalid raw test has a claimed result')
                                    elif pair_neg[j].sum()>0:
                                        require(test['status']=='unavailable_signed_weights',context+' signed raw test has a claimed result')
                                    report['raw_pair_missing_support_checks'] += 1
                                probability=test['p_value']
                                require(probability is None or (np.isfinite(probability) and 0<=probability<=1 and test['ndof']>0),context+' invalid p-value')
                                if j==ref and test['status']=='proportional_within_precision':
                                    require(test['ndof']==0 and probability is None,context+' fabricated CV p-value')
                                if j==ref and np.isfinite(value[j]).all() and value[j].sum()>0 and (
                                        policy=='fitter_policy' or (pair_bad[j].sum()==0 and pair_neg[j].sum()==0)):
                                    require(test['status']=='proportional_within_precision',context+' actual CV fails self-comparison')
            if closure_fc is not None:
                closure_fc.close(); closure_pc.close()
        report['dataset_summaries'][identifier]=summary
        print(f'VALIDATED {identifier}: {len(curves)} curve records',flush=True)
    require(dimensions_seen==expected_dimensions,'Not every dataset dimension was validated')
    report.update(status='passed',elapsed_seconds=time.monotonic()-started,
                  flatness_status_counts=dict(status_counts))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--cv-map',type=Path,default=Path(__file__).parents[2]/'atmospheric_response/metadata/dial_cv_map.json')
    parser.add_argument('--allow-pilot',action='store_true',help='Validate one-file 101-dial output; never claim complete production')
    parser.add_argument('--expected-source-files',type=int,default=145)
    parser.add_argument('--expected-production-height-km',type=float)
    args=parser.parse_args()
    try:
        report=validate(args.output,args.cv_map,allow_pilot=args.allow_pilot,
                        expected_source_files=args.expected_source_files,
                        expected_production_height_km=args.expected_production_height_km)
    except Exception as error:
        args.report.write_text(json.dumps({'status':'failed','error':str(error),'output':str(args.output)},indent=2)+'\n')
        raise
    args.report.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='dataset_summaries'},indent=2),flush=True)


if __name__=='__main__':
    main()
