#!/usr/bin/env python3
"""Reaggregate reusable local caches into configurable atmospheric canvases.

ROOT is never opened here. Every histogram uses its requested edges directly;
coarse response histograms and weighted quantiles are not interpolated from a
previous fine plot. One sparse dial is resident at a time. The two policies
share event selections and observables, and retain their different semantics.
"""
from __future__ import annotations
import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import resource
import time
import numpy as np

from .v2_config import load_config, resolve_edges, profile_has_samples, OBSERVABLE_IDS, SELECTIONS
from .v2_physics import derive_observables
from .v2_statistics import paired_shape_flatness, energy_stratified_flatness
from .v2_moments import get_difference_moments, get_response_quality_counts
from .sparse_response_aggregation import SparseResponseStudy

HERE = Path(__file__).resolve().parent
POLICIES = ('fitter_policy', 'raw')
RAW_FIELDS = ['E','q0','inelasticity','baseline_m','pdg','iscc', 'true_momentum',
              'reco_heshw','reco_lngtrk','reco_calo','vtx','enu_calo','enu_e_calo',
              'enu_e_had','enu_lep_calo','enu_mu_range','enu_mu_had',
              'n_particles','min_wall_cm','wall_valid']


def write_json(path, value):
    """Atomic publication tolerates brief Windows/OneDrive sharing locks."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    for attempt in range(40):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(min(.05*(attempt+1), .5))


def mmap(path, shape, dtype):
    return np.lib.format.open_memmap(path, mode='w+', dtype=dtype, shape=shape)


def close_map(array):
    if isinstance(array, np.memmap) and not array._mmap.closed:
        array.flush()
        array._mmap.close()


def reference_index(record):
    if record['kind'] == 'coupled_categorical_scenarios':
        return record['reference_index']
    return record['cv_index'] if record['cv_is_stored'] else None


def dial_definition(record):
    categorical = record['kind'] == 'coupled_categorical_scenarios'
    values = []
    for i, value in enumerate(record['values']):
        item = {'id':f'v{i:03d}', 'value':value, 'label':f'{value:.7g}'}
        if categorical:
            point = record['scenario_coordinates'][i]
            item.update(value=f'scenario_{i}', label=(f"scenario {i}: "
                f"({point['DecayAng2MEC']:g}, {point['DecayAngMEC']:g})"))
        values.append(item)
    index = reference_index(record)
    if categorical:
        cv = {'value_id':f'v{index:03d}', 'label':'Reference scenario (1,0); not a declared CV'}
    elif index is None:
        cv = {'value_id':None, 'value':record['reference_coordinate'],
              'label':'Nominal unity reference (CV not stored; fitter convention)'}
    else:
        cv = {'value_id':f'v{index:03d}', 'label':f"Stored CV ({record['cv_coordinate']:.7g})"}
    return {'id':f"d{record['index']:03d}", 'label':record['name'],
            'units':record['coordinate_label'], 'values':values, 'cv':cv,
            'reference_is_cv':index is not None and not categorical,
            'metadata':copy.deepcopy(record)}


def _npz_shape(archive, key):
    with archive.zip.open(key+'.npy') as stream:
        version = np.lib.format.read_magic(stream)
        reader = np.lib.format.read_array_header_1_0 if version == (1,0) else np.lib.format.read_array_header_2_0
        shape, _, dtype = reader(stream)
        if dtype.hasobject:
            raise ValueError('Object arrays are forbidden')
        return shape


def materialize(files, cfg, work):
    """Derive observables once, retaining a raw-cache -> selected-row mapping."""
    maximum = sum(r['counts']['selected'] for r in files)
    specs = {'category':('i1',()), 'combined_category':('i1',()),
             'containment':('i1',()), 'energy':('f8',()), 'residual':('f8',()),
             'base_weight':('f8',()), 'observable_values':('f8',(13,)),
             'observable_mask':('?',(13,)), 'run':('i4',()), 'subrun':('i4',()),
             'event':('i4',()), 'source_entry':('i8',()), 'source_part':('i2',())}
    arrays = {k:mmap(work/(k+'.npy'), (maximum,)+tail, dtype) for k,(dtype,tail) in specs.items()}
    sources, sparse_counts, offset = [], np.zeros((len(files),101),dtype=np.int64), 0
    bounds = np.array([cfg['selection']['fiducial_bounds_cm'][x] for x in 'xyz'])
    cut = cfg['selection']
    audit = {'cached_events':maximum, 'selected_events':0, 'files':[],
             'true_baseline_finite':0, 'true_y_discrepancy_above_1e_5':0,
             'true_y_max_abs_discrepancy':0., 'chosen_methods':[0,0,0]}
    for part, record in enumerate(files):
        with np.load(record['cache'], allow_pickle=False) as archive:
            raw = {k:archive[k] for k in RAW_FIELDS}
            derived = derive_observables(raw, cfg['physics'])
            E = raw['E']
            keep = (derived['category'] >= 0) & np.isfinite(E) & (E >= cut['true_energy_min_GeV'])
            if cut.get('true_energy_max_GeV') is not None:
                keep &= E <= cut['true_energy_max_GeV']
            keep &= np.all((raw['vtx'] > bounds[:,0]) & (raw['vtx'] < bounds[:,1]),axis=1)
            if cut.get('require_valid_wall',True):
                keep &= derived['containment'] >= 0
            if cut.get('require_valid_direction',True):
                keep &= derived['masks']['angular_resolution']
            base = archive['base_weight']
            if not np.all(np.isfinite(base) & (base > 0)):
                raise ValueError('Raw-cache baseline must be finite and positive')
            n = int(keep.sum())
            mapping = np.full(len(keep), -1, dtype=np.int64)
            mapping[keep] = np.arange(offset,offset+n)
            map_path = work/f'selection_map_{part:03d}.npy'
            np.save(map_path,mapping)
            cols = {'category':derived['category'], 'containment':derived['containment'],
                    'combined_category':derived['category']*3+derived['containment']+1,
                    'energy':E, 'residual':derived['signed_residual_deg'],
                    'base_weight':base, 'observable_values':derived['observable_values'],
                    'observable_mask':derived['observable_mask']}
            for key in ('run','subrun','event','source_entry'):
                cols[key] = archive[key]
            for key,value in cols.items():
                arrays[key][offset:offset+n] = value[keep]
            arrays['source_part'][offset:offset+n] = part
            for slot in range(101):
                sparse_counts[part,slot] = _npz_shape(archive,f'd{slot:03d}_rows')[0]
            delta_y = np.abs(derived['true_y_minus_q0_over_E'][keep])
            finite = delta_y[np.isfinite(delta_y)]
            audit['true_y_discrepancy_above_1e_5'] += int(np.sum(finite>1e-5))
            audit['true_y_max_abs_discrepancy'] = max(audit['true_y_max_abs_discrepancy'],float(finite.max(initial=0)))
            audit['true_baseline_finite'] += int(np.isfinite(raw['baseline_m'][keep]).sum())
            methods = np.bincount(derived['chosen_method'][keep],minlength=3)
            audit['chosen_methods'] = (np.array(audit['chosen_methods'])+methods).tolist()
            audit['files'].append({'part':part,'cached':len(keep),'selected':n,
                                  'invalid_wall_cached':int((derived['containment']<0).sum())})
            sources.append({**record, 'part':part,'selected':n,'start':offset,'stop':offset+n,
                            'selection_map':str(map_path)})
            offset += n
            print(f'BASELINE {part+1}/{len(files)}: {n:,}; selected total {offset:,}',flush=True)
    for key in arrays:
        arrays[key].flush()
        arrays[key] = arrays[key][:offset]
    audit['selected_events'] = offset
    audit['events_by_truth_and_containment_unknown_pc_fc'] = np.bincount(arrays['combined_category'],minlength=15).reshape(5,3).tolist()
    audit['sumw_by_truth_and_containment_unknown_pc_fc'] = np.bincount(arrays['combined_category'],weights=arrays['base_weight'],minlength=15).reshape(5,3).tolist()
    # Row identities, not weights.run (historically unfilled), identify events.
    order = np.lexsort((arrays['event'],arrays['subrun'],arrays['run']))
    repeated = np.ones(max(0,len(order)-1),dtype=bool)
    for name in ('run','subrun','event'):
        x = arrays[name][order]
        repeated &= x[1:]==x[:-1]
    audit['duplicate_event_key_adjacencies'] = int(repeated.sum())
    if repeated.any():
        raise ValueError('Repeated selected event keys; inspect before pooling')
    return arrays,sources,sparse_counts,audit


def direct_bin_codes(values, mask, category, edge_by_category):
    """Encode exact requested bins without rebinning previous histograms.

    Different channels can have different edges. A shared padded integer axis
    is only an internal grouping device; published edges are always physical.
    Underflow=-1 and overflow=max_bins give shared flow locations. Unused
    padded visible bins remain empty and are never published.
    """
    max_bins = max(len(e)-1 for e in edge_by_category.values())
    code = np.zeros(len(values),dtype=np.int16 if max_bins<32766 else np.int32)
    valid = np.zeros(len(values),dtype=bool)
    for cat, edges in edge_by_category.items():
        rows = np.flatnonzero(mask & (category==cat))
        x = values[rows]
        b = np.searchsorted(edges,x,side='right')-1
        b[x==edges[-1]] = len(edges)-2
        b[b>=len(edges)-1] = max_bins
        code[rows] = b
        valid[rows] = True
    return code,valid,np.arange(max_bins+1,dtype=float)-.5


def build_studies(arrays,cfg):
    specs, descriptions = {}, {}
    combined = arrays['combined_category']
    for profile in cfg['binning_profiles']:
        for j,oid in enumerate(OBSERVABLE_IDS):
            if oid=='angular_resolution':
                continue
            edge_by_category = {}
            for c,selection in enumerate(SELECTIONS):
                for sub,containment in enumerate(('unknown','pc','fc')):
                    edges,note = resolve_edges(cfg,profile,containment,selection,oid)
                    if edges is not None:
                        edge_by_category[3*c+sub] = edges
            if not edge_by_category:
                continue
            name = profile+'__'+oid
            code,mask,edges = direct_bin_codes(arrays['observable_values'][:,j],
                arrays['observable_mask'][:,j],combined,edge_by_category)
            specs[name] = {'values':code,'mask':mask,'edges':edges}
            descriptions[name] = edge_by_category
    # Independent coarse cosZ grid for the optional true-energy-conditioned
    # diagnostic. This does not silently merge bins in the plotted marginal test.
    ee = np.asarray(cfg['statistics']['conditional_energy_edges'])
    ce = np.asarray(cfg['statistics']['conditional_coszenith_edges'])
    eb = np.searchsorted(ee,arrays['energy'],side='right')-1
    eb[arrays['energy']==ee[-1]] = len(ee)-2
    cos = arrays['observable_values'][:,4]
    cb = np.searchsorted(ce,cos,side='right')-1
    cb[cos==ce[-1]] = len(ce)-2
    valid = arrays['observable_mask'][:,4] & (eb>=0)&(eb<len(ee)-1)&(cb>=0)&(cb<len(ce)-1)
    code = eb*(len(ce)-1)+cb
    joint_bins = (len(ee)-1)*(len(ce)-1)
    specs['conditional_zenith'] = {'values':code.astype(np.int32),'mask':valid,
                                 'edges':np.arange(joint_bins+1)-.5}
    grids, profile_grid = {}, {}
    for name in cfg['binning_profiles']:
        edges = first_available_edges(cfg,name,('fc','pc'),'angular_resolution')
        key = hashlib.sha256(edges.tobytes()).hexdigest()[:12]
        grids[key] = edges
        profile_grid[name] = key
    studies = {}
    first = next(iter(grids))
    for key,edges in grids.items():
        profile = {'energy':arrays['energy'],'residual':arrays['residual'],
                   'mask':arrays['observable_mask'][:,6],'edges':edges}
        studies[key] = {
            'split':SparseResponseStudy(combined,arrays['base_weight'],
                                       specs if key==first else {},profile,category_codes=tuple(range(15))),
            'all':SparseResponseStudy(arrays['category'],arrays['base_weight'],{},profile),
        }
    return studies,first,profile_grid,descriptions


def load_dial(sources,sparse_counts,slot,nvalues,work):
    maximum = int(sparse_counts[:,slot].sum())
    # Distinct names avoid a Windows/WSL delete-pending handle race between
    # consecutive dials on a mounted NTFS volume.
    rows = mmap(work/f'dial_{slot:03d}_rows.npy',(maximum,),'i8')
    responses = mmap(work/f'dial_{slot:03d}_responses.npy',(maximum,nvalues),'f4')
    offset = 0
    for part,record in enumerate(sources):
        if not sparse_counts[part,slot]:
            continue
        mapping = np.load(record['selection_map'],mmap_mode='r')
        with np.load(record['cache'],allow_pickle=False) as archive:
            local = archive[f'd{slot:03d}_rows']
            raw = archive[f'd{slot:03d}_responses']
            if (local.ndim != 1 or local.dtype.kind not in 'iu'
                    or (len(local) and (local.min()<0 or local.max()>=len(mapping)))):
                raise ValueError('Sparse cached row indices must be integers within their source cache')
            if len(np.unique(local))!=len(local) or raw.shape!=(len(local),nvalues):
                raise ValueError('Sparse cache shape or row uniqueness mismatch')
            mapped = mapping[local]
            keep = mapped>=0
            n = int(keep.sum())
            rows[offset:offset+n] = mapped[keep]
            responses[offset:offset+n] = raw[keep]
            offset += n
        close_map(mapping)
    rows.flush(); responses.flush()
    return rows[:offset],responses[:offset]


def raw_missing(study,result,rows,raw):
    """Invalidate every output touched by a nonfinite raw knot."""
    for j in range(raw.shape[1]):
        bad = rows[~np.isfinite(raw[:,j])]
        if not len(bad):
            continue
        for name,spec in study.histograms.items():
            group = spec['group'][bad]
            count = np.bincount(group[group>=0],minlength=study.nc*spec['width']).reshape(study.nc,spec['width'])
            for field in ('sumw_with_flow','sumw2_with_flow'):
                result['histograms'][name][field][:,j,:][count>0] = np.nan
        rank = study.profile_rank[bad]
        rank = rank[rank>=0]
        group = np.searchsorted(study.boundaries,rank,side='right')-1
        count = np.bincount(group,minlength=study.nc*study.nb).reshape(study.nc,study.nb)
        for field in ('q16','q84','r68','sumw','sumw2','n_eff'):
            result['profile'][field][:,j,:][count>0] = np.nan


def compact_test(result):
    """Keep useful statistics without repeating binwise ratios in every JSON."""
    keep = ('status','statistic','ndof','p_value','rate_scale','effective_bins',
            'min_effective_count','low_stat_bins','effective_count_threshold','notes',
            'p_value_underflow','n_strata','used_strata','invalid_strata',
            'ratio_shape_se')
    return {k:result[k] for k in keep if k in result}


def selected_groups(c,containment):
    return [3*c,3*c+1,3*c+2] if containment=='all' else [3*c+(2 if containment=='fc' else 1)]


def summed_hist(result,name,groups,field='sumw_with_flow'):
    return result['histograms'][name][field][groups].sum(axis=0)


def observable_definition(oid):
    titles = ['True charged-lepton energy','Reco charged-lepton contribution',
              'True hadronic energy transfer','Reco hadronic energy (NC: visible proxy)',
              'True neutrino zenith cosine','Reconstructed neutrino zenith cosine',
              'Weighted signed angular 68% half-width','True neutrino energy',
              'Reconstructed neutrino energy (NC: visible proxy)','True generator y',
              'Reconstructed inelasticity (CC estimator)',
              'True-angle geometric L/E proxy','Reco geometric L/E proxy']
    label = ('True neutrino energy [GeV]' if oid=='angular_resolution' else
             'L / E [km / GeV; fixed-height geometry]' if 'l_over_e' in oid else
             'y' if 'inelasticity' in oid else 'cos(zenith)' if 'coszenith' in oid else 'Energy [GeV]')
    return {'id':oid,'title':titles[OBSERVABLE_IDS.index(oid)],'x_label':label,
            'y_label':'(q84 − q16) / 2 [degrees]' if oid=='angular_resolution' else 'Stored-normalization weighted sum / bin',
            'shape_y_label':'Shape density / unit x',
            'kind':'profile' if oid=='angular_resolution' else 'histogram',
            'edges_key':'default_edges__'+oid,'y_scale':'linear','x_scale':'linear'}


def first_available_edges(cfg,profile,containments,observable):
    """A custom sample-specific preset need not contain an electron FC sample."""
    for containment in containments:
        for selection in SELECTIONS:
            edges,_ = resolve_edges(cfg,profile,containment,selection,observable)
            if edges is not None:
                return edges
    raise ValueError(f'{profile}: no available {observable} bins for {containments}')


def make_outputs(cfg,cv_map,audit,output):
    outputs = {}
    for profile in cfg['binning_profiles']:
        contains = ('fc','pc') if profile_has_samples(cfg,profile) else ('all','fc','pc')
        for containment in contains:
            for policy in POLICIES:
                name = f'{profile}_{containment}_{policy}'
                path = output/name
                path.mkdir(parents=True,exist_ok=True)
                observables = [observable_definition(x) for x in OBSERVABLE_IDS]
                if cfg['physics'].get('true_baseline_mode')=='stored':
                    observables[11]['title'] = 'Stored true baseline / true energy'
                    observables[11]['x_label'] = 'Stored L / true E [km / GeV]'
                data = {}
                for obs in observables:
                    edges = first_available_edges(cfg,profile,(containment,),obs['id'])
                    data[obs['edges_key']] = edges
                selections = copy.deepcopy(SELECTIONS)
                for c,selection in enumerate(selections):
                    groups = selected_groups(c,containment)
                    count = np.asarray(audit['events_by_truth_and_containment_unknown_pc_fc']).ravel()[groups].sum()
                    selection['description'] = (f'Truth category, not a CVN fitter sample; {containment.upper()}; '
                        f'{count:,} selected events; {cfg["physics"]["direction_method"]} direction. '+
                        ('NC uses heshw in this explicit legacy comparison. ' if cfg['physics']['direction_method']=='all_particles'
                         else 'NC uses hits only. ')+ 'Observable-specific masks and flows are audited.')
                rules = cfg['weight_policy']
                policy_text = ((f"MaCh3 knot policy: nonfinite→{rules['nonfinite_knot_replacement']:g}, "
                    f"clamp[{rules['knot_min']:g},{rules['knot_max']:g}]; nonpositive final weight→0.")
                    if policy=='fitter_policy' else 'RAW diagnostic: signed rates; nonfinite-affected bins and negative-weight quantiles undefined.')
                manifest = {'schema_version':2,'data_file':'curves.npz',
                    'title':f'Atmospheric response explorer · {profile} · {containment.upper()} · {policy}',
                    'output_prefix':name,'allow_missing_histogram_bins':policy=='raw',
                    'dimensions':{'policy':policy,'binning':profile,'containment':containment,
                                  'direction':cfg['physics']['direction_method']},
                    'dimension_labels':{
                        'policy':{'fitter_policy':'MaCh3 numerical policy',
                                  'raw':'Raw responses (diagnostic)'}[policy],
                        'binning':cfg['binning_profiles'][profile].get('label',profile),
                        'containment':{'all':'All containment','fc':'Fully contained',
                                       'pc':'Partially contained'}[containment],
                        'direction':{'hybrid':'Pierre hybrid; NC hits only',
                                     'all_particles':'Legacy all particles (NC heshw)',
                                     'hits':'Hits only'}.get(cfg['physics']['direction_method'],
                                                          cfg['physics']['direction_method'])},
                    'weight_description':'Unoscillated matching-flavor Honda flux × stored xsec × one stored response. '+policy_text,
                    'provenance':{'config_sha256':cfg['_sha256'],'selected_events':audit['selected_events'],
                        'categories':'Five truth channels; Miles edges applied by truth flavor, not an assertion of CVN fit selection',
                        'geometry':cfg['physics']['geometry'],
                        'true_baseline_mode':cfg['physics'].get('true_baseline_mode'),
                        'baseline':('Fixed production-height geometry proxy' if cfg['physics'].get('true_baseline_mode')=='geometric'
                            else 'CAF stored baseline converted metres to km; missing values remain unavailable'),
                        'selection':cfg['selection'],'weight_policy':cfg['weight_policy'],
                        'energy_estimator':cfg['physics']['energy_estimator'],
                        'flatness':'Paired-event finite-MC test of proportional marginal shapes; not a test of nuclear angular dependence',
                        'reference_warning':'Shared CV factors are not multiplied; exact absolute exposure and combined tune remain unverified'},
                    'selections':selections,'observables':observables,
                    'dials':[dial_definition(r) for r in cv_map['records']],'curves':[]}
                outputs[name] = {'manifest':manifest,'data':data,'path':path,'policy':policy,
                                 'profile':profile,'containment':containment,'diagnostics':{},'response_audits':[]}
    return outputs


def response_audit(category,combined,rows,raw):
    answer = []
    for j in range(raw.shape[1]):
        x = raw[:,j]
        finite = np.isfinite(x)
        masks = {'nonfinite':~finite,'negative_finite':finite&(x<0),
                 'zero_finite':finite&(x==0),'finite_nonunity':finite&(x!=1),
                 'below_minus9999':finite&(x<-9999),'above_9999':finite&(x>9999)}
        answer.append({'point_index':j, **{name:{'total':int(m.sum()),
            'by_category':np.bincount(category[rows][m],minlength=5).tolist(),
            'by_truth_and_containment':np.bincount(combined[rows][m],minlength=15).tolist()}
            for name,m in masks.items()}})
    return answer


def make_policy(raw,policy,cfg):
    """Only finite computational arrays enter the numerical accumulator.

    In raw mode nonfinite cells are placeholders and every affected reported
    output is subsequently invalidated. They never become claimed raw data.
    """
    response = np.empty(raw.shape,dtype=np.float32)
    rules = cfg['weight_policy']
    for j in range(raw.shape[1]):
        response[:,j] = np.where(np.isfinite(raw[:,j]),raw[:,j],rules['nonfinite_knot_replacement'])
        if policy=='fitter_policy':
            np.clip(response[:,j],rules['knot_min'],rules['knot_max'],out=response[:,j])
            # External nominal weights are strictly positive in this study,
            # so max(response,0) is equivalent to the one-dial total-weight cut.
            np.maximum(response[:,j],0,out=response[:,j])
    return response


def save_diagnostics(path,result,moments,quality):
    arrays = {}
    for name,spec in result['histograms'].items():
        for field,value in spec.items():
            arrays['hist__'+name+'__'+field] = value
    for field,value in result['profile'].items():
        arrays['profile__'+field] = value
    for name,fields in moments.items():
        for field,value in fields.items():
            arrays['paired__'+name+'__'+field] = value
    for name,fields in quality.items():
        for field,value in fields.items():
            arrays['raw_quality__'+name+'__'+field] = value
    np.savez_compressed(path,**arrays)


def _reference(matrix,baseline,index):
    return baseline[0] if index is None else matrix[index]


def make_flatness(output,c,slot,record,result,baseline,moments,quality,cfg):
    """Use the actual per-dial CV in both moments and normalization."""
    profile,containment,policy = output['profile'],output['containment'],output['policy']
    groups = selected_groups(c,containment)
    index = reference_index(record)
    name = profile+'__true_coszenith'
    edges,_ = resolve_edges(cfg,profile,containment,SELECTIONS[c],'true_coszenith')
    nb = len(edges)-1
    # Padded bins are excluded; normalization uses exactly the displayed bins.
    A = summed_hist(result,name,groups)[:,1:nb+1]
    A2 = summed_hist(result,name,groups,'sumw2_with_flow')[:,1:nb+1]
    baseB = summed_hist(baseline,name,groups)[:,1:nb+1]
    baseB2 = summed_hist(baseline,name,groups,'sumw2_with_flow')[:,1:nb+1]
    B,B2 = _reference(A,baseB,index),_reference(A2,baseB2,index)
    D2 = moments[name]['sumw2_difference'][groups].sum(axis=0)[:,1:nb+1]
    BD = moments[name]['sumw_ref_difference'][groups].sum(axis=0)[:,1:nb+1]
    C = moments[name]['sumw_cross'][groups].sum(axis=0)[:,1:nb+1]
    bad = quality[name]['nonfinite_events'][groups].sum(axis=0)[:,1:nb+1].sum(axis=1)
    neg = quality[name]['negative_events'][groups].sum(axis=0)[:,1:nb+1].sum(axis=1)
    # Conditional checks use their own explicitly declared coarse angular grid.
    en = len(cfg['statistics']['conditional_energy_edges'])-1
    zn = len(cfg['statistics']['conditional_coszenith_edges'])-1
    joint = 'conditional_zenith'
    JA = summed_hist(result,joint,groups)[:,1:en*zn+1].reshape(-1,en,zn)
    JA2 = summed_hist(result,joint,groups,'sumw2_with_flow')[:,1:en*zn+1].reshape(-1,en,zn)
    JB0 = summed_hist(baseline,joint,groups)[:,1:en*zn+1].reshape(1,en,zn)
    JB20 = summed_hist(baseline,joint,groups,'sumw2_with_flow')[:,1:en*zn+1].reshape(1,en,zn)
    JB,JB2 = _reference(JA,JB0,index),_reference(JA2,JB20,index)
    JD2 = moments[joint]['sumw2_difference'][groups].sum(axis=0)[:,1:en*zn+1].reshape(-1,en,zn)
    JBD = moments[joint]['sumw_ref_difference'][groups].sum(axis=0)[:,1:en*zn+1].reshape(-1,en,zn)
    JC = moments[joint]['sumw_cross'][groups].sum(axis=0)[:,1:en*zn+1].reshape(-1,en,zn)
    jbad = quality[joint]['nonfinite_events'][groups].sum(axis=0)[:,1:en*zn+1].reshape(-1,en,zn).sum(axis=2)
    jneg = quality[joint]['negative_events'][groups].sum(axis=0)[:,1:en*zn+1].reshape(-1,en,zn).sum(axis=2)
    tests = []
    for j in range(A.shape[0]):
        test = paired_shape_flatness(A[j],B,A2[j],B2,C[j],
            sumw2_difference=D2[j],sumw_ref_difference=BD[j],
            negative_events=int(neg[j]) if policy=='raw' else 0,
            nonfinite_events=int(bad[j]) if policy=='raw' else 0,
            min_effective_count=cfg['statistics']['minimum_effective_count'])
        conditioned = energy_stratified_flatness(JA[j],JB,JA2[j],JB2,JC[j],
            sumw2_difference=JD2[j],sumw_ref_difference=JBD[j],
            negative_events=jneg[j] if policy=='raw' else None,
            nonfinite_events=jbad[j] if policy=='raw' else None,
            min_effective_count=cfg['statistics']['minimum_effective_count'])
        compact = compact_test(test)
        compact['raw_nonfinite_pair_events'] = int(bad[j])
        compact['raw_negative_pair_events'] = int(neg[j])
        compact['energy_conditioned'] = compact_test(conditioned)
        compact['energy_conditioned']['empty_strata'] = conditioned['empty_strata']
        compact['energy_conditioned']['problem_strata'] = conditioned['problem_strata']
        compact['energy_conditioned']['standardized_ratio'] = conditioned['standardized_ratio']
        tests.append(compact)
    return {'values':tests,
            'normalization':'Each variation rescaled to its actual CV/reference visible yield; shared-event covariance included',
            'conditional_energy_edges_GeV':cfg['statistics']['conditional_energy_edges'],
            'conditional_coszenith_edges':cfg['statistics']['conditional_coszenith_edges'],
            'interpretation':'Descriptive finite-MC proportional-shape test; energy/flux/selection correlations can produce real marginal changes without nuclear incoming-angle dependence.'}


def add_curves(output,slot,record,results,baselines,first,profile_grid,moments,quality,cfg):
    manifest,data = output['manifest'],output['data']
    profile,containment = output['profile'],output['containment']
    hist = results[first]['split']
    hist_base = baselines[first]['split']
    grid = profile_grid[profile]
    for c,selection in enumerate(SELECTIONS):
        groups = selected_groups(c,containment)
        for oid in OBSERVABLE_IDS:
            curve = {'selection_id':selection['id'],'dial_id':f'd{slot:03d}','observable_id':oid}
            edges,note = resolve_edges(cfg,profile,containment,selection,oid)
            reason = None
            if edges is None:
                reason = note
            elif c==2 and oid in ('true_lepton_energy','reco_lepton_energy'):
                reason = 'NC has no primary charged lepton.'
            elif c==2 and oid=='reco_inelasticity':
                reason = 'NC calorimetric visible energy does not determine the outgoing neutrino energy; calo/calo=1 would not measure y.'
            if reason:
                curve.update(available=False,reason=reason)
                manifest['curves'].append(curve)
                continue
            edgekey = 'edges__'+selection['id']+'__'+oid
            data[edgekey] = edges
            curve['edges_key'] = edgekey
            if oid=='angular_resolution':
                centerkey = edgekey+'__centers'
                # Preserve the log-energy profile's geometric bin centers.
                # A user-supplied zero edge instead uses arithmetic centers.
                data[centerkey] = (np.sqrt(edges[:-1]*edges[1:]) if np.all(edges>0)
                                   else (edges[:-1]+edges[1:])/2)
                curve['centers_key'] = centerkey
                source = 'all' if containment=='all' else 'split'
                group = c if source=='all' else groups[0]
                matrix = results[grid][source]['profile']['r68'][group]
                base = baselines[grid][source]['profile']['r68'][group,0]
            else:
                name = profile+'__'+oid
                matrix = summed_hist(hist,name,groups)[:,1:len(edges)]
                base = summed_hist(hist_base,name,groups)[0,1:len(edges)]
            key = f'd{slot:03d}__{selection["id"]}__{oid}'
            data[key] = matrix.copy()
            curve['values_key'] = key
            if reference_index(record) is None:
                basekey = f'baseline__{selection["id"]}__{oid}'
                data[basekey] = base.copy()
                curve['cv_key'] = basekey
            energy_axis = ('energy' in oid or 'l_over_e' in oid or oid=='angular_resolution'
                           or oid=='true_hadronic_transfer')
            curve['x_scale'] = ('log' if edges[0]>0 else 'symlog') if energy_axis else 'linear'
            if curve['x_scale']=='symlog':
                curve['x_linthresh'] = .1
            curve['note'] = note
            if 'l_over_e' in oid and (oid=='reco_l_over_e' or cfg['physics'].get('true_baseline_mode')=='geometric'):
                geom = cfg['physics']['geometry']
                curve['note'] += (f"; fixed-height geometric proxy: R={geom['earth_radius_km']:g}km, "
                    f"h={geom['production_height_km']:g}km, depth={geom['detector_depth_km']:g}km. "
                    'Not a reconstructed production point or height-averaged oscillation probability.')
            if oid=='true_coszenith':
                curve['flatness'] = make_flatness(output,c,slot,record,hist,hist_base,moments,quality,cfg)
            if output['policy']=='raw':
                curve['missing_reason'] = 'Raw nonfinite responses invalidate touched bins; any negative event weight invalidates probability quantiles.'
            manifest['curves'].append(curve)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index',type=Path,required=True,help='Local v2 extractor scan_index.json')
    parser.add_argument('--config',type=Path,default=HERE/'configs/analysis_v2.yaml')
    parser.add_argument('--cv-map',type=Path,default=HERE/'metadata/dial_cv_map.json')
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--work-dir',type=Path,required=True)
    parser.add_argument('--limit-files',type=int,help='Explicitly labelled pilot, never a complete production claim')
    parser.add_argument('--limit-dials',type=int,help='Development pilot only; retain complete metadata order')
    args = parser.parse_args()
    started = time.monotonic()
    calculation_modules = ('v2_aggregate.py','v2_physics.py','v2_config.py',
        'v2_moments.py','v2_statistics.py','sparse_response_aggregation.py','response_aggregation.py')
    build = {name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in calculation_modules}
    cfg = load_config(args.config)
    index = json.loads(args.index.read_text())
    cv_map = json.loads(args.cv_map.read_text())
    files = sorted(index['files'],key=lambda r:r['cache'])
    expected = index.get('total_production_files',145)
    if args.limit_files is not None:
        files = files[:args.limit_files]
    elif len(files)!=expected:
        raise ValueError(f'Full aggregation requires all{expected} files; currently{len(files)}')
    if not files:
        raise ValueError('No cached input files')
    if len(cv_map['records'])!=101 or sum(len(r['values']) for r in cv_map['records'])!=623:
        raise ValueError('Unexpected dial metadata; preserve actual101-slot/623-point production contract')
    if len({x['cache'] for x in files})!=len(files):
        raise ValueError('Duplicate cache index entry')
    for record in files:
        if Path(record['cache']).stat().st_size!=record['cache_bytes']:
            raise ValueError('Cached file size differs from extraction record')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    args.work_dir.mkdir(parents=True,exist_ok=True)
    write_json(args.output_dir/'calculation_build.json',{'source_sha256':build,'numpy_version':np.__version__})
    config_out = args.output_dir/'analysis_config.yaml'
    config_out.write_bytes(args.config.read_bytes())
    write_json(args.output_dir/'input_snapshot.json',{'index':str(args.index.resolve()),
        'files':files,'config_sha256':cfg['_sha256'],'extractor_schema':index.get('schema_name'),
        'pilot':args.limit_files is not None or args.limit_dials is not None})
    arrays,sources,sparse_counts,audit = materialize(files,cfg,args.work_dir)
    write_json(args.output_dir/'selection_audit.json',audit)
    write_json(args.output_dir/'source_order.json',sources)
    studies,first,profile_grid,descriptions = build_studies(arrays,cfg)
    print('INDEXED',audit['selected_events'],'events;',len(studies[first]['split'].histograms),'histogram grids',flush=True)
    for key in list(arrays):
        if key not in ('base_weight','category','combined_category'):
            close_map(arrays.pop(key))
    baselines = {key:{part:study.aggregate(np.array([],dtype=np.int64),np.empty((0,1)))
                     for part,study in group.items()} for key,group in studies.items()}
    outputs = make_outputs(cfg,cv_map,audit,args.output_dir)
    records = cv_map['records'][:args.limit_dials] if args.limit_dials is not None else cv_map['records']
    if args.limit_files is not None or args.limit_dials is not None:
        for output in outputs.values():
            output['manifest']['title'] = f'PILOT ({len(files)}/{expected} files) · '+output['manifest']['title']
            output['manifest']['dials'] = output['manifest']['dials'][:len(records)]
    diag = args.output_dir/'diagnostics'
    diag.mkdir(exist_ok=True)
    for key,group in baselines.items():
        for part,result in group.items():
            save_diagnostics(diag/f'baseline_{key}_{part}.npz',result,{}, {})
    main_study = studies[first]['split']
    test_names = [k for k in main_study.histograms if k.endswith('__true_coszenith') or k=='conditional_zenith']
    for slot,record in enumerate(records):
        tick = time.monotonic()
        rows,raw = load_dial(sources,sparse_counts,slot,len(record['values']),args.work_dir)
        counts = response_audit(arrays['category'],arrays['combined_category'],rows,raw)
        ref = reference_index(record)
        quality = {name:get_response_quality_counts(main_study.histograms[name]['group'],rows,raw,
                    reference_index=ref,group_shape=(15,main_study.histograms[name]['width'])) for name in test_names}
        for policy in POLICIES:
            response = make_policy(raw,policy,cfg)
            results = {}
            for key,group in studies.items():
                results[key] = {}
                for part,study in group.items():
                    result = study.aggregate(rows,response,negative_policy='preserve')
                    if policy=='raw':
                        raw_missing(study,result,rows,raw)
                    results[key][part] = result
            moments = {name:get_difference_moments(main_study.histograms[name]['group'],rows,
                arrays['base_weight'],response,reference_index=ref,
                group_shape=(15,main_study.histograms[name]['width'])) for name in test_names}
            save_diagnostics(diag/f'd{slot:03d}_{policy}.npz',results[first]['split'],moments,quality)
            # The all-containment profile is a distinct weighted quantile, not
            # an average of FC/PC widths. Retain it explicitly for validation.
            for key in studies:
                np.savez_compressed(diag/f'd{slot:03d}_{policy}_{key}_all_profile.npz',**results[key]['all']['profile'])
            for output in outputs.values():
                if output['policy']==policy:
                    add_curves(output,slot,record,results,baselines,first,profile_grid,moments,quality,cfg)
                    output['response_audits'].append({'slot':slot,'name':record['name'],'response_counts':counts})
            del response,results,moments
        close_map(raw); close_map(rows)
        for name in (f'dial_{slot:03d}_rows.npy',f'dial_{slot:03d}_responses.npy'):
            (args.work_dir/name).unlink(missing_ok=True)
        gc.collect()
        print(f'DIAL {slot+1}/{len(records)} {record["name"]}: {time.monotonic()-tick:.2f}s',flush=True)
    manifest_list = []
    for name,output in outputs.items():
        np.savez_compressed(output['path']/'curves.npz',**output['data'])
        write_json(output['path']/'manifest.json',output['manifest'])
        write_json(output['path']/'aggregation_audit.json',{
            'response_count_scope':'global selected sample across ALL containment and five truth channels',
            'dials':output['response_audits'],'mode':output['policy'],
            'selection_audit':'../selection_audit.json','config_sha256':cfg['_sha256']})
        manifest_list.append({'id':name,'manifest':str((output['path']/'manifest.json').resolve())})
    write_json(args.output_dir/'manifest_list.json',manifest_list)
    write_json(args.output_dir/'completion.json',{'complete_production':len(files)==expected and len(records)==101,
        'source_files':len(files),'selected_events':audit['selected_events'],
        'dials':len(records),'actual_points':sum(len(r['values']) for r in records),
        'datasets':len(outputs),'observables':len(OBSERVABLE_IDS),'config_sha256':cfg['_sha256'],
        'elapsed_seconds':time.monotonic()-started,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
    print(f'COMPLETE {args.output_dir} in{time.monotonic()-started:.1f}s',flush=True)


if __name__=='__main__':
    main()
