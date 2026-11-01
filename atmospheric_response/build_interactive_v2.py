#!/usr/bin/env python3
"""Build an offline, multi-binning Plotly viewer from aggregate manifests.

Example:
  python -m atmospheric_response.build_interactive_v2 \
    --manifest-list outputs/collection.json --output outputs/explorer.html

Combine separately aggregated methods without manually merging their lists:
  python -m atmospheric_response.build_interactive_v2 \
    --manifest-list outputs/hybrid/manifest_list.json \
    --manifest-list outputs/previous/manifest_list.json \
    --output outputs/explorer.html

Repeated list inputs namespace colliding dataset IDs as collection1__ID,
collection2__ID, and so on. One-list IDs remain unchanged. Each manifest path
is resolved beside its own list, not beside the output HTML.

Only aggregate files are read. Each selection/dial canvas is compressed and
decoded on demand. By default, data live in local JavaScript sidecars so the
HTML remains small and works with file:// without a local web server. Keep the
HTML next to its .assets directory. --embed is available for small collections.
The default view uses the verified fitter numerical policy. Raw responses are
available only by explicitly selecting their diagnostic policy.

The controls answer different questions. ``Weight policy`` selects a separately
aggregated dataset whose event responses were handled upstream; this builder
never applies an event weight. ``Display`` changes how those saved bin sums
are presented: unchanged sums (rate), or each histogram divided by its own
visible-bin sum and physical bin width (shape). It neither reruns the paired
shape tests nor changes event selection. See ``normalise``, ``curvesFor`` and
``flatnessSummary`` in the embedded JavaScript below for the exact UI rules;
the corresponding static-renderer rules are in ``render_canvases_v2.py``.
"""
from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import html
import json
from pathlib import Path

import numpy as np

try:  # Installed package and direct-script use are both supported.
    from .render_canvases_v2 import line_styles, load_inputs, require, canvas_groups, ORIGINAL_ORDER, KINEMATICS_ORDER
except ImportError:
    from render_canvases_v2 import line_styles, load_inputs, require, canvas_groups, ORIGINAL_ORDER, KINEMATICS_ORDER

def json_array(a):
    a = np.asarray(a, dtype=float)
    b = a.astype(object)
    b[~np.isfinite(a)] = None
    return b.tolist()


def plain(text):
    """Render these scientific labels without a network-loaded MathJax dependency."""
    replacements = [(r"\bar{\nu}", "ν̄"), (r"\bar\nu", "ν̄"), (r"\nu", "ν"),
                    (r"\mu", "μ"), (r"\theta", "θ"), (r"\ell", "ℓ"),
                    (r"\cos", "cos"), (r"\mathrm", ""), (r"\rm", ""),
                    ("$", ""), ("{", ""), ("}", ""), ("\\", "")]
    for before, after in replacements:
        text = text.replace(before, after)
    return text


def collection_specs(paths):
    """Resolve independent manifest lists while preserving source-local paths.

    Dataset IDs name browser caches and sidecars. Two method runs normally use
    the same binning/policy IDs, so only those colliding names receive a stable
    collection-number prefix. The manifest's physical dimensions are unchanged;
    duplicate actual dimension combinations are still rejected by ``main``.
    """
    from collections import Counter
    records = []
    for number, source in enumerate(paths, start=1):
        source = Path(source)
        collection = json.loads(source.read_text())
        if isinstance(collection, dict):
            collection = collection['datasets']
        require(isinstance(collection, list), f'{source}: expected a dataset list')
        seen = set()
        for item in collection:
            identifier = item['id']
            require(isinstance(identifier, str) and identifier and '=' not in identifier,
                    f'{source}: dataset IDs must be nonempty strings without =')
            require(identifier not in seen, f'{source}: duplicate dataset ID {identifier}')
            seen.add(identifier)
            path = Path(item['manifest'])
            if not path.is_absolute():
                path = source.parent / path
            records.append((number, identifier, path.resolve()))
    counts = Counter(identifier for _, identifier, _ in records)
    # Reserve unchanged IDs first so an automatically generated prefix cannot
    # accidentally take a unique user-supplied ID from another input list.
    assigned = {identifier for _, identifier, _ in records if counts[identifier] == 1}
    specs = []
    for number, identifier, path in records:
        name = identifier
        if counts[identifier] > 1:
            stem = f'collection{number}__{identifier}'
            name, suffix = stem, 2
            while name in assigned:
                name = f'{stem}__{suffix}'
                suffix += 1
            assigned.add(name)
        specs.append(name + '=' + str(path))
    return specs


def pack_dataset(name, manifest):
    import matplotlib
    from matplotlib.colors import to_hex
    doc, selections, observables, dials, curves, audit = load_inputs(manifest)
    ordered = [k for k in ORIGINAL_ORDER if k in observables]
    ordered += [k for k in KINEMATICS_ORDER if k in observables]
    ordered += [k for k in observables if k not in ordered]
    require(1 <= len(ordered) <= 14, "Expected 1–14 observable panels")
    obs_metadata = []
    for oid in ordered:
        obs = {k: v for k, v in observables[oid].items() if not k.startswith("_")}
        obs["edges"] = observables[oid]["_edges"].tolist()
        obs["centers"] = observables[oid]["_centers"].tolist()
        obs["x_label"] = plain(obs["x_label"])
        obs["y_label"] = plain(obs["y_label"])
        if "shape_y_label" in obs:
            obs["shape_y_label"] = plain(obs["shape_y_label"])
        obs_metadata.append(obs)
    dial_metadata = []
    for dial in dials.values():
        item = dict(dial)
        item["styles"] = []
        for style in line_styles(dial, matplotlib.colormaps["turbo"]):
            item["styles"].append({"color": to_hex(style["color"]),
                                   "dash": {"-": "solid", "--": "dash", ":": "dot"}[style["linestyle"]],
                                   "width": style["linewidth"]})
        dial_metadata.append(item)
    default_label = {"raw": "Raw stored responses", "fitter_policy": "Fitter numerical handling",
                     "nominal": "Nominal baseline"}.get(name, name.replace("_", " ").title())
    metadata = {"id": name, "label": doc.get("policy_label", default_label),
                "title": doc.get("title", "Cross-section responses"),
                "weight_description": doc.get("weight_description", ""),
                "provenance": doc.get("provenance", {}),
                "allow_missing_histogram_bins": doc.get("allow_missing_histogram_bins", False),
                "nominal_only": doc.get("nominal_only", False),
                "dimensions": {"policy":name,"binning":"fine","containment":"all","direction":"unspecified",**doc.get("dimensions",{})},
                "dimension_labels": doc.get("dimension_labels",{}),
                "canvas_groups": [{"id":cid,"label":label,"observable_ids":keys} for cid,label,keys in canvas_groups(observables)],
                "selections": [{**s, "label": plain(s["label"])} for s in selections.values()],
                "observables": obs_metadata, "dials": dial_metadata, "pages": {}}
    compressed_bytes = raw_bytes = missing_hist_bins = missing_profile_bins = 0
    for sid in selections:
        for did in dials:
            page = {}
            for oid in ordered:
                record = curves[(sid, did, oid)]
                if not record["available"]:
                    page[oid] = {"available": False, "reason": record["reason"]}
                    continue
                nmissing = int(np.count_nonzero(~np.isfinite(record["values"])))
                if observables[oid]["kind"] == "histogram": missing_hist_bins += nmissing
                else: missing_profile_bins += nmissing
                page[oid] = {"available": True, "values": json_array(record["values"]),
                             "cv": json_array(record["cv"]),
                             "edges": record["edges"].tolist(), "centers":record["centers"].tolist(),
                             "x_scale":record["x_scale"], "x_linthresh":record["x_linthresh"],
                             **({"flatness":record["flatness"]} if "flatness" in record else {}),
                             **{k:record[k] for k in ("note", "missing_reason") if k in record}}
            blob = json.dumps(page, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
            packed = gzip.compress(blob, compresslevel=6, mtime=0)
            metadata["pages"][sid + "\x1f" + did] = base64.b64encode(packed).decode("ascii")
            raw_bytes += len(blob); compressed_bytes += len(packed)
    audit.update({"id": name, "selections": len(selections), "dials": len(dials),
                  "stored_parameter_points": sum(len(d["values"]) for d in dials.values()),
                  "canvases": len(metadata["pages"]), "uncompressed_page_bytes": raw_bytes,
                  "compressed_page_bytes": compressed_bytes,
                  "missing_histogram_value_bins": missing_hist_bins,
                  "missing_profile_value_bins": missing_profile_bins})
    return metadata, audit


TEMPLATE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{color-scheme:light;font-family:system-ui,-apple-system,"Segoe UI",sans-serif;color:#1c2633;background:#edf1f5}
*{box-sizing:border-box}body{margin:0}header{background:#fff;padding:18px 24px 14px;border-bottom:1px solid #ccd5de}
h1{font-size:23px;margin:0 0 8px}p{margin:6px 0;line-height:1.45}.intro{font-size:13px;color:#47586b;max-width:1150px}
.controls{display:flex;gap:12px;align-items:end;flex-wrap:wrap;margin-top:14px}.control{display:grid;gap:4px;font-size:12px;font-weight:600}
select,input,button{font:inherit;color:inherit;background:#fff;border:1px solid #aebbc8;border-radius:5px;padding:7px 9px}
select{min-height:34px;max-width:100%}#dial{width:min(680px,70vw)}#search{width:210px}.buttons{display:flex;gap:5px}button{cursor:pointer;font-size:12px}
button:hover{background:#edf3f8}button:disabled{opacity:.45;cursor:default}.status{font-size:12px;color:#47586b;margin-top:10px;min-height:20px}
#grid{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:12px;padding:16px;max-width:1900px;margin:auto}
.panel{background:#fff;border:1px solid #d1d9e2;border-radius:6px;min-width:0;height:380px;padding:9px 8px 3px;position:relative}
.panel h2{font-size:13px;font-weight:650;margin:0 8px 4px;min-height:19px}.chart{height:315px}.panel-note{font-size:10px;color:#795318;line-height:1.3;min-height:20px;margin:0 8px;max-height:29px;overflow:auto}
.empty{font-size:13px;color:#546579;padding:70px 22px;line-height:1.6}.notes-panel{overflow:auto;padding:12px 16px}.notes-panel h2{margin:0 0 8px}
.notes-panel p{font-size:11px;color:#394b60}.legend{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:5px 10px;margin:9px 0}
.legend-item{display:flex;align-items:center;gap:5px;font-size:11px;min-width:0;overflow-wrap:anywhere}.swatch{width:28px;min-width:28px;border-top:2px solid}.legend-item input{padding:0;margin:0}
.warning{color:#93450f!important}.small{font-size:10px!important;color:#657388!important}details{font-size:11px;margin:8px 0}details div{white-space:pre-wrap;overflow-wrap:anywhere;margin-top:7px}summary{cursor:pointer}
footer{font-size:11px;color:#5c6b7b;padding:4px 24px 16px}.loading{opacity:.5;pointer-events:none}.fatal{margin:24px;padding:16px;background:#fff4ec;border:1px solid #c67035;color:#7f3110}
.acceptance-note{font-size:12px;line-height:1.4;color:#795318;margin:7px 0 0;max-width:1250px}
@media(max-width:800px){#grid{grid-template-columns:1fr}.panel{height:370px}#dial{width:90vw}.legend{grid-template-columns:repeat(2,minmax(0,1fr))}header{padding:14px}}
@media print{@page{size:A3 portrait;margin:8mm}body{background:#fff}header{padding:0 4px 5px}h1{font-size:16px}.intro,.controls,.status,footer{display:none}#grid{display:grid;grid-template-columns:1fr 1fr;gap:4px;padding:0}.panel{height:350px;border-radius:0;break-inside:avoid}.chart{height:290px}.notes-panel{overflow:visible}.notes-panel p{font-size:8px;line-height:1.2;margin:3px 0}.notes-panel .legend{gap:2px 8px;margin:5px 0}.notes-panel .legend-item{font-size:8px;line-height:1.1}.notes-panel .legend-item input{width:10px;height:10px}.notes-panel details{display:none}.modebar{display:none!important}}
</style></head><body>
<header><h1 id="title">__TITLE__</h1>
<p class="intro">Two canvases cover the original observables and neutrino energy, inelasticity and L/E, with every stored parameter point. Black shows the stated central value or reference. Select a parameter and event selection below. Plotting code and local data assets work offline. Keep this HTML file beside its assets folder. Raw responses are an optional diagnostic; the default applies the documented fitter numerical handling.</p>
<div class="controls">
<label class="control">Binning<select id="binning"></select></label>
<label class="control">Containment<select id="containment"></select></label>
<label class="control">Angle reconstruction<select id="direction"></select></label>
<label class="control">Weight policy<select id="policy"></select></label>
<label class="control">Canvas<select id="canvas"></select></label>
<label class="control">Selection<select id="selection"></select></label>
<label class="control">Display<select id="mode"><option value="rate">Weighted rates</option><option value="shape">Unit-area shapes</option></select></label>
<label class="control">Find parameter<input id="search" type="search" placeholder="Type part of a dial name"></label>
<label class="control">Parameter<select id="dial"></select></label>
<div class="buttons"><button id="previous" title="Previous parameter">←</button><button id="next" title="Next parameter">→</button><button id="reset">Reset zoom</button><button id="download">Download canvas CSV</button><button id="print">Print canvas (A3)</button></div>
</div><p id="direction-acceptance" class="acceptance-note" role="note" hidden></p><div id="status" class="status" role="status">Loading embedded data…</div></header>
<main id="grid"></main><footer>Each histogram’s shape is normalized over its displayed range. Energy underflow/overflow and sample exclusions remain in the accompanying analysis audit. Ratios mask zero or undefined reference bins. Resolution profiles are never area-normalized.</footer>
<script id="embedded-data" type="application/json">__DATA__</script>
<script>__PLOTLY__</script>
<script>
'use strict';
const DATA=JSON.parse(document.getElementById('embedded-data').textContent);
const el=id=>document.getElementById(id), finite=x=>typeof x==='number'&&Number.isFinite(x);
const state={dataset:null,page:null,selection:null,dial:null,mode:'rate',canvas:'original',hidden:new Set(),ticket:0};
const decoded=new Map();
function esc(s){return String(s).replace(/[&<>"']/g,x=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[x]));}
function option(id,label){const o=document.createElement('option');o.value=id;o.textContent=label;return o;}
function fillSelect(target,items,keep){target.replaceChildren(...items.map(x=>option(x.id,x.label)));if(items.some(x=>x.id===keep))target.value=keep;}
// Preserve complete manifest wording while keeping long rotated titles inside
// the upper plot's height. Plotly accepts explicit line breaks in axis titles.
function verticalTitle(text){const lines=[''];for(const word of String(text).split(/\s+/)){const i=lines.length-1;if(lines[i]&&lines[i].length+word.length+1>26)lines.push(word);else lines[i]+=(lines[i]?' ':'')+word;}return lines.map(esc).join('<br>');}
// Input H[j] is the already weighted bin SUM from the chosen policy dataset.
// "Weighted rates" returns H unchanged: no extra flux, response, exposure,
// division by bin width, or normalization is applied in the browser.
// For "Unit-area shapes", return f[j] = H[j] / (sum_k H[k] * Delta_x[j]),
// so sum_j f[j]*Delta_x[j] = 1. Delta_x is the PHYSICAL edge difference even
// on a logarithmic/symlog x axis; this is density per dx, not per d(log x).
// The sum covers this observable's full stored/displayed bin range, excluding
// audited under/overflow. Zooming, hiding curves or opening another canvas
// cannot change that sum. Each dial point and the reference normalize separately.
// Missing bins make the whole integral unknown: never drop them and silently
// renormalize the remaining support. Raw negative bins are retained when their
// signed total is positive, giving a signed unit-area density, not a PDF.
// The angular-resolution profile is a quantile statistic in degrees, not a
// histogram of events; both display modes must leave its values unchanged.
function normalise(values,obs,mode){
 const out=values.slice();if(mode!=='shape'||obs.kind==='profile')return {values:out,warning:null};
 if(out.some(x=>!finite(x)))return {values:out.map(()=>null),warning:'Shape unavailable: missing bins make its integral unknown.'};
 const sum=out.reduce((a,b)=>a+b,0);if(!finite(sum)||sum<=0)return {values:out.map(()=>null),warning:'Shape unavailable: its signed total is nonpositive or nonfinite.'};
 return {values:out.map((x,i)=>x/sum/(obs.edges[i+1]-obs.edges[i])),warning:null};
}
// Undefined/zero-reference bins stay gaps; do not replace an undefined 0/0 by 1.
// Finite negative raw reference values remain usable for a signed rate ratio.
function ratios(values,reference){return values.map((x,i)=>finite(x)&&finite(reference[i])&&reference[i]!==0?x/reference[i]:null);}
// Match Matplotlib's base-10 SymmetricalLogTransform with linscale=1.
// Only display coordinates change. Histogram widths, hover bounds and CSV
// values always retain the original physical coordinates, including zero.
function symlogTransform(x,threshold=.1){const a=Math.abs(x),adjustment=1/(1-1/10);return a<=threshold?x*adjustment:Math.sign(x)*threshold*(adjustment+Math.log10(a/threshold));}
function displayX(x,obs){return obs.x_scale==='symlog'?symlogTransform(x,obs.x_linthresh||.1):x;}
function expanded(values,obs){
 if(obs.kind==='profile')return {x:obs.centers,y:values,custom:obs.centers.map((_,i)=>[obs.edges[i],obs.edges[i+1]])};
 const x=[],y=[],custom=[];for(let i=0;i<values.length;i++){x.push(obs.edges[i],obs.edges[i+1]);y.push(values[i],values[i]);custom.push([obs.edges[i],obs.edges[i+1]],[obs.edges[i],obs.edges[i+1]]);}return {x,y,custom};
}
// The manifest, not a presumed dial coordinate of zero or an array offset,
// supplies the reference. It may be an actual in-grid CV (possibly nonzero),
// or an explicitly labelled separate reference if no actual CV was supplied.
// Draw an in-grid CV only once, as the black reference. "raw" below means the
// unnormalized saved bin sums; it does NOT mean the selected policy is raw.
// Keep all three ratios distinct:
//   rate_ratio = A[j]/B[j];
//   shape_ratio = (A[j]/sum A)/(B[j]/sum B), since equal bin widths cancel;
//   ratio = the ratio of values in the current Display mode.
// This is equivalent to rescaling A to the reference total before dividing.
function curvesFor(record,dial,obs,mode){
 const reference=normalise(record.cv,obs,mode),shapeReference=normalise(record.cv,obs,'shape'),out=[];
 const referenceIndex=dial.values.findIndex(v=>v.id===dial.cv.value_id);
 const referenceTest=record.flatness?.reference||(referenceIndex>=0?record.flatness?.values?.[referenceIndex]:null)||null;
 for(let i=0;i<dial.values.length;i++){
  const value=dial.values[i];if(value.id===dial.cv.value_id)continue;
  const c=normalise(record.values[i],obs,mode),shape=normalise(record.values[i],obs,'shape');
  out.push({id:value.id,label:value.label,value:value.value,values:c.values,raw:record.values[i],shape:shape.values,
   warning:c.warning,style:dial.styles[i],reference:false,test:record.flatness?.values?.[i]||null});
 }
 out.push({id:'__reference__',label:dial.cv.label,value:referenceValue(dial),values:reference.values,raw:record.cv,shape:shapeReference.values,
  warning:reference.warning,style:{color:'#000000',dash:'solid',width:2.3},reference:true,test:referenceTest});
 return out.map(c=>({...c,ratio:ratios(c.values,reference.values),rate_ratio:ratios(c.raw,record.cv),shape_ratio:obs.kind==='profile'?c.values.map(()=>null):ratios(c.shape,shapeReference.values)}));
}
function referenceValue(dial){return dial.cv.value_id===null?dial.cv.value:dial.values.find(x=>x.id===dial.cv.value_id).value;}
// Sidecars use ordinary local script loading rather than fetch, so opening this
// HTML by file:// works without a web server. Only one compressed dataset and
// three decoded selection/dial pages are retained, even after many switches.
const loadedPayloads=new Map(),payloadRequests=new Map(),payloadResolvers=new Map();
window.__xsecRegisterPayload=(id,pages)=>{
 loadedPayloads.set(id,pages);while(loadedPayloads.size>1)loadedPayloads.delete(loadedPayloads.keys().next().value);
 const pending=payloadResolvers.get(id);if(pending)pending.resolve(pages);
};
async function loadPayload(dataset){
 if(dataset.pages)return dataset.pages;
 if(loadedPayloads.has(dataset.id))return loadedPayloads.get(dataset.id);
 if(payloadRequests.has(dataset.id))return payloadRequests.get(dataset.id);
 const request=new Promise((resolve,reject)=>{
  payloadResolvers.set(dataset.id,{resolve,reject});const script=document.createElement('script');
  script.src=dataset.payload_script;script.async=true;script.onload=()=>script.remove();
  script.onerror=()=>{script.remove();reject(Error('Cannot load local data asset '+dataset.payload_script+'. Keep the assets folder beside this HTML file.'));};
  document.head.append(script);
 }).finally(()=>{payloadRequests.delete(dataset.id);payloadResolvers.delete(dataset.id);});
 payloadRequests.set(dataset.id,request);return request;
}
async function decodePage(dataset,sid,did){
 const cacheKey=dataset.id+'\x1f'+sid+'\x1f'+did;if(decoded.has(cacheKey))return decoded.get(cacheKey);
 if(typeof DecompressionStream==='undefined')throw Error('This offline viewer needs a current browser with DecompressionStream.');
 const pages=await loadPayload(dataset),b64=pages[sid+'\x1f'+did];
 if(!b64)throw Error('Selected canvas is absent from the supplied manifest.');
 const bytes=Uint8Array.from(atob(b64),c=>c.charCodeAt(0));
 const stream=new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
 const page=JSON.parse(await new Response(stream).text());decoded.set(cacheKey,page);
 while(decoded.size>3)decoded.delete(decoded.keys().next().value);
 return page;
}
function formatP(value){return value===0?'below numerical range':value.toExponential(2);}
function testStatus(test){return ({ok_asymptotic:'Asymptotic test',low_statistics:'Low statistics; p withheld',proportional_within_precision:'Proportional; no p-value'}[test?.status]||(test?.status||'unavailable').replaceAll('_',' '));}
function rawQualityNote(flatness,policy){
 const tests=flatness?.values||[],nonfinite=Math.max(0,...tests.map(t=>t.raw_nonfinite_pair_events||0)),negative=Math.max(0,...tests.map(t=>t.raw_negative_pair_events||0));
 if(!nonfinite&&!negative)return '';
 return 'Selected true-zenith range: max per stored point is '+nonfinite.toLocaleString()+' nonfinite-response and '+negative.toLocaleString()+' negative raw-weight events in the variation/reference pair; these are not sums over points. '+
  (policy==='fitter_policy'?'The numerical policy replaces nonfinite responses by 1 and sets nonpositive final weights to 0. Handled flatness can reflect these replacements, not physical insensitivity.':'Raw nonfinite contributions leave undefined bins; negative event weights make probability quantiles undefined.');
}
// This is a synopsis of precomputed MARGINAL tests, not a new statistical test.
// "N point(s)" counts stored dial coordinates/scenarios with a finite p_value
// in flat.values. It does not count events, zenith bins, degrees of freedom,
// independent trials, or the separately supplied black reference. There is
// no explicit CV-ID filter: the actual CV self-comparison normally has rank 0
// and p_value=null upstream, so it is naturally absent from this finite-p count.
// The shown interval is min(p)..max(p) across those eligible stored points;
// it is neither a confidence interval nor a p-value combined across the dial.
// "Descriptive" flags this uncorrected synopsis of correlated comparisons.
// Each individual p is the upstream asymptotic tail probability, subject to
// its validity screens. Invalid/low-statistic/zero-rank tests have no finite p.
// Hidden legend curves, Display mode and zoom do NOT filter this summary.
// Energy-stratified results are separate and appear in the details table/CSV.
function flatnessSummary(flat){
 if(!flat?.values?.length)return '';
 const valid=flat.values.filter(t=>finite(t?.p_value));
 if(valid.length){const ps=valid.map(t=>t.p_value),low=Math.min(...ps),high=Math.max(...ps);return 'Paired shape test: p '+formatP(low)+(low===high?'':'–'+formatP(high))+'; '+valid.length+' point(s), descriptive.';}
 if(flat.values.every(t=>t?.status==='proportional_within_precision'))return 'All points proportional within numerical precision; no nondegenerate p-value.';
 return 'Shape test unavailable or low statistics; see per-point notes.';
}
// Preserve every stored point in the details table, including ineligible tests
// omitted from the finite-p synopsis. Missing p-values remain em dashes with
// explicit statuses; an exact reference match is not assigned a fabricated p=1.
function appendFlatnessTable(box,dial,flat){
 const details=document.createElement('details'),summary=document.createElement('summary');summary.textContent='True-zenith shape tests (shared-event covariance)';details.append(summary);
 const text=document.createElement('p');text.textContent='Null: proportional marginal zenith spectra. Dial settings use the same MC events, not independent random draws. Rejection can reflect energy, flux and selection correlations; it does not establish an incoming-angle dependence of the nucleus.';details.append(text);
 const table=document.createElement('table');table.style.cssText='font-size:10px;width:100%;border-collapse:collapse';
 const conditioned=flat.values?.some(test=>test.energy_conditioned);
 const labels=['Point','χ²','dof','Marginal p','Status'];if(conditioned)labels.push('Energy-stratified p','Stratified status');
 const header=table.insertRow();for(const label of labels){const cell=document.createElement('th');cell.textContent=label;cell.style.textAlign='left';header.append(cell);}
 dial.values.forEach((point,i)=>{const test=flat.values?.[i]||{},row=table.insertRow();
  const values=[point.label,finite(test.statistic)?test.statistic.toPrecision(4):'—',test.ndof??'—',finite(test.p_value)?formatP(test.p_value):'—',testStatus(test)];
  if(conditioned){const sub=test.energy_conditioned;values.push(finite(sub?.p_value)?formatP(sub.p_value):'—',testStatus(sub));}
  for(const value of values){const cell=row.insertCell();cell.textContent=value;cell.style.cssText='padding:3px;vertical-align:top;overflow-wrap:anywhere';}
  row.title=[...(test.notes||[]),...(test.energy_conditioned?.notes||[])].join(' ');
 });details.append(table);box.append(details);
 if(conditioned){const note=document.createElement('p');note.className='small';note.textContent='Energy-stratified tests fit a separate normalization inside each supplied energy interval. Their coarse angular grid can differ from the displayed spectrum. Energy edges [GeV]: '+JSON.stringify(flat.conditional_energy_edges_GeV||[])+'; zenith-cosine edges: '+JSON.stringify(flat.conditional_coszenith_edges||[])+'.';details.append(note);}
}
function activeObservables(dataset=state.dataset){const group=dataset.canvas_groups.find(g=>g.id===el('canvas').value)||dataset.canvas_groups[0];return group.observable_ids.map(id=>dataset.observables.find(o=>o.id===id));}
function recordObservable(obs,record){return {...obs,edges:record.edges||obs.edges,centers:record.centers||obs.centers,x_scale:record.x_scale||obs.x_scale,x_linthresh:record.x_linthresh||obs.x_linthresh||.1};}
function makeGrid(dataset){
 document.querySelectorAll('.chart').forEach(p=>{if(p.data)Plotly.purge(p);});
 el('grid').replaceChildren();const observables=activeObservables(dataset);observables.forEach((obs,i)=>{
  const panel=document.createElement('section');panel.className='panel';const h=document.createElement('h2');h.textContent=(i+1)+' · '+obs.title;
  const chart=document.createElement('div');chart.className='chart';chart.id='plot-'+i;
  const note=document.createElement('div');note.className='panel-note';note.id='note-'+i;panel.append(h,chart,note);el('grid').append(panel);
 });for(let i=observables.length;i<7;i++){const spacer=document.createElement('section');spacer.className='panel';spacer.setAttribute('aria-hidden','true');el('grid').append(spacer);}const notes=document.createElement('section');notes.className='panel notes-panel';notes.id='notes';el('grid').append(notes);
}
function buildNotes(dataset,selection,dial,page){
 const box=el('notes');box.replaceChildren();const title=document.createElement('h2');title.textContent='8 · Legend and notes';box.append(title);
 const summary=document.createElement('p');summary.textContent=dial.label+(dial.units?' ['+dial.units+']':'');box.append(summary);
 const legend=document.createElement('div');legend.className='legend';
 const items=[{id:'__reference__',label:dial.cv.label,style:{color:'#000000',dash:'solid'}},...dial.values.map((v,i)=>({...v,style:dial.styles[i]})).filter(v=>v.id!==dial.cv.value_id)];
 for(const item of items){const label=document.createElement('label');label.className='legend-item';const check=document.createElement('input');check.type='checkbox';check.checked=!state.hidden.has(item.id);check.addEventListener('change',()=>{check.checked?state.hidden.delete(item.id):state.hidden.add(item.id);renderCharts();});const line=document.createElement('span');line.className='swatch';line.style.borderColor=item.style.color;line.style.borderTopStyle=item.style.dash==='dash'?'dashed':item.style.dash==='dot'?'dotted':'solid';const text=document.createElement('span');text.textContent=item.label;label.append(check,line,text);legend.append(label);}box.append(legend);
 const selectedQuality=rawQualityNote(page.true_coszenith?.flatness,dataset.dimensions.policy);
 const qualityNotes=[selectedQuality,...(dial.metadata?.response_quality_notes||[])].filter(Boolean);const notes=[...qualityNotes];notes.push(dial.reference_is_cv===false?'Black is the stated reference, not an identified physical central value.':'Black is the explicitly supplied central/reference curve.');
 if(dial.cv.value_id===null)notes.push('The black reference is supplied separately; it is not an additional stored parameter point.');
 notes.push(dial.values.every(v=>finite(v.value))?'Matching ± offsets share colors; lower values are dashed and higher values dotted. No interpolation is applied.':'Stored discrete scenarios use distinct colors; no one-dimensional ± pairing is assigned. No interpolation is applied.');
 if(state.mode==='shape')notes.push('Histograms are divided by their own signed total and bin width. Negative densities are not probabilities. A missing bin makes the whole unit-area curve unavailable. Profiles retain degree units.');
 notes.push(selection.description||'');
 const geometry=dataset.provenance?.geometry;
 if(geometry&&activeObservables(dataset).some(o=>o.id.includes('l_over_e')))notes.push('L/E geometry proxy: Earth radius '+geometry.earth_radius_km+' km, fixed production height '+geometry.production_height_km+' km, detector depth '+geometry.detector_depth_km+' km; no production-height average.');
 notes.push(dataset.weight_description||'');
 if(Array.isArray(dial.metadata?.notes))notes.push(...dial.metadata.notes.filter(n=>!qualityNotes.includes(n)));
 if(dataset.allow_missing_histogram_bins)notes.push('Raw nonfinite responses can leave histogram gaps. A weighted quantile is undefined in slices with negative event weights; those profile points remain gaps.');
 for(const text of notes.filter(Boolean)){const p=document.createElement('p');p.textContent=text;if(qualityNotes.includes(text))p.className='warning';box.append(p);}
 const details=document.createElement('details');const ds=document.createElement('summary');ds.textContent='Canvas definitions and missing-value notes';const body=document.createElement('div');
 body.textContent=activeObservables(dataset).map(o=>{const r=page[o.id];return o.title+': '+(r.available?[r.note,r.missing_reason].filter(Boolean).join(' '):r.reason);}).join('\n\n')+'\n\nSource manifest: '+dataset.source_manifest;
 details.append(ds,body);box.append(details);
 const flat=page.true_coszenith?.flatness;if(flat)appendFlatnessTable(box,dial,flat);
}
// Hiding a curve changes only drawn traces. It does not renormalize another
// curve, refit a shape test, or change the summary's eligible-point count.
// True cosZ always shows BOTH rate/reference and shape/reference ratios,
// regardless of the spectrum Display mode. Other histogram panels show the
// ratio appropriate to Display; the profile ratio is always width/reference.
function tracesFor(record,dial,obs,mode){
 const traces=[],curves=curvesFor(record,dial,obs,mode),triple=obs.id==='true_coszenith';
 for(const c of curves){if(state.hidden.has(c.id))continue;
  const panels=[{values:c.values,x:'x',y:'y',label:'Value'},
   {values:triple?c.rate_ratio:c.ratio,x:'x2',y:'y2',label:triple?'Rate ratio':'Ratio'}];
  if(triple)panels.push({values:c.shape_ratio,x:'x3',y:'y3',label:'Shape-only ratio'});
  for(const panel of panels){const points=expanded(panel.values,obs);
   traces.push({type:'scatter',mode:obs.kind==='profile'?'lines+markers':'lines',name:esc(c.label),x:points.x.map(x=>displayX(x,obs)),y:points.y,customdata:points.custom,connectgaps:false,
    xaxis:panel.x,yaxis:panel.y,showlegend:false,legendgroup:c.id,
    line:{color:c.style.color,dash:c.style.dash,width:c.style.width},...(obs.kind==='profile'?{marker:{size:3,color:c.style.color}}:{}),
    hovertemplate:esc(c.label)+'<br>Bin [%{customdata[0]:.5g}, %{customdata[1]:.5g}]<br>'+panel.label+': %{y:.6g}<extra></extra>'});
  }
 }return {traces,curves};
}
async function renderCharts(){
 const {dataset,selection,dial,page,mode}=state;if(!page)return;const renderingTicket=state.ticket;
 const promises=[];
 activeObservables(dataset).forEach((declared,i)=>{
  const target=el('plot-'+i),note=el('note-'+i),record=page[declared.id],obs=recordObservable(declared,record);
  if(!record.available){if(target.data)Plotly.purge(target);target.replaceChildren();const div=document.createElement('div');div.className='empty';div.textContent=record.reason;target.append(div);note.textContent='';return;}
  const {traces,curves}=tracesFor(record,dial,obs,mode),triple=obs.id==='true_coszenith';
  const isLog=obs.x_scale==='log',isSymlog=obs.x_scale==='symlog';
  const range=isLog?[Math.log10(obs.edges[0]),Math.log10(obs.edges.at(-1))]:[displayX(obs.edges[0],obs),displayX(obs.edges.at(-1),obs)];
  const label=mode==='shape'&&obs.kind==='histogram'?(obs.shape_y_label||'Signed unit-area density'):obs.y_label;
  const refTerm=dial.reference_is_cv===false?'ref.':'CV';
  const layout={margin:{l:72,r:18,t:8,b:43},paper_bgcolor:'#fff',plot_bgcolor:'#fff',font:{family:'Arial, sans-serif',size:10,color:'#233449'},hovermode:'closest',showlegend:false,
   xaxis:{type:isLog?'log':'linear',range,domain:[0,1],anchor:'y',showticklabels:false,showgrid:true,gridcolor:'#e8edf2',zeroline:false},
   xaxis2:{type:isLog?'log':'linear',range,domain:[0,1],anchor:'y2',matches:'x',title:{text:obs.x_label,standoff:3},showgrid:true,gridcolor:'#e8edf2',zeroline:false},
   yaxis:{domain:[.35,1],title:{text:verticalTitle(label),standoff:4},type:obs.y_scale||'linear',automargin:true,gridcolor:'#e8edf2',zeroline:true,zerolinecolor:'#b6c0cb'},
   yaxis2:{domain:[0,.22],title:{text:'/ '+refTerm,standoff:4},automargin:true,gridcolor:'#e8edf2',zeroline:false},
   shapes:[{type:'line',xref:'paper',yref:'y2',x0:0,x1:1,y0:1,y1:1,line:{color:'#b9c5d0',width:1}}],
   uirevision:dataset.id+'|'+selection.id+'|'+dial.id+'|'+mode};
  if(triple){
   layout.yaxis.domain=[.53,1];layout.yaxis.title.text=mode==='shape'?'Unit-area density':'Weighted sum / bin';
   layout.yaxis2.domain=[.28,.43];layout.yaxis2.title.text='Rate / '+refTerm;layout.yaxis2.nticks=3;
   layout.xaxis2.showticklabels=false;delete layout.xaxis2.title;
   layout.xaxis3={type:isLog?'log':'linear',range,domain:[0,1],anchor:'y3',matches:'x',title:{text:obs.x_label,standoff:3},showgrid:true,gridcolor:'#e8edf2',zeroline:false};
   layout.yaxis3={domain:[0,.17],title:{text:'Shape / '+refTerm,standoff:4},nticks:3,automargin:true,gridcolor:'#e8edf2',zeroline:false};
   layout.shapes.push({type:'line',xref:'paper',yref:'y3',x0:0,x1:1,y0:1,y1:1,line:{color:'#b9c5d0',width:1}});
  }
  if(isSymlog){
   const ticks=[-100,-10,-5,-2,-1,-.5,-.2,-.1,0,.1,.2,.5,1,2,5,10,100].filter(x=>x>=obs.edges[0]&&x<=obs.edges.at(-1));
   for(const axis of ['xaxis','xaxis2','xaxis3'])if(layout[axis])Object.assign(layout[axis],{tickmode:'array',tickvals:ticks.map(x=>displayX(x,obs)),ticktext:ticks.map(String)});
  }
  if(obs.ratio_limits)layout.yaxis2.range=obs.ratio_limits;
  const failed=curves.filter(c=>c.warning),missing=record.values.flat().filter(x=>!finite(x)).length;
  const noteParts=[];if(triple){const summary=flatnessSummary(record.flatness);if(summary)noteParts.push(summary);}
  if(isSymlog)noteParts.push('Symlog x: linear within ±'+obs.x_linthresh+'; physical bin edges and CSV are unchanged.');
  if(failed.length)noteParts.push(failed.length+' shape curve(s) unavailable: missing bins or nonpositive total.');
  if(missing)noteParts.push(record.missing_reason||'Undefined values are shown as gaps.');
  if(mode==='shape'&&obs.kind==='profile')noteParts.push('Resolution statistic: no area normalization.');
  note.textContent=noteParts.join(' ');note.title=note.textContent;
  promises.push(Plotly.react(target,traces,layout,{responsive:true,displaylogo:false,scrollZoom:false,modeBarButtonsToRemove:['select2d','lasso2d'],toImageButtonOptions:{format:'svg',filename:selection.id+'_'+dial.id+'_'+obs.id}}));
 });await Promise.all(promises);
 if(renderingTicket===state.ticket){window.__viewerState={policy:dataset.id,selection:selection.id,dial:dial.id,mode,panels:activeObservables(dataset).length,dimensions:dataset.dimensions,canvas:state.canvas,stored_points:dial.values.length};window.__viewerReady=true;}
}
const dimensionKeys=['binning','containment','direction','policy'];
function selectedDataset(){return DATA.datasets.find(d=>dimensionKeys.every(k=>String(d.dimensions[k])===el(k).value));}
function directionAcceptanceNote(dataset){
 if(new Set(DATA.datasets.map(d=>d.dimensions.direction)).size<2)return '';
 const supplied=dataset.provenance?.direction_acceptance;
 if(typeof supplied==='string'&&supplied.trim())return supplied;
 return dataset.provenance?.selection?.require_valid_direction===true?
  'Each method requires a valid reconstructed direction; event counts and truth spectra can therefore differ. Use the common-valid comparison to isolate changes of reconstructed values.':'';
}
function dimensionLabel(dataset,key,value){const labels=dataset.dimension_labels?.[key];return typeof labels==='string'?labels:labels?.[value]||({all:'All containment',fc:'Fully contained',pc:'Partially contained',FC:'Fully contained',PC:'Partially contained',fine:'Fine',miles:'Miles',hybrid:'Pierre hybrid; NC hits only',all_particles:'Previous all-particle method (NC heshw)',all_particle:'Previous all-particle method (NC heshw)',hits:'Hits only',raw:'Raw responses (diagnostic)',fitter_policy:'MaCh3 numerical policy'}[value]||value);}
function syncDimensions(preferred={}){
 let candidates=DATA.datasets;
 for(const key of dimensionKeys){const items=[];const seen=new Set();for(const d of candidates){const id=String(d.dimensions[key]);if(!seen.has(id)){seen.add(id);items.push({id,label:dimensionLabel(d,key,id)});}}
  fillSelect(el(key),items,preferred[key]??el(key).value);candidates=candidates.filter(d=>String(d.dimensions[key])===el(key).value);el(key).disabled=items.length<2;
 }
 const dataset=candidates[0];if(!dataset)throw Error('No supplied dataset matches these dimensions.');
 const keepSelection=preferred.selection??el('selection').value,keepDial=preferred.dial??el('dial').value;
 fillSelect(el('selection'),dataset.selections,keepSelection);fillSelect(el('canvas'),dataset.canvas_groups,preferred.canvas??el('canvas').value);
 updateDialList(keepDial);
}
async function showPage(){
 const ticket=++state.ticket,dataset=selectedDataset();if(!dataset)return;
 const sid=el('selection').value,did=el('dial').value;if(!did){el('status').textContent='No parameter matches that search.';return;}
 el('grid').classList.add('loading');el('status').textContent='Loading selected local canvas…';window.__viewerReady=false;
 try{const page=await decodePage(dataset,sid,did);if(ticket!==state.ticket)return;
  const selection=dataset.selections.find(x=>x.id===sid),dial=dataset.dials.find(x=>x.id===did),canvas=el('canvas').value;
  if(state.dataset?.id!==dataset.id||state.canvas!==canvas)makeGrid(dataset);
  Object.assign(state,{dataset,page,selection,dial,mode:el('mode').value,canvas});state.hidden=new Set();
  const acceptance=directionAcceptanceNote(dataset);el('direction-acceptance').textContent=acceptance;el('direction-acceptance').hidden=!acceptance;
  buildNotes(dataset,selection,dial,page);await renderCharts();if(ticket!==state.ticket)return;
  const index=Array.from(el('dial').options).findIndex(x=>x.value===did);
  // This header count includes EVERY stored dial setting. It can exceed the
  // true-zenith synopsis count, which includes only settings with a finite p.
  el('status').textContent=dimensionKeys.map(k=>dimensionLabel(dataset,k,String(dataset.dimensions[k]))).join(' · ')+' · '+selection.label+' · '+(index+1)+'/'+el('dial').options.length+' parameters · '+dial.values.length+' stored point(s)';
  el('previous').disabled=index<=0;el('next').disabled=index>=el('dial').options.length-1;
  history.replaceState(null,'','#'+new URLSearchParams({...dataset.dimensions,selection:sid,dial:did,mode:state.mode,canvas}).toString());
  window.__viewerError=null;
 }catch(error){el('status').textContent='Cannot display this canvas: '+error.message;console.error(error);window.__viewerError=String(error);}
 finally{if(ticket===state.ticket)el('grid').classList.remove('loading');}
}
function updateDialList(keep){const dataset=selectedDataset(),q=el('search').value.toLowerCase();if(!dataset)return;const dials=dataset.dials.filter(d=>(d.id+' '+d.label).toLowerCase().includes(q));fillSelect(el('dial'),dials,keep);showPage();}

// Export all stored curves for this canvas, including those hidden in the
// legend. "value" follows Display (bin sum, unit-area density, or unchanged
// profile statistic); bin coordinates stay physical. Both rate_ratio and
// shape_ratio are explicit columns so switching Display cannot obscure which
// comparison was used. Test columns copy upstream results without refitting.
function downloadCSV(){if(!state.page)return;const {dataset,selection,dial,page,mode}=state;
 const rows=[['dataset','binning','containment','direction','policy','selection','dial','mode','canvas','observable','point_label','point_value','is_reference','bin_low','bin_high','bin_center','value','display_ratio','rate_ratio','shape_ratio','shape_ratio_se','shape_chi2','shape_ndof','shape_p_value','shape_test_status','energy_stratified_chi2','energy_stratified_ndof','energy_stratified_p_value','energy_stratified_status','raw_nonfinite_pair_events','raw_negative_pair_events']];
 for(const declared of activeObservables(dataset)){const r=page[declared.id];if(!r.available)continue;const obs=recordObservable(declared,r);
  // The primary ratio in the true-zenith panel stays a rate ratio in both
  // display modes; its additional lower panel supplies the shape-only ratio.
  for(const c of curvesFor(r,dial,obs,mode)){for(let i=0;i<c.values.length;i++)rows.push([dataset.id,dataset.dimensions.binning,dataset.dimensions.containment,dataset.dimensions.direction,dataset.dimensions.policy,selection.id,dial.id,mode,state.canvas,obs.id,c.label,JSON.stringify(c.value),c.reference,obs.edges[i],obs.edges[i+1],obs.centers[i],c.values[i],obs.id==='true_coszenith'?c.rate_ratio[i]:c.ratio[i],c.rate_ratio[i],c.shape_ratio[i],c.test?.ratio_shape_se?.[i],c.test?.statistic,c.test?.ndof,c.test?.p_value,c.test?.status,c.test?.energy_conditioned?.statistic,c.test?.energy_conditioned?.ndof,c.test?.energy_conditioned?.p_value,c.test?.energy_conditioned?.status,c.test?.raw_nonfinite_pair_events,c.test?.raw_negative_pair_events]);}
 }
 const cell=x=>x===null||x===undefined?'':'"'+String(x).replace(/"/g,'""')+'"',csv=rows.map(r=>r.map(cell).join(',')).join('\r\n');
 const url=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'})),a=document.createElement('a');a.href=url;a.download=[dataset.id,selection.id,dial.id,mode,state.canvas].join('_').replace(/[^A-Za-z0-9_.-]/g,'_').slice(0,210)+'.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}
function boot(){
 const hash=new URLSearchParams(location.hash.slice(1)),fallback=DATA.datasets.find(d=>d.id===DATA.default_dataset)||DATA.datasets[0];
 if(['rate','shape'].includes(hash.get('mode')))el('mode').value=hash.get('mode');
 for(const key of dimensionKeys)el(key).addEventListener('change',()=>syncDimensions());
 for(const id of ['selection','dial','mode','canvas'])el(id).addEventListener('change',showPage);
 let searchTimer;el('search').addEventListener('input',()=>{clearTimeout(searchTimer);searchTimer=setTimeout(()=>updateDialList(el('dial').value),120);});
 for(const [id,step] of [['previous',-1],['next',1]])el(id).addEventListener('click',()=>{el('dial').selectedIndex+=step;showPage();});
 el('reset').addEventListener('click',()=>{activeObservables(state.dataset).forEach((o,i)=>{const p=el('plot-'+i);if(p.data){const update={'xaxis.autorange':true,'xaxis2.autorange':true,'yaxis.autorange':true,'yaxis2.autorange':true};if(o.id==='true_coszenith')Object.assign(update,{'xaxis3.autorange':true,'yaxis3.autorange':true});Plotly.relayout(p,update);}});});
 el('download').addEventListener('click',downloadCSV);el('print').addEventListener('click',()=>window.print());
 window.addEventListener('beforeprint',()=>{const width=Math.floor(((297-16)*96/25.4-4)/2-18);document.querySelectorAll('.chart').forEach(p=>{if(p.data)Plotly.relayout(p,{width,height:290});});});
 window.addEventListener('afterprint',()=>{document.querySelectorAll('.chart').forEach(p=>{if(p.data)Plotly.relayout(p,{width:null,height:null,autosize:true});});});
 const preferred={...fallback.dimensions};for(const [key,value] of hash)preferred[key]=value;syncDimensions(preferred);
}
boot();
</script></body></html>'''


def main():
    """Package aggregate files; never load event-level data or contact ROOT."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", default=[], metavar="NAME=MANIFEST")
    parser.add_argument("--manifest-list", type=Path, action="append", default=[],
                        help="JSON list of {id,manifest} entries; repeat to combine reconstruction methods")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="Atmospheric cross-section response atlas")
    parser.add_argument("--default-dataset")
    parser.add_argument("--embed", action="store_true", help="Embed all payloads; use only for small collections")
    args = parser.parse_args()
    specs = list(args.dataset)
    specs.extend(collection_specs(args.manifest_list))
    require(specs, "Supply --dataset or --manifest-list")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    assets = args.output.with_suffix(".assets")
    if not args.embed:
        assets.mkdir(parents=True, exist_ok=True)
    datasets, audits, names, dimension_combinations = [], [], set(), set()
    for spec in specs:
        require("=" in spec, "Each --dataset needs NAME=MANIFEST")
        name, path = spec.split("=", 1)
        require(name and name not in names, "Dataset names must be nonempty and unique")
        names.add(name)
        data, audit = pack_dataset(name, Path(path))
        require(all(isinstance(data["dimensions"].get(k), str) for k in ("binning","containment","direction","policy")),
                "All four dataset dimensions must be explicit strings")
        combination = tuple(data["dimensions"][k] for k in ("binning","containment","direction","policy"))
        require(combination not in dimension_combinations, f"Duplicate dataset dimensions: {combination}")
        dimension_combinations.add(combination)
        data["source_manifest"] = str(Path(path).resolve())
        if not args.embed:
            filename = hashlib.sha256(name.encode()).hexdigest()[:20] + ".js"
            pages = data.pop("pages")
            script = "globalThis.__xsecRegisterPayload(" + json.dumps(name) + "," + json.dumps(pages,separators=(",",":")) + ");\n"
            target = assets / filename
            target.write_text(script, encoding="utf-8")
            data["payload_script"] = assets.name + "/" + filename
            audit["payload_file"] = str(target.resolve())
            audit["payload_file_bytes"] = target.stat().st_size
            audit["payload_sha256"] = hashlib.sha256(script.encode()).hexdigest()
            del pages, script  # Do not retain a previous large dataset while packing the next.
        datasets.append(data); audits.append(audit)
        print(f"Packed {name}: {audit['canvases']} canvases, {audit['stored_parameter_points']} stored points, {audit['compressed_page_bytes']/2**20:.2f} MiB compressed", flush=True)
    preferred = sorted(datasets, key=lambda d: (
        d["dimensions"]["policy"] != "fitter_policy",
        d["dimensions"]["binning"] != "fine",
        d["dimensions"]["containment"] != "all",
        d["dimensions"]["direction"] != "hybrid"))
    default = args.default_dataset or preferred[0]["id"]
    require(default in names, "Unknown --default-dataset")
    from plotly.offline import get_plotlyjs
    import plotly
    payload = json.dumps({"datasets": datasets, "default_dataset": default}, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    payload = payload.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    js = get_plotlyjs().replace("</script", "<\\/script")
    page = TEMPLATE.replace("__TITLE__", html.escape(args.title)).replace("__DATA__", payload).replace("__PLOTLY__", js)
    args.output.write_text(page, encoding="utf-8")
    summary = {"output": str(args.output.resolve()), "bytes": args.output.stat().st_size,
               "sha256": hashlib.sha256(page.encode()).hexdigest(), "plotly_version": plotly.__version__,
               "offline": True, "payload_mode": "embedded" if args.embed else "local_script_sidecars",
               "lazy_page_decoding": True, "maximum_decoded_page_cache": 3,
               "maximum_compressed_dataset_cache": None if args.embed else 1,
               "datasets": audits, "canvas_order": [ORIGINAL_ORDER,KINEMATICS_ORDER],
               "shape_nan_policy": "Any missing histogram bin makes the whole normalized curve unavailable",
               "shape_test_scope": "Paired marginal-shape test; does not identify angular nuclear physics",
               "signed_density_note": "Negative normalized bins are signed densities, not probabilities"}
    args.output.with_suffix(".audit.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(f"Wrote {args.output} ({summary['bytes']/2**20:.2f} MiB HTML), fully offline", flush=True)


if __name__ == "__main__":
    main()
