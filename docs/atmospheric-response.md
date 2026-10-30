# Atmospheric cross-section response explorer

This package compares every **stored** dial point in the October 2026 atmospheric
flat-CAF production. It builds two selectable canvases with 13 observables,
five truth-channel selections, FC/PC containment and configurable physical bins.
The HTML works offline. It reads aggregated results, never the ROOT files.

This is a diagnostic preparation study, not an oscillation fit. The samples are
truth nue CC, numu CC, NC, nuebar CC and numubar CC. They are **not** the four
CVN-selected MaCh3 fitter samples. Absolute exposure and the combined tuned CV
normalization have not been established for this production.

## Install and start with one file

Use Python 3.10 or later on a filesystem supporting virtual environments:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest
```

Make a small inventory JSON. File sizes are optional in the supplied inventory;
the extractor records actual size and modification time before reading.
Relative source directories resolve beside the inventory JSON, and recorded source/cache paths are absolute.

```json
{
  "directory": "/path/to/AtmInputs_cafnusyst_flat_zstd/v1",
  "entries": [{"name": "first-file.root"}, {"name": "second-file.root"}]
}
```

Use the actual filenames. Do not join unrelated CAF and response productions by
row number. The extractor requires same-file CAF and weight trees, checks tree
lengths, compares generator weights, and checks every response header and ragged
offset. A mismatch stops the run. Historical `weights.run` is unfilled and is
not used as an event identifier.

```bash
atmo-extract --inventory inventory.json --cache-dir cache/v2 --limit-files 1
atmo-aggregate --index cache/v2/scan_index.json \
  --config atmospheric_response/configs/analysis_v2.yaml \
  --work-dir work/pilot --output-dir outputs/pilot --limit-files 1
atmo-viewer --manifest-list outputs/pilot/manifest_list.json \
  --output outputs/pilot/explorer.html
```

The bundled header and CV metadata describe **101 response slots and 623 stored
points**. The guard deliberately rejects another production until its metadata
and central-value conventions are reviewed. Parameter IDs are not array offsets:
always use the actual slot order. Missing sparse rows mean unit response, not
missing events. Pilot outputs are prominently labelled.

After inspecting the pilot, remove `--limit-files` from extraction and
aggregation. Extraction supports one to three workers and resumes only if its
code, source fingerprints and cache sizes agree. It reads only named branches,
with 25,000-event chunks, a 16 MiB single-read ceiling and a 256 MiB logical-read
budget per file. It does not map or copy complete ROOT files. Aggregation opens
only local NPZ caches, processes one dial at a time, and uses local memory maps.
Keep `explorer.html` beside its `explorer.assets` directory when sharing it.

## Controls and static export

The default is the source-backed **MaCh3 numerical policy**. The raw option is a
diagnostic view with explicit missing/undefined values. Choose a truth channel,
dial, containment, binning and rate or unit-area display. Equal-magnitude positive
and negative displacements share colors with different line styles; the actual
stored CV is black and solid. A missing CV uses an explicitly labelled unity
reference, and the coupled MEC scenarios are labelled as scenarios, not sigmas.

The first canvas retains the requested order: true/reco lepton energy,
true/reco hadronic energy, true/reco zenith cosine, signed angular 68% half-width,
and notes. The second contains true/reco neutrino energy, y and L/E. CSV export
retains physical coordinates, reference yields, both zenith ratios and paired
shape-test information. Browser print exports the current canvas. For PDFs:

```bash
atmo-render outputs/pilot/fine_fc_fitter_policy/manifest.json \
  --output-dir outputs/review --selections nue_cc --dials d031 \
  --modes rate --canvas all --individual-formats png pdf
```

Use dataset paths from `manifest_list.json`; it is the authoritative list.
The renderer can export every stored dial and selection by omitting those
filters, but this makes a large book. It never needs to reread the CAF.

## Change bins without rereading ROOT

Copy `atmospheric_response/configs/analysis_v2.yaml` and change an axis to an
explicit list, or use `{spacing: log, min: 0.1, max: 100, bins: 80}`. Linear axes
use `spacing: linear`. Edges must be finite and strictly increasing; a log
specification cannot start at zero. The supplied Miles samples retain their
**exact zero and 100 GeV edges**. A symlog display keeps the zero-edge bin visible;
it never substitutes an arbitrary positive epsilon. Bin sums and densities use
the original physical widths, not transformed display widths.

Miles provides separate FC/PC electron/muon bins. Both charge signs use the same
flavour bins. True and reco counterparts use matching edges for comparison.
There is no supplied Miles NC or combined-FC/PC binning; those combinations are
explicitly unavailable. The fine preset provides all five truth channels and
ALL, FC and PC views. Other observables inherit the fine axes.

```bash
atmo-bins --config my_binning.yaml > effective_binning.json
atmo-aggregate --index cache/v2/scan_index.json --config my_binning.yaml \
  --work-dir work/my_binning --output-dir outputs/my_binning
atmo-viewer --manifest-list outputs/my_binning/manifest_list.json \
  --output outputs/my_binning/explorer.html
```

Every histogram is filled directly from cached event observables with the
requested edges. It is not rebinned or interpolated from the earlier fine plot.
Underflow/overflow and per-observable masks are retained in diagnostics. The
last bin includes the final edge. Weighted quantiles are recalculated from
individual events, including a distinct ALL-containment calculation.

## Selection, reconstruction and weights

The default common selection is one MC and one Pandora interaction, a supported
truth category, positive finite `xsec * matching-flavour Honda flux`, true energy
0.1–100 GeV, the configured vertex fiducial volume, a valid containment minimum,
and a finite nonzero true/reco direction. Thus the plotted truth spectra also
carry reconstruction/containment acceptance. These common cuts are configurable;
each observable additionally has its own finite-value mask. The signed angular
profile retains its earlier 0.1–10 GeV range unless its YAML axis is changed.

For containment, exclude the unassociated-hit object (`origRecoObjType == 3`)
and take the minimum particle `walldist` in cm. FC means minimum **>10 cm**;
PC means **0–10 cm inclusive**. Missing, negative or >10000 cm minima are unknown.
The source loop ignores a NaN/+infinite particle wall if another particle gives
a valid minimum. This is a spacepoint-to-wall quantity, not vertex distance.

The new hybrid direction follows Pierre's supplied update literally: use
`dir.calo` for fewer than two non-hit particles, or when `!(Enu.calo < 1.3)`;
otherwise use `dir.heshw` for electron truth channels and `dir.lngtrk` for muon
truth channels. NC always uses `dir.calo`. Normalize all three components and
take `cosZ = -d.y / |d|`. A horizontal direction is valid. The exact final paper
figure generator was not recovered; the paper's hybrid-method text supports
NC hits-only, while the new switching rule comes from Pierre's message.

Default reconstructed energy follows the checked Pierre/Dan/Miles code: total
`Enu.e_calo` for electron channels and `Enu.lep_calo` for muon channels; hadronic
`Enu.e_had` for both. Lepton energy is total minus hadronic. Reco y is hadronic
over total when total >0, otherwise 0; finite values outside [0,1] are retained
in flow accounting. This documented muon pairing is a fitter convention, not a
claim that it is a matched range/hadronic estimator. The optional `matched_range`
choice uses `mu_range/mu_had` with its explicitly different validity mask.
NC calorimetric energy is a **visible-energy proxy**; charged-lepton energy and
reco y are unavailable. True hadronic energy is the transfer `q0`; true charged
lepton energy is `E-q0`. True y uses stored `rec.mc.nu.inelasticity`, with `q0/E`
checked separately rather than silently substituted.

Nominal event weight is `xsec * flux_nue` for |PDG|=12, or `xsec * flux_numu` for
|PDG|=14. Tau-generated events have zero unoscillated atmospheric source flux
in this study. No oscillation probability or extra GENIE factor is multiplied.
An Earth-density dial acts through propagation and is not an additional
independent unoscillated rate factor. Preserve the supplied cross-section
normalization: do not divide by the number of files. `xsec_cvwgt` is nonfinite in
this production, so a combined tuned CV is not established. Shared central
factors must not be multiplied once per dial; these are **one-dial comparisons**.

The verified numerical policy is:

```python
response = np.where(np.isfinite(raw_response), raw_response, 1.0)
response = np.clip(response, -9999.0, 9999.0)
event_weight = positive_nominal_weight * response
event_weight = np.where(event_weight > 0, event_weight, 0.0)
```

The first two steps match the checked CAF knot loader and its configured
default bounds. The last matches the core nonpositive-total-weight cut for
this single-dial, positive-baseline study. **Nonfinite knots become 1, not 0.**
This is not a claim that arbitrary final NaNs are repaired by MaCh3. Here the
nominal baseline is validated before aggregation. Malformed lengths and offsets
raise errors. Raw negatives invalidate probability quantiles, and raw nonfinite
responses invalidate touched histogram bins. Counts are retained by dial point,
truth channel and containment. Bounds can differ in another fitter configuration.

## What L/E means here

The CAF's stored baseline is in metres but is unfilled in the inspected
production. The configured plots therefore explicitly use a fixed-height
geometric proxy, using true angle/energy for the truth proxy and reconstructed
angle/energy for the reconstruction proxy. With down-going cosZ=+1:

```python
r = earth_radius_km - detector_depth_km
L_km = np.sqrt((r*cosZ)**2 + (earth_radius_km + height_km)**2 - r*r) - r*cosZ
L_over_E = L_km / energy_GeV
```

Defaults are R=6371 km, height=15 km (the requested effective production altitude)
and depth=0 km. The implementation uses a
stable equivalent expression down-going. This is not a recovered production
vertex, a surveyed detector depth, or the fitter's production-height average.
At depth zero this is exactly the supplied slide's total-path formula,
`sqrt((R+h)^2 - R^2*sin(pi-theta_z)^2) + R*cos(pi-theta_z)`, including the
atmospheric segment. Its downward, horizontal and upward limits are 15 km,
approximately 437 km and 12,757 km respectively for the default geometry.
Set `true_baseline_mode: stored` to leave true L/E unavailable when stored values
are absent. Never label the geometric curve as an exact stored true baseline.

## Angular width and paired shape test

The seventh panel is `(Q84-Q16)/2` of the **signed zenith-angle residual** in
each true-energy bin, in degrees. It is neither a mean nor a median nor a 3D
opening angle. It uses event-level weighted inverted-CDF quantiles:

```python
delta = np.degrees(np.arccos(reco_unit_y) - np.arccos(true_unit_y))
keep = selection & (Etrue >= low_edge) & (Etrue < high_edge)
# Include Etrue == high_edge for the last displayed bin.
q16, q84 = np.quantile(delta[keep], [0.16, 0.84],
                       weights=event_weight[keep], method="inverted_cdf")
width68 = (q84 - q16) / 2
```

NumPy >=2 is required for that reference calculation. The sparse implementation
reuses the sorted residuals and gives the same empirical quantile convention;
it does not compute a quantile from a coarse 2D histogram or average FC/PC widths.
Profile centers are geometric for positive edges and arithmetic if a zero edge
is configured. `atmo-bins` prints both edges and centers.

For each true-zenith bin, A is the varied weighted yield and B is the actual
reference yield. The rate ratio is A/B. The shape ratio is
`(A/sum(A)) / (B/sum(B))`, equivalent to rescaling A to the reference yield first.
Normalization uses visible bins only. The two ratios remain distinct even when
the spectrum display is switched to unit area.

The tests account for the fact that A and B use **the same MC events**. They
accumulate paired weight moments and include the fitted common normalization
in the covariance; treating A and B as independent histograms would be wrong.
Low effective counts, signed raw weights, nonfinite weights and zero-rank exact
agreement receive explicit statuses, not manufactured p-values. The additional
coarse true-energy-stratified diagnostic normalizes within each energy stratum.
Its exact grid is recorded, and it cannot remove all within-stratum kinematic
correlations. See [the derivation](paired-shape-test.txt).

Spherical nuclear symmetry does **not** require a flat marginal zenith response:
energy-dependent flux, interactions and selection correlate energy with zenith.
A significant result can reflect those correlations. Dial points themselves are
deterministic; statistical uncertainty here is finite weighted MC sampling.
P-values are asymptotic, unadjusted for the many correlated dial tests, and are
diagnostics rather than a proof of spurious angular physics.

## Reproducibility and extending the code

Each run saves the YAML, its SHA256, input source fingerprints, selection audit,
all bin sums and squared-weight sums including flows, quantile statistics,
paired test moments, raw-response quality counts, and a completion record.
The app embeds the calculation metadata. Source decisions and pinned URLs are
in [source-audit.txt](source-audit.txt). The exact producer version is not fully
pinned merely because the observed header matches the candidate FCL.

Module boundaries make changes reviewable: `v2_extract` handles bounded I/O;
`v2_physics` derives unweighted values and masks; `v2_config` resolves bins;
`sparse_response_aggregation` fills rates/quantiles; `v2_moments` and
`v2_statistics` calculate paired uncertainties; `v2_aggregate` assembles the
study; the viewer and renderer only consume aggregates. Unit tests use small
arrays and temporary byte files, without any remote CAF reads. Browser test
fixtures are explicitly synthetic and must never be shipped as scientific
results. Large generated products and machine-specific paths are not committed.

The original notebooks and `systs.py` remain for their historical input schema.
Their older hybrid helper is **not the implementation used here**: its NC
direction alias loads heshw, its particle threshold differs, and it rejects
horizontal directions. Use this package for the updated flat-CAF analysis;
do not combine those legacy definitions with these outputs.

For the reviewed 101-dial production, the saved-output validator checks every
curve, central-value reference, exact edge, FC+PC histogram closure and raw
missing-value support using only the aggregates:

```bash
python tests/integration/validate_saved_aggregates.py \
  --output outputs/full --report outputs/full/validation.json \
  --expected-production-height-km 15
```

For an explicitly one-file pilot, compare the sparse calculation against direct
event-by-event histograms and quantiles without opening ROOT:

```bash
python tests/integration/validate_pilot.py --output outputs/pilot \
  --dials 0 15 60 61 62 78 91 92 --report outputs/pilot/dense_validation.json
```

The pilot must contain the requested dials. These integration tools target the
reviewed production and default fine/Miles collection; their docstrings state
the limits. Browser tests are optional and require Playwright/Chromium. Their
scripts live in `tests/browser/`, with separate synthetic and actual-data checks.
