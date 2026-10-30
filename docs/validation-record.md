# Validation record — 30 October 2026

The full 15 km production-height aggregation, its saved-output audit and the
delivered app's automated browser checks passed. Browser checks establish
presentation/export behavior within their tested cases, not upstream physics.

The reviewed run contains **145 files, 9,743,963 selected events, 101 response
slots, 623 stored points, 13 observables and 10 datasets**. Its configuration
SHA256 is `f008c09976e6834ac73d70f6221e1536ce34c65a15c3728f19d055c0b1e4786d`.
Geometry is R=6371 km, height=15 km and depth=0 km; the height change reused the
raw cache without rereading ROOT. FC/PC totals are 7,512,063 and 2,231,900.

## Numerical checks and their scope

| Check | Result | Scope and evidence |
|---|---|---|
| Unit suite | 45 tests and 86 subtests passed | Main run's console result: `45 passed, 86 subtests passed in 3.27s`. No separate pytest log was saved. Subsequent changes were documentation/configuration notes. Reproduce with `python -m pytest`. |
| Independent dense 15 km pilot | Passed | One local source cache, 67,633 selected events and slots 0, 1, 2. The installed-package pilot matched direct event calculations across all 10 datasets and 13 observables. Report: `pilot_dense_validation_h15.json`. |
| Targeted reference/response pilot | Passed | Eight slots from the 101-slot pilot: 0, 15, 60, 61, 62, 78, 91, 92. Covers absent/nonzero CVs, coupled scenarios and malformed responses. Report: `pilot_dense_validation_all101_targets.json`. This is eight independently checked targets, not 101 dense dial checks; its report does not record geometry. The separate 15 km pilot validates the final geometry. |
| Full saved aggregates | Passed | All 101 slots and all 623 points in the full run. Report: `saved_full_aggregate_validation.json`; it records complete production and the configuration/geometry above. |
| Full-app browser QA | Passed | The actual delivered app recorded 52 browser states across 10 datasets/five truth channels and both canvases, plus 10 special-dial/policy cases. No browser errors or external requests; exported ratios match saved arrays and geometry is 15 km. Reports: `validation/browser/validation.json` and `validation/special_dials/special_dials_validation.json`. |

The 15 km dense pilot checked 9,612 histogram vectors, 828 profile vectors,
828 flatness comparisons, 1,740 baseline vectors and 2,052 FC+PC closure vectors.
Its maximum absolute histogram difference was `5.65e-13`; weighted profile
widths agreed exactly. The eight-target pilot checked 27,768 histogram vectors
and 2,392 profile/flatness vectors; its maximum histogram difference was
`1.43e-12`, with exact profile agreement.

The full saved-output validator checked 332,682 histogram vectors, 28,658 profile
vectors, 20,300 baseline references, 1,010 actual-CV metadata cases, 58,580 exact
curve-edge cases, 71,022 FC+PC histogram closures, 14,329 raw missing-support
cases and 3,738 raw-response count cases. It visited 65,650 curve records;
7,070 are explicitly unavailable combinations, not silently filled results.
This audit uses saved aggregates only; it is distinct from independent dense
event recomputation. Reports are retained with the delivered run/provenance,
outside the portable source repository.

The special browser cases cover d061, d062, d078, d091 and d092 in handled and
raw modes, including nonzero references and the difference between unity
replacement and undefined raw support. These are targeted UI checks; every
curve is covered by the saved-array validator, not by exhaustive browser clicks.
The app has two selectable canvases, one visible at a time; review PDFs provide
the corresponding two pages for each selection/dial comparison.

Reproduce the two integration checks with
[`tests/integration/validate_pilot.py`](../tests/integration/validate_pilot.py)
and [`tests/integration/validate_saved_aggregates.py`](../tests/integration/validate_saved_aggregates.py).
Their docstrings state the supported production and binning assumptions.

## Integrity and scientific limits

Bounded extraction completed all 145 files, retaining 11,979,891 broad-cache
rows. All 15,424,615 generator-weight row-alignment checks passed; headers,
particle offsets and response offsets closed, and source fingerprints remained
unchanged during reads. The original pilot's 101 raw dial arrays agreed exactly
with the previous cache on shared rows. Resuming the packaged extractor also
checks input-metadata hashes and full local-cache SHA256, including same-size
corruption and orphan-sidecar regression tests.

Validation does not establish the physical intention of every supplied dial.
For example, the full response audit observes NC changes from
`EnergyDependence_CCMEC`, and `FracDelta_CCMEC` has 143,230 nonfinite responses
at every stored point with no finite nonunity responses. The handled policy
therefore makes that latter response flat by replacement; raw diagnostics expose
the missing support. See [the source audit](source-audit.txt).

These are truth-channel diagnostics, not CVN fitter samples. Combined tuned CV
normalization, absolute exposure, exact production build and the final paper
figure generator remain unverified. L/E is a declared geometric proxy. See the
[analysis guide](atmospheric-response.md) for selections, estimators and statistical
limitations. Numerical agreement alone does not remove those limitations.
