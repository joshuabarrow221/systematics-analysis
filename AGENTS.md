# Working on the atmospheric response analysis

Read `docs/atmospheric-response.md` and `docs/source-audit.txt` before changing
physics definitions. The new workflow lives in `atmospheric_response/`; the
historical top-level `systs.py` uses an older schema and different angle helper.

- Preserve raw response arrays. Apply numerical handling at aggregation time,
  with separate raw and handled views and explicit quality counts.
- Never infer a CV from array order or assume that every coordinate is a sigma.
  Keep `metadata/dial_cv_map.json` attached to the actual header slot order.
- Keep truth categories, reconstructed CVN samples, detector containment and
  observable validity separate. Do not silently change the selected population.
- Keep production altitude, energy choices, cuts and bin edges in the YAML.
  The requested default altitude is 15 km; L/E uses a labelled geometric proxy.
- Rebin from local cached events. Only the bounded extractor opens ROOT; plot
  builders and renderers consume local aggregates. Do not use whole-file reads
  or mmap on an SSHFS ROOT source.
- Preserve event identity and source fingerprints. A failed alignment or cache
  integrity check is a reason to investigate, not to truncate or reorder arrays.
- Generated ROOT/NPZ/HTML/PDF products belong outside tracked source. Synthetic
  browser fixtures are tests, never substitutes for scientific output.

Run `python -m pytest` after calculation changes. Build with
`python -m build` to check that the YAML and metadata ship with the wheel.
The browser-test scripts document their own offline fixture and real-manifest
commands. Physical bin edges and CSV values must stay unchanged by display-axis
transformations. A zero-edge energy bin uses linear or symlog display.

Source claims need a pinned commit or an explicit distinction between a supplied
author instruction, a checked implementation, and an inference. The exact final
paper figure generator and the production's combined tuned CV remain unverified.
