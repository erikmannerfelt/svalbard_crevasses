# Repository Instructions

## Project

This repository analyzes annual Google Earth Engine predictions of *visible*
crevasse extent for Svalbard glaciers. It combines 30-model raster ensembles
with RGI7 glacier geometry and the Svalbard surge database described by
Harcourt et al. (2026).

**Framing:** the model predicts detectable/visible crevassing, not total
structural crevassing. Snow and firn bridging hide real crevasses without
erasing them, so predicted area and its trend reflect both genuine
structural change and year-to-year exposure/snow conditions -- never treat a
predicted-area change as direct proof of new crevasse formation. A second
gap is timing: labels are digitized from late-summer scenes, when crevasses
are most exposed, while the embedding is an annual composite, so a correctly
labeled polygon can carry a snow/firn-dominated signature. This is a likely
source of false positives in firn-covered terrain. A Svalbard-wide low-snow
year inflates predicted area on surging and non-surging glaciers alike,
which is the main confound for the surge-period comparison. Use "visible crevasse(s)"/"visible crevasse extent" in
new text describing the predicted or trend quantity; "crevasse" alone is
still fine for the training labels themselves, glacier/zone names, and
component names (e.g. `svalbard_crevasse_detection.js`).

The project is offline-first. Source data must already be present in `input/`.
Never download data or add a network-dependent step to the workflow.

## Workflow

The user-facing workflow is the root `Justfile`:

```text
just all
just analyze
just validation-table
just validation-table-holdout2024
just raster-validation
just validation-table-raster
just figures
just check
just format
just typecheck
just label-audit
just label-audit-map
```

Python modules intentionally do not parse command-line arguments. Change fixed
paths, years, thresholds, selected glaciers, and figure settings in
`src/settings.py` instead.

`just all` runs the analysis, both validation tables, and all figure recipes
in that order. The analysis must run before the validation table or figures because
they consume its generated artifacts.

Lightweight checks that do not require the input bundle are:

```text
ruff check src
ruff format --check src
pytest -q
python -B -m compileall -q src
```

Pyrefly is configured as a future type-checking step but may not be installed.
Do not install dependencies or make large downloads during routine work.

## Source Layout

- `src/settings.py` contains repository-relative paths and fixed scientific and
  presentation choices.
- `src/surge.py` reads the RGI surge archive, applies documented overrides, and
  classifies surge intervals.
- `src/raster.py` validates source rasters, creates vote-fraction rasters, and
  aggregates model areas over glacier polygons.
- `src/analysis.py` is the canonical analysis pipeline and writes all data
  products, including the reusable surge-period summary CSV.
- `src/figures/timeseries.py` renders the six-glacier timeseries and
  Arnesenbreen raster panels.
- `src/figures/trend_map.py` renders the trend map and reads the reusable
  surge-period summary rather than recalculating its table values.
- `src/figures/per_surge.py` renders configured per-glacier surge figures.
- `src/tables/validation.py` renders the ensemble validation LaTeX table
  from the in-training-year (`repeated_polygon_holdout`) validation CSV.
- `src/tables/validation_holdout2024.py` renders the same table format for
  the 2024 temporal-holdout validation CSV (2024 is excluded from
  `TRAINING_YEARS`; this checks generalization to a year no model has
  seen). Reuses `validation.py`'s renderer with different paths -- the two
  validations are kept in separate CSVs/tables rather than merged, since
  they answer different questions and `validation.py` assumes one row per
  model.
- `src/raster_validation.py` scores the exported ensemble rasters directly
  against the label polygons, offline; see "Raster Validation" below.
- `src/tables/validation_raster.py` renders its two LaTeX tables, reusing
  `validation.py`'s renderer for the per-model one.
- `src/label_audit.py` audits spatial/temporal coverage of the hand-digitized
  crevasse training labels; see "Label Coverage Audit" below.
- `src/figures/label_map.py` renders the label coverage audit map.
- Focused tests for deterministic analysis and formatting helpers live beside
  their implementations in `src/`; tests must not require the full ignored
  input bundle.

Keep transformations and scientific calculations in analysis/domain modules.
Figure modules should load prepared products and handle presentation. Avoid
new generic framework modules unless repeated behavior clearly warrants one.

## Input Contracts

The expected prediction dataset contains 1,583 glaciers, 30 models, and every
year from 2017 through 2025. Prediction rows are keyed by
`(anlys_id, model_id, year)` and must be unique.

Join predictions to the RGI surge database through `anlys_id`. Verify that
prediction `glac_id` equals RGI `glims_id`; these are independent identity
fields and must not be silently substituted.

The source model rasters are named
`crevasse_model_probabilities_{year}.tif`. Each raster must contain 30 `uint8`
bands named `prob_m00` through `prob_m29`, use `EPSG:32633`, and have 50 m
pixels. A model vote is positive at the configured uint8 threshold of 128.

Derived vote-fraction rasters are named
`crevasse_vote_fraction_{year}.tif`, use one `uint8` band named
`vote_fraction`, and decode as `value / 255`.

RGI geometries must be transformed to `EPSG:32633` before rasterization,
sampling, or centroid calculations.

## Surge Semantics

The manual overrides in `src/surge.py` are provenance data. Preserve them
unless the source interpretation is deliberately revised and documented.

Preserve these interval rules:

- `Ongoing` has no upper bound.
- `Not observed` is treated as ending ten years after onset.
- Semicolon-separated onset and termination entries are paired by position.
- Invalid or non-numeric onset entries do not create an interval.

Use `rgi_id` for glacier identity. Glacier names are not unique in RGI7. The
largest-polygon rule is allowed only for the configured name-based six-glacier
timeseries selection.

## Analysis Products

Canonical data products are written beneath `output/data/`:

- `annual-model.arrow`: one row per glacier, model, and year.
- `glacier-model-trends.arrow`: one trend row per glacier and model.
- `glacier-trends.arrow`: one ensemble summary row per glacier.
- `regional-annual-model.arrow`: annual regional values by model.
- `regional-model-trends.arrow`: regional trend rows by model.
- `regional-trends.arrow`: regional ensemble summaries.
- `representative-glaciers.arrow`: selected positive and negative examples.
- `surge-period-trend-summary.csv`: six rows split by `pre`, `recent`, and
  `none` surge periods and `negative` or `positive` trend signs.
- `surge-magnitude-ratio.csv`: per-model ratio of mean absolute trend of
  significant surging (`recent` or `pre`) to significant non-surging glaciers,
  summarized as median and `ENSEMBLE_QUANTILES` bounds.
- `elevation-band-trends.csv` (from `src/elevation.py`, `just elevation-bands`):
  visible crevasse trend per elevation band for the significant non-surging
  gaining and losing glaciers, in absolute 100 m bands and in tenths of each
  glacier's own elevation range. Elevation is the NPI S0 50 m DTM
  (`input/NP_S0_DTM50.zip`, fetched by `just fetch-inputs`), resampled
  bilinearly onto the prediction grid. The per-band areas are cached in
  `cache/elevation-band-areas.arrow` and sum exactly to `annual-model.arrow`.
- `elevation-zone-trends.csv`: the same split into lower and upper zones
  (`ELEVATION_ZONE_SPLIT_M`, `RELATIVE_ZONE_SPLIT_BAND`), overall and by
  terminus type (RGI7 `term_type` 1 = tidewater, otherwise land), with
  per-model shares of each group's trend.

The surge-period CSV is an analysis result that may be consumed by maps,
tables, or future reports. It contains model rate summaries, uncertainty,
significant-glacier counts, glacier counts, and model counts.

Use atomic writes for expensive canonical Arrow products and return/re-read the
finalized artifact where the code represents a serialized data product. Keep
serialization decisions visible in the owning analysis function.

## Figures

The six-glacier timeseries intentionally combines two data sources. Its chart
uses sampled RGI7-linked annual model data. Its Arnesenbreen raster panels use
the modified annual outlines from `glacier_outlines_mod.sqlite`, including the
explicit fallback outline years in `settings.py`.

The trend map uses the RGI geometry and trend products, with recent and earlier
surge centroids, fixed glacier annotations, a compact colorbar, legend, and
the colored uncertainty matrix. The matrix reads
`surge-period-trend-summary.csv`; do not duplicate its calculations in the
renderer.

Per-surge outputs are written beneath `output/figures/per-surge/` using stable
RGI-based directory names.

Preserve the established figure layout and annotations when refactoring figure
data access. Visual changes should be intentional, not incidental results of
replacing the plotting code.

## Raster Validation

`src/raster_validation.py` is a second, independent validation of the same
models the Earth Engine script validates. It runs entirely offline from
`input/crevasse_model_probabilities_{year}.tif` and costs no GEE quota.

```text
just raster-validation
just validation-table-raster
```

It differs from `src/tables/validation.py` in ways that matter:

- It scores the **published 50 m vote-fraction rasters**, over every
  labelled pixel, at the class ratio the labels imply. The Earth Engine
  validation samples the 10 m embedding at a balanced 50/50 ratio. The two
  sets of numbers describe different things and must not be presented as
  comparable -- state which one every quoted figure comes from.
- It reports two domains. `glacier` restricts scoring to RGI7 glacierized
  area and is the one to publish: it matches the only domain
  `src/analysis.py` reports over. `all_labelled` keeps every labelled pixel
  and exists for comparison. Negative labels spill onto rock, tundra and
  sea, where the ensemble is near-perfect for trivial reasons, so
  `all_labelled` overstates specificity by roughly five points.
- It reports three positive-class strata via the label asset's
  `big_crevasses` flag. **Compare recall across strata, never precision or
  F1**: every stratum is scored against the same pool of negatives, so a
  stratum holding fewer positive pixels shows lower precision for that
  reason alone.

2024 is the temporal holdout and is the honest number; training years are
optimistic because each model saw 70% of those polygons.

`big_crevasses` reaches the audit through `label_audit.read_labels`, which
carries it into the label cache when the export provides it. It is optional
there -- exports predating the flag simply omit it, and `src/label_audit.py`
does not need it -- but `src/raster_validation.py` requires it and says so.

## Training Labels And Earth Engine Scripts

The label chain, from primary to derived:

1. `gee/imports.js`: the digitized polygons and big-crevasse points as Earth
   Engine Imports (`poly_pos_*`, `poly_neg_*`, `big_crevasses_*` per year).
   This is the authoritative record; it is archived here, not read by the
   pipeline.
2. `gee/svalbard_crevasse_labelling.js` tags them (`polyg_id` =
   `year_class_index`, `crevassed`, `big_crevasses` when a positive polygon
   touches one of that year's points) and exports one collection to the asset
   `users/erikmann/crevasse_training_labels`, which
   `gee/svalbard_crevasse_detection.js` trains on, and to Drive as GeoJSON.
3. `shapes/crevasse_training_labels.geojson`: that Drive export, committed. It
   is the primary label file for the offline pipeline. As of 2026-10-13 it
   matches `gee/imports.js` exactly (757 polygons; ids, classes, flags, and
   geometry within 1e-9 degrees; three planar rectangles start at a different
   corner).
4. `cache/crevasse_area_labels.arrow`: derived by `label_audit.read_labels`
   on first use, or rebuilt with `just labels` after a new export.

After re-digitizing in Earth Engine, re-run the labelling export, replace the
GeoJSON, update `gee/imports.js`, and run `just labels`.

`gee/svalbard_crevasse_detection.js` is the authoritative source for the
method (random forest settings, sampling, export grid). It runs in Earth
Engine, not in this pipeline.

## Label Coverage Audit

`src/label_audit.py` and `src/figures/label_map.py` audit the spatial and
temporal coverage of the hand-digitized crevasse training labels in
`shapes/crevasse_training_labels.geojson` (separate from the RGI/surge trend
pipeline above). Run via:

```text
just label-audit
just label-audit-map
```

Outputs: `output/data/label-zone-year-summary.csv` (one row per named zone x
year with positive/negative counts, `has_labels`, `single_class_only`) and
`output/figures/label-audit-map.png`/`.pdf` (a year-faceted map of zone
outlines with label locations colored by class).

Notes for interpreting and re-running this audit:

- A meaningful fraction of raw labels (131 of 508 as of 2026-08-24) are known,
  expected remnants of an earlier pan-Arctic labeling attempt, sitting far
  outside Svalbard. `label_audit.filter_to_svalbard` drops them using a
  bounding box matching `gee/svalbard_crevasse_detection.js`'s
  `SVALBARD_BOUNDS`. This is intentional filtering, not a bug to fix at the
  source.
- A few remaining labels (e.g. on Hopen) legitimately fall outside all 9
  zones in `shapes/glacier_zones.geojson`, which only covers major glaciated
  regions. `label_audit.join_zones` reports and drops these; it is not an
  error.
- `single_class_only` is descriptive, not automatically a defect. Some zones
  (e.g. C Spitsbergen) have very few real crevasses, so an all-negative cell
  there can be a correct reflection of nature. The same pattern in a
  known-crevassed zone (N/NE/NW Spitsbergen, Austfonna) is more likely a real
  annotation gap. Judge case by case against the map figure.
- The target training-year set is every-other-year: 2017, 2019, 2021, 2023,
  2025. 2024 is an intentional holdout and should not be scored for
  labeling-gap severity.
- When asked to prioritize labeling effort from this audit, rank by: (1)
  fully blank zone/year cells, weighted toward known-crevassed zones, (2)
  single-class cells in known-crevassed zones, (3) years with low total label
  volume overall. Report it as a compact ranked table with one-line reasoning
  per year/zone, not a full statistical writeup.

## Generated Files And Safety

Analysis and figure outputs under `output/` are reproducible generated
artifacts and should not be committed. Do not commit Python bytecode such as
`src/**/__pycache__/` or `*.pyc`.

Preserve the `.gitignore` exclusions for `input/`, `output/`, `cache/`,
`crevasses_svalbard.qgz`, and `scripts/` unless artifact policy is intentionally
changed. `scripts/` holds local helpers only; the Earth Engine scripts live in
the tracked `gee/` directory.

Do not delete source data, rewrite Git history, or install dependencies as part
of ordinary analysis or refactoring work.

## Archived Analyses

Three analyses were developed while revising the manuscript and then removed
so that the submitted code does only what the paper uses. They are complete
and tested in commit `0b62b5272ab5` ("Add mass-change, elevation and supporting
analyses") and can be restored from there if a reviewer asks, e.g.
`git show 0b62b5272ab5:src/mass_balance.py`.

- **Mass-change comparison** (`src/mass_balance.py`, WGMS AMCE release fetched
  by `fetch_inputs`, `settings.MASS_CHANGE_*`). Gaining non-surging glaciers
  lose less mass than losing ones (median -0.70 vs -0.89 m w.e. yr-1), also
  within size classes and in all 30 models, but the relationship is weak
  (partial r = 0.16). It cannot separate exposure from new crevassing in
  accumulation areas, which is why the elevation analysis replaced it. RGI6
  series are matched to RGI7 by centroid; values are multi-year rates, not
  observed annual balance.
- **Common-mode decomposition** (`analysis.common_mode_factors`,
  `common_mode_model_factors`, `common_mode_summary`). A Svalbard-wide year
  factor, the median of each glacier-model timeseries divided by its own mean.
  It is not identifiable as exposure. Recomputed without the gaining glaciers
  it has no trend, so the rise it showed came from those glaciers themselves.
- **Year-to-year sensitivity per elevation band**
  (`elevation.band_year_sensitivity`). Detrended band anomalies against the
  group-excluded year factor: 300-800 m bands on gaining glaciers follow it
  nearly in step (r 0.72-0.83, swing 15-19%), low bands do not.
