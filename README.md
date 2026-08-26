# Svalbard Crevasse Analysis

This project analyzes annual predictions of *exposed* crevasse extent for
glaciers in Svalbard. It combines Google Earth Engine raster exports from a
30-model ensemble with RGI7 glacier geometry and surge histories from the
Svalbard surge database.

The analysis asks how predicted exposed-crevasse-covered area changes through
time, and whether those changes differ between glaciers with recent, earlier,
or no recorded surge activity.

## Framing: Exposed Vs. Structural Crevassing

The predicted quantity is *detectable/exposed* crevasse extent, not total
structural crevassing. Snow and firn bridging hide real crevasses without
erasing them, so a predicted-area change can reflect genuine structural
change, year-to-year exposure/snow conditions, or both -- it is not on its
own evidence of new crevasse formation. See
[`docs/data-contracts.md`](docs/data-contracts.md) for why this matters and
how it can confound the surge-period comparison specifically.

## Data And Provenance

The surge database used by this project is described in:

> Harcourt, W. D., Pearce, D. M., Gajek, W., Lovell, H., Mannerfelt, E. S.,
> Kääb, A., Benn, D. I., Luckman, A., Hann, R., Kohler, J., Strozzi, T.,
> McCerery, R., & Davies, B. J. (2026). Surging glaciers in Svalbard:
> Observing their distribution, characteristics and evolution.
> *Earth-Science Reviews*, 275, 105410.
> [https://doi.org/10.1016/j.earscirev.2026.105410](https://doi.org/10.1016/j.earscirev.2026.105410)

The project is offline-first. The required source bundle must already be
available in `input/`. The workflow never downloads data or depends on a
network connection.

Expected source files include:

- `RGI_V7_Surge_Database.zip`
- `glacier_outlines_mod.sqlite`
- `S100_Land_f.gpkg`
- `crevasse_model_probabilities_2017.tif` through
  `crevasse_model_probabilities_2025.tif`
- `crevasse_rf_ensemble_validation_summary.csv`

The raster and tabular input contracts are documented in
[`docs/data-contracts.md`](docs/data-contracts.md).

## Quick Start

The project uses fixed settings rather than runtime command-line arguments.
Review `src/settings.py` before running the workflow. It defines paths, study
years, raster thresholds, selected glaciers, output names, and figure choices.

The complete workflow is:

```text
just all
```

The individual stages are useful when iterating:

```text
just analyze
just validation-table
just trend-map
just timeseries-figure
just per-surge-figures
```

The analysis must run before the validation table or figures because those
stages consume its generated products. See
[`docs/workflow.md`](docs/workflow.md) for the dependency order and checks.

## Repository Layout

```text
src/
  settings.py                 fixed paths and scientific choices
  surge.py                    surge database and interval semantics
  raster.py                   raster validation and area aggregation
  analysis.py                 canonical analysis pipeline
  figures/
    trend_map.py              regional trend map and matrix rendering
    timeseries.py             six-glacier figure and raster panels
    per_surge.py              configured per-glacier figures
  tables/
    validation.py             validation LaTeX table
tests are colocated in modules   focused tests without the input bundle
docs/                          workflow, data, and figure documentation
```

## Generated Products

`just analyze` writes canonical data products below `output/data/`:

- `annual-model.arrow`: annual positive and valid area by glacier, model, and
  year.
- `glacier-model-trends.arrow`: fitted trend statistics for every glacier and
  model.
- `glacier-trends.arrow`: ensemble trend summaries for every glacier.
- `regional-annual-model.arrow`: annual regional values by model and group.
- `regional-model-trends.arrow`: regional trend statistics by model.
- `regional-trends.arrow`: regional ensemble summaries.
- `representative-glaciers.arrow`: selected positive and negative examples.
- `surge-period-trend-summary.csv`: six cells split by surge period and trend
  sign, including rates, uncertainty, and significant-glacier counts.

Derived vote-fraction rasters are written below
`output/rasters/vote-fractions/`. Validation output is written below
`output/tables/`. Figures are written below `output/figures/`, including
`per-surge/` directories named with stable RGI identifiers.

All files beneath `output/` are reproducible generated artifacts and are
ignored by Git.

## Verification

The lightweight checks do not need the full source bundle:

```text
ruff check src tests
ruff format --check src tests
pytest -q
python -B -m compileall -q src
```

Run them together with:

```text
just check
```

Pyrefly has a `just typecheck` recipe for future use, but it is not required
for routine execution and may not be installed in the local environment.
