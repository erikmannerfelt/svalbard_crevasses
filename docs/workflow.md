# Workflow

The workflow is intentionally fixed and offline. Python modules do not parse
command-line arguments. Change paths, years, thresholds, selected glaciers,
or presentation settings in `src/settings.py` and then rerun the relevant
recipe.

## Recipe Order

`just all` runs these stages in order:

1. `just analyze` reads the source bundle, validates the prediction/RGI join,
   aggregates annual model areas, fits trends, and writes canonical data
   products.
2. `just validation-table` reads the validation CSV from `input/` and writes
   `output/tables/validation.tex`.
3. `just figures` renders the trend map, selected six-glacier timeseries, and
   configured per-surge figures from the analysis products.

The individual figure recipes are:

```text
just trend-map
just timeseries-figure
just per-surge-figures
```

The analysis creates the vote-fraction rasters consumed by the figures. Do not
run figure recipes against a missing or stale analysis output when comparing a
changed analysis configuration.

## Settings

`src/settings.py` derives the repository root from the module location. This
means the workflow does not depend on the caller's current working directory.
It contains:

- Input, output, data, raster, figure, and table directories.
- The 2017-2025 study years and 30-model count.
- The expected 1,583-glacier count.
- EPSG:32633, 50 m pixels, and the uint8 vote threshold of 128.
- Fixed source filenames and derived output filename templates.
- The six-glacier selection and RGI name correction.
- Arnesenbreen outline fallback years.
- The pre-surge window and presentation choices.

## Verification

Run the checks that do not require source data with:

```text
just check
```

This runs Ruff, pytest, and non-writing Python compilation. `just typecheck`
runs Pyrefly when it is installed. Routine work should not install packages or
make large network downloads.
