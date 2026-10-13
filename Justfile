default:
    @just --list

format:
    ruff format .
    ruff check --fix .

lint:
    ruff check .
    ruff format --check .

test:
    pytest

check: lint test
    python -B -m compileall -q src

fetch-inputs:
    python -m src.fetch_inputs

analyze:
    python -m src.analysis

validation-table:
    python -m src.tables.validation

validation-table-holdout2024:
    python -m src.tables.validation_holdout2024

raster-validation:
    python -m src.raster_validation

raster-validation-sentinel1:
    python -c "from src import raster_validation; raster_validation.run_sentinel1()"

elevation-bands:
    python -m src.elevation

raster-validation-zones:
    python -c "from src import raster_validation; raster_validation.zone_validation()"

validation-table-raster: raster-validation
    python -m src.tables.validation_raster

trend-map:
    python -m src.figures.trend_map

timeseries-figure:
    python -m src.figures.timeseries

per-surge-figures:
    python -m src.figures.per_surge

sentinel-examples-figure:
    python -m src.figures.sentinel_examples

publication:
    python -m src.publication

figures: trend-map timeseries-figure per-surge-figures

all: analyze validation-table validation-table-raster figures

labels:
    python -c "from src import label_audit; label_audit.read_labels(force_redo='self')"

label-audit:
    python -m src.label_audit

label-audit-map:
    python -m src.figures.label_map

typecheck:
    pyrefly check
