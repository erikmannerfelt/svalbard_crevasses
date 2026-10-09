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

analyze:
    python -m src.analysis

validation-table:
    python -m src.tables.validation

validation-table-holdout2024:
    python -m src.tables.validation_holdout2024

trend-map:
    python -m src.figures.trend_map

timeseries-figure:
    python -m src.figures.timeseries

per-surge-figures:
    python -m src.figures.per_surge

sentinel-examples-figure:
    python -m src.figures.sentinel_examples

figures: trend-map timeseries-figure per-surge-figures

all: analyze validation-table figures

label-audit:
    python -m src.label_audit

label-audit-map:
    python -m src.figures.label_map

typecheck:
    pyrefly check
