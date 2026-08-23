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

trend-map:
    python -m src.figures.trend_map

timeseries-figure:
    python -m src.figures.timeseries

per-surge-figures:
    python -m src.figures.per_surge

figures: trend-map timeseries-figure per-surge-figures

all: analyze validation-table figures

typecheck:
    pyrefly check
