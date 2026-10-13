"""Render LaTeX tables from the offline raster validation.

Two tables come out of `src.raster_validation`'s CSV:

``validation-raster-holdout-2024.tex``
    Per-model metrics on the 2024 temporal holdout, over glacierized area
    only. Same layout as `validation.py`'s table, so the two can sit side
    by side -- but see `src.raster_validation`'s module docstring for why
    the numbers are not directly comparable.

``validation-raster-predictors.tex``
    The embedding run against the Sentinel-1 comparison run on the same
    2024 holdout, same rasters and same domain. This is the only
    like-for-like comparison of the two predictors.

``validation-raster-strata.tex``
    Recall by crevasse size class. Only recall and specificity are shown:
    every stratum is scored against the same negatives, so precision and
    F1 differ between strata for reasons that have nothing to do with
    model quality.
"""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import Final

import pandas as pd

from src import raster_validation, settings

from . import validation

HOLDOUT_YEAR: Final = 2024
PUBLISHED_DOMAIN: Final = "glacier"

STRATUM_LABELS: Final = {
    "all": "All positives",
    "big_crevasses": "Large, clear fields",
    "small_crevasses": "Small or ambiguous",
}


def select(
    rows: pd.DataFrame, year: int = HOLDOUT_YEAR, domain: str = PUBLISHED_DOMAIN, stratum: str = "all"
) -> pd.DataFrame:
    """Take the one-row-per-model slice matching a year, domain and stratum."""
    selected = rows[(rows.year == year) & (rows.domain == domain) & (rows.stratum == stratum)]
    if len(selected) != settings.N_MODELS:
        raise ValueError(f"Expected {settings.N_MODELS} rows for {year}/{domain}/{stratum}, found {len(selected)}")
    return selected.sort_values("model_id")


def render_strata_table(rows: pd.DataFrame, year: int = HOLDOUT_YEAR, domain: str = PUBLISHED_DOMAIN) -> str:
    """Render the size-class comparison as ensemble mean plus standard deviation."""
    lines = [
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Positive class & Recall & Specificity \\",
        r"\midrule",
    ]

    for stratum, label in STRATUM_LABELS.items():
        selected = select(rows, year=year, domain=domain, stratum=stratum)
        cells = [
            validation.format_average(selected[column].tolist())
            for column in ("recall_crevasse", "specificity_noncrevasse")
        ]
        lines.append(" & ".join([validation.latex_escape(label), *cells]) + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


PREDICTOR_METRICS: Final = (
    ("accuracy", "Accuracy"),
    ("f1_crevasse", "F1"),
    ("precision_crevasse", "Precision"),
    ("recall_crevasse", "Recall"),
    ("specificity_noncrevasse", "Specificity"),
    ("false_positive_rate", "FPR"),
)


def render_predictor_table(embedding: pd.DataFrame, sentinel1: pd.DataFrame) -> str:
    """Render the embedding/Sentinel-1 comparison as ensemble mean plus spread.

    Both columns come from the same protocol, so unlike the Earth Engine
    tables these numbers may be compared directly.
    """
    header = " & ".join(["Predictor", *(label for _, label in PREDICTOR_METRICS)])
    lines = [
        r"\begin{tabular}{l*{6}{c}}",
        r"\toprule",
        header + r" \\",
        r"\midrule",
    ]
    for label, rows in (("AlphaEarth embeddings", embedding), ("Sentinel-1 backscatter", sentinel1)):
        cells = [validation.format_average(rows[column].tolist()) for column, _ in PREDICTOR_METRICS]
        lines.append(" & ".join([validation.latex_escape(label), *cells]) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def run(
    input_path: Path = raster_validation.RASTER_VALIDATION_FILE,
    table_dir: Path = settings.TABLE_DIR,
    sentinel1_path: Path = raster_validation.SENTINEL1_VALIDATION_FILE,
) -> pd.DataFrame:
    """Write both raster-validation tables and return the rows behind them."""
    if not input_path.exists():
        raise FileNotFoundError(f"{input_path} not found; run `just raster-validation` first")
    rows = pd.read_csv(input_path)

    table_dir.mkdir(parents=True, exist_ok=True)
    holdout = select(rows)
    (table_dir / "validation-raster-holdout-2024.tex").write_text(validation.render_table(holdout.to_dict("records")))
    (table_dir / "validation-raster-strata.tex").write_text(render_strata_table(rows))
    if sentinel1_path.exists():
        sentinel1 = select(pd.read_csv(sentinel1_path))
        (table_dir / "validation-raster-predictors.tex").write_text(render_predictor_table(holdout, sentinel1))
    return holdout


if __name__ == "__main__":
    written = run()
    mean_recall = statistics.fmean(written.recall_crevasse) * 100
    print(
        f"Wrote raster-validation tables for {HOLDOUT_YEAR} ({PUBLISHED_DOMAIN} domain); "
        f"ensemble recall {mean_recall:.1f}%"
    )


def test_stratum_labels_match_the_analysis() -> None:
    assert set(STRATUM_LABELS) == set(raster_validation.STRATA)
