"""Generate a LaTeX table for ensemble validation performance."""

from __future__ import annotations

import csv
import statistics
from pathlib import Path

from src import settings

METRICS = (
    ("accuracy", "Accuracy"),
    ("f1_crevasse", "F1"),
    ("false_negative_rate", "False negative rate"),
    ("false_positive_rate", "False positive rate"),
    ("precision_crevasse", "Precision"),
    ("recall_crevasse", "Recall"),
    ("specificity_noncrevasse", "Specificity"),
)


def latex_escape(value: str) -> str:
    """Escape the LaTeX characters used in table text."""
    return value.replace("\\", r"\textbackslash{}").replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")


def format_percentage(value: float) -> str:
    return f"{value * 100:.0f}\\%"


def format_average(values: list[float]) -> str:
    mean = format_percentage(statistics.fmean(values))
    deviation = format_percentage(statistics.stdev(values))
    return f"{mean} $\\pm$ {deviation}"


def render_table(rows: list[dict[str, str]]) -> str:
    """Render validation rows and a mean-plus-standard-deviation row."""
    lines = [
        r"\begin{tabular}{c*{7}{c}}",
        r"\toprule",
        r"Model & Accuracy & F1 & FNR & FPR & Precision & Recall & Specificity \\",
        r"\midrule",
    ]

    for row in rows:
        values = [str(int(row["model_id"]))]
        values.extend(format_percentage(float(row[column])) for column, _ in METRICS)
        lines.append(" & ".join(values) + r" \\")

    averages = [r"\textbf{Average}"]
    averages.extend(r"\textbf{" + format_average([float(row[column]) for row in rows]) + r"}" for column, _ in METRICS)
    lines.extend(
        [
            r"\midrule",
            " & ".join(averages) + r" \\",
            r"\bottomrule",
            r"\end{tabular}",
            "",
        ]
    )
    return "\n".join(lines)


def select_stratum(rows: list[dict[str, str]], stratum: str) -> list[dict[str, str]]:
    """Keep one row per model, dropping the positive-class strata.

    Runs of `svalbard_crevasse_detection.js` from 2026-08-27 onward emit
    three rows per model -- `all`, `big_crevasses`, `small_crevasses` --
    sharing one sampled test set. The renderer assumes one row per model,
    so a stratum has to be chosen before rendering. CSVs produced before
    that change have no `stratum` column and are passed through unchanged.
    """
    if not rows or "stratum" not in rows[0]:
        return rows

    selected = [row for row in rows if row["stratum"] == stratum]
    if not selected:
        available = sorted({row["stratum"] for row in rows})
        raise ValueError(f"No rows for stratum {stratum!r}; available: {', '.join(available)}")
    return selected


def run(
    input_path: Path = settings.INPUT_DIR / "crevasse_rf_ensemble_validation_summary.csv",
    output_path: Path = settings.TABLE_DIR / "validation.tex",
    stratum: str = "all",
) -> list[dict[str, str]]:
    """Read validation data and write the corresponding LaTeX table."""
    with input_path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"No validation rows found in {input_path}")

    required_columns = {"model_id", *(column for column, _ in METRICS)}
    missing = required_columns - set(rows[0])
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    rows = select_stratum(rows, stratum)
    rows.sort(key=lambda row: int(row["model_id"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_table(rows))
    return rows


if __name__ == "__main__":
    run()


def test_validation_formatting() -> None:
    assert format_percentage(0.25) == "25\\%"
    assert latex_escape("a_b & c") == r"a\_b \& c"


def test_select_stratum_passes_through_legacy_rows() -> None:
    legacy = [{"model_id": "0", "accuracy": "0.9"}]
    assert select_stratum(legacy, "all") == legacy


def test_select_stratum_filters_and_rejects_unknown() -> None:
    rows = [
        {"model_id": "0", "stratum": "all"},
        {"model_id": "0", "stratum": "big_crevasses"},
    ]
    assert select_stratum(rows, "big_crevasses") == [rows[1]]

    try:
        select_stratum(rows, "nonexistent")
    except ValueError as error:
        assert "available" in str(error)
    else:
        raise AssertionError("expected ValueError for an unknown stratum")
