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


def run(
    input_path: Path = settings.INPUT_DIR / "crevasse_rf_ensemble_validation_summary.csv",
    output_path: Path = settings.TABLE_DIR / "validation.tex",
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

    rows.sort(key=lambda row: int(row["model_id"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_table(rows))
    return rows


if __name__ == "__main__":
    run()


def test_validation_formatting() -> None:
    assert format_percentage(0.25) == "25\\%"
    assert latex_escape("a_b & c") == r"a\_b \& c"
