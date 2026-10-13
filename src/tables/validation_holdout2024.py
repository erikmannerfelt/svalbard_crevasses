"""Render the LaTeX table for the 2024 temporal-holdout validation.

Reuses `validation.py`'s renderer as-is (it already takes
`input_path`/`output_path` and doesn't assume a particular
`validation_type`); only the input/output paths differ from the main
in-training-year `crevasse_rf_ensemble_validation_summary.csv` table. See
`gee/svalbard_crevasse_detection.js`'s `runHoldoutValidation` for how
this CSV is produced.
"""

from __future__ import annotations

from src import settings

from . import validation


def run() -> list[dict[str, str]]:
    """Read the 2024 holdout validation CSV and write its LaTeX table."""
    return validation.run(
        input_path=settings.VALIDATION_HOLDOUT_2024_FILE,
        output_path=settings.TABLE_DIR / "validation-holdout-2024.tex",
    )


if __name__ == "__main__":
    run()
