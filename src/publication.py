"""Assemble the data deposit that accompanies the manuscript.

Everything is derived from existing pipeline outputs and input rasters; this
module computes no new science. It writes ``settings.PUBLICATION_DIR``:

- ``crevasse_vote_fraction_{year}.tif``: the ensemble vote-fraction rasters,
  copied with self-describing GeoTIFF metadata.
- ``crevasse_tree_vote_fractions_2017-2025.zip``: the 30-band per-model source
  rasters, stored uncompressed (they are already LZW-compressed) and renamed so
  that neither product is called a probability.
- ``glacier_trends.csv`` and ``glacier_annual_area.csv``: per-glacier tables.
- ``README.md``.

Uploading the deposit is deliberately not part of this module, since the
repository is offline-first.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import rasterio
import tqdm

from src import settings

TREE_VOTE_ARCHIVE: Final = f"crevasse_tree_vote_fractions_{settings.YEARS[0]}-{settings.YEARS[-1]}.zip"
TREE_VOTE_MEMBER_TEMPLATE: Final = "crevasse_tree_vote_fractions_{year}.tif"
GLACIER_TRENDS_FILE: Final = "glacier_trends.csv"
GLACIER_ANNUAL_AREA_FILE: Final = "glacier_annual_area.csv"
GITHUB_URL: Final = "https://github.com/erikmannerfelt/svalbard_crevasses"
MANUSCRIPT_URL: Final = "https://doi.org/10.5194/egusphere-2026-6305"
SURGE_DATABASE_URL: Final = "https://doi.org/10.1016/j.earscirev.2026.105410"

TREND_COLUMNS: Final = (
    "rgi_id",
    "anlys_id",
    "glac_name",
    "area_km2",
    "reference_valid_area_km2",
    "surging",
    "S_Onset",
    "S_Term",
    "surge_period",
    "slope_km2_yr",
    "slope_km2_yr_low",
    "slope_km2_yr_high",
    "slope_percent_points_yr",
    "slope_percent_points_yr_low",
    "slope_percent_points_yr_high",
    "significant",
    "trend_sign",
    "r2_median",
    "rmse_median",
    "n_models",
    "mean_coverage_percent",
)


def _write_csv(data: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Atomically write a CSV and return the finalized table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.csv")
    try:
        data.to_csv(temporary, index=False)
        os.replace(temporary, path)
        return pd.read_csv(path)
    finally:
        temporary.unlink(missing_ok=True)


def classify_trends(trends: pd.DataFrame) -> pd.DataFrame:
    """Add ``significant`` and a three-way ``trend_sign`` to glacier trends.

    A trend is significant when the `settings.ENSEMBLE_QUANTILES` range of the
    model-wise slopes excludes zero, the same rule the analysis uses. Where most
    models predict no visible crevasses in any year, the median slope is exactly
    zero; the analysis groups those glaciers with positive trends, but the
    published table labels them ``zero`` so the sign is not misread.
    """
    result = trends.copy()
    result["significant"] = (result.slope_km2_yr_low > 0) | (result.slope_km2_yr_high < 0)
    result["trend_sign"] = np.select(
        [result.slope_km2_yr > 0, result.slope_km2_yr < 0], ["positive", "negative"], default="zero"
    )
    return result


def glacier_trends_table() -> pd.DataFrame:
    """Write the per-glacier trend table, one row per RGI7 glacier."""
    trends = pd.read_feather(settings.DATA_DIR / "glacier-trends.arrow")
    table = classify_trends(trends)[list(TREND_COLUMNS)].sort_values("rgi_id")
    return _write_csv(table, settings.PUBLICATION_DIR / GLACIER_TRENDS_FILE)


def glacier_annual_area_table() -> pd.DataFrame:
    """Write annual visible crevasse area per glacier as an ensemble summary.

    Returns
    -------
    pd.DataFrame
        One row per ``(rgi_id, year)`` with the median and
        `settings.ENSEMBLE_QUANTILES` bounds of the 30 model areas (km2).
    """
    annual = pd.read_feather(settings.DATA_DIR / "annual-model.arrow")
    area = annual.groupby(["rgi_id", "anlys_id", "glac_name", "year"], dropna=False).positive_area_km2
    table = pd.concat(
        {
            "visible_crevasse_area_km2": area.median(),
            "visible_crevasse_area_km2_low": area.quantile(settings.ENSEMBLE_QUANTILES[0]),
            "visible_crevasse_area_km2_high": area.quantile(settings.ENSEMBLE_QUANTILES[1]),
        },
        axis=1,
    ).reset_index()
    return _write_csv(table.sort_values(["rgi_id", "year"]), settings.PUBLICATION_DIR / GLACIER_ANNUAL_AREA_FILE)


def vote_fraction_rasters() -> list[Path]:
    """Copy the vote-fraction rasters and embed their encoding as metadata."""
    paths = []
    for year in tqdm.tqdm(settings.YEARS, desc="Vote-fraction rasters"):
        source = settings.RASTER_DIR / settings.VOTE_RASTER_TEMPLATE.format(year=year)
        target = settings.PUBLICATION_DIR / source.name
        temporary = target.with_name(f".{target.stem}.tmp.tif")
        try:
            shutil.copyfile(source, temporary)
            with rasterio.open(temporary, "r+") as raster:
                raster.scales = (1 / 255,)
                raster.offsets = (0.0,)
                raster.update_tags(
                    year=str(year),
                    n_models=str(settings.N_MODELS),
                    description=(
                        f"Fraction of {settings.N_MODELS} random-forest models that classify the pixel as visibly "
                        f"crevassed (per-model tree-vote fraction >= {settings.POSITIVE_THRESHOLD_U8}/255). "
                        "Decode as value / 255. No glacier mask is applied."
                    ),
                    source=GITHUB_URL,
                )
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        paths.append(target)
    return paths


def tree_vote_archive() -> Path:
    """Store the 30-band per-model rasters in one zip under their published names."""
    target = settings.PUBLICATION_DIR / TREE_VOTE_ARCHIVE
    temporary = target.with_name(f".{target.stem}.tmp.zip")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            for year in tqdm.tqdm(settings.YEARS, desc="Tree-vote archive"):
                source = settings.INPUT_DIR / settings.MODEL_RASTER_TEMPLATE.format(year=year)
                archive.write(source, arcname=TREE_VOTE_MEMBER_TEMPLATE.format(year=year))
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def code_version() -> str:
    """Return the checked-out commit, marked when tracked files differ from it.

    The deposit cites the code by commit rather than by a separate archive, so
    a deposit built from uncommitted code must say so.
    """

    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=settings.ROOT, check=True, capture_output=True, text=True).stdout

    commit = git("rev-parse", "--short=7", "HEAD").strip()
    dirty = git("status", "--porcelain", "--untracked-files=no").strip()
    return f"`{commit}` (with uncommitted changes)" if dirty else f"`{commit}`"


def readme() -> Path:
    """Write the deposit README from the fixed project settings."""
    low, high = (round(100 * quantile) for quantile in settings.ENSEMBLE_QUANTILES)
    first, last = settings.YEARS[0], settings.YEARS[-1]
    text = README_TEMPLATE.format(
        first=first,
        last=last,
        n_models=settings.N_MODELS,
        n_glaciers=f"{settings.EXPECTED_GLACIERS:,}",
        pixel=settings.PIXEL_SIZE_M,
        crs=settings.RASTER_CRS,
        threshold=settings.POSITIVE_THRESHOLD_U8,
        low=low,
        high=high,
        archive=TREE_VOTE_ARCHIVE,
        member=TREE_VOTE_MEMBER_TEMPLATE.format(year="{year}"),
        recent_start=settings.RECENT_SURGE_START,
        end=settings.END_YEAR,
        pre_start=first - settings.PRE_SURGE_YEARS,
        pre_end=settings.EARLIER_SURGE_END,
        github=GITHUB_URL,
        manuscript=MANUSCRIPT_URL,
        surge_database=SURGE_DATABASE_URL,
        commit=code_version(),
        last_band=f"{settings.N_MODELS - 1:02d}",
        n_models_plus_one=settings.N_MODELS + 1,
    )
    path = settings.PUBLICATION_DIR / "README.md"
    path.write_text(text, encoding="utf-8")
    return path


README_TEMPLATE: Final = """\
# Annual visible crevasse extent of Svalbard glaciers, {first}–{last}

Annual 50 m maps of visible crevasses on Svalbard glaciers, predicted from the
AlphaEarth Foundations satellite embedding
(`GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`) by an ensemble of {n_models} random-forest
models, and per-glacier tables of visible crevasse area and its {first}–{last}
trend for {n_glaciers} RGI7 glaciers.

The method, validation and interpretation are described in the accompanying
manuscript, <{manuscript}>. The analysis code, Earth Engine scripts and
training labels are at <{github}>, at commit {commit}.

## Files

| File | Contents |
| --- | --- |
| `crevasse_vote_fraction_{{year}}.tif` | Ensemble vote fraction, one file per year. Start here. |
| `{archive}` | Per-model tree-vote fractions, one {n_models}-band file per year. |
| `glacier_trends.csv` | {first}–{last} trend of visible crevasse area, one row per glacier. |
| `glacier_annual_area.csv` | Visible crevasse area per glacier and year. |

## Rasters

All rasters share one grid: {crs}, {pixel} m pixels, `uint8`, decoded as
`value / 255`. They cover the full Svalbard bounding box and **have no glacier
mask and no nodata value**: 0 means either no positive vote or no prediction.
Clip to glacier outlines (e.g. RGI7) before use. The {pixel} m grid is an export
choice; the embedding is 10 m.

The two raster products describe model agreement at two levels:

- **Tree-vote fraction** (`{member}` in the zip, bands `prob_m00`–`prob_m{last_band}`):
  for each of the {n_models} models, the fraction of its trees that classify the pixel
  as crevassed.
- **Vote fraction** (`crevasse_vote_fraction_{{year}}.tif`): the fraction of the
  {n_models} models whose tree-vote fraction is at least {threshold}/255. It takes
  {n_models_plus_one} discrete values; `round(value / 255 * {n_models})` recovers the
  number of agreeing models.

The ensemble spread measures how sensitive a prediction is to the training
subset, since each model saw a random 70% of the label polygons.

## Tables

Areas are km2 and rates km2 yr-1 unless stated. `_low` and `_high` are the
{low}th and {high}th percentiles across the {n_models} models; unsuffixed values are the
ensemble median. Per-model areas use the {threshold}/255 threshold above.

`glacier_trends.csv`:

| Column | Description |
| --- | --- |
| `rgi_id` | RGI 7.0 glacier ID. |
| `anlys_id` | Glacier ID in the Svalbard surge database ([Harcourt et al., 2026]({surge_database})). |
| `glac_name` | Glacier name, where one exists. |
| `area_km2` | RGI7 outline area. |
| `reference_valid_area_km2` | Area of the {pixel} m pixels inside the outline. |
| `surging`, `S_Onset`, `S_Term` | Surge record from the surge database, with the corrections documented in the code. |
| `surge_period` | `recent`: surge overlapping {recent_start}–{end}; `pre`: surge overlapping {pre_start}–{pre_end}; `none`: neither. |
| `slope_km2_yr` | Linear {first}–{last} trend of visible crevasse area. |
| `slope_percent_points_yr` | The same trend as a share of `reference_valid_area_km2`. |
| `significant` | `True` when the `_low`–`_high` slope range excludes zero. |
| `trend_sign` | Sign of the median slope: `positive`, `negative`, or `zero` (median slope exactly zero, because most models predict no visible crevasses in any year). |
| `r2_median`, `rmse_median` | Median goodness of fit of the model-wise linear trends (RMSE in km2). |
| `n_models` | Number of models. |
| `mean_coverage_percent` | Median across models of the {first}–{last} mean visible crevasse area as a share of `reference_valid_area_km2`. |

`glacier_annual_area.csv` has `rgi_id`, `anlys_id`, `glac_name`, `year`, and
`visible_crevasse_area_km2` with its `_low` and `_high` bounds.

## Caveats

- The maps show *visible* crevasses. Snow and firn bridges hide crevasses without
  removing them, so a change in area can reflect structural change, snow
  conditions, or both.
- Models were trained on labels from 2017, 2019, 2021, 2023 and 2025
  (`TRAINING_YEARS` in the Earth Engine script). 2024 was withheld for validation;
  in training years, predictions inside labelled polygons are in-sample.
"""  # noqa: E501


def run() -> None:
    """Build the full deposit in `settings.PUBLICATION_DIR`."""
    settings.PUBLICATION_DIR.mkdir(parents=True, exist_ok=True)
    glacier_trends_table()
    glacier_annual_area_table()
    vote_fraction_rasters()
    readme()
    tree_vote_archive()


def test_classify_trends_separates_zero_and_significance() -> None:
    table = pd.DataFrame(
        {"slope_km2_yr": [0.0, 0.2, -0.1], "slope_km2_yr_low": [0.0, 0.1, -0.2], "slope_km2_yr_high": [0.0, 0.3, 0.05]}
    )
    result = classify_trends(table)
    assert result.significant.tolist() == [False, True, False]
    assert result.trend_sign.tolist() == ["zero", "positive", "negative"]
    assert "significant" not in table


if __name__ == "__main__":
    run()
