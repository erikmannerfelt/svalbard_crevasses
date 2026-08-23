"""Run the fixed Svalbard crevasse trend analysis."""

from __future__ import annotations

import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from src import raster, settings, surge


def _write_arrow(data: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Atomically write an Arrow table and return the finalized table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.arrow")
    try:
        normalized = data.reset_index(drop=True).copy()
        for column in normalized.select_dtypes(include="str"):
            normalized[column] = normalized[column].astype("string")
        normalized.to_feather(temporary)
        os.replace(temporary, path)
        return pd.read_feather(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_predictions(metadata: gpd.GeoDataFrame) -> pd.DataFrame:
    """Aggregate and validate complete model trajectories."""
    data = raster.aggregate_model_areas(settings.INPUT_DIR, metadata, settings.YEARS, settings.RASTER_DIR)
    expected = len(settings.YEARS) * settings.N_MODELS
    counts = data.groupby("anlys_id").size()
    if len(counts) != settings.EXPECTED_GLACIERS or counts.ne(expected).any():
        raise ValueError("Expected 1,583 complete glacier trajectories with 30 models and 9 years")
    if data.duplicated(["year", "model_id", "anlys_id"]).any():
        raise ValueError("Duplicate glacier/model/year prediction rows")
    data["positive_area_km2"] = data.positive_area_m2 / 1e6
    data["valid_area_km2"] = data.valid_area_m2 / 1e6
    return data


def attach_metadata(data: pd.DataFrame, metadata: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join RGI metadata and enforce the two independent glacier identities."""
    summary = data.groupby("anlys_id", as_index=False).agg(
        glac_id=("glac_id", "first"),
        glac_name=("glac_name", "first"),
        reference_valid_area_m2=("valid_area_m2", "median"),
    )
    joined = summary.merge(metadata, on="anlys_id", how="left", suffixes=("_gee", "_rgi"), validate="one_to_one")
    if joined.rgi_id.isna().any() or joined.glims_id.ne(joined.glac_id).any():
        raise ValueError("Prediction IDs did not join uniquely to the RGI surge database")
    joined["reference_valid_area_km2"] = joined.reference_valid_area_m2 / 1e6
    joined["glac_name"] = joined.glac_name_rgi.combine_first(joined.glac_name_gee)
    joined = joined.drop(columns=["glac_name_gee", "glac_name_rgi", "glims_id"])
    columns = [
        "anlys_id",
        "rgi_id",
        "glac_name",
        "area_km2",
        "S_Onset",
        "S_Term",
        "term_type",
        "surging",
        "reference_valid_area_m2",
        "reference_valid_area_km2",
    ]
    result = data.drop(columns=["glac_name"], errors="ignore").merge(
        joined[columns], on="anlys_id", how="left", validate="many_to_one"
    )
    return result, joined


def _fit(group: pd.DataFrame, value: str) -> pd.Series:
    x = group.year.to_numpy(dtype=float)
    y = group[value].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    residual = y - (slope * x + intercept)
    total = np.sum((y - y.mean()) ** 2)
    return pd.Series(
        {
            "slope": slope,
            "intercept": intercept,
            "r2": 1 - np.sum(residual**2) / total if total else np.nan,
            "rmse": np.sqrt(np.mean(residual**2)),
            "n_years": len(y),
        }
    )


def fit_trends(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit model-wise trends and summarize their ensemble distributions."""
    model = (
        data.groupby(["anlys_id", "model_id"], sort=False)
        .apply(_fit, value="positive_area_km2", include_groups=False)
        .reset_index()
        .rename(columns={"slope": "slope_km2_yr"})
    )
    area = data[
        [
            "anlys_id",
            "reference_valid_area_km2",
            "surging",
            "rgi_id",
            "glac_name",
            "area_km2",
            "S_Onset",
            "S_Term",
        ]
    ].drop_duplicates()
    model = model.merge(area, on="anlys_id", validate="many_to_one")
    model["slope_percent_points_yr"] = 100 * model.slope_km2_yr / model.reference_valid_area_km2
    quantiles = {
        "low": lambda values: values.quantile(0.025),
        "high": lambda values: values.quantile(0.975),
    }
    summary = model.groupby("anlys_id", as_index=False).agg(
        rgi_id=("rgi_id", "first"),
        glac_name=("glac_name", "first"),
        area_km2=("area_km2", "first"),
        reference_valid_area_km2=("reference_valid_area_km2", "first"),
        surging=("surging", "first"),
        S_Onset=("S_Onset", "first"),
        S_Term=("S_Term", "first"),
        slope_km2_yr=("slope_km2_yr", "median"),
        slope_km2_yr_low=("slope_km2_yr", quantiles["low"]),
        slope_km2_yr_high=("slope_km2_yr", quantiles["high"]),
        slope_percent_points_yr=("slope_percent_points_yr", "median"),
        slope_percent_points_yr_low=("slope_percent_points_yr", quantiles["low"]),
        slope_percent_points_yr_high=("slope_percent_points_yr", quantiles["high"]),
        r2_median=("r2", "median"),
        rmse_median=("rmse", "median"),
        n_models=("model_id", "nunique"),
    )
    coverage = data.assign(coverage_percent=100 * data.positive_area_km2 / data.reference_valid_area_km2)
    coverage = coverage.groupby(["anlys_id", "model_id"]).coverage_percent.mean().groupby("anlys_id").median()
    summary = summary.merge(coverage.rename("mean_coverage_percent"), on="anlys_id")
    return model, summary


def regional_trends(data: pd.DataFrame, metadata: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Calculate annual and model-wise trends for all surge groups."""
    groups = {
        "all": np.ones(len(metadata), dtype=bool),
        "surging": metadata.surging.to_numpy(),
        "non_surging": ~metadata.surging.to_numpy(),
    }
    rows = []
    for name, mask in groups.items():
        ids = set(metadata.loc[mask, "anlys_id"])
        subset = data[data.anlys_id.isin(ids)]
        denominator = metadata.loc[mask, "reference_valid_area_km2"].sum()
        annual = (
            subset.groupby(["year", "model_id"], as_index=False)
            .positive_area_km2.sum()
            .rename(columns={"positive_area_km2": "coverage_km2"})
        )
        annual["coverage_percent"] = 100 * annual.coverage_km2 / denominator
        annual["group"] = name
        annual["reference_area_km2"] = denominator
        rows.append(annual)
    annual = pd.concat(rows, ignore_index=True)
    model = (
        annual.groupby(["group", "model_id"], sort=False)
        .apply(_fit, value="coverage_km2", include_groups=False)
        .reset_index()
        .rename(columns={"slope": "slope_km2_yr"})
    )
    model["reference_area_km2"] = model.group.map(annual.groupby("group").reference_area_km2.first())
    model["slope_percent_points_yr"] = 100 * model.slope_km2_yr / model.reference_area_km2
    summary = model.groupby("group", as_index=False).agg(
        slope_km2_yr=("slope_km2_yr", "median"),
        slope_km2_yr_low=("slope_km2_yr", lambda x: x.quantile(0.025)),
        slope_km2_yr_high=("slope_km2_yr", lambda x: x.quantile(0.975)),
        slope_percent_points_yr=("slope_percent_points_yr", "median"),
        slope_percent_points_yr_low=("slope_percent_points_yr", lambda x: x.quantile(0.025)),
        slope_percent_points_yr_high=("slope_percent_points_yr", lambda x: x.quantile(0.975)),
        r2_median=("r2", "median"),
        rmse_median=("rmse", "median"),
        reference_area_km2=("reference_area_km2", "first"),
    )
    return annual, model, summary


def representative_glaciers(summaries: pd.DataFrame) -> pd.DataFrame:
    """Select the two strongest positive and negative trends per surge status."""
    selected = []
    for surging in (True, False):
        subset = summaries[summaries.surging == surging]
        selected.extend([subset.nlargest(2, "slope_km2_yr"), subset.nsmallest(2, "slope_km2_yr")])
    result = pd.concat(selected).drop_duplicates("anlys_id").copy()
    result["trend_direction"] = np.where(result.slope_km2_yr >= 0, "most_positive", "most_negative")
    return result.sort_values(["surging", "trend_direction", "slope_km2_yr"], ascending=[False, True, False])


def add_surge_periods(summary: pd.DataFrame) -> pd.DataFrame:
    """Classify each glacier by surge timing and trend sign for downstream results."""
    result = summary.copy()
    recent = [
        surge.surge_overlaps(term, onset, settings.MIDPOINT_YEAR - 1, settings.END_YEAR)
        for term, onset in zip(result.S_Term, result.S_Onset, strict=True)
    ]
    before = [
        not is_recent
        and surge.surge_overlaps(term, onset, settings.YEARS[0] - settings.PRE_SURGE_YEARS, settings.MIDPOINT_YEAR)
        for is_recent, term, onset in zip(recent, result.S_Term, result.S_Onset, strict=True)
    ]
    result["surge_period"] = np.select([recent, before], ["recent", "pre"], default="none")
    result["trend_sign"] = np.where(result.slope_km2_yr >= 0, "positive", "negative")
    return result


def surge_period_summary(summary: pd.DataFrame, model: pd.DataFrame) -> pd.DataFrame:
    """Summarize model rates and significant-glacier counts in six result cells."""
    classified = add_surge_periods(summary)
    model = model.merge(
        classified[["anlys_id", "surge_period", "trend_sign"]],
        on="anlys_id",
        validate="many_to_one",
    )
    rates = model.groupby(["model_id", "surge_period", "trend_sign"], as_index=False).slope_km2_yr.sum()
    rows = []
    for period in ("pre", "recent", "none"):
        for sign in ("negative", "positive"):
            cell = rates[(rates.surge_period == period) & (rates.trend_sign == sign)]
            values = cell.slope_km2_yr
            glacier = classified[(classified.surge_period == period) & (classified.trend_sign == sign)]
            if values.empty:
                low = median = high = np.nan
            else:
                low, median, high = values.quantile(0.025), values.median(), values.quantile(0.975)
            significant = (
                (glacier.slope_km2_yr_high < 0) if sign == "negative" else (glacier.slope_km2_yr_low > 0)
            ).sum()
            rows.append(
                {
                    "surge_period": period,
                    "trend_sign": sign,
                    "median_rate_km2_yr": median,
                    "rate_low_km2_yr": low,
                    "rate_high_km2_yr": high,
                    "rate_uncertainty_km2_yr": max(median - low, high - median) if values.size else np.nan,
                    "significant_glacier_count": significant,
                    "glacier_count": len(glacier),
                    "model_count": len(values),
                }
            )
    return pd.DataFrame(rows)


def run() -> dict[str, pd.DataFrame]:
    """Run the complete analysis using the fixed project settings."""
    metadata = surge.read_database()
    data, metadata = attach_metadata(read_predictions(metadata), metadata)
    glacier_model, glacier = fit_trends(data)
    annual, regional_model, regional = regional_trends(data, metadata)
    glacier = add_surge_periods(glacier)
    tables = {
        "annual-model": data,
        "glacier-model-trends": glacier_model,
        "glacier-trends": glacier,
        "regional-annual-model": annual,
        "regional-model-trends": regional_model,
        "regional-trends": regional,
        "representative-glaciers": representative_glaciers(glacier),
    }
    settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        tables[name] = _write_arrow(table, settings.DATA_DIR / f"{name}.arrow")
    summary = surge_period_summary(glacier, glacier_model)
    summary.to_csv(settings.DATA_DIR / "surge-period-trend-summary.csv", index=False)
    tables["surge-period-trend-summary"] = summary
    return tables


if __name__ == "__main__":
    run()


def test_surge_period_summary_has_six_cells() -> None:
    summary = pd.DataFrame(
        {
            "anlys_id": [1, 2, 3],
            "S_Onset": ["2013", "2022", "n/a"],
            "S_Term": ["2014", "Ongoing", "n/a"],
            "slope_km2_yr": [-1.0, 1.0, 0.0],
            "slope_km2_yr_low": [-2.0, 0.5, -0.1],
            "slope_km2_yr_high": [-0.5, 2.0, 0.1],
        }
    )
    model = pd.DataFrame(
        {
            "anlys_id": np.repeat([1, 2, 3], 2),
            "model_id": [0, 1] * 3,
            "slope_km2_yr": [-1.0, -1.2, 1.0, 1.2, 0.0, 0.1],
        }
    )
    classified = add_surge_periods(summary)
    result = surge_period_summary(classified, model)
    assert len(result) == 6
    assert set(result.surge_period) == {"pre", "recent", "none"}
    assert (
        result.loc[
            (result.surge_period == "recent") & (result.trend_sign == "positive"),
            "significant_glacier_count",
        ].item()
        == 1
    )
