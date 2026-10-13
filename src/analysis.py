"""Run the fixed Svalbard crevasse trend analysis."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import geopandas as gpd
import numpy as np
import pandas as pd

from src import mass_balance, raster, settings, surge

type ForceRedo = Literal[False, "self", "all"]


def _write_arrow(data: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Atomically write an Arrow table and return the finalized table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.arrow")
    try:
        normalized = data.reset_index(drop=True).convert_dtypes()
        normalized.to_feather(temporary)
        os.replace(temporary, path)
        return pd.read_feather(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_predictions(metadata: gpd.GeoDataFrame, force_redo: ForceRedo = False) -> pd.DataFrame:
    """Aggregate and validate complete model trajectories."""
    cache_path = settings.CACHE_DIR / "predictions.arrow"
    if force_redo is False and cache_path.exists():
        return pd.read_feather(cache_path)

    data = raster.aggregate_model_areas(settings.INPUT_DIR, metadata, settings.YEARS, settings.RASTER_DIR)
    expected = len(settings.YEARS) * settings.N_MODELS
    counts = data.groupby("anlys_id").size()
    if len(counts) != settings.EXPECTED_GLACIERS or counts.ne(expected).any():
        raise ValueError("Expected 1,583 complete glacier trajectories with 30 models and 9 years")
    if data.duplicated(["year", "model_id", "anlys_id"]).any():
        raise ValueError("Duplicate glacier/model/year prediction rows")
    data["positive_area_km2"] = data.positive_area_m2 / 1e6
    data["valid_area_km2"] = data.valid_area_m2 / 1e6
    return _write_arrow(data, cache_path)


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
        "low": lambda values: values.quantile(settings.ENSEMBLE_QUANTILES[0]),
        "high": lambda values: values.quantile(settings.ENSEMBLE_QUANTILES[1]),
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
        slope_km2_yr_low=("slope_km2_yr", lambda x: x.quantile(settings.ENSEMBLE_QUANTILES[0])),
        slope_km2_yr_high=("slope_km2_yr", lambda x: x.quantile(settings.ENSEMBLE_QUANTILES[1])),
        slope_percent_points_yr=("slope_percent_points_yr", "median"),
        slope_percent_points_yr_low=("slope_percent_points_yr", lambda x: x.quantile(settings.ENSEMBLE_QUANTILES[0])),
        slope_percent_points_yr_high=("slope_percent_points_yr", lambda x: x.quantile(settings.ENSEMBLE_QUANTILES[1])),
        r2_median=("r2", "median"),
        rmse_median=("rmse", "median"),
        reference_area_km2=("reference_area_km2", "first"),
    )
    return annual, model, summary


def common_mode_factors(data: pd.DataFrame) -> pd.DataFrame:
    """Extract the Svalbard-wide year factor shared by all glaciers.

    A regional snow or melt anomaly changes how much of an unchanged crevasse
    field is visible, and does so for many glaciers in the same year. Glacier
    dynamics act on each glacier's own schedule. The component shared between
    glaciers therefore bounds how much of a trend an exposure effect could
    explain, without identifying that shared component as exposure: a genuine
    region-wide dynamic response is equally shared and is not separable here.

    Each glacier-model timeseries is divided by its own mean so that glaciers
    of different size contribute equally, and the year factor is the median of
    those ratios across glaciers, taken separately for every model so that the
    ensemble spread propagates.

    Parameters
    ----------
    data
        Annual model areas, one row per glacier, model, and year, as written to
        `annual-model.arrow`.

    Returns
    -------
    pandas.DataFrame
        One row per year with the ensemble median factor, its
        `settings.ENSEMBLE_QUANTILES` model percentiles, and the glacier count entering the median. A factor
        of 1.0 means the typical glacier showed its own average visible
        crevasse area that year.
    """
    per_model, n_glaciers = common_mode_model_factors(data)
    return (
        per_model.groupby("year", as_index=False)
        .agg(
            factor=("ratio", "median"),
            factor_low=("ratio", lambda values: values.quantile(settings.ENSEMBLE_QUANTILES[0])),
            factor_high=("ratio", lambda values: values.quantile(settings.ENSEMBLE_QUANTILES[1])),
            n_models=("ratio", "size"),
        )
        .assign(n_glaciers=n_glaciers)
    )


def common_mode_model_factors(data: pd.DataFrame, exclude: frozenset[int] = frozenset()) -> tuple[pd.DataFrame, int]:
    """Per-model Svalbard-wide year factor behind `common_mode_factors`.

    Parameters
    ----------
    data
        Annual area per glacier, model and year, as in `annual-model.arrow`.
    exclude
        `anlys_id` values left out of the factor, so that a group of glaciers
        can be compared with a factor it did not contribute to.

    Returns
    -------
    tuple
        Columns `model_id`, `year` and `ratio` (the median across glaciers of
        each glacier-model timeseries divided by its own mean), and the number
        of glaciers entering the median.
    """
    frame = data.loc[~data.anlys_id.isin(exclude), ["anlys_id", "model_id", "year", "positive_area_km2"]].copy()
    frame["mean_area_km2"] = frame.groupby(["anlys_id", "model_id"], sort=False).positive_area_km2.transform("mean")
    frame = frame[frame.mean_area_km2 >= settings.COMMON_MODE_MIN_MEAN_AREA_KM2]
    if frame.empty:
        raise ValueError(
            f"No glacier exceeds COMMON_MODE_MIN_MEAN_AREA_KM2="
            f"{settings.COMMON_MODE_MIN_MEAN_AREA_KM2}; cannot normalize."
        )
    frame["ratio"] = frame.positive_area_km2 / frame.mean_area_km2
    per_model = frame.groupby(["model_id", "year"], as_index=False).ratio.median()
    return per_model, frame.anlys_id.nunique()


def common_mode_summary(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Refit glacier trends after dividing out the shared year factor.

    The corrected trends answer one question only: how much of the aggregate
    signal survives when everything Svalbard did in common is removed. Because
    the shared component may itself be dynamic, the corrected aggregate is a
    lower bound on real change and the difference from the raw aggregate is an
    upper bound on what exposure could explain.

    Returns
    -------
    factors
        The year factors from `common_mode_factors`.
    comparison
        One row per model with the raw and corrected aggregate trend and the
        negative and positive glacier counts under each, so that the ensemble
        spread of the comparison is available rather than a point estimate.
    """
    factors = common_mode_factors(data)
    frame = data[["anlys_id", "model_id", "year", "positive_area_km2"]].copy()
    frame["mean_area_km2"] = frame.groupby(["anlys_id", "model_id"], sort=False).positive_area_km2.transform("mean")
    frame = frame[frame.mean_area_km2 >= settings.COMMON_MODE_MIN_MEAN_AREA_KM2]
    frame = frame.merge(factors[["year", "factor"]], on="year", validate="many_to_one")
    frame["corrected_area_km2"] = frame.positive_area_km2 / frame.factor
    rows = []
    for model_id, group in frame.groupby("model_id", sort=True):
        raw = group.groupby("anlys_id", sort=False).apply(_fit, value="positive_area_km2", include_groups=False)
        corrected = group.groupby("anlys_id", sort=False).apply(_fit, value="corrected_area_km2", include_groups=False)
        rows.append(
            {
                "model_id": model_id,
                "glacier_count": len(raw),
                "raw_total_km2_yr": raw.slope.sum(),
                "corrected_total_km2_yr": corrected.slope.sum(),
                "raw_negative_count": int((raw.slope < 0).sum()),
                "corrected_negative_count": int((corrected.slope < 0).sum()),
                "sign_flip_count": int((np.sign(raw.slope) != np.sign(corrected.slope)).sum()),
            }
        )
    return factors, pd.DataFrame(rows)


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
        surge.surge_overlaps(term, onset, settings.RECENT_SURGE_START, settings.END_YEAR)
        for term, onset in zip(result.S_Term, result.S_Onset, strict=True)
    ]
    before = [
        not is_recent
        and surge.surge_overlaps(term, onset, settings.YEARS[0] - settings.PRE_SURGE_YEARS, settings.EARLIER_SURGE_END)
        for is_recent, term, onset in zip(recent, result.S_Term, result.S_Onset, strict=True)
    ]
    result["surge_period"] = np.select([recent, before], ["recent", "pre"], default="none")
    result["trend_sign"] = np.where(result.slope_km2_yr >= 0, "positive", "negative")
    return result


def surge_magnitude_ratio(summary: pd.DataFrame, model: pd.DataFrame) -> pd.DataFrame:
    """Compare trend magnitudes of surging and non-surging glaciers.

    Only glaciers whose trend is significant across the ensemble enter, so the
    ratio compares glaciers that are changing rather than diluting the
    non-surging group with the many near-zero trends of small glaciers. Group
    membership is fixed by the ensemble; the magnitude is evaluated per model so
    the ensemble spread propagates into the ratio.

    Parameters
    ----------
    summary
        Ensemble summary per glacier, as returned by `fit_trends`.
    model
        Trend per glacier and model, as returned by `fit_trends`.

    Returns
    -------
    pandas.DataFrame
        One row with the median and `settings.ENSEMBLE_QUANTILES` bounds of the
        per-model ratio of mean absolute trend (km2 yr-1) of significant
        glaciers with a surge in either window (`recent` or `pre`) to that of
        significant glaciers with no recorded surge, plus the group sizes.
    """
    classified = add_surge_periods(summary)
    significant = classified[(classified.slope_km2_yr_low > 0) | (classified.slope_km2_yr_high < 0)]
    surged = significant.surge_period.isin(["recent", "pre"]).rename("surged")
    members = pd.concat([significant.anlys_id, surged], axis=1)
    magnitude = (
        model.merge(members, on="anlys_id", validate="many_to_one")
        .assign(abs_slope_km2_yr=lambda frame: frame.slope_km2_yr.abs())
        .groupby(["model_id", "surged"])
        .abs_slope_km2_yr.mean()
        .unstack("surged")
    )
    ratio = magnitude[True] / magnitude[False]
    return pd.DataFrame(
        [
            {
                "ratio_median": ratio.median(),
                "ratio_low": ratio.quantile(settings.ENSEMBLE_QUANTILES[0]),
                "ratio_high": ratio.quantile(settings.ENSEMBLE_QUANTILES[1]),
                "surging_significant_glacier_count": int(surged.sum()),
                "non_surging_significant_glacier_count": int((~surged).sum()),
                "model_count": len(ratio),
            }
        ]
    )


def _latest_exact_termination(term_value: object, onset_value: object) -> int | None:
    """Return the latest exact completed termination, if one exists."""
    intervals = surge.terminated_intervals(term_value, onset_value)
    terminations = [termination for _, termination, inferred in intervals if not inferred]
    return max(terminations, default=None)


def _surge_period_summary_cells(
    classified: pd.DataFrame,
    model: pd.DataFrame,
    group_columns: list[str],
    group_values: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build surge-period summary cells, optionally grouped by spatial zone."""
    rates = model.groupby([*group_columns, "model_id", "trend_sign"], as_index=False).slope_km2_yr.sum()
    rows = []
    if group_values is None:
        group_values = classified[group_columns].drop_duplicates()
    for group_value in group_values.itertuples(index=False, name=None):
        group_filter = np.ones(len(classified), dtype=bool)
        for column, value in zip(group_columns, group_value, strict=True):
            group_filter &= classified[column].eq(value)
        for sign in ("negative", "positive"):
            cell_filter = group_filter & classified.trend_sign.eq(sign)
            group_rates = rates
            for column, value in zip(group_columns, group_value, strict=True):
                group_rates = group_rates[group_rates[column] == value]
            cell = group_rates[group_rates.trend_sign == sign]
            values = cell.slope_km2_yr
            glacier = classified[cell_filter]
            if values.empty:
                low = median = high = np.nan
            else:
                low, median, high = (
                    values.quantile(settings.ENSEMBLE_QUANTILES[0]),
                    values.median(),
                    values.quantile(settings.ENSEMBLE_QUANTILES[1]),
                )
            significant = (
                (glacier.slope_km2_yr_high < 0) if sign == "negative" else (glacier.slope_km2_yr_low > 0)
            ).sum()
            historical_glacier = glacier[glacier.historical_termination]
            historical_significant = (
                (historical_glacier.slope_km2_yr_high < 0)
                if sign == "negative"
                else (historical_glacier.slope_km2_yr_low > 0)
            ).sum()
            significant_glacier = glacier[
                (glacier.slope_km2_yr_high < 0) if sign == "negative" else (glacier.slope_km2_yr_low > 0)
            ]
            significant_historical_glacier = significant_glacier[significant_glacier.historical_termination]
            historical_model = model[model.anlys_id.isin(significant_historical_glacier.anlys_id)]
            significant_model = model[model.anlys_id.isin(significant_glacier.anlys_id)]
            for column, value in zip(group_columns, group_value, strict=True):
                historical_model = historical_model[historical_model[column] == value]
                significant_model = significant_model[significant_model[column] == value]
            historical_rates = (
                historical_model[historical_model.trend_sign == sign]
                .groupby("model_id", as_index=False)
                .slope_km2_yr.sum()
            )
            historical_values = historical_rates.slope_km2_yr
            historical_rate = historical_values.median() if len(historical_values) else np.nan
            significant_rates = (
                significant_model[significant_model.trend_sign == sign]
                .groupby("model_id", as_index=False)
                .slope_km2_yr.sum()
            )
            total_magnitude = significant_rates.slope_km2_yr.abs().median() if len(significant_rates) else np.nan
            row = {
                **dict(zip(group_columns, group_value, strict=True)),
                "trend_sign": sign,
                "median_rate_km2_yr": median,
                "rate_low_km2_yr": low,
                "rate_high_km2_yr": high,
                "rate_uncertainty_km2_yr": max(median - low, high - median) if values.size else np.nan,
                "significant_glacier_count": significant,
                "glacier_count": len(glacier),
                "model_count": len(values),
                "historical_termination_years": settings.HISTORICAL_TERMINATION_YEARS,
                "historical_termination_start_year": settings.MIDPOINT_YEAR - settings.HISTORICAL_TERMINATION_YEARS,
                "historical_termination_glacier_count": len(historical_glacier),
                "historical_termination_significant_glacier_count": historical_significant,
                "historical_termination_significant_rate_km2_yr": historical_rate,
                "historical_termination_significant_rate_share": (
                    abs(historical_rate) / total_magnitude if total_magnitude else np.nan
                ),
            }
            rows.append(row)
    return pd.DataFrame(rows)


def surge_period_summary(summary: pd.DataFrame, model: pd.DataFrame) -> pd.DataFrame:
    """Summarize model rates and significant-glacier counts in six result cells."""
    classified = add_surge_periods(summary)
    classified["latest_exact_termination"] = [
        _latest_exact_termination(term, onset)
        for term, onset in zip(classified.S_Term, classified.S_Onset, strict=True)
    ]
    historical_start = settings.MIDPOINT_YEAR - settings.HISTORICAL_TERMINATION_YEARS
    classified["historical_termination"] = classified.latest_exact_termination.between(
        historical_start, settings.MIDPOINT_YEAR, inclusive="both"
    )
    model = model.merge(
        classified[["anlys_id", "surge_period", "trend_sign"]],
        on="anlys_id",
        validate="many_to_one",
    )
    periods = pd.DataFrame({"surge_period": ("pre", "recent", "none")})
    return _surge_period_summary_cells(classified, model, ["surge_period"], periods)


def surge_zone_summary(
    summary: pd.DataFrame, model: pd.DataFrame, glaciers: gpd.GeoDataFrame, zones: gpd.GeoDataFrame
) -> pd.DataFrame:
    """Summarize surge periods and trend signs by the centroid's glacier zone."""
    classified = add_surge_periods(summary)
    classified["latest_exact_termination"] = [
        _latest_exact_termination(term, onset)
        for term, onset in zip(classified.S_Term, classified.S_Onset, strict=True)
    ]
    historical_start = settings.MIDPOINT_YEAR - settings.HISTORICAL_TERMINATION_YEARS
    classified["historical_termination"] = classified.latest_exact_termination.between(
        historical_start, settings.MIDPOINT_YEAR, inclusive="both"
    )
    points = gpd.GeoDataFrame(glaciers[["rgi_id", "geometry"]].copy(), geometry="geometry", crs=glaciers.crs)
    zone_columns = zones[["zone_label", "zone_name", "geometry"]].copy()
    if points.crs != zone_columns.crs:
        points = points.to_crs(zone_columns.crs)
    points.geometry = points.geometry.centroid
    joined = gpd.sjoin(points, zone_columns, how="left", predicate="within")
    classified = classified.merge(
        joined[["rgi_id", "zone_label", "zone_name"]], on="rgi_id", how="left", validate="one_to_one"
    )
    if classified.zone_label.isna().any():
        raise ValueError("Some glacier centroids did not fall within a glacier zone")
    model = model.merge(
        classified[["anlys_id", "surge_period", "trend_sign", "zone_label", "zone_name"]],
        on="anlys_id",
        validate="many_to_one",
    )
    periods = pd.DataFrame({"surge_period": ("pre", "recent", "none")})
    spatial_groups = zones[["zone_label", "zone_name"]].drop_duplicates().merge(periods, how="cross")
    return _surge_period_summary_cells(
        classified,
        model,
        ["zone_label", "zone_name", "surge_period"],
        spatial_groups,
    )


def run() -> dict[str, pd.DataFrame]:
    """Run the complete analysis using the fixed project settings."""
    rgi_metadata = surge.read_database()
    data, metadata = attach_metadata(read_predictions(rgi_metadata), rgi_metadata)
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
    magnitude = surge_magnitude_ratio(glacier, glacier_model)
    magnitude.to_csv(settings.DATA_DIR / "surge-magnitude-ratio.csv", index=False)
    tables["surge-magnitude-ratio"] = magnitude
    mass_by_model = mass_balance.compare_by_model_trend_sign(glacier, glacier_model, rgi_metadata)
    mass_by_model.to_csv(settings.DATA_DIR / "mass-balance-by-model.csv", index=False)
    tables["mass-balance-by-model"] = mass_by_model
    zones = gpd.read_file(settings.GLACIER_ZONES_FILE)
    zone_summary = surge_zone_summary(glacier, glacier_model, rgi_metadata, zones)
    zone_summary.to_csv(settings.DATA_DIR / "surge-period-trend-zone-summary.csv", index=False)
    tables["surge-period-trend-zone-summary"] = zone_summary
    mass_change = mass_balance.compare_by_trend_sign(glacier, rgi_metadata)
    mass_change.to_csv(settings.DATA_DIR / "mass-balance-trend-comparison.csv", index=False)
    tables["mass-balance-trend-comparison"] = mass_change
    factors, comparison = common_mode_summary(data)
    factors.to_csv(settings.DATA_DIR / "common-mode-year-factors.csv", index=False)
    comparison.to_csv(settings.DATA_DIR / "common-mode-trend-comparison.csv", index=False)
    tables["common-mode-year-factors"] = factors
    tables["common-mode-trend-comparison"] = comparison
    return tables


if __name__ == "__main__":
    run()


def test_write_arrow_normalizes_numpy_string_dtypes(tmp_path: Path) -> None:
    """Arrow serialization accepts NumPy's native string dtype."""
    data = pd.DataFrame({"name": np.array(["a", "b"], dtype="str"), "value": [1, 2]})

    result = _write_arrow(data, tmp_path / "table.arrow")

    assert result.to_dict(orient="list") == {"name": ["a", "b"], "value": [1, 2]}


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


def test_surge_magnitude_ratio_uses_significant_glaciers_per_model() -> None:
    summary = pd.DataFrame(
        {
            "anlys_id": [1, 2, 3],
            "S_Onset": ["2022", "n/a", "n/a"],
            "S_Term": ["Ongoing", "n/a", "n/a"],
            "slope_km2_yr": [10.0, -1.0, 0.0],
            "slope_km2_yr_low": [8.0, -1.5, -0.1],
            "slope_km2_yr_high": [12.0, -0.5, 0.1],
        }
    )
    model = pd.DataFrame(
        {
            "anlys_id": np.repeat([1, 2, 3], 2),
            "model_id": [0, 1] * 3,
            # Glacier 3 is not significant, so its large model-1 slope must not
            # enter the non-surging magnitude.
            "slope_km2_yr": [10.0, 12.0, -1.0, -2.0, 0.0, 50.0],
        }
    )

    result = surge_magnitude_ratio(summary, model).iloc[0]

    assert result.ratio_median == 8.0
    assert result.surging_significant_glacier_count == 1
    assert result.non_surging_significant_glacier_count == 1
