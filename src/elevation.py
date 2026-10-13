"""Locate visible crevasse change by elevation on each glacier.

The mass-change comparison in `src/mass_balance.py` asks whether glaciers that
lose more mass gain more visible crevasses. It cannot test whether gains come
from crevasses in high accumulation areas being exposed as the firn line rises,
because that mechanism depends on where on a glacier the change happens, not on
the glacier-wide mass loss rate. This module bins every model's positive pixels
by glacier and elevation band so that trends can be located in elevation:
gains concentrated in upper glacier reaches are consistent with exposure, while
gains near termini and in ablation areas point to dynamic causes.

Two band definitions are used. Absolute bands of `settings.ELEVATION_BAND_M`
locate change relative to regional snow and firn line heights. Relative bands
split each glacier's own elevation range into
`settings.RELATIVE_ELEVATION_BANDS` intervals, so that a low and a high glacier
can be compared by position between terminus and top.
"""

from __future__ import annotations

import os
from typing import Literal

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import tqdm
from rasterio.features import rasterize
from rasterio.warp import Resampling, reproject

from src import analysis, raster, settings, surge

type ForceRedo = Literal[False, "self", "all"]
type BandKind = Literal["absolute", "relative"]

BAND_KINDS: tuple[BandKind, ...] = ("absolute", "relative")


def read_dem(transform: rasterio.Affine, shape: tuple[int, int]) -> np.ndarray:
    """Resample the S0 DTM onto the prediction grid.

    The DTM uses ETRS89 / UTM 33N (EPSG:25833) and the predictions WGS 84 /
    UTM 33N (EPSG:32633); the two differ by well under a pixel, but the grids
    are offset by 10 m, so the DTM is resampled bilinearly rather than read by
    index.

    Returns
    -------
    numpy.ndarray
        Elevation in m a.s.l. as float32, NaN outside the DTM.
    """
    if not settings.DEM_FILE.exists():
        raise FileNotFoundError(f"{settings.DEM_FILE} is missing; run `just fetch-inputs` to download it")
    elevation = np.full(shape, np.nan, dtype=np.float32)
    with rasterio.open(f"zip://{settings.DEM_FILE.resolve()}!{settings.DEM_MEMBER}") as dem:
        reproject(
            source=rasterio.band(dem, 1),
            destination=elevation,
            dst_transform=transform,
            dst_crs=settings.RASTER_CRS,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
    # The S0 nodata value is a float32 sentinel near its maximum; anything
    # implausible for Svalbard is treated as missing as well.
    elevation[(elevation < -100) | (elevation > 2000)] = np.nan
    return elevation


def assign_bands(labels: np.ndarray, elevation: np.ndarray, n_glaciers: int) -> dict[BandKind, np.ndarray]:
    """Assign each glacier pixel to an absolute and a relative elevation band.

    Parameters
    ----------
    labels
        Flat glacier index per pixel (0-based), restricted to glacier pixels.
    elevation
        Flat elevation per glacier pixel (m a.s.l.), NaN where missing.
    n_glaciers
        Number of glaciers, for the per-glacier elevation range.

    Returns
    -------
    dict
        Band index per pixel for each band kind, -1 where elevation is missing.
        Relative bands use each glacier's DEM minimum and maximum; a glacier
        with a flat or single-pixel range falls entirely into band 0.
    """
    valid = np.isfinite(elevation)
    absolute = np.full(labels.shape, -1, dtype=np.int32)
    absolute[valid] = np.floor(elevation[valid] / settings.ELEVATION_BAND_M).astype(np.int32).clip(min=0)

    low = np.full(n_glaciers, np.inf)
    high = np.full(n_glaciers, -np.inf)
    np.minimum.at(low, labels[valid], elevation[valid])
    np.maximum.at(high, labels[valid], elevation[valid])
    span = high - low
    position = np.zeros(labels.shape)
    with np.errstate(invalid="ignore", divide="ignore"):
        position[valid] = (elevation[valid] - low[labels[valid]]) / span[labels[valid]]
    position = np.nan_to_num(position, nan=0.0, posinf=0.0, neginf=0.0)
    relative = np.full(labels.shape, -1, dtype=np.int32)
    relative[valid] = np.minimum(
        (position[valid] * settings.RELATIVE_ELEVATION_BANDS).astype(np.int32), settings.RELATIVE_ELEVATION_BANDS - 1
    )
    return {"absolute": absolute, "relative": relative}


def band_areas(polygons: gpd.GeoDataFrame, force_redo: ForceRedo = False) -> pd.DataFrame:
    """Positive area per glacier, elevation band, model and year.

    This is the expensive step: it reads every annual 30-band raster once. The
    result is cached as an intermediate product and re-read after writing.

    Returns
    -------
    pandas.DataFrame
        Columns `anlys_id`, `band_kind`, `band`, `model_id`, `year`,
        `positive_area_km2` and `band_area_km2` (the glacier area in that band).
        Only glacier/band combinations with at least one pixel are kept.
    """
    cache_path = settings.CACHE_DIR / "elevation-band-areas.arrow"
    if force_redo is False and cache_path.exists():
        return pd.read_feather(cache_path)

    polygons = polygons.to_crs(settings.RASTER_CRS).reset_index(drop=True)
    n_glaciers = len(polygons)
    pixel_km2 = raster.PIXEL_AREA_M2 / 1e6
    frames = []
    labels = glacier_pixels = bands = None
    for year in tqdm.tqdm(settings.YEARS, desc="Elevation bands"):
        values, transform = raster.read_model_votes(
            settings.INPUT_DIR / settings.MODEL_RASTER_TEMPLATE.format(year=year)
        )
        if labels is None:
            grid = rasterize(
                zip(polygons.geometry, np.arange(1, n_glaciers + 1, dtype=np.int32), strict=True),
                out_shape=values.shape[1:],
                transform=transform,
                fill=0,
                dtype="int32",
            ).ravel()
            glacier_pixels = grid > 0
            labels = grid[glacier_pixels] - 1
            elevation = read_dem(transform, values.shape[1:]).ravel()[glacier_pixels]
            bands = assign_bands(labels, elevation, n_glaciers)
        for kind in BAND_KINDS:
            n_bands = int(bands[kind].max()) + 2  # slot 0 holds missing elevation (-1)
            key = labels * n_bands + bands[kind] + 1
            size = n_glaciers * n_bands
            band_pixels = np.bincount(key, minlength=size)
            present = np.flatnonzero(band_pixels)
            for model_id, model in enumerate(values):
                positive = np.bincount(
                    key, weights=model.ravel()[glacier_pixels] >= settings.POSITIVE_THRESHOLD_U8, minlength=size
                )
                frames.append(
                    pd.DataFrame(
                        {
                            "anlys_id": polygons.anlys_id.to_numpy()[present // n_bands],
                            "band_kind": kind,
                            "band": (present % n_bands - 1).astype(np.int16),
                            "model_id": np.int16(model_id),
                            "year": np.int16(year),
                            "positive_area_km2": positive[present] * pixel_km2,
                            "band_area_km2": band_pixels[present] * pixel_km2,
                        }
                    )
                )
        del values
    result = pd.concat(frames, ignore_index=True)
    result["band_kind"] = result.band_kind.astype("category")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_name(f".{cache_path.name}.tmp.arrow")
    try:
        result.to_feather(temporary)
        os.replace(temporary, cache_path)
    finally:
        temporary.unlink(missing_ok=True)
    return pd.read_feather(cache_path)


def _slope(years: np.ndarray, values: np.ndarray) -> float:
    return float(np.polyfit(years, values, 1)[0])


def trend_groups(summary: pd.DataFrame) -> pd.DataFrame:
    """Gaining and losing glaciers, defined as in `mass_balance.compare_by_trend_sign`.

    Glaciers with no recorded surge in the analysis window and a trend whose
    ensemble range excludes zero, split by the sign of the median trend.
    """
    significant = (summary.slope_km2_yr_low > 0) | (summary.slope_km2_yr_high < 0)
    groups = summary.loc[significant & (summary.surge_period == "none"), ["anlys_id", "slope_km2_yr"]]
    return groups.assign(group=np.where(groups.slope_km2_yr > 0, "gain", "loss"))[["anlys_id", "group"]]


def _grouped_trends(data: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Sum area per group key, model and year, fit a trend per model, summarize the ensemble."""
    summed = data.groupby([*keys, "model_id", "year"], observed=True, as_index=False).agg(
        positive_area_km2=("positive_area_km2", "sum"), band_area_km2=("band_area_km2", "sum")
    )
    per_model = (
        summed.groupby([*keys, "model_id"], observed=True)
        .apply(
            lambda frame: pd.Series(
                {
                    "slope_km2_yr": _slope(frame.year.to_numpy(float), frame.positive_area_km2.to_numpy(float)),
                    "band_area_km2": frame.band_area_km2.iloc[0],
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    per_model["model_total_km2_yr"] = per_model.groupby(
        [key for key in keys if key not in {"band", "zone"}] + ["model_id"], observed=True
    ).slope_km2_yr.transform("sum")
    per_model["share_of_group_trend"] = per_model.slope_km2_yr / per_model.model_total_km2_yr
    low, high = settings.ENSEMBLE_QUANTILES
    return per_model.groupby(keys, observed=True, as_index=False).agg(
        band_area_km2=("band_area_km2", "first"),
        slope_km2_yr=("slope_km2_yr", "median"),
        slope_km2_yr_low=("slope_km2_yr", lambda values: values.quantile(low)),
        slope_km2_yr_high=("slope_km2_yr", lambda values: values.quantile(high)),
        share_of_group_trend=("share_of_group_trend", "median"),
        share_of_group_trend_low=("share_of_group_trend", lambda values: values.quantile(low)),
        share_of_group_trend_high=("share_of_group_trend", lambda values: values.quantile(high)),
    )


def band_trends(areas: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    """Trend of visible crevasse area per elevation band for gaining and losing glaciers.

    Within each group (see `trend_groups`) the positive area is summed per band,
    model and year, a least-squares trend is fitted per band and model, and the
    ensemble median and `settings.ENSEMBLE_QUANTILES` bounds are reported. The
    share of the group trend is computed per model before summarizing.

    Returns
    -------
    pandas.DataFrame
        One row per group, band kind and band with `band_area_km2`, the trend
        median and bounds (km2 yr-1), and the band's share of the group trend.
    """
    data = areas.merge(trend_groups(summary), on="anlys_id", validate="many_to_one")
    return _grouped_trends(data, ["group", "band_kind", "band"])


def zone_trends(areas: pd.DataFrame, summary: pd.DataFrame, terminus: pd.DataFrame) -> pd.DataFrame:
    """Trends split into a lower and an upper elevation zone, overall and by terminus type.

    Absolute bands split at `settings.ELEVATION_ZONE_SPLIT_M` and relative bands
    at `settings.RELATIVE_ZONE_SPLIT_BAND`; see `band_trends` for the method.
    RGI7 marks marine-terminating glaciers with `term_type` 1; in Svalbard every
    other glacier carries 9 (not assigned) and is treated as land-terminating.
    Shares of the group trend are computed within each terminus subset.

    Parameters
    ----------
    areas
        Output of `band_areas`.
    summary
        Ensemble summary per glacier with `surge_period`.
    terminus
        `anlys_id` and RGI7 `term_type` per glacier.
    """
    data = areas.merge(trend_groups(summary), on="anlys_id", validate="many_to_one")
    split = np.where(
        data.band_kind == "absolute",
        settings.ELEVATION_ZONE_SPLIT_M // settings.ELEVATION_BAND_M,
        settings.RELATIVE_ZONE_SPLIT_BAND,
    )
    data = data.assign(zone=np.where(data.band >= split, "upper", "lower"))
    types = terminus[["anlys_id", "term_type"]].assign(
        terminus=lambda frame: np.where(frame.term_type.astype(int) == 1, "tidewater", "land")
    )
    data = data.merge(types[["anlys_id", "terminus"]], on="anlys_id", validate="many_to_one")
    keys = ["group", "terminus", "band_kind", "zone"]
    overall = _grouped_trends(data.assign(terminus="all"), keys)
    by_terminus = _grouped_trends(data, keys)
    return pd.concat([overall, by_terminus], ignore_index=True)


def _detrend(years: np.ndarray, values: np.ndarray) -> np.ndarray:
    return values - np.polyval(np.polyfit(years, values, 1), years)


def band_year_sensitivity(areas: pd.DataFrame, annual: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    """How strongly each band's year-to-year visible crevasse area follows the regional year factor.

    For each group and band, the summed positive area per model is divided by
    its mean and linearly detrended, leaving the interannual anomaly. It is
    compared with the detrended Svalbard-wide year factor of
    `analysis.common_mode_model_factors`, recomputed without the group's own
    glaciers so that they are not correlated with themselves. Crevasse fields
    cannot open and close within a year, so a band that swings with the
    regional factor has visibility controlled by snow and melt conditions.

    Parameters
    ----------
    areas
        Output of `band_areas`.
    annual
        Annual area per glacier, model and year, as in `annual-model.arrow`.
    summary
        Ensemble summary per glacier with `surge_period`.

    Returns
    -------
    pandas.DataFrame
        One row per group, band kind and band (absolute bands at or above
        `settings.SENSITIVITY_POOL_ABOVE_M` pooled) with the ensemble median
        and bounds of the correlation `r`, the regression slope `beta` of the
        band anomaly on the factor anomaly (1 means it moves in step), and the
        anomaly standard deviation `swing_fraction`. `factor_trend_per_yr` is
        the median least-squares trend of the group-excluded factor itself.
    """
    years = np.array(settings.YEARS, dtype=float)
    groups = trend_groups(summary)
    pooled_band = settings.SENSITIVITY_POOL_ABOVE_M // settings.ELEVATION_BAND_M
    low, high = settings.ENSEMBLE_QUANTILES
    rows = []
    for group, members in groups.groupby("group"):
        factors, _ = analysis.common_mode_model_factors(annual, exclude=frozenset(members.anlys_id))
        factor = factors.pivot(index="model_id", columns="year", values="ratio")[list(settings.YEARS)]
        factor_trend = float(np.median([_slope(years, row) for row in factor.to_numpy(float)]))
        factor_anomaly = {
            model_id: _detrend(years, row) for model_id, row in zip(factor.index, factor.to_numpy(float), strict=True)
        }
        data = areas[areas.anlys_id.isin(members.anlys_id)]
        data = data.assign(band=np.where(data.band_kind == "absolute", np.minimum(data.band, pooled_band), data.band))
        series = (
            data.groupby(["band_kind", "band", "model_id", "year"], observed=True)
            .agg(positive_area_km2=("positive_area_km2", "sum"), band_area_km2=("band_area_km2", "sum"))
            .reset_index()
        )
        for (kind, band), frame in series.groupby(["band_kind", "band"], observed=True):
            values = frame.pivot(index="model_id", columns="year", values="positive_area_km2")[list(settings.YEARS)]
            statistics = []
            for model_id, row in zip(values.index, values.to_numpy(float), strict=True):
                if row.mean() <= 0:
                    continue
                anomaly = _detrend(years, row / row.mean())
                regional = factor_anomaly[model_id]
                statistics.append(
                    (np.corrcoef(anomaly, regional)[0, 1], np.polyfit(regional, anomaly, 1)[0], anomaly.std())
                )
            if not statistics:
                continue
            r, beta, swing = (pd.Series(column) for column in zip(*statistics, strict=True))
            rows.append(
                {
                    "group": group,
                    "band_kind": kind,
                    "band": band,
                    "band_area_km2": frame.band_area_km2.iloc[0],
                    "r": r.median(),
                    "r_low": r.quantile(low),
                    "r_high": r.quantile(high),
                    "beta": beta.median(),
                    "beta_low": beta.quantile(low),
                    "beta_high": beta.quantile(high),
                    "swing_fraction": swing.median(),
                    "factor_trend_per_yr": factor_trend,
                    "model_count": len(statistics),
                }
            )
    return pd.DataFrame(rows)


def run(force_redo: ForceRedo = False) -> dict[str, pd.DataFrame]:
    """Write the elevation-band products from existing analysis products."""
    summary = pd.read_feather(settings.DATA_DIR / "glacier-trends.arrow")
    annual = pd.read_feather(settings.DATA_DIR / "annual-model.arrow")
    glaciers = surge.read_database()
    areas = band_areas(glaciers, force_redo=force_redo)
    products = {
        "elevation-band-trends": band_trends(areas, summary),
        "elevation-zone-trends": zone_trends(areas, summary, glaciers),
        "elevation-band-year-sensitivity": band_year_sensitivity(areas, annual, summary),
    }
    result = {}
    for name, table in products.items():
        path = settings.DATA_DIR / f"{name}.csv"
        table.to_csv(path, index=False)
        result[name] = pd.read_csv(path)
    return result


if __name__ == "__main__":
    run()


def test_assign_bands_absolute_and_relative() -> None:
    labels = np.array([0, 0, 0, 1, 1])
    elevation = np.array([50.0, 150.0, 250.0, 400.0, np.nan])

    bands = assign_bands(labels, elevation, n_glaciers=2)

    assert bands["absolute"].tolist() == [0, 1, 2, 4, -1]
    # Glacier 0 spans 50-250 m: bottom, middle and top of its range. Glacier 1
    # has a single valid pixel, so its range is flat and it falls in band 0.
    assert bands["relative"].tolist() == [0, 5, 9, 0, -1]


def test_band_trends_groups_by_ensemble_sign() -> None:
    summary = pd.DataFrame(
        {
            "anlys_id": [1, 2, 3],
            "surge_period": ["none", "none", "recent"],
            "slope_km2_yr": [1.0, -1.0, 5.0],
            "slope_km2_yr_low": [0.5, -1.5, 4.0],
            "slope_km2_yr_high": [1.5, -0.5, 6.0],
        }
    )
    years = np.array(settings.YEARS[:3])
    rows = []
    for anlys_id, rate in ((1, 1.0), (2, -1.0), (3, 5.0)):
        for model_id in (0, 1):
            for year in years:
                rows.append(
                    {
                        "anlys_id": anlys_id,
                        "band_kind": "absolute",
                        "band": 3,
                        "model_id": model_id,
                        "year": year,
                        "positive_area_km2": 10 + rate * (year - years[0]),
                        "band_area_km2": 20.0,
                    }
                )

    result = band_trends(pd.DataFrame(rows), summary).set_index("group")

    assert set(result.index) == {"gain", "loss"}
    assert np.isclose(result.loc["gain", "slope_km2_yr"], 1.0)
    assert np.isclose(result.loc["loss", "slope_km2_yr"], -1.0)


def test_band_year_sensitivity_detects_in_step_band() -> None:
    years = np.array(settings.YEARS)
    swing = np.array([-1, 1] * (len(years) // 2) + [-1] * (len(years) % 2), dtype=float) * 0.2
    summary = pd.DataFrame(
        {
            "anlys_id": [1, 2],
            "surge_period": ["none", "none"],
            "slope_km2_yr": [1.0, -1.0],
            "slope_km2_yr_low": [0.5, -1.5],
            "slope_km2_yr_high": [1.5, -0.5],
        }
    )
    # Glaciers 9 and 10 are outside both groups and set the regional factor;
    # glacier 1 swings in step with it, glacier 2 does not swing at all.
    annual = pd.DataFrame(
        [
            {"anlys_id": anlys_id, "model_id": 0, "year": year, "positive_area_km2": value}
            for anlys_id, values in (
                (9, 10 * (1 + swing)),
                (10, 5 * (1 + swing)),
                (1, 10 * (1 + swing)),
                (2, np.full(len(years), 10.0)),
            )
            for year, value in zip(years, values, strict=True)
        ]
    )
    areas = annual[annual.anlys_id.isin([1, 2])].assign(band_kind="relative", band=5, band_area_km2=20.0)

    with np.errstate(invalid="ignore"):
        result = band_year_sensitivity(areas, annual, summary).set_index("group")

    assert np.isclose(result.loc["gain", "r"], 1.0)
    assert np.isclose(result.loc["gain", "beta"], 1.0)
    assert np.isclose(result.loc["loss", "beta"], 0.0)
    assert np.isclose(result.loc["loss", "swing_fraction"], 0.0)


def test_zone_trends_splits_by_terminus() -> None:
    summary = pd.DataFrame(
        {
            "anlys_id": [1, 2],
            "surge_period": ["none", "none"],
            "slope_km2_yr": [1.0, 1.0],
            "slope_km2_yr_low": [0.5, 0.5],
            "slope_km2_yr_high": [1.5, 1.5],
        }
    )
    years = np.array(settings.YEARS[:3])
    # Glacier 1 (tidewater) loses low and gains high; glacier 2 (land) only gains high.
    rates = {(1, 0): -2.0, (1, 9): 3.0, (2, 0): 0.0, (2, 9): 1.0}
    areas = pd.DataFrame(
        [
            {
                "anlys_id": anlys_id,
                "band_kind": "relative",
                "band": band,
                "model_id": 0,
                "year": year,
                "positive_area_km2": 10 + rate * (year - years[0]),
                "band_area_km2": 20.0,
            }
            for (anlys_id, band), rate in rates.items()
            for year in years
        ]
    )
    terminus = pd.DataFrame({"anlys_id": [1, 2], "term_type": [1, 9]})

    result = zone_trends(areas, summary, terminus).set_index(["terminus", "zone"])

    assert np.isclose(result.loc[("all", "lower"), "slope_km2_yr"], -2.0)
    assert np.isclose(result.loc[("tidewater", "lower"), "slope_km2_yr"], -2.0)
    assert np.isclose(result.loc[("land", "lower"), "slope_km2_yr"], 0.0)
    assert np.isclose(result.loc[("all", "upper"), "slope_km2_yr"], 4.0)
