"""Test the visible-crevasse decreases against per-glacier mass change.

If the decreases in visible crevasse extent were an artifact of firn loss
revealing crevasses that already existed, then glaciers losing mass fastest
would reveal the most crevassing and show the most positive trends. The
comparison here comes out the other way, which is why it is worth making: it
discriminates between an exposure explanation and a dynamic one rather than
merely bounding the first.

The comparison is restricted to glaciers with no surge recorded in the
analysis window. Surging glaciers gain and lose crevasses for a reason that
has nothing to do with either explanation, and including them would let a
handful of large surges dominate the comparison.

The mass-change series are keyed by RGI6 outline identifiers while this
project uses RGI7, so glaciers are matched on centroid position; for Svalbard
the two inventories agree closely enough that this is unambiguous.
"""

from __future__ import annotations

import io
import zipfile

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import stats

from src import settings, surge


def read_mass_change() -> gpd.GeoDataFrame:
    """Read Svalbard annual mass-change series as points in the project CRS.

    Returns
    -------
    geopandas.GeoDataFrame
        One row per RGI6 glacier with `mass_balance_m_we_yr`, the mean specific
        mass balance over `settings.YEARS`, and a centroid geometry.

    Raises
    ------
    FileNotFoundError
        If the source archive is absent. Run `just fetch-inputs` first.
    """
    if not settings.MASS_CHANGE_FILE.exists():
        raise FileNotFoundError(f"{settings.MASS_CHANGE_FILE} is missing; run `just fetch-inputs` to download it")

    with zipfile.ZipFile(settings.MASS_CHANGE_FILE) as archive:
        series = pd.read_csv(io.BytesIO(archive.read(settings.MASS_CHANGE_SERIES_MEMBER)))
        metadata = pd.read_csv(io.BytesIO(archive.read(settings.MASS_CHANGE_METADATA_MEMBER)))

    columns = [str(year) for year in settings.YEARS]
    missing = [column for column in columns if column not in series.columns]
    if missing:
        raise ValueError(f"Mass-change series does not cover {missing}; the release may predate the study period")

    series = series[["outline_id", *columns]].copy()
    series["mass_balance_m_we_yr"] = series[columns].mean(axis=1)
    # A few RGI6 outlines carry two WGMS glacier records with slightly
    # different series (3 of 1615 for Svalbard). Averaging them is preferable
    # to letting a join pick one arbitrarily, and the choice is visible here
    # rather than hidden in a merge.
    series = series.groupby("outline_id", as_index=False).mass_balance_m_we_yr.mean()
    metadata = metadata.groupby("outline_id", as_index=False)[["latitude", "longitude"]].first()
    joined = metadata.merge(series, on="outline_id", validate="one_to_one")
    return gpd.GeoDataFrame(
        joined[["outline_id", "mass_balance_m_we_yr"]],
        geometry=gpd.points_from_xy(joined.longitude, joined.latitude),
        crs="EPSG:4326",
    ).to_crs(settings.RASTER_CRS)


def attach_mass_change(summary: pd.DataFrame, geometry: gpd.GeoDataFrame) -> pd.DataFrame:
    """Match each RGI7 glacier to its nearest RGI6 mass-change centroid."""
    centroids = geometry[geometry.rgi_id.isin(summary.rgi_id)][["rgi_id", "geometry"]].to_crs(settings.RASTER_CRS)
    centroids = centroids.assign(geometry=centroids.geometry.centroid)
    matched = (
        gpd.sjoin_nearest(centroids, read_mass_change(), how="left", max_distance=settings.MASS_CHANGE_MATCH_DISTANCE_M)
        .dropna(subset=["mass_balance_m_we_yr"])
        .drop_duplicates("rgi_id")
    )
    return summary.merge(matched[["rgi_id", "mass_balance_m_we_yr"]], on="rgi_id", validate="one_to_one")


def compare_by_trend_sign(summary: pd.DataFrame, geometry: gpd.GeoDataFrame) -> pd.DataFrame:
    """Compare mass change between crevasse decliners and gainers, non-surge only.

    Only glaciers whose trend sign is significant across the ensemble enter the
    comparison, because a glacier whose sign the ensemble cannot agree on
    belongs in neither group.

    Returns
    -------
    pandas.DataFrame
        Two rows, `negative` and `positive`, with the glacier count and the
        median and quartile mass balance of each group, plus the Mann-Whitney
        one-sided p-value for decliners losing mass faster and the partial
        correlation between the normalized crevasse trend and mass balance
        after removing the effect of glacier size.
    """
    data = attach_mass_change(summary, geometry)
    significant = data[(data.slope_km2_yr_low > 0) | (data.slope_km2_yr_high < 0)]
    quiescent = significant[significant.surge_period == "none"]
    negative = quiescent[quiescent.slope_km2_yr < 0]
    positive = quiescent[quiescent.slope_km2_yr > 0]
    if negative.empty or positive.empty:
        raise ValueError("Both trend signs must be present among non-surging glaciers to compare them")

    p_value = stats.mannwhitneyu(
        negative.mass_balance_m_we_yr, positive.mass_balance_m_we_yr, alternative="less"
    ).pvalue

    design = np.column_stack([np.ones(len(quiescent)), np.log(quiescent.area_km2.to_numpy(dtype=float))])

    def _residual(values: np.ndarray) -> np.ndarray:
        return values - design @ np.linalg.lstsq(design, values, rcond=None)[0]

    partial = np.corrcoef(
        _residual(quiescent.slope_percent_points_yr.to_numpy(dtype=float)),
        _residual(quiescent.mass_balance_m_we_yr.to_numpy(dtype=float)),
    )[0, 1]

    return pd.DataFrame(
        [
            {
                "trend_sign": name,
                "glacier_count": len(group),
                "mass_balance_median_m_we_yr": group.mass_balance_m_we_yr.median(),
                "mass_balance_q1_m_we_yr": group.mass_balance_m_we_yr.quantile(0.25),
                "mass_balance_q3_m_we_yr": group.mass_balance_m_we_yr.quantile(0.75),
                "mannwhitney_p_negative_lower": p_value,
                "partial_r_trend_vs_mass_balance": partial,
                "matched_glacier_count": len(data),
                "start_year": settings.YEARS[0],
                "end_year": settings.YEARS[-1],
            }
            for name, group in (("negative", negative), ("positive", positive))
        ]
    )


def _ols_p_value(r2: pd.Series, n_years: pd.Series) -> pd.Series:
    """Two-sided p-value of a least-squares slope from its r2 and sample size.

    For simple linear regression, t = sqrt(r2 (n - 2) / (1 - r2)) with n - 2
    degrees of freedom. A constant series has undefined r2 and gets p = 1.
    """
    # Floating-point noise can push r2 of a near-flat series slightly below zero.
    r2 = r2.astype(float).clip(lower=0, upper=1 - 1e-12)
    degrees = n_years.astype(float) - 2
    t = np.sqrt(r2 * degrees / (1 - r2))
    return pd.Series(2 * stats.t.sf(t, degrees), index=r2.index).fillna(1.0)


def compare_by_model_trend_sign(summary: pd.DataFrame, model: pd.DataFrame, geometry: gpd.GeoDataFrame) -> pd.DataFrame:
    """Repeat the decliner/gainer mass-change comparison within each model.

    The ensemble comparison in `compare_by_trend_sign` fixes group membership
    from the ensemble, and since mass change does not depend on the model, its
    medians are identical for every model. Testing whether the ordering holds
    "in every model" therefore needs groups drawn from each model alone. Two
    membership rules are reported:

    - `ols_p05`: glaciers whose single-model trend is significant at p < 0.05
      (two-sided t-test on the least-squares slope over the study years).
    - `sign_only`: every glacier with a non-zero single-model trend, split by
      its sign. This keeps the many small, noisy trends and is the weaker test.

    Parameters
    ----------
    summary
        Ensemble summary per glacier with `surge_period`, as written to
        `glacier-trends.arrow`.
    model
        Trend per glacier and model with `r2` and `n_years`, as written to
        `glacier-model-trends.arrow`.
    geometry
        RGI7 glacier geometry with `rgi_id`, used to match mass change.

    Returns
    -------
    pandas.DataFrame
        One row per model and membership rule with group sizes, median mass
        balance per group (m w.e. yr-1), the one-sided Mann-Whitney p-value for
        decliners losing mass faster, and `ordering_holds`, true when the
        decliners' median is the more negative.
    """
    data = attach_mass_change(summary, geometry)
    quiescent = data.loc[data.surge_period == "none", ["anlys_id", "mass_balance_m_we_yr"]]
    per_model = model.merge(quiescent, on="anlys_id", validate="many_to_one")
    per_model = per_model.assign(p_value=_ols_p_value(per_model.r2, per_model.n_years))
    rules = {
        "ols_p05": per_model.p_value < 0.05,
        "sign_only": per_model.slope_km2_yr != 0,
    }
    rows = []
    for rule, mask in rules.items():
        for model_id, group in per_model[mask].groupby("model_id"):
            negative = group[group.slope_km2_yr < 0].mass_balance_m_we_yr
            positive = group[group.slope_km2_yr > 0].mass_balance_m_we_yr
            if negative.empty or positive.empty:
                raise ValueError(f"Model {model_id} lacks one trend sign under the {rule} rule")
            rows.append(
                {
                    "rule": rule,
                    "model_id": model_id,
                    "negative_count": len(negative),
                    "positive_count": len(positive),
                    "negative_median_m_we_yr": negative.median(),
                    "positive_median_m_we_yr": positive.median(),
                    "mannwhitney_p_negative_lower": stats.mannwhitneyu(negative, positive, alternative="less").pvalue,
                    "ordering_holds": negative.median() < positive.median(),
                }
            )
    return pd.DataFrame(rows)


def run_by_model() -> pd.DataFrame:
    """Write the per-model comparison from the existing analysis products.

    Reads `glacier-trends.arrow` and `glacier-model-trends.arrow` rather than
    recomputing them, so it can be rerun without `just analyze`.
    """
    summary = pd.read_feather(settings.DATA_DIR / "glacier-trends.arrow")
    model = pd.read_feather(settings.DATA_DIR / "glacier-model-trends.arrow")
    result = compare_by_model_trend_sign(summary, model, surge.read_database())
    path = settings.DATA_DIR / "mass-balance-by-model.csv"
    result.to_csv(path, index=False)
    return pd.read_csv(path)


def test_ols_p_value_matches_scipy_linregress() -> None:
    x = np.arange(9.0)
    y = np.array([0.0, 1.2, 1.9, 3.4, 3.8, 5.3, 5.9, 7.4, 7.7])
    reference = stats.linregress(x, y)

    p_value = _ols_p_value(pd.Series([reference.rvalue**2]), pd.Series([9])).item()

    assert np.isclose(p_value, reference.pvalue)
