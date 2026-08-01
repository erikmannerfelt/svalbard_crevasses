"""Build uncertainty-aware crevasse trends from the GEE CSV exports."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

YEARS = np.arange(2017, 2026)
N_MODELS = 30
START_YEAR = 2013
END_YEAR = 2025
SURGE_OVERRIDES = {
    "RGI2000-v7.0-G-07-00741": {"term_type": 1},  # Scheelebreen became tidewater during the study period; treat it as tidewater throughout.
    "RGI2000-v7.0-G-07-00780": {"S_Term": 2017},  # Approximate for Vasilievbreen.
    "RGI2000-v7.0-G-07-01312": {"glac_name": "Lilliehöökbreen"},
    "RGI2000-v7.0-G-07-01379": {"S_Onset": "2015", "S_Term": "Ongoing", "glac_name": "Austfonna Basin-7"},
    "RGI2000-v7.0-G-07-01383": {"glac_name": "Storisstraumen"},
    "RGI2000-v7.0-G-07-01385": {"glac_name": "Austfonna Basin-2"},
    "RGI2000-v7.0-G-07-01381": {"glac_name": "Austfonna Basin-5"},
    "RGI2000-v7.0-G-07-01384": {"glac_name": "Bråsvellbreen"},
    "RGI2000-v7.0-G-07-00888": {"S_Onset": "2016", "S_Term": "2019"},  # Sveitsarfonna mirroring Penckbreen.
    "RGI2000-v7.0-G-07-00899": {"S_Onset": "1890; 2008", "S_Term": "1890; 2017"},  # Zawadzkibreen mirroring Nathorstbreen.
    "RGI2000-v7.0-G-07-00902": {"S_Onset": "1890; 2008", "S_Term": "1890; 2017"},  # Polakkbreen mirroring Nathorstbreen.
    "RGI2000-v7.0-G-07-00912": {"S_Onset": "1890; 2008", "S_Term": "1890; 2017"},  # Dobrowolskibreen mirroring Nathorstbreen.
    # Orsabreen's reported 2017 surge was considered a misclassification by Sieglova et al. (2026).
    "RGI2000-v7.0-G-07-01211": {"S_Onset": "n/a", "S_Term": "n/a"},
}


def _surge_overlaps(term_value, onset_value, min_year=START_YEAR, max_year=END_YEAR):
    def tokens(value):
        if pd.isna(value):
            return []
        return [part.strip() for part in str(value).split(";") if part.strip()]

    terms, onsets = tokens(term_value), tokens(onset_value)
    for i, onset in enumerate(onsets):
        try:
            onset_year = int(onset)
        except ValueError:
            continue
        if onset_year > max_year:
            continue
        term = terms[i] if i < len(terms) else None
        if term is None or term.lower() in {"", "ongoing"}:
            term_year = np.inf
        elif term.lower() == "not observed":
            term_year = onset_year + 10
        else:
            try:
                term_year = int(term)
            except ValueError:
                term_year = np.inf
        if onset_year <= max_year and term_year > min_year:
            return True
    return False


def read_predictions(input_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(input_dir.glob("crevasse_model_polygon_positive_fraction_*.csv")):
        year = int(path.stem.rsplit("_", 1)[-1])
        frame = pd.read_csv(path).drop(columns=["Unnamed: 0"], errors="ignore")
        frame["year"] = year
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No prediction CSVs found in {input_dir}")
    data = pd.concat(frames, ignore_index=True)
    required = {"year", "model_id", "anlys_id", "glac_id", "positive_area_m2", "valid_area_m2"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")
    expected = len(YEARS) * N_MODELS
    counts = data.groupby("anlys_id").size()
    if len(data.anlys_id.unique()) != 1583 or counts.ne(expected).any():
        raise ValueError("Expected 1,583 complete glacier trajectories with 30 models and 9 years")
    if data.duplicated(["year", "model_id", "anlys_id"]).any():
        raise ValueError("Duplicate glacier/model/year prediction rows")
    data["positive_area_km2"] = data.positive_area_m2 / 1e6
    data["valid_area_km2"] = data.valid_area_m2 / 1e6
    return data


def read_surge_database(input_path: Path) -> pd.DataFrame:
    member = "RGI_V7_Surge_Database/RGI2000-v7.0-G-07_svalbard_jan_mayen_Surge_Database.shp"
    shp = f"zip://{input_path.resolve()}!{member}"
    columns = ["rgi_id", "glims_id", "anlys_id", "glac_name", "area_km2", "S_Onset", "S_Term", "term_type"]
    surge = gpd.read_file(shp, ignore_geometry=True)[columns]
    for rgi_id, overrides in SURGE_OVERRIDES.items():
        idx = surge.rgi_id == rgi_id
        for key, value in overrides.items():
            surge.loc[idx, key] = value
    for column in ["S_Onset", "S_Term"]:
        surge[column] = surge[column].astype("string")
    surge["surging"] = [
        _surge_overlaps(term, onset) for term, onset in zip(surge.S_Term, surge.S_Onset)
    ]
    return surge


def attach_metadata(data: pd.DataFrame, surge: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata = data.groupby("anlys_id", as_index=False).agg(
        glac_id=("glac_id", "first"), glac_name=("glac_name", "first"),
        reference_valid_area_m2=("valid_area_m2", "median")
    )
    joined = metadata.merge(surge, on="anlys_id", how="left", suffixes=("_gee", "_rgi"), validate="one_to_one")
    if joined.rgi_id.isna().any() or joined.glims_id.ne(joined.glac_id).any():
        raise ValueError("Prediction IDs did not join uniquely to the RGI surge database")
    joined["reference_valid_area_km2"] = joined.reference_valid_area_m2 / 1e6
    joined["glac_name"] = joined.glac_name_rgi.combine_first(joined.glac_name_gee)
    joined = joined.drop(columns=["glac_name_gee", "glac_name_rgi", "glims_id"])
    data = data.drop(columns=["glac_name"], errors="ignore").merge(
        joined[["anlys_id", "rgi_id", "glac_name", "area_km2", "S_Onset", "S_Term", "term_type", "surging", "reference_valid_area_m2", "reference_valid_area_km2"]],
        on="anlys_id", how="left", validate="many_to_one"
    )
    return data, joined


def _fit(group: pd.DataFrame, value: str) -> pd.Series:
    x = group.year.to_numpy(dtype=float)
    y = group[value].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    residual = y - fitted
    ss_tot = np.sum((y - y.mean()) ** 2)
    return pd.Series({"slope": slope, "intercept": intercept, "r2": 1 - np.sum(residual**2) / ss_tot if ss_tot else np.nan, "rmse": np.sqrt(np.mean(residual**2)), "n_years": len(y)})


def fit_trends(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_groups = data.groupby(["anlys_id", "model_id"], sort=False)
    model_trends = model_groups.apply(_fit, value="positive_area_km2", include_groups=False).reset_index()
    model_trends = model_trends.rename(columns={"slope": "slope_km2_yr"})
    area = data[["anlys_id", "reference_valid_area_km2", "surging", "rgi_id", "glac_name", "area_km2"]].drop_duplicates()
    model_trends = model_trends.merge(area, on="anlys_id", validate="many_to_one")
    model_trends["slope_percent_points_yr"] = 100 * model_trends.slope_km2_yr / model_trends.reference_valid_area_km2
    summaries = model_trends.groupby("anlys_id", as_index=False).agg(
        rgi_id=("rgi_id", "first"), glac_name=("glac_name", "first"), area_km2=("area_km2", "first"),
        reference_valid_area_km2=("reference_valid_area_km2", "first"), surging=("surging", "first"),
        slope_km2_yr=("slope_km2_yr", "median"), slope_km2_yr_low=("slope_km2_yr", lambda x: x.quantile(.025)), slope_km2_yr_high=("slope_km2_yr", lambda x: x.quantile(.975)),
        slope_percent_points_yr=("slope_percent_points_yr", "median"), slope_percent_points_yr_low=("slope_percent_points_yr", lambda x: x.quantile(.025)), slope_percent_points_yr_high=("slope_percent_points_yr", lambda x: x.quantile(.975)),
        r2_median=("r2", "median"), rmse_median=("rmse", "median"), n_models=("model_id", "nunique")
    )
    mean_coverage = data.assign(coverage_percent=100 * data.positive_area_km2 / data.reference_valid_area_km2).groupby(["anlys_id", "model_id"]).coverage_percent.mean().groupby("anlys_id").median().rename("mean_coverage_percent")
    summaries = summaries.merge(mean_coverage, on="anlys_id")
    return model_trends, summaries


def regional_trends(data: pd.DataFrame, metadata: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = data.copy()
    groups = {"all": np.ones(len(metadata), dtype=bool), "surging": metadata.surging.to_numpy(), "non_surging": ~metadata.surging.to_numpy()}
    rows = []
    for group_name, mask in groups.items():
        ids = set(metadata.loc[mask, "anlys_id"])
        subset = data[data.anlys_id.isin(ids)]
        denominator = metadata.loc[mask, "reference_valid_area_km2"].sum()
        annual = subset.groupby(["year", "model_id"], as_index=False).positive_area_km2.sum().rename(columns={"positive_area_km2": "coverage_km2"})
        annual["coverage_percent"] = 100 * annual.coverage_km2 / denominator
        annual["group"] = group_name
        annual["reference_area_km2"] = denominator
        rows.append(annual)
    annual = pd.concat(rows, ignore_index=True)
    trends = annual.groupby(["group", "model_id"], sort=False).apply(_fit, value="coverage_km2", include_groups=False).reset_index().rename(columns={"slope": "slope_km2_yr"})
    denominator = annual[["group", "reference_area_km2"]].drop_duplicates()
    trends = trends.merge(denominator, on="group")
    trends["slope_percent_points_yr"] = 100 * trends.slope_km2_yr / trends.reference_area_km2
    summary = trends.groupby("group", as_index=False).agg(
        slope_km2_yr=("slope_km2_yr", "median"), slope_km2_yr_low=("slope_km2_yr", lambda x: x.quantile(.025)), slope_km2_yr_high=("slope_km2_yr", lambda x: x.quantile(.975)),
        slope_percent_points_yr=("slope_percent_points_yr", "median"), slope_percent_points_yr_low=("slope_percent_points_yr", lambda x: x.quantile(.025)), slope_percent_points_yr_high=("slope_percent_points_yr", lambda x: x.quantile(.975)),
        r2_median=("r2", "median"), rmse_median=("rmse", "median"), reference_area_km2=("reference_area_km2", "first")
    )
    return annual, trends, summary


def representative_glaciers(summaries: pd.DataFrame) -> pd.DataFrame:
    selections = []
    for surging, group in [(True, "surging"), (False, "non-surging")]:
        subset = summaries[summaries.surging == surging]
        selections.extend([subset.nlargest(2, "slope_km2_yr"), subset.nsmallest(2, "slope_km2_yr")])
    result = pd.concat(selections).drop_duplicates("anlys_id").copy()
    result["trend_direction"] = np.where(result.slope_km2_yr >= 0, "most_positive", "most_negative")
    return result.sort_values(["surging", "trend_direction", "slope_km2_yr"], ascending=[False, True, False])


def run(input_dir=Path("input"), output_dir=Path("output")) -> dict[str, pd.DataFrame]:
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = read_predictions(input_dir)
    metadata = read_surge_database(input_dir / "RGI_V7_Surge_Database.zip")
    data, metadata = attach_metadata(data, metadata)
    model_trends, glacier_trends = fit_trends(data)
    annual, regional_model, regional_summary = regional_trends(data, metadata)
    reps = representative_glaciers(glacier_trends)
    tables = {"annual_model": data, "glacier_model_trends": model_trends, "glacier_trends": glacier_trends, "regional_annual_model": annual, "regional_model_trends": regional_model, "regional_trends": regional_summary, "representative_glaciers": reps}
    for name, table in tables.items():
        table = table.reset_index(drop=True).copy()
        for column in table.select_dtypes(include="object"):
            table[column] = table[column].astype("string")
        table.to_feather(output_dir / f"{name}.arrow")
    return tables


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("input"))
    parser.add_argument("--output", type=Path, default=Path("output"))
    args = parser.parse_args()
    run(args.input, args.output)
