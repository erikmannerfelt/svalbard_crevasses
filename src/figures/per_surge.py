"""Prototype per-glacier figures for surging RGI7 glaciers."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from rasterio.transform import array_bounds
from rasterio.windows import from_bounds

YEARS = range(2017, 2026)
SURGE_MEMBER = "RGI_V7_Surge_Database/RGI2000-v7.0-G-07_svalbard_jan_mayen_Surge_Database.shp"


def slugify(name: str, rgi_id: str) -> str:
    """Create a stable, filesystem-safe output directory name."""
    ascii_name = name.encode("ascii", "ignore").decode("ascii") if name else "unnamed"
    ascii_name = re.sub(r"[^A-Za-z0-9]+", "_", ascii_name).strip("_").lower() or "unnamed"
    suffix = str(rgi_id).split("-")[-1]
    return f"{suffix}_{ascii_name}"


def load_rgi_polygons(surge_zip: Path) -> gpd.GeoDataFrame:
    return gpd.read_file(f"zip://{surge_zip.resolve()}!{SURGE_MEMBER}")


def select_glaciers(trends: pd.DataFrame, min_area_km2: float, glacier: str | None) -> pd.DataFrame:
    if glacier:
        selected = (
            trends[trends.glac_name == glacier]
            .sort_values("area_km2", ascending=False)
            .drop_duplicates("glac_name")
            .copy()
        )
    else:
        selected = trends[(trends.surging) & (trends.area_km2 >= min_area_km2)].copy()
    if selected.empty:
        raise ValueError("No surging glaciers match the requested selector and area threshold")
    return selected


def sample_vote_fraction(geometry, raster_path: Path) -> tuple[np.ndarray, np.ndarray, rasterio.Affine]:
    with rasterio.open(raster_path) as raster:
        window = from_bounds(*geometry.bounds, transform=raster.transform).round_offsets().round_lengths()
        values = raster.read(3, window=window, boundless=True)
        transform = raster.window_transform(window)
        inside = geometry_mask(
            [geometry], out_shape=values.shape, transform=transform, invert=True
        )
    return values.astype(float) / 255, inside, transform


def model_coverage(data: pd.DataFrame, anlys_id: int) -> pd.DataFrame:
    subset = data[data.anlys_id == anlys_id].copy()
    subset["crevassed_area_km2"] = subset.positive_area_m2 / 1e6
    return subset


def write_figure(
    glacier: pd.Series,
    model_data: pd.DataFrame,
    geometry,
    input_dir: Path,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    annual = model_data.groupby("year").crevassed_area_km2.agg(
        median="median",
        low=lambda values: values.quantile(0.025),
        high=lambda values: values.quantile(0.975),
    ).reset_index()
    annual["rgi_id"] = glacier.rgi_id
    annual["glac_name"] = glacier.glac_name
    annual["area_km2"] = glacier.area_km2
    annual.to_feather(output_dir / "annual_summary.arrow")

    fig = plt.figure(figsize=(9, 8))
    grid = fig.add_gridspec(4, 3, height_ratios=[1.05, 1, 1, 1], hspace=0.35, wspace=0.18)
    trend_ax = fig.add_subplot(grid[0, :])
    y_max = min(100, max(1, 1.1 * annual["high"].max()))
    trend_ax.fill_between(annual.year, annual["low"], annual["high"], color="#d98c8c", alpha=0.35, linewidth=0)
    trend_ax.plot(annual.year, annual["median"], color="#963b3b", marker="o", markersize=3)
    trend_ax.set_xlim(2016.7, 2025.3)
    trend_ax.set_ylim(0, y_max)
    trend_ax.set_ylabel("Crevassed area (km²)")
    trend_ax.set_xticks(list(YEARS))
    trend_ax.grid(color="#dddddd", linewidth=0.5)
    trend_ax.set_title(f"{glacier.glac_name} ({glacier.rgi_id})", fontsize=10)

    for index, year in enumerate(YEARS):
        ax = fig.add_subplot(grid[1 + index // 3, index % 3])
        image, inside, transform = sample_vote_fraction(
            geometry, input_dir / f"crevasse_ensemble_metrics_{year}.tif"
        )
        display = np.ma.masked_where(~inside, image)
        height, width = display.shape
        left, bottom, right, top = array_bounds(height, width, transform)
        ax.imshow(
            display,
            cmap="YlOrRd",
            vmin=0,
            vmax=1,
            origin="upper",
            interpolation="none",
            extent=(left, right, bottom, top),
        )
        gpd.GeoSeries([geometry], crs="EPSG:32633").boundary.plot(
            ax=ax, color="#333333", linewidth=0.5
        )
        ax.set_title(str(year), fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(output_dir / f"{output_dir.name}.png", dpi=250, bbox_inches="tight")
    fig.savefig(output_dir / f"{output_dir.name}.pdf", bbox_inches="tight")
    plt.close(fig)


def run(
    input_dir: Path = Path("input"),
    output_root: Path = Path("output/figures/persurge"),
    min_area_km2: float = 50,
    glacier: str | None = None,
) -> pd.DataFrame:
    trends = pd.read_feather(output_root.parent.parent / "glacier_trends.arrow")
    annual_model = pd.read_feather(output_root.parent.parent / "annual_model.arrow")
    selected = select_glaciers(trends, min_area_km2, glacier)
    polygons = load_rgi_polygons(input_dir / "RGI_V7_Surge_Database.zip")
    output_rows = []
    for _, row in selected.iterrows():
        polygon = polygons[polygons.rgi_id == row.rgi_id]
        if len(polygon) != 1:
            raise ValueError(f"Expected one RGI polygon for {row.rgi_id}, found {len(polygon)}")
        polygon = polygon.to_crs("EPSG:32633")
        output_dir = output_root / slugify(row.glac_name, row.rgi_id)
        write_figure(row, model_coverage(annual_model, row.anlys_id), polygon.geometry.iloc[0], input_dir, output_dir)
        output_rows.append({
            "rgi_id": row.rgi_id,
            "anlys_id": row.anlys_id,
            "glac_name": row.glac_name,
            "area_km2": row.area_km2,
            "output_dir": str(output_dir),
        })
    index = pd.DataFrame(output_rows)
    output_root.mkdir(parents=True, exist_ok=True)
    index.to_feather(output_root / "index.arrow")
    return index


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("input"))
    parser.add_argument("--output-root", type=Path, default=Path("output/figures/persurge"))
    parser.add_argument("--min-area-km2", type=float, default=50)
    parser.add_argument("--glacier", default=None)
    args = parser.parse_args()
    run(args.input, args.output_root, args.min_area_km2, args.glacier)
