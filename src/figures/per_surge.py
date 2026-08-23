"""Render configured per-surge glacier figures."""

from __future__ import annotations

import re

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio.transform

from src import raster, settings, surge


def slugify(name: str, rgi_id: str) -> str:
    """Create a stable, filesystem-safe output directory name."""
    ascii_name = name.encode("ascii", "ignore").decode("ascii") if name else "unnamed"
    ascii_name = re.sub(r"[^A-Za-z0-9]+", "_", ascii_name).strip("_").lower() or "unnamed"
    return f"{str(rgi_id).split('-')[-1]}_{ascii_name}"


def select_glaciers(trends: pd.DataFrame) -> pd.DataFrame:
    """Select all configured surging glaciers above the area threshold."""
    selected = trends[(trends.surging) & (trends.area_km2 >= 50)].copy()
    if selected.empty:
        raise ValueError("No surging glaciers meet the configured area threshold")
    return selected


def render(glacier: pd.Series, data: pd.DataFrame, geometry: object) -> None:
    """Render one glacier's annual model envelope and annual raster panels."""
    output_dir = settings.FIGURE_DIR / "per-surge" / slugify(glacier.glac_name, glacier.rgi_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    annual = data.groupby("year").crevassed_area_km2.agg(
        median="median", low=lambda x: x.quantile(0.025), high=lambda x: x.quantile(0.975)
    )
    fig = plt.figure(figsize=(9, 8))
    grid = fig.add_gridspec(4, 3, height_ratios=[1.05, 1, 1, 1])
    trend_ax = fig.add_subplot(grid[0, :])
    trend_ax.fill_between(annual.index, annual["low"], annual["high"], color="#d98c8c", alpha=0.35)
    trend_ax.plot(annual.index, annual["median"], color="#963b3b", marker="o", markersize=3)
    trend_ax.set(
        title=f"{glacier.glac_name} ({glacier.rgi_id})",
        ylabel="Crevassed area (km²)",
        xlim=(2016.7, 2025.3),
        ylim=(0, max(1, 1.1 * annual["high"].max())),
    )
    trend_ax.grid(color="#ddd", linewidth=0.5)
    for index, year in enumerate(settings.YEARS):
        ax = fig.add_subplot(grid[1 + index // 3, index % 3])
        image, inside, transform = raster.crop_vote_fraction(
            geometry, settings.RASTER_DIR / settings.VOTE_RASTER_TEMPLATE.format(year=year)
        )
        height, width = image.shape
        left, bottom, right, top = rasterio.transform.array_bounds(height, width, transform)
        ax.imshow(
            np.ma.masked_where(~inside, image),
            cmap="YlOrRd",
            vmin=0,
            vmax=1,
            origin="upper",
            interpolation="none",
            extent=(left, right, bottom, top),
        )
        gpd.GeoSeries([geometry], crs=settings.RASTER_CRS).boundary.plot(ax=ax, color="#333", linewidth=0.5)
        ax.set(title=str(year), xticks=[], yticks=[], aspect="equal")
    fig.tight_layout()
    fig.savefig(output_dir / f"{output_dir.name}.png", dpi=250, bbox_inches="tight")
    fig.savefig(output_dir / f"{output_dir.name}.pdf", bbox_inches="tight")
    plt.close(fig)


def run() -> None:
    """Render all configured per-surge figures."""
    trends = pd.read_feather(settings.DATA_DIR / "glacier-trends.arrow")
    annual = pd.read_feather(settings.DATA_DIR / "annual-model.arrow")
    polygons = surge.read_database().to_crs(settings.RASTER_CRS)
    for _, glacier in select_glaciers(trends).iterrows():
        matches = polygons[polygons.rgi_id == glacier.rgi_id]
        if len(matches) != 1:
            raise ValueError(f"Expected one RGI polygon for {glacier.rgi_id}, found {len(matches)}")
        data = annual[annual.anlys_id == glacier.anlys_id].copy()
        data["crevassed_area_km2"] = data.positive_area_m2 / 1e6
        render(glacier, data, matches.geometry.iloc[0])


if __name__ == "__main__":
    run()
