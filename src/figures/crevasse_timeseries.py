"""Crevasse coverage timeseries with Arnesenbreen raster panels."""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from rasterio.transform import array_bounds
from rasterio.windows import from_bounds

YEARS = range(2017, 2026)
ARNESEN_YEARS = range(2018, 2026)
ARNESEN_OUTLINE_YEARS = {
    2018: 2018,
    2019: 2020,
    2020: 2020,
    2021: 2021,
    2022: 2022,
    2023: 2023,
    2024: 2023,
    2025: 2023,
}
GLACIERS = [
    "Recherchebreen",
    "Moršnevbreen",
    "Arnesenbreen",
    "Borebreen",
    "Vallåkrabreen",
    "Liestølbreen",
]
RGI_NAME_MAP = {
    "Morsjnevbreen / Strongbreen": "Moršnevbreen",
}


def rgi_timeseries_data(annual_model_path: Path) -> pd.DataFrame:
    """Build model-wise coverage timeseries from the RGI-linked exports."""
    data = pd.read_feather(annual_model_path)
    data["glac_name"] = data["glac_name"].replace(RGI_NAME_MAP)
    data = data[data.glac_name.isin(GLACIERS)].copy()
    # Several RGI7 polygons share a name; use the largest named polygon.
    largest_ids = (
        data[["rgi_id", "glac_name", "area_km2"]]
        .drop_duplicates()
        .sort_values("area_km2", ascending=False)
        .drop_duplicates("glac_name")
        .rgi_id
    )
    data = data[data.rgi_id.isin(largest_ids)].copy()
    data["crevasse_area_km2"] = data.positive_area_m2 / 1e6
    return data


def annual_arnesen_outlines(path: Path) -> dict[int, object]:
    """Select the requested annual outline, with explicit fallback years."""
    outlines = gpd.read_file(path)
    outlines["outline_year"] = pd.to_datetime(outlines["src_date"]).dt.year
    outlines = outlines[outlines.glac_name == "Arnesenbreen"]
    selected = {}
    for display_year, outline_year in ARNESEN_OUTLINE_YEARS.items():
        matches = outlines[outlines.outline_year == outline_year]
        if len(matches) != 1:
            raise ValueError(f"Expected one Arnesenbreen outline for {outline_year}, found {len(matches)}")
        selected[display_year] = matches.iloc[0]
    return selected


def sample_outline(outline, raster_path: Path) -> tuple[np.ndarray, rasterio.Affine]:
    """Read the vote-fraction raster only within the outline bounds."""
    with rasterio.open(raster_path) as raster:
        window = from_bounds(*outline.geometry.bounds, transform=raster.transform)
        window = window.round_offsets().round_lengths()
        vote_fraction = raster.read(1, window=window).astype(float) / 255
        return vote_fraction, raster.window_transform(window)


def sample_arnesen_maps(annual_maps: dict[int, object], vote_fraction_dir: Path) -> dict[int, tuple[np.ndarray, np.ndarray, rasterio.Affine, object]]:
    arnesen_maps = {}
    for year in ARNESEN_YEARS:
        raster_path = vote_fraction_dir / f"crevasse_vote_fraction_{year}.tif"
        map_outline = annual_maps[year]
        map_vote_image, map_transform = sample_outline(map_outline, raster_path)
        map_inside = geometry_mask([map_outline.geometry], out_shape=map_vote_image.shape, transform=map_transform, invert=True)
        arnesen_maps[year] = (map_vote_image, map_inside, map_transform, map_outline)
    return arnesen_maps


def make_figure(values: pd.DataFrame, arnesen_maps: dict, output_path: Path) -> None:
    fig = plt.figure(figsize=(10, 3.9))
    outer = fig.add_gridspec(1, 2, width_ratios=[4, 4], wspace=0.05)
    timeseries_grid = outer[0].subgridspec(2, 3, wspace=0.35, hspace=0.18)
    map_grid = outer[1].subgridspec(2, 4, wspace=0.02, hspace=-0.18)
    timeseries_axes = timeseries_grid.subplots()
    map_axes = map_grid.subplots()

    map_bounds = np.array([arnesen_maps[year][3].geometry.bounds for year in ARNESEN_YEARS])
    left, bottom = map_bounds[:, :2].min(axis=0)
    right, top = map_bounds[:, 2:].max(axis=0)
    padding = 0.03 * max(right - left, top - bottom)
    map_extent = (left - padding, right + padding, bottom - padding, top + padding)

    for index, glacier in enumerate(GLACIERS):
        ax = timeseries_axes.ravel()[index]
        subset = values[values.glac_name == glacier]
        annual = subset.groupby("year").crevasse_area_km2.agg(
            median="median", low=lambda x: x.quantile(0.025), high=lambda x: x.quantile(0.975)
        ).reset_index()
        ax.fill_between(annual.year, annual["low"], annual["high"], color="#d98c8c", alpha=0.35, linewidth=0)
        ax.plot(annual.year, annual["median"], color="#963b3b", marker="o", markersize=2.5)
        ax.set_title(glacier, fontsize=8, pad=3)
        ax.set_ylim(0, max(1, 1.1 * annual["high"].max()))
        ax.set_xlim(2016.7, 2025.3)
        ax.set_xticks([2019, 2023])
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.set_xticks(list(YEARS), minor=True)
        ax.grid(zorder=0, color="#dddddd", linewidth=0.5)
        if index % 3 == 0:
            ax.set_ylabel("Crevassed area (km²)")
        if index < 3:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Year")

    for map_index, year in enumerate(ARNESEN_YEARS):
        ax = map_axes.ravel()[map_index]
        image, inside, transform, map_outline = arnesen_maps[year]
        display = np.ma.masked_where(~inside, image)
        height, width = display.shape
        left, bottom, right, top = array_bounds(height, width, transform)
        img = ax.imshow(display, cmap="YlOrRd", vmin=0, vmax=1, origin="upper", interpolation="none", extent=(left, right, bottom, top))
        boundary = gpd.GeoSeries([map_outline.geometry], crs="EPSG:32633")
        boundary.boundary.plot(ax=ax, color="#333333", linewidth=0.6)
        plt.text(0.03, 0.97, str(year), transform=ax.transAxes, ha="left", va="top")

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(map_extent[0], map_extent[1])
        ax.set_ylim(map_extent[2], map_extent[3])
        ax.set_aspect("equal")

        if map_index == (len(ARNESEN_YEARS) - 1):
            ax_inset = ax.inset_axes([0.52, 0.07, 0.1, 0.3])
            cbar = plt.colorbar(img, ax_inset)
            cbar.set_ticks(ticks=[0, 1],labels=["0%", "100%"], fontsize=8) 
        elif map_index == (len(ARNESEN_YEARS) - 2):
            scale_km = 3
            scale_xmax = ax.get_xlim()[1] - 1200

            scale_y = ax.get_ylim()[0] + 1500
            ax.plot([scale_xmax - scale_km * 1000, scale_xmax], [scale_y] * 2, color="black")
            for (label, xcoord) in [(f"{scale_km} km", scale_xmax), ("0 km", scale_xmax - scale_km * 1000)]:
                ax.text(xcoord, scale_y - 250,label , transform=ax.transData, fontsize=8, ha="center", va="top")
                ax.plot([xcoord] * 2, [scale_y, scale_y + 250], color="black")
            

    fig.subplots_adjust(left=0.07, right=0.99, top=0.99, bottom=0.08)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def run(input_dir: Path = Path("input"), output_dir: Path = Path("output/figures"), annual_model_path: Path | None = None) -> pd.DataFrame:
    annual_model_path = annual_model_path or output_dir.parent / "annual_model.arrow"
    values = rgi_timeseries_data(annual_model_path)
    annual_maps = annual_arnesen_outlines(input_dir / "glacier_outlines_mod.sqlite")
    arnesen_maps = sample_arnesen_maps(annual_maps, output_dir.parent / "vote_fractions")
    output_dir.mkdir(parents=True, exist_ok=True)
    values.to_feather(output_dir / "crevasse_timeseries.arrow")
    make_figure(values, arnesen_maps, output_dir / "crevasse_timeseries.png")
    return values


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("input"))
    parser.add_argument("--output", type=Path, default=Path("output/figures"))
    args = parser.parse_args()
    run(args.input, args.output)
