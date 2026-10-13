"""Crevasse coverage timeseries with Arnesenbreen raster panels."""

from __future__ import annotations

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch
from rasterio.features import geometry_mask
from rasterio.transform import array_bounds
from rasterio.windows import from_bounds

from src import settings


def rgi_timeseries_data() -> pd.DataFrame:
    """Build model-wise coverage timeseries from the RGI-linked exports."""
    data = pd.read_feather(settings.DATA_DIR / "annual-model.arrow")
    data["glac_name"] = data["glac_name"].replace(settings.RGI_NAME_MAP)
    data = data[data.glac_name.isin(settings.SELECTED_GLACIERS)].copy()
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


def annual_arnesen_outlines() -> dict[int, object]:
    """Select the requested annual outline, with explicit fallback years."""
    outlines = gpd.read_file(settings.OUTLINE_FILE)
    outlines["outline_year"] = pd.to_datetime(outlines["src_date"]).dt.year
    outlines = outlines[outlines.glac_name == "Arnesenbreen"]
    selected = {}
    for display_year, outline_year in settings.ARNESEN_OUTLINE_YEARS.items():
        matches = outlines[outlines.outline_year == outline_year]
        if len(matches) != 1:
            raise ValueError(f"Expected one Arnesenbreen outline for {outline_year}, found {len(matches)}")
        selected[display_year] = matches.iloc[0]
    return selected


def sample_outline(outline: object, raster_path: object) -> tuple[np.ndarray, rasterio.Affine]:
    """Read the vote-fraction raster only within the outline bounds."""
    with rasterio.open(raster_path) as raster_file:
        window = from_bounds(*outline.geometry.bounds, transform=raster_file.transform)
        window = window.round_offsets().round_lengths()
        vote_fraction = raster_file.read(1, window=window).astype(float) / 255
        return vote_fraction, raster_file.window_transform(window)


def sample_arnesen_maps(
    annual_maps: dict[int, object],
) -> dict[int, tuple[np.ndarray, np.ndarray, rasterio.Affine, object]]:
    """Prepare masked annual vote-fraction images for the map panels."""
    arnesen_maps = {}
    for year in settings.ARNESEN_YEARS:
        raster_path = settings.RASTER_DIR / settings.VOTE_RASTER_TEMPLATE.format(year=year)
        map_outline = annual_maps[year]
        map_vote_image, map_transform = sample_outline(map_outline, raster_path)
        map_inside = geometry_mask(
            [map_outline.geometry],
            out_shape=map_vote_image.shape,
            transform=map_transform,
            invert=True,
        )
        arnesen_maps[year] = (map_vote_image, map_inside, map_transform, map_outline)
    return arnesen_maps


def make_figure(values: pd.DataFrame, arnesen_maps: dict[int, tuple], output_path: object) -> None:
    """Render the timeseries and annual Arnesenbreen panels."""
    fig = plt.figure(figsize=(8, 3))
    outer = fig.add_gridspec(1, 2, width_ratios=[3.2, 4], wspace=0.05)
    timeseries_grid = outer[0].subgridspec(2, 3, wspace=0.36, hspace=0.18)
    map_grid = outer[1].subgridspec(2, 4, wspace=0.02, hspace=-0.00)
    timeseries_axes = timeseries_grid.subplots()
    map_axes = map_grid.subplots()

    map_bounds = np.array([arnesen_maps[year][3].geometry.bounds for year in settings.ARNESEN_YEARS])
    left, bottom = map_bounds[:, :2].min(axis=0)
    right, top = map_bounds[:, 2:].max(axis=0)
    padding = 0.03 * max(right - left, top - bottom)
    map_extent = (left - padding, right + padding, bottom - padding, top + padding)

    for index, glacier in enumerate(settings.SELECTED_GLACIERS):
        ax = timeseries_axes.ravel()[index]
        subset = values[values.glac_name == glacier]
        annual = (
            subset.groupby("year")
            .crevasse_area_km2.agg(
                median="median",
                low=lambda x: x.quantile(settings.ENSEMBLE_QUANTILES[0]),
                high=lambda x: x.quantile(settings.ENSEMBLE_QUANTILES[1]),
            )
            .reset_index()
        )
        ax.fill_between(annual.year, annual["low"], annual["high"], color="#d98c8c", alpha=0.35, linewidth=0)
        ax.plot(annual.year, annual["median"], color="#963b3b", marker="o", markersize=2.5)
        ax.set_title(glacier, fontsize=8, pad=3)
        ax.set_ylim(0, max(1, 1.1 * annual["high"].max()))
        ax.set_xlim(2016.7, 2025.3)
        ax.set_xticks([2019, 2023])
        ax.tick_params(labelsize=8)
        ax.set_xticks(list(settings.YEARS), minor=True)
        ax.grid(zorder=0, color="#dddddd", linewidth=0.5)
        if index < 3:
            ax.set_xticklabels([])

    fig.text(
        0.00,
        0.5,
        "Crevassed area (km²)",
        fontsize=8,
        rotation=90,
        ha="left",
        va="center",
        transform=fig.transFigure,
    )

    for map_index, year in enumerate(settings.ARNESEN_YEARS):
        ax = map_axes.ravel()[map_index]
        image, inside, transform, map_outline = arnesen_maps[year]
        display = np.ma.masked_where(~inside, image)
        height, width = display.shape
        left, bottom, right, top = array_bounds(height, width, transform)
        image_artist = ax.imshow(
            display,
            cmap="YlOrRd",
            vmin=0,
            vmax=1,
            origin="upper",
            interpolation="none",
            extent=(left, right, bottom, top),
        )
        boundary = gpd.GeoSeries([map_outline.geometry], crs=settings.RASTER_CRS)
        boundary.boundary.plot(ax=ax, color="#333333", linewidth=0.6)
        ax.text(0.03, 0.97, str(year), transform=ax.transAxes, ha="left", va="top")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(map_extent[0], map_extent[1])
        ax.set_ylim(map_extent[2], map_extent[3])
        ax.set_aspect("equal")
        ax.set_axis_off()

        if map_index == 0:
            ax.annotate(
                "Ice\nflow",
                xytext=(0.6, 0.2),
                xy=(0.9, 0.6),
                arrowprops=dict(arrowstyle="->", color="#339", lw=2),
                color="#339",
                xycoords="axes fraction",
                textcoords="axes fraction",
                ha="center",
                fontsize=8,
            )
        elif map_index == 2:
            ax_inset = ax.inset_axes([0.52, 0.07, 0.1, 0.3])
            colorbar = fig.colorbar(image_artist, ax_inset)
            colorbar.set_ticks(ticks=[0, 1], labels=["0%", "100%"], fontsize=8)
        elif map_index == len(settings.ARNESEN_YEARS) - 2:
            scale_km = 3
            scale_xmax = ax.get_xlim()[1] - 1200
            scale_y = ax.get_ylim()[0] + 1500
            ax.plot([scale_xmax - scale_km * 1000, scale_xmax], [scale_y] * 2, color="black")
            for label, xcoord in [
                (f"{scale_km} km", scale_xmax),
                ("0 km", scale_xmax - scale_km * 1000),
            ]:
                ax.text(
                    xcoord,
                    scale_y - 250,
                    label,
                    transform=ax.transData,
                    fontsize=8,
                    ha="center",
                    va="top",
                )
                ax.plot([xcoord] * 2, [scale_y, scale_y + 250], color="black")

    arnesen_ax = timeseries_axes[0, 2]
    for source_corner, target_ax, target_corner in [
        ((1, 1), map_axes[0, 0], (0, 1)),
        ((1, 0), map_axes[1, 0], (0, 0)),
    ]:
        fig.add_artist(
            ConnectionPatch(
                xyA=source_corner,
                coordsA=arnesen_ax.transAxes,
                xyB=target_corner,
                coordsB=target_ax.transAxes,
                color="k",
                linewidth=0.8,
                clip_on=False,
                arrowstyle="-",
                zorder=0,
            )
        )

    fig.subplots_adjust(left=0.05, right=0.999, top=0.99, bottom=0.08)
    fig.canvas.draw()
    map_positions = [ax.get_position() for ax in map_axes.ravel()]
    map_left = min(position.x0 for position in map_positions)
    map_bottom = min(position.y0 for position in map_positions)
    map_right = max(position.x1 for position in map_positions)
    map_top = max(position.y1 for position in map_positions)
    fig.add_artist(
        Line2D(
            [map_left, map_right, map_right, map_left],
            [map_bottom, map_bottom, map_top, map_top],
            transform=fig.transFigure,
            color="k",
            linewidth=0.8,
            clip_on=False,
            zorder=3,
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def run() -> pd.DataFrame:
    """Render the configured six-glacier figure."""
    values = rgi_timeseries_data()
    arnesen_maps = sample_arnesen_maps(annual_arnesen_outlines())
    output_path = settings.FIGURE_DIR / "timeseries.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    values.to_feather(settings.FIGURE_DIR / "timeseries.arrow")
    make_figure(values, arnesen_maps, output_path)
    return values


if __name__ == "__main__":
    run()
