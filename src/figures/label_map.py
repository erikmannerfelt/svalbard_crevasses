"""Render a spatial/temporal coverage map of crevasse training labels."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Final

import geopandas as gpd
import matplotlib.colors
import matplotlib.patches
import matplotlib.pyplot as plt
import pandas as pd
import shapely.geometry

from src import label_audit, settings, surge

# Okabe-Ito vermillion and blue. Chosen over a red/green pair because they stay
# distinguishable under deuteranopia, protanopia and tritanopia, where hue
# separation along the blue-orange axis survives and red-green does not.
POSITIVE_COLOR: Final = "#d55e00"
NEGATIVE_COLOR: Final = "#0072b2"
FILL_ALPHA: Final = 0.55
COLUMNS: Final = 3
# The supplement's \textwidth is 503.6 pt. Authoring the figure at exactly that
# width means LaTeX includes it at scale 1, so the font sizes set here are the
# sizes that reach the page.
FIGURE_WIDTH_IN: Final = 503.6 / 72.27
# Vertical room for the panel title, on top of the map itself.
TITLE_ALLOWANCE_IN: Final = 0.30
TITLE_FONTSIZE: Final = 9
LEGEND_FONTSIZE: Final = 8
# The land and glacier outlines are ~1600 polygons repeated per panel, which as
# vectors dominate the PDF. Rasterizing only the basemap keeps the label
# polygons, text and legend as vectors. 500 dpi matches `figures.trend_map`;
# 300 was visibly soft at print size and the returns fall off well before 900.
BASEMAP_DPI: Final = 500
LAND_COLOR: Final = "#e6e6e1"
LAND_EDGE: Final = "#aaaaa5"
GLACIER_EDGE: Final = "#bbbbb8"
# Matches the trend map, so the two figures are read at the same extent.
MAP_EXTENT: Final = (400000, 848000, 8480000, 8970000)


def render(
    zoned: gpd.GeoDataFrame,
    land: gpd.GeoSeries,
    glaciers: gpd.GeoDataFrame,
    summary: pd.DataFrame,
    output_path: Path,
) -> None:
    """Render a year-faceted grid of zone outlines and label locations.

    One panel per year that actually carries labels. Labelling is
    every-other-year by design, so faceting over `settings.YEARS` would
    leave empty panels that read as gaps in coverage rather than as years
    nobody set out to label. Each panel draws land and RGI7 glacier outlines
    over the same extent as the trend map, with that year's label polygons
    overlaid and colored by class.
    """
    years = sorted(zoned.year.unique())
    columns = min(COLUMNS, len(years))
    rows = math.ceil(len(years) / columns)
    minx, maxx, miny, maxy = MAP_EXTENT
    # Give each panel the aspect of the mapped extent. A squarer panel would
    # leave whitespace to the left and right of every map, because the axes
    # hold their equal aspect and the extent is taller than it is wide.
    panel_width = FIGURE_WIDTH_IN / columns
    panel_height = panel_width * (maxy - miny) / (maxx - minx) + TITLE_ALLOWANCE_IN
    fig, axes = plt.subplots(rows, columns, figsize=(FIGURE_WIDTH_IN, panel_height * rows), squeeze=False)
    for ax, year in zip(axes.flat, years, strict=False):
        land.plot(ax=ax, color=LAND_COLOR, edgecolor=LAND_EDGE, linewidth=0.25, rasterized=True)
        glaciers.plot(ax=ax, color="white", edgecolor=GLACIER_EDGE, linewidth=0.1, rasterized=True)
        year_labels = zoned[zoned.year == year]
        negative, positive = year_labels[~year_labels.crevassed], year_labels[year_labels.crevassed]
        for subset, color in ((negative, NEGATIVE_COLOR), (positive, POSITIVE_COLOR)):
            if subset.empty:
                continue
            subset.plot(
                ax=ax,
                facecolor=matplotlib.colors.to_rgba(color, FILL_ALPHA),
                edgecolor=color,
                linewidth=0.5,
            )
        counts = summary.loc[summary.year == year, ["positive_count", "negative_count"]].sum()
        ax.set_title(
            f"{year}  (+{int(counts.positive_count)} / -{int(counts.negative_count)})",
            fontsize=TITLE_FONTSIZE,
        )
        ax.set(xlim=(minx, maxx), ylim=(miny, maxy), xticks=[], yticks=[], aspect="equal")
    for unused in axes.flat[len(years) :]:
        unused.set_axis_off()
    handles = [
        matplotlib.patches.Patch(
            facecolor=matplotlib.colors.to_rgba(POSITIVE_COLOR, FILL_ALPHA),
            edgecolor=POSITIVE_COLOR,
            label="Crevassed",
        ),
        matplotlib.patches.Patch(
            facecolor=matplotlib.colors.to_rgba(NEGATIVE_COLOR, FILL_ALPHA),
            edgecolor=NEGATIVE_COLOR,
            label="Not crevassed",
        ),
    ]
    # Inside the last panel rather than under the figure: the lower right of
    # every panel is open water, so the legend costs no map area.
    axes.flat[len(years) - 1].legend(
        handles=handles, loc="lower right", ncol=1, fontsize=LEGEND_FONTSIZE, framealpha=0.9, borderpad=0.5
    )
    fig.tight_layout(w_pad=0.4, h_pad=0.8, pad=0.3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # No bbox_inches="tight": the saved size must stay exactly FIGURE_WIDTH_IN
    # so that \includegraphics does not rescale it and shrink the fonts.
    fig.savefig(output_path, dpi=250)
    fig.savefig(output_path.with_suffix(".pdf"), dpi=BASEMAP_DPI)
    plt.close(fig)


def run() -> None:
    """Assemble label-audit inputs and render the coverage map figure."""
    zones = gpd.read_file(settings.GLACIER_ZONES_FILE)
    labels = label_audit.read_labels()
    in_bounds, _out_of_bounds = label_audit.filter_to_svalbard(labels)
    zoned = label_audit.join_zones(in_bounds, zones)
    summary = label_audit.zone_year_summary(zoned, zones)
    land = gpd.read_file(settings.LAND_FILE).difference(shapely.geometry.box(730000, 8740000, 900000, 8820000))
    glaciers = surge.read_database().to_crs(settings.RASTER_CRS)
    render(zoned, land.to_crs(settings.RASTER_CRS), glaciers, summary, settings.FIGURE_DIR / "label-audit-map.png")


if __name__ == "__main__":
    run()
