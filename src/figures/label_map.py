"""Render a spatial/temporal coverage map of crevasse training labels."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

from src import label_audit, settings

POSITIVE_COLOR: Final = "#d1495b"
NEGATIVE_COLOR: Final = "#2b7a78"


def render(zoned: gpd.GeoDataFrame, zones: gpd.GeoDataFrame, summary: pd.DataFrame, output_path: Path) -> None:
    """Render a year-faceted grid of zone outlines and label locations.

    One panel per `settings.YEARS`, reusing the 3x3 layout established in
    `figures.per_surge.render`. Each panel draws the named zone polygons
    (colored per `glacier_zones.geojson`'s own ``color`` column) with that
    year's label polygons overlaid, colored by class. Years with zero
    labels render zone outlines only, annotated "no labels", so a full
    temporal gap is visible without cross-referencing the summary table.
    """
    fig, axes = plt.subplots(3, 3, figsize=(11, 11))
    minx, miny, maxx, maxy = zones.total_bounds
    for ax, year in zip(axes.flat, settings.YEARS, strict=True):
        zones.plot(ax=ax, color=zones["color"], edgecolor="#333333", linewidth=0.4, alpha=0.5)
        year_labels = zoned[zoned.year == year]
        negative, positive = year_labels[~year_labels.crevassed], year_labels[year_labels.crevassed]
        if not negative.empty:
            negative.plot(ax=ax, facecolor="none", edgecolor=NEGATIVE_COLOR, linewidth=0.8)
        if not positive.empty:
            positive.plot(ax=ax, facecolor="none", edgecolor=POSITIVE_COLOR, linewidth=0.8)
        if year_labels.empty:
            ax.text(
                0.5, 0.5, "no labels", transform=ax.transAxes, ha="center", va="center", fontsize=10, color="#888888"
            )
        counts = summary.loc[summary.year == year, ["positive_count", "negative_count"]].sum()
        ax.set(
            title=f"{year}  (+{int(counts.positive_count)} / -{int(counts.negative_count)})",
            xlim=(minx, maxx),
            ylim=(miny, maxy),
            xticks=[],
            yticks=[],
            aspect="equal",
        )
    handles = [
        plt.Line2D([0], [0], color=POSITIVE_COLOR, lw=1.5, label="Crevassed"),
        plt.Line2D([0], [0], color=NEGATIVE_COLOR, lw=1.5, label="Not crevassed"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def run() -> None:
    """Assemble label-audit inputs and render the coverage map figure."""
    zones = gpd.read_file(settings.GLACIER_ZONES_FILE)
    labels = label_audit.read_labels()
    in_bounds, _out_of_bounds = label_audit.filter_to_svalbard(labels)
    zoned = label_audit.join_zones(in_bounds, zones)
    summary = label_audit.zone_year_summary(zoned, zones)
    render(zoned, zones, summary, settings.FIGURE_DIR / "label-audit-map.png")


if __name__ == "__main__":
    run()
