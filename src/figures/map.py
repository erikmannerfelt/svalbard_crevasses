"""Map-data helpers and a first-pass crevasse trend map."""

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patheffects
import shapely.geometry
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pandas as pd

from src.analysis import END_YEAR, YEARS, _surge_overlaps, read_surge_database
import src.figures.crevasse_timeseries


PRE_YEARS = 5
MIDPOINT_YEAR = int(YEARS[len(YEARS) // 2])

GLACIER_ANNOTATIONS = {
    "Recherchebreen": {"xy": [500000, 8592000], "xytext": [460000, 8592000]},
    "Moršnevbreen": {"xy": [562000, 8621000], "xytext": [590000, 8621000]},
    "Arnesenbreen": {"xy": [574000, 8639000], "xytext": [594000, 8655000]},
    "Borebreen": {"xy": [473000, 8710000], "xytext": [430000, 8710000]},
    "Vallåkrabreen": {"xy": [550320, 8644000], "xytext": [520000, 8635000]},
    "Liestølbreen": {"xy": [545000, 8596000], "xytext": [575000, 8586000]},
}


def glacier_trends_with_geometry(trends_path: Path, surge_zip: Path) -> gpd.GeoDataFrame:
    """Join glacier trend summaries to the RGI polygons for map plotting."""
    polygons = read_surge_database(surge_zip)
    trends = pd.read_feather(trends_path)
    return polygons.merge(trends, on="rgi_id", how="left", validate="one_to_one")


def classify_surge_periods(glaciers: gpd.GeoDataFrame, pre_years: int = PRE_YEARS) -> gpd.GeoDataFrame:
    """Classify surges after the dataset midpoint and in the prior window."""
    pre_start = int(YEARS[0]) - pre_years
    glaciers = glaciers.copy()
    glaciers["surge_after_midpoint"] = [
        # _surge_overlaps uses an exclusive lower bound for the end year.
        _surge_overlaps(term, onset, min_year=MIDPOINT_YEAR - 1, max_year=END_YEAR)
        for term, onset in zip(glaciers["S_Term"], glaciers["S_Onset"])
    ]
    glaciers["surge_before_midpoint"] = [
        not after and _surge_overlaps(term, onset, min_year=pre_start, max_year=MIDPOINT_YEAR)
        for after, term, onset in zip(
            glaciers["surge_after_midpoint"], glaciers["S_Term"], glaciers["S_Onset"]
        )
    ]
    return glaciers


def add_agreement_matrix(ax, glaciers: gpd.GeoDataFrame, cmap: str) -> None:
    """Add a compact count and trend-magnitude agreement matrix."""
    glaciers = glaciers.copy()
    glaciers["trend_class"] = "uncertain"
    glaciers.loc[glaciers["slope_km2_yr_high"] < 0, "trend_class"] = "negative"
    glaciers.loc[glaciers["slope_km2_yr_low"] > 0, "trend_class"] = "positive"
    periods = [f"Surges\n{int(YEARS[0]) - PRE_YEARS}-{MIDPOINT_YEAR - 1}", f"Surges\n{MIDPOINT_YEAR}-{END_YEAR}", "No recorded\nsurge"]
    glaciers["surge_class"] = periods[-1]
    glaciers.loc[glaciers["surge_before_midpoint"], "surge_class"] = periods[0]
    glaciers.loc[glaciers["surge_after_midpoint"], "surge_class"] = periods[1]

    classified = glaciers[glaciers["trend_class"] != "uncertain"].copy()
    classified["absolute_trend"] = classified["slope_km2_yr"].abs()
    total_absolute_trend = classified["absolute_trend"].sum()
    directions = ["negative", "positive"]
    colormap = plt.get_cmap(cmap)
    max_share = 100 * classified.groupby("surge_class")["absolute_trend"].sum().div(total_absolute_trend).max()
    share_norm = TwoSlopeNorm(vmin=-max_share, vcenter=0, vmax=max_share if max_share else 1)

    ax.set_xlim(0, 3)
    ax.set_ylim(0, 4.05)
    ax.axis("off")
    # ax.text(1.5, 3.72, "Crevasse-area trend", fontsize=7.5, ha="center", va="center")
    for column, direction in enumerate(directions, start=1):
        ax.text(column + 0.5, 0.4 * 3 + 0.8, f"{direction.capitalize()}\ntrend", fontsize=7.5, ha="center", va="center")

    for row, period in enumerate(periods):
        y = 0.4 * 3 - row * 0.60
        ax.text(0.47, y + 0.30, period, fontsize=7, ha="center", va="center")
        for column, direction in enumerate(directions, start=1):
            subset = classified[(classified["surge_class"] == period) & (classified["trend_class"] == direction)]
            count = len(subset)
            share = 100 * subset["absolute_trend"].sum() / total_absolute_trend if total_absolute_trend else 0
            # Keep even small nonzero shares visibly on their directional side
            # of the diverging colormap rather than at its white midpoint.
            minimum_visible_share = 0.03 * max_share
            signed_share = max(share, minimum_visible_share) if direction == "positive" else -max(share, minimum_visible_share)
            ax.add_patch(Rectangle((column, y), 1, 0.60, facecolor=colormap(share_norm(signed_share)), edgecolor="#888888", linewidth=0.7))
            ax.text(column + 0.5, y + 0.37, f"n = {count}", fontsize=8, fontweight="bold", ha="center", va="center", path_effects=[matplotlib.patheffects.withStroke(foreground="#444", linewidth=1.5)], color="white")
            ax.text(column + 0.5, y + 0.14, f"{share:.1f}%", fontsize=8, ha="center", va="center", path_effects=[matplotlib.patheffects.withStroke(foreground="#444", linewidth=1.5)], color="white")



def plot_crevasse_trend_map(
    trends_path: Path,
    surge_zip: Path,
    land_path: Path,
    output_path: Path,
    limit: float = 2.0,
    pre_years: int = PRE_YEARS,
) -> None:
    """Plot glacier trend polygons over the S100 land polygons."""
    glaciers = classify_surge_periods(glacier_trends_with_geometry(trends_path, surge_zip), pre_years=pre_years)
    land = gpd.read_file(land_path).difference(
        # Remove Kong Karls Land because it's in the way of the colorbar
        shapely.geometry.box(730000, 8740000, 900000, 8820000)
    )

    print(land.iloc[0])

    if glaciers.crs != land.crs:
        glaciers = glaciers.to_crs(land.crs)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.set_ylim(8480000, 8970000)
    ax.set_xlim(400000, 848000)
    land.plot(ax=ax, color="#e6e6e1", edgecolor="#aaaaa5", linewidth=0.25)
    cmap = "PuOr_r"
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    glaciers.plot(
        ax=ax,
        column="slope_km2_yr",
        cmap=cmap,
        norm=norm,
        linewidth=0.08,
        edgecolor="#555555",
        missing_kwds={"color": "none", "edgecolor": "none"},
    )
    later_surges = glaciers[glaciers["surge_after_midpoint"]].copy()
    later_surges.geometry.centroid.plot(
        ax=ax,
        # marker="*",
        marker="o",
        markersize=40,
        color="#ffd92f",
        edgecolor="#222222",
        linewidth=0.5,
        zorder=3,
        label=f"Surges {MIDPOINT_YEAR}-{END_YEAR}",
    )
    earlier_surges = glaciers[glaciers["surge_before_midpoint"]].copy()
    earlier_surges.geometry.centroid.plot(
        ax=ax,
        # marker="*",
        marker="o",
        markersize=40,
        color="#377eb8",
        edgecolor="#222222",
        linewidth=0.5,
        zorder=3,
        label=f"Surges {int(YEARS[0]) - pre_years}-{MIDPOINT_YEAR - 1}",
    )

    for glacier_name in src.figures.crevasse_timeseries.GLACIERS:
        ax.annotate(
            glacier_name[0],
            **GLACIER_ANNOTATIONS[glacier_name],
             arrowprops=dict(arrowstyle="->", color="#333", lw=2)

        )
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    # Keep the colorbar's upper edge in the map's upper-right corner, but leave
    # room below it for the agreement matrix.
    colorbar_ax = ax.inset_axes([0.87, 0.52, 0.04, 0.32])
    fig.colorbar(sm, cax=colorbar_ax, label="Crevasse area trend (km² year$^{-1}$)")
    ax.set_axis_off()
    ax.legend(loc="lower left", bbox_to_anchor=(0.02, 0.02), fontsize=8, framealpha=0.9)
    agreement_ax = ax.inset_axes([0.67, 0, 0.33, 0.60])
    add_agreement_matrix(agreement_ax, glaciers, cmap)
    fig.savefig(output_path, dpi=400, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trends", type=Path, default=Path("output/glacier_trends.arrow"))
    parser.add_argument("--surge-database", type=Path, default=Path("input/RGI_V7_Surge_Database.zip"))
    parser.add_argument("--land", type=Path, default=Path("input/S100_Land_f.gpkg"))
    parser.add_argument("--output", type=Path, default=Path("output/figures/crevasse_area_trend_map.png"))
    parser.add_argument("--limit", type=float, default=2.0)
    parser.add_argument("--pre-years", type=int, default=PRE_YEARS)
    args = parser.parse_args()
    plot_crevasse_trend_map(args.trends, args.surge_database, args.land, args.output, args.limit, args.pre_years)
