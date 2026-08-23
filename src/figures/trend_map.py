"""Map-data helpers and the crevasse trend map."""

from __future__ import annotations

import geopandas as gpd
import matplotlib.patheffects
import matplotlib.pyplot as plt
import pandas as pd
import shapely.geometry
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle

from src import settings, surge

GLACIER_ANNOTATIONS = {
    "Recherchebreen": {"xy": [500000, 8592000], "xytext": [460000, 8592000]},
    "Moršnevbreen": {"xy": [562000, 8621000], "xytext": [590000, 8621000]},
    "Arnesenbreen": {"xy": [574000, 8639000], "xytext": [594000, 8655000]},
    "Borebreen": {"xy": [473000, 8710000], "xytext": [430000, 8710000]},
    "Vallåkrabreen": {"xy": [550320, 8644000], "xytext": [520000, 8635000]},
    "Liestølbreen": {"xy": [545000, 8596000], "xytext": [575000, 8586000]},
}


def classify_surge_periods(glaciers: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Classify surges after the dataset midpoint and in the prior window."""
    result = glaciers.copy()
    recent = [
        surge.surge_overlaps(term, onset, settings.MIDPOINT_YEAR - 1, settings.END_YEAR)
        for term, onset in zip(result["S_Term"], result["S_Onset"], strict=True)
    ]
    result["surge_after_midpoint"] = recent
    result["surge_before_midpoint"] = [
        not after
        and surge.surge_overlaps(term, onset, settings.YEARS[0] - settings.PRE_SURGE_YEARS, settings.MIDPOINT_YEAR)
        for after, term, onset in zip(recent, result["S_Term"], result["S_Onset"], strict=True)
    ]
    return result


def add_agreement_matrix(ax: object, glaciers: gpd.GeoDataFrame, summary: pd.DataFrame, cmap: str) -> None:
    """Add the colored trend-rate matrix from the analysis CSV."""
    glaciers = glaciers.copy()
    glaciers["trend_class"] = "uncertain"
    glaciers.loc[glaciers["slope_km2_yr_high"] < 0, "trend_class"] = "negative"
    glaciers.loc[glaciers["slope_km2_yr_low"] > 0, "trend_class"] = "positive"
    glaciers["trend_direction"] = glaciers["slope_km2_yr"].ge(0).map({True: "positive", False: "negative"})
    labels = {
        "pre": (f"Surges\n{settings.YEARS[0] - settings.PRE_SURGE_YEARS}–{settings.MIDPOINT_YEAR - 1}"),
        "recent": f"Surges\n{settings.MIDPOINT_YEAR}–{settings.END_YEAR}",
        "none": (f"No recorded\nsurge\n{settings.YEARS[0] - settings.PRE_SURGE_YEARS}–{settings.END_YEAR}"),
    }
    periods = ["pre", "recent", "none"]
    directions = ["negative", "positive"]
    colormap = plt.get_cmap(cmap)
    max_rate = summary["median_rate_km2_yr"].abs().max()
    rate_norm = TwoSlopeNorm(vmin=-max_rate, vcenter=0, vmax=max_rate if max_rate else 1)
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 4.05)
    ax.axis("off")
    for column, direction in enumerate(directions, start=1):
        ax.text(
            column + 0.5,
            2.0,
            f"{direction.capitalize()}\ntrend",
            fontsize=7.5,
            ha="center",
            va="center",
        )
    for row, period in enumerate(periods):
        y = 1.2 - row * 0.60
        for i in range(2):
            ax.text(
                0.47,
                y + 0.30,
                labels[period],
                fontsize=7,
                ha="center",
                va="center",
                path_effects=[matplotlib.patheffects.withStroke(foreground="white", linewidth=7)] if i == 0 else None,
                color="white" if i == 0 else "black",
            )
        for column, direction in enumerate(directions, start=1):
            cell = summary[(summary.surge_period == period) & (summary.trend_sign == direction)]
            if cell.empty:
                continue
            value = cell.iloc[0]
            rate = value.median_rate_km2_yr
            uncertainty = value.rate_uncertainty_km2_yr
            minimum_visible_rate = 0.03 * max_rate
            signed_rate = (
                max(rate, minimum_visible_rate) if direction == "positive" else min(rate, -minimum_visible_rate)
            )
            ax.add_patch(
                Rectangle(
                    (column, y),
                    1,
                    0.60,
                    facecolor=colormap(rate_norm(signed_rate)),
                    edgecolor="#888888",
                    linewidth=0.7,
                )
            )
            ax.text(
                column + 0.5,
                y + 0.37,
                f"{rate:+.0f} ± {uncertainty:.0f}",
                fontsize=7.5,
                fontweight="bold",
                ha="center",
                va="center",
                path_effects=[matplotlib.patheffects.withStroke(foreground="#444", linewidth=1.5)],
                color="white",
            )
            ax.text(
                column + 0.5,
                y + 0.14,
                f"(n$_{{sig}}$ = {int(value.significant_glacier_count)})",
                fontsize=6.5,
                ha="center",
                va="center",
                path_effects=[matplotlib.patheffects.withStroke(foreground="#444", linewidth=1.5)],
                color="white",
            )


def plot() -> None:
    """Render the configured glacier trend map."""
    trends_path = settings.DATA_DIR / "glacier-trends.arrow"
    trends = pd.read_feather(trends_path)
    summary = pd.read_csv(settings.DATA_DIR / "surge-period-trend-summary.csv")
    glaciers = surge.read_database().merge(
        trends.drop(columns=["glac_name", "S_Onset", "S_Term", "surging"], errors="ignore"),
        on="rgi_id",
        how="left",
        validate="one_to_one",
    )
    glaciers = classify_surge_periods(glaciers)
    land = gpd.read_file(settings.LAND_FILE).difference(shapely.geometry.box(730000, 8740000, 900000, 8820000))
    if glaciers.crs != land.crs:
        glaciers = glaciers.to_crs(land.crs)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_ylim(8480000, 8970000)
    ax.set_xlim(400000, 848000)
    land.plot(ax=ax, color="#e6e6e1", edgecolor="#aaaaa5", linewidth=0.25)
    cmap = "PuOr_r"
    norm = TwoSlopeNorm(vmin=-2, vcenter=0, vmax=2)
    glaciers.plot(
        ax=ax,
        column="slope_km2_yr",
        cmap=cmap,
        norm=norm,
        linewidth=0.08,
        edgecolor="#555555",
        missing_kwds={"color": "none", "edgecolor": "none"},
    )
    glaciers[glaciers["surge_after_midpoint"]].geometry.centroid.plot(
        ax=ax,
        marker="o",
        markersize=30,
        color="#ffd92f",
        edgecolor="#222222",
        linewidth=0.5,
        zorder=3,
        label=f"Surges {settings.MIDPOINT_YEAR}–{settings.END_YEAR}",
    )
    glaciers[glaciers["surge_before_midpoint"]].geometry.centroid.plot(
        ax=ax,
        marker="o",
        markersize=30,
        color="#377eb8",
        edgecolor="#222222",
        linewidth=0.5,
        zorder=3,
        label=f"Surges {settings.YEARS[0] - settings.PRE_SURGE_YEARS}–{settings.MIDPOINT_YEAR - 1}",
    )
    for glacier_name in settings.SELECTED_GLACIERS:
        ax.annotate(
            glacier_name[0],
            **GLACIER_ANNOTATIONS[glacier_name],
            arrowprops=dict(arrowstyle="->", color="#333", lw=2),
        )
    scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array([])
    colorbar_ax = ax.inset_axes([0.87, 0.52, 0.04, 0.32])
    fig.colorbar(scalar, cax=colorbar_ax, label="Crevasse area trend (km² year$^{-1}$)")
    ax.set_axis_off()
    ax.legend(loc="lower left", bbox_to_anchor=(0.02, 0.02), fontsize=8, framealpha=0.9)
    agreement_ax = ax.inset_axes([0.65, 0, 0.35, 0.60])
    add_agreement_matrix(agreement_ax, glaciers, summary, cmap)
    output_path = settings.FIGURE_DIR / "trend-map.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=400, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    plot()
