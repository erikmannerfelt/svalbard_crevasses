"""Map-data helpers and a first-pass crevasse trend map."""

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import pandas as pd


def glacier_trends_with_geometry(trends_path: Path, surge_zip: Path) -> gpd.GeoDataFrame:
    """Join glacier trend summaries to the RGI polygons for map plotting."""
    member = "RGI_V7_Surge_Database/RGI2000-v7.0-G-07_svalbard_jan_mayen_Surge_Database.shp"
    polygons = gpd.read_file(f"zip://{Path(surge_zip).resolve()}!{member}")
    trends = pd.read_feather(trends_path)
    return polygons.merge(trends, on="rgi_id", how="left", validate="one_to_one")


def plot_crevasse_trend_map(
    trends_path: Path,
    surge_zip: Path,
    land_path: Path,
    output_path: Path,
    limit: float = 2.0,
) -> None:
    """Plot glacier trend polygons over the S100 land polygons."""
    glaciers = glacier_trends_with_geometry(trends_path, surge_zip)
    land = gpd.read_file(land_path)

    if glaciers.crs != land.crs:
        glaciers = glaciers.to_crs(land.crs)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 10))

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
    surging = glaciers[glaciers["surging"]].copy()
    surging.geometry.centroid.plot(
        ax=ax,
        marker="*",
        markersize=150,
        color="#ffd92f",
        edgecolor="#222222",
        linewidth=0.5,
        zorder=3,
        label="Surges 2017-2025 (Harcourt et al. 2026)",
    )
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02, label="Crevasse area trend (km²/year)")
    ax.set_axis_off()
    print(ax.get_ylim(), ax.get_xlim())
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=400, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trends", type=Path, default=Path("output/glacier_trends.arrow"))
    parser.add_argument("--surge-database", type=Path, default=Path("input/RGI_V7_Surge_Database.zip"))
    parser.add_argument("--land", type=Path, default=Path("input/S100_Land_f.gpkg"))
    parser.add_argument("--output", type=Path, default=Path("output/figures/crevasse_area_trend_map.png"))
    parser.add_argument("--limit", type=float, default=2.0)
    args = parser.parse_args()
    plot_crevasse_trend_map(args.trends, args.surge_database, args.land, args.output, args.limit)
