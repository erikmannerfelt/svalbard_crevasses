"""Audit spatial and temporal coverage of hand-digitized crevasse training labels.

Barentsøya/Edgeøya, Vestfonna and Kvitøya are thin or blank in most target
years. That matches their low real crevassing and is a standing pattern, not
a labeling gap to chase.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final, Literal

import geopandas as gpd
import numpy as np
import pandas as pd

from src import settings

type ForceRedo = Literal[False, "self", "all"]

# The GeoJSON is the Drive export of the tagged label collection written by
# gee/svalbard_crevasse_labelling.js from the digitized polygons in
# gee/imports.js. It is the primary label file; the Arrow below is derived.
LABELS_SOURCE: Final = settings.ROOT / "shapes" / "crevasse_training_labels.geojson"
LABELS_FILE: Final = settings.CACHE_DIR / "crevasse_area_labels.arrow"
REQUIRED_LABEL_COLUMNS: Final = frozenset({"polyg_id", "crevassed", "year", "geometry"})
# Carried through when the export provides them. `big_crevasses` marks positives
# containing large, unambiguous crevasse fields and stratifies the raster
# validation by crevasse size class.
OPTIONAL_LABEL_COLUMNS: Final = ("big_crevasses",)
ZONE_YEAR_SUMMARY_FILE: Final = settings.DATA_DIR / "label-zone-year-summary.csv"

# Matches gee/svalbard_crevasse_detection.js's SVALBARD_BOUNDS. Keep in sync if that changes.
SVALBARD_BOUNDS_32633: Final = (390_000.0, 8_480_000.0, 850_000.0, 8_970_000.0)


def read_labels(force_redo: ForceRedo = False) -> gpd.GeoDataFrame:
    """Read hand-digitized crevasse training-label polygons.

    The labels are read from `LABELS_SOURCE` and cached as `LABELS_FILE` with
    a fixed schema: ``polyg_id`` (str), ``crevassed`` (bool), ``year`` (int),
    the optional columns present in the export (bool), and ``geometry``
    (Polygon, EPSG:4326). Earth Engine's internal feature ``id`` is dropped.
    The cache is cheap to rebuild and is written directly.

    Parameters
    ----------
    force_redo
        ``False`` reuses an existing cache; ``"self"`` or ``"all"`` rebuild it
        from the GeoJSON (there are no derived dependencies to propagate to).
    """
    if force_redo is False and LABELS_FILE.exists():
        return gpd.read_feather(LABELS_FILE)

    frame = gpd.read_file(LABELS_SOURCE)
    missing = REQUIRED_LABEL_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{LABELS_SOURCE} is missing required columns: {sorted(missing)}")
    frame = frame.set_crs("EPSG:4326") if frame.crs is None else frame.to_crs("EPSG:4326")
    frame["polyg_id"] = frame["polyg_id"].astype(str)
    frame["crevassed"] = frame["crevassed"].astype(bool)
    frame["year"] = frame["year"].astype(int)
    optional = [column for column in OPTIONAL_LABEL_COLUMNS if column in frame.columns]
    for column in optional:
        frame[column] = frame[column].astype(bool)
    frame = frame[["polyg_id", "crevassed", "year", *optional, "geometry"]]

    LABELS_FILE.parent.mkdir(parents=True, exist_ok=True)
    frame.to_feather(LABELS_FILE)
    return gpd.read_feather(LABELS_FILE)


def filter_to_svalbard(
    labels: gpd.GeoDataFrame, bounds: tuple[float, float, float, float] = SVALBARD_BOUNDS_32633
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Split labels by whether their centroid falls inside the Svalbard bounding box.

    A handful of labels are known, expected leftovers from an earlier
    pan-Arctic labeling attempt and sit far outside Svalbard (Arctic Canada,
    Greenland, the Kara Sea coast, the Yukon, Tibet). They are not a
    labeling error to fix here -- they are filtered out the same way
    ``gee/svalbard_crevasse_detection.js``'s ``SVALBARD_BOUNDS`` already
    excludes them from the GEE training pipeline.

    Returns
    -------
    in_bounds, out_of_bounds
        Both reprojected to ``settings.RASTER_CRS`` (``bounds``'s CRS).
    """
    reprojected = labels.to_crs(settings.RASTER_CRS)
    minx, miny, maxx, maxy = bounds
    centroid = reprojected.geometry.centroid
    inside = centroid.x.between(minx, maxx) & centroid.y.between(miny, maxy)
    return reprojected[inside].copy(), reprojected[~inside].copy()


def join_zones(labels: gpd.GeoDataFrame, zones: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Attach each label's containing glacier zone by polygon centroid.

    Parameters
    ----------
    labels
        Svalbard-filtered labels (see `filter_to_svalbard`), in the same
        CRS as ``zones``.
    zones
        Named zone polygons from `settings.GLACIER_ZONES_FILE`.

    Returns
    -------
    Labels with ``zone_label``/``zone_name`` columns attached. Rows whose
    centroid falls outside all named zones are dropped and reported (e.g. a
    label on Hopen, a minor outlying island not covered by any zone in
    `settings.GLACIER_ZONES_FILE`) -- unlike `filter_to_svalbard`, this
    isn't necessarily a data problem, just a zone polygon that doesn't
    happen to cover every scrap of Svalbard land.
    """
    if labels.crs != zones.crs:
        raise ValueError("labels and zones must share a CRS; reproject before calling join_zones")
    points = labels.copy()
    points["geometry"] = points.geometry.centroid
    joined = gpd.sjoin(points, zones[["zone_label", "zone_name", "geometry"]], how="left", predicate="within")
    unmatched = joined.zone_label.isna()
    if unmatched.any():
        print(
            f"{unmatched.sum()} Svalbard-bounded label(s) fell outside all named zones "
            f"(e.g. outlying islands like Hopen not covered by glacier_zones.geojson): "
            f"{joined.loc[unmatched, 'polyg_id'].tolist()}"
        )
    zoned = labels.copy()
    zoned["zone_label"] = joined["zone_label"].to_numpy()
    zoned["zone_name"] = joined["zone_name"].to_numpy()
    return zoned[zoned.zone_label.notna()].copy()


def zone_year_summary(
    zoned: gpd.GeoDataFrame, zones: gpd.GeoDataFrame, years: Sequence[int] = settings.YEARS
) -> pd.DataFrame:
    """Count positive/negative labels per zone x year, including empty cells.

    Parameters
    ----------
    zoned
        Labels with ``zone_label``/``zone_name`` attached (see
        `join_zones`).
    zones
        Named zone polygons; used only for the full, ordered zone list so
        zones with zero labels still appear.
    years
        Years to report; defaults to the full analysis year range so gaps
        outside the training-year subset (e.g. 2019 today) are still
        visible.

    Returns
    -------
    One row per (zone, year): ``positive_count``, ``negative_count``,
    ``total_count``, ``positive_share`` (NaN when ``total_count`` is 0),
    ``has_labels``, and ``single_class_only`` (True when ``total_count`` is
    positive and one of the two counts is 0). ``single_class_only`` is
    descriptive only -- some zones (e.g. C Spitsbergen) plausibly have very
    few real crevasses, so an all-negative cell there can be an accurate
    reflection of nature rather than a labeling gap; use it together with
    the label-audit map figure to judge which flagged cells actually need
    new labels.
    """
    zone_columns = zones[["zone_label", "zone_name"]].drop_duplicates()
    grid = zone_columns.merge(pd.DataFrame({"year": list(years)}), how="cross")
    counts = (
        zoned.groupby(["zone_label", "year", "crevassed"], as_index=False)
        .size()
        .pivot(index=["zone_label", "year"], columns="crevassed", values="size")
        .reindex(columns=[False, True], fill_value=0)
        .rename(columns={False: "negative_count", True: "positive_count"})
        .reset_index()
    )
    summary = grid.merge(counts, on=["zone_label", "year"], how="left")
    summary[["negative_count", "positive_count"]] = summary[["negative_count", "positive_count"]].fillna(0).astype(int)
    summary["total_count"] = summary.negative_count + summary.positive_count
    summary["positive_share"] = summary.positive_count / summary.total_count.replace(0, np.nan)
    summary["has_labels"] = summary.total_count > 0
    summary["single_class_only"] = summary.has_labels & ((summary.negative_count == 0) | (summary.positive_count == 0))
    return summary.sort_values(["zone_label", "year"]).reset_index(drop=True)


def run() -> dict[str, pd.DataFrame | gpd.GeoDataFrame]:
    """Audit label spatial/temporal coverage and write the zone x year summary."""
    zones = gpd.read_file(settings.GLACIER_ZONES_FILE)
    labels = read_labels()
    in_bounds, out_of_bounds = filter_to_svalbard(labels)
    zoned = join_zones(in_bounds, zones)
    summary = zone_year_summary(zoned, zones)
    settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(ZONE_YEAR_SUMMARY_FILE, index=False)
    print(
        f"Dropped {len(out_of_bounds)} of {len(labels)} labels outside the Svalbard bounding box "
        "(expected pan-Arctic remnants)."
    )
    return {"zoned": zoned, "out_of_bounds": out_of_bounds, "zone_year_summary": summary}


if __name__ == "__main__":
    run()


def test_filter_to_svalbard_drops_out_of_bounds_centroid() -> None:
    import shapely.geometry

    points = gpd.GeoDataFrame(
        {"id": ["inside", "outside"]},
        geometry=[shapely.geometry.Point(600_000, 8_700_000), shapely.geometry.Point(0, 0)],
        crs=settings.RASTER_CRS,
    )
    inside, outside = filter_to_svalbard(points)
    assert list(inside.id) == ["inside"]
    assert list(outside.id) == ["outside"]


def test_zone_year_summary_includes_zero_count_cells() -> None:
    zones = gpd.GeoDataFrame(
        {"zone_label": ["A", "B"], "zone_name": ["Zone A", "Zone B"]},
        geometry=gpd.points_from_xy([0, 0], [0, 0]),
        crs=settings.RASTER_CRS,
    )
    zoned = gpd.GeoDataFrame(
        {"zone_label": ["A"], "zone_name": ["Zone A"], "year": [2020], "crevassed": [True]},
        geometry=gpd.points_from_xy([0], [0]),
        crs=settings.RASTER_CRS,
    )
    summary = zone_year_summary(zoned, zones, years=[2020, 2021])
    empty_cell = summary[(summary.zone_label == "B") & (summary.year == 2020)].iloc[0]
    assert empty_cell.total_count == 0
    assert not empty_cell.has_labels
    filled_cell = summary[(summary.zone_label == "A") & (summary.year == 2020)].iloc[0]
    assert filled_cell.positive_count == 1
    assert filled_cell.single_class_only
