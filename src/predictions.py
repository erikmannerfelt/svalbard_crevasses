"""Read and aggregate the per-model GEE prediction rasters."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize

N_MODELS = 30
POSITIVE_THRESHOLD_U8 = 128
PIXEL_AREA_M2 = 50 * 50


def read_model_votes(path: Path) -> tuple[np.ndarray, rasterio.Affine]:
    """Read one complete annual raster of uint8 model vote fractions."""
    with rasterio.open(path) as raster:
        if raster.count != N_MODELS or raster.dtypes != ("uint8",) * N_MODELS:
            raise ValueError(f"Expected {N_MODELS} uint8 model bands in {path}")
        expected_names = tuple(f"prob_m{model:02d}" for model in range(N_MODELS))
        if raster.descriptions != expected_names:
            raise ValueError(f"Unexpected model band names in {path}: {raster.descriptions}")
        if raster.crs.to_string() != "EPSG:32633" or raster.res != (50.0, 50.0):
            raise ValueError(f"Unexpected raster grid in {path}")
        return raster.read(), raster.transform


def vote_fraction_image(path: Path) -> tuple[np.ndarray, rasterio.Affine]:
    """Return the fraction of models classifying each pixel as positive."""
    values, transform = read_model_votes(path)
    return (values >= POSITIVE_THRESHOLD_U8).mean(axis=0), transform


def write_vote_fraction_raster(
    values: np.ndarray, source_path: Path, output_path: Path
) -> None:
    """Write a compact uint8 vote-fraction raster using a source grid."""
    vote_fraction = np.rint((values >= POSITIVE_THRESHOLD_U8).mean(axis=0) * 255).astype(np.uint8)
    with rasterio.open(source_path) as source:
        profile = source.profile.copy()
        profile.update(
            count=1,
            dtype="uint8",
            compress="lzw",
            tiled=True,
            blockxsize=256,
            blockysize=256,
            nodata=None,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_path, "w", **profile) as output:
            output.write(vote_fraction, 1)
            output.set_band_description(1, "vote_fraction")


def aggregate_model_areas(
    input_dir: Path,
    polygons: gpd.GeoDataFrame,
    years: np.ndarray,
    vote_fraction_dir: Path | None = None,
) -> dict[str, np.ndarray]:
    """Aggregate all model rasters into one row per glacier/model/year."""
    polygons = polygons.to_crs("EPSG:32633").reset_index(drop=True)
    labels_by_index = np.arange(1, len(polygons) + 1, dtype=np.int32)
    rows = []
    glacier_count = len(polygons)

    for year in years:
        path = input_dir / f"crevasse_model_probabilities_{year}.tif"
        values, transform = read_model_votes(path)
        if vote_fraction_dir is not None:
            write_vote_fraction_raster(
                values,
                path,
                vote_fraction_dir / f"crevasse_vote_fraction_{year}.tif",
            )
        labels = rasterize(
            zip(polygons.geometry, labels_by_index),
            out_shape=values.shape[1:],
            transform=transform,
            fill=0,
            dtype="int32",
        )
        valid_counts = np.bincount(labels.ravel(), minlength=len(polygons) + 1)[1:]
        if not valid_counts.all():
            raise ValueError(f"Some glacier polygons have no pixels in {path}")

        labelled = labels.ravel()
        glacier_pixels = labelled > 0
        glacier_ids = labelled[glacier_pixels] - 1
        positive_counts = np.stack(
            [
                np.bincount(
                    glacier_ids,
                    weights=(values_for_model.ravel()[glacier_pixels] >= POSITIVE_THRESHOLD_U8),
                    minlength=glacier_count,
                )
                for values_for_model in values
            ]
        )
        rows.append(
            {
                "year": np.full(N_MODELS * glacier_count, year, dtype=np.int16),
                "model_id": np.repeat(np.arange(N_MODELS), glacier_count),
                "anlys_id": np.tile(polygons.anlys_id.to_numpy(), N_MODELS),
                "glac_id": np.tile(polygons.glims_id.to_numpy(), N_MODELS),
                "glac_name": np.tile(polygons.glac_name.to_numpy(), N_MODELS),
                "positive_area_m2": (positive_counts * PIXEL_AREA_M2).ravel(),
                "valid_area_m2": np.tile(valid_counts * PIXEL_AREA_M2, N_MODELS),
            }
        )
    return {key: np.concatenate([row[key] for row in rows]) for key in rows[0]}
