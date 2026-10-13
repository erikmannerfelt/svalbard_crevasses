"""Raster validation, vote-fraction creation, and polygon aggregation.

Each source raster ``crevasse_model_probabilities_{year}.tif`` holds 30
``uint8`` bands named ``prob_m00`` through ``prob_m29``, in ``EPSG:32633`` with
50 m pixels. A model vote is positive at ``settings.POSITIVE_THRESHOLD_U8``
(128). Derived vote-fraction rasters hold one ``uint8`` band named
``vote_fraction`` that decodes as ``value / 255``.

RGI geometries must be in ``EPSG:32633`` before rasterization, sampling, or
centroid calculations.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import tqdm
from rasterio.features import geometry_mask, rasterize
from rasterio.windows import from_bounds

from src import settings

PIXEL_AREA_M2 = settings.PIXEL_SIZE_M**2


def read_model_votes(path: Path) -> tuple[np.ndarray, rasterio.Affine]:
    """Read one annual raster and enforce the source-grid contract."""
    with rasterio.open(path) as raster:
        expected_names = tuple(f"prob_m{model:02d}" for model in range(settings.N_MODELS))
        if raster.count != settings.N_MODELS or raster.dtypes != ("uint8",) * settings.N_MODELS:
            raise ValueError(f"Expected {settings.N_MODELS} uint8 model bands in {path}")
        if raster.descriptions != expected_names:
            raise ValueError(f"Unexpected model band names in {path}: {raster.descriptions}")
        if raster.crs is None or raster.crs.to_string() != settings.RASTER_CRS or raster.res != (50.0, 50.0):
            raise ValueError(f"Unexpected raster grid in {path}")
        return raster.read(), raster.transform


def write_vote_fraction(values: np.ndarray, source_path: Path, output_path: Path) -> None:
    """Write the uint8 encoded vote fraction using the source raster grid."""
    fraction = np.rint((values >= settings.POSITIVE_THRESHOLD_U8).mean(axis=0) * 255).astype(np.uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp.tif")
    try:
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
            with rasterio.open(temporary, "w", **profile) as output:
                output.write(fraction, 1)
                output.set_band_description(1, "vote_fraction")
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)


def crop_vote_fraction(
    geometry: object, raster_source: Path | tuple[np.ndarray, rasterio.Affine]
) -> tuple[np.ndarray, np.ndarray, rasterio.Affine]:
    """Read or crop a vote-fraction raster around a geometry and mask its exterior."""
    bounds = geometry.bounds
    if isinstance(raster_source, Path):
        with rasterio.open(raster_source) as raster:
            window = from_bounds(*bounds, transform=raster.transform)
            window = window.round_offsets().round_lengths()
            image = raster.read(1, window=window).astype(float) / 255
            transform = raster.window_transform(window)
    else:
        image, transform = raster_source
        window = from_bounds(*bounds, transform=transform).round_offsets().round_lengths()
        row, column = int(window.row_off), int(window.col_off)
        image = image[row : row + int(window.height), column : column + int(window.width)]
        transform = rasterio.windows.transform(window, transform)
    inside = geometry_mask([geometry], out_shape=image.shape, transform=transform, invert=True)
    return image, inside, transform


def aggregate_model_areas(
    input_dir: Path, polygons: gpd.GeoDataFrame, years: tuple[int, ...], vote_fraction_dir: Path
) -> pd.DataFrame:
    """Aggregate source model rasters into a glacier/model/year table."""
    polygons = polygons.to_crs(settings.RASTER_CRS).reset_index(drop=True)
    labels_by_index = np.arange(1, len(polygons) + 1, dtype=np.int32)
    rows: list[dict[str, np.ndarray]] = []
    for year in tqdm.tqdm(years, desc="Sampling data"):
        path = input_dir / settings.MODEL_RASTER_TEMPLATE.format(year=year)
        values, transform = read_model_votes(path)
        vote_path = vote_fraction_dir / settings.VOTE_RASTER_TEMPLATE.format(year=year)
        if not vote_path.exists():
            write_vote_fraction(values, path, vote_path)
        labels = rasterize(
            zip(polygons.geometry, labels_by_index, strict=True),
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
                    weights=(model.ravel()[glacier_pixels] >= settings.POSITIVE_THRESHOLD_U8),
                    minlength=len(polygons),
                )
                for model in values
            ]
        )
        rows.append(
            {
                "year": np.full(settings.N_MODELS * len(polygons), year, dtype=np.int16),
                "model_id": np.repeat(np.arange(settings.N_MODELS), len(polygons)),
                "anlys_id": np.tile(polygons.anlys_id.to_numpy(), settings.N_MODELS),
                "glac_id": np.tile(polygons.glims_id.to_numpy(), settings.N_MODELS),
                "glac_name": np.tile(polygons.glac_name.to_numpy(), settings.N_MODELS),
                "positive_area_m2": (positive_counts * PIXEL_AREA_M2).ravel(),
                "valid_area_m2": np.tile(valid_counts * PIXEL_AREA_M2, settings.N_MODELS),
            }
        )
    return pd.DataFrame({key: np.concatenate([row[key] for row in rows]) for key in rows[0]})
