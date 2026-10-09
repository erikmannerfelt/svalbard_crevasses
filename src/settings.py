"""Fixed project configuration and repository-relative paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

ROOT: Final = Path(__file__).resolve().parent.parent
INPUT_DIR: Final = ROOT / "input"
OUTPUT_DIR: Final = ROOT / "output"
CACHE_DIR: Final = ROOT / "cache"
DATA_DIR: Final = OUTPUT_DIR / "data"
RASTER_DIR: Final = OUTPUT_DIR / "rasters" / "vote-fractions"
FIGURE_DIR: Final = OUTPUT_DIR / "figures"
TABLE_DIR: Final = OUTPUT_DIR / "tables"
GLACIER_ZONES_FILE: Final = ROOT / "shapes" / "glacier_zones.geojson"

YEARS: Final = tuple(range(2017, 2026))
N_MODELS: Final = 30
EXPECTED_GLACIERS: Final = 1_583
START_YEAR: Final = 2013
END_YEAR: Final = 2025
MIDPOINT_YEAR: Final = YEARS[len(YEARS) // 2]
PRE_SURGE_YEARS: Final = 5
HISTORICAL_TERMINATION_YEARS: Final = 50
PIXEL_SIZE_M: Final = 50
POSITIVE_THRESHOLD_U8: Final = 128
RASTER_CRS: Final = "EPSG:32633"

SURGE_DATABASE: Final = INPUT_DIR / "RGI_V7_Surge_Database.zip"
LAND_FILE: Final = INPUT_DIR / "S100_Land_f.gpkg"
OUTLINE_FILE: Final = INPUT_DIR / "glacier_outlines_mod.sqlite"
VALIDATION_HOLDOUT_2024_FILE: Final = INPUT_DIR / "crevasse_rf_ensemble_validation_holdout_2024.csv"

MODEL_RASTER_TEMPLATE: Final = "crevasse_model_probabilities_{year}.tif"
VOTE_RASTER_TEMPLATE: Final = "crevasse_vote_fraction_{year}.tif"
SELECTED_GLACIERS: Final = (
    "Recherchebreen",
    "Moršnevbreen",
    "Arnesenbreen",
    "Borebreen",
    "Vallåkrabreen",
    "Liestølbreen",
)
RGI_NAME_MAP: Final = {"Morsjnevbreen / Strongbreen": "Moršnevbreen"}
ARNESEN_YEARS: Final = tuple(range(2018, 2026))
ARNESEN_OUTLINE_YEARS: Final = {
    2018: 2018,
    2019: 2018,
    2020: 2020,
    2021: 2021,
    2022: 2022,
    2023: 2023,
    2024: 2023,
    2025: 2023,
}


@dataclass(frozen=True)
class SentinelExample:
    """One Sentinel-2 image pair shown beside the vote-fraction rasters.

    `bounds` is (left, bottom, right, top) in metres in the image's own UTM
    zone, which differs between sites; the vote fraction is resampled onto
    the image grid rather than the other way round. `stretch` applies a
    linear contrast stretch shared by both dates, for scenes the export
    leaves washed out. `stretch_exclude`, in the same coordinates as
    `bounds`, leaves a box out of the stretch limits, e.g. open sea that
    would otherwise set the black point and wash out the ice.
    """

    name: str
    file_stem: str
    cause: str
    dates: tuple[str, str]
    bounds: tuple[float, float, float, float]
    stretch: bool = False
    stretch_exclude: tuple[float, float, float, float] | None = None


# Copernicus Browser "Highlight Optimized Natural Color" L1C exports. These are
# demonstration examples chosen for unambiguous visible change, not a sample.
SENTINEL_DIR: Final = INPUT_DIR / "sentinel2"
SENTINEL_EXAMPLES: Final = (
    SentinelExample(
        "Wahlenbergbreen",
        "wahlenbergbreen",
        "surge",
        ("2017-07-28", "2025-07-22"),
        (479_000, 8_709_700, 485_500, 8_716_200),
    ),
    SentinelExample(
        "Petermannbreen",
        "petermannbreen",
        "surge",
        ("2018-07-30", "2025-08-28"),
        (443_700, 8_710_000, 451_500, 8_717_800),
    ),
    SentinelExample(
        "Conwayjøkulen",
        "conwayjokulen",
        "likely snow cover",
        ("2019-08-05", "2024-08-04"),
        (545_500, 8_752_700, 549_000, 8_756_200),
        stretch=True,
    ),
    SentinelExample(
        "NE Kvitøyjøkulen",
        "kvitoyjokulen-ne",
        "likely snow cover",
        ("2018-07-30", "2025-08-06"),
        (488_500, 8_909_000, 493_500, 8_914_000),
        stretch=True,
        stretch_exclude=(488_500, 8_913_100, 491_000, 8_914_000),
    ),
)
