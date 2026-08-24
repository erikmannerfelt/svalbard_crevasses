"""Fixed project configuration and repository-relative paths."""

from __future__ import annotations

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
    2019: 2020,
    2020: 2020,
    2021: 2021,
    2022: 2022,
    2023: 2023,
    2024: 2023,
    2025: 2023,
}
