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
# The earlier-surge window starts this many years before the study period.
# Surge termination is gradual and crevasses take years to close, so a surge
# that ended shortly before 2017 can still drive a decline within it.
PRE_SURGE_YEARS: Final = 5
# Surge-period boundaries. Crevassing outlasts a catalogued termination by
# about a year, so the recent/earlier boundary sits one year before the study
# midpoint rather than at it. Tested against the alternative: moving it to the
# midpoint raises recent-class purity from 93% to 95% but drops earlier-class
# purity from 77% to 67%, and moves +37 km2/yr of positive trend into the
# earlier-surge class, where crevasses are supposed to be closing.
RECENT_SURGE_START: Final = MIDPOINT_YEAR - 1
EARLIER_SURGE_END: Final = RECENT_SURGE_START - 1
# Ensemble spread is reported as this percentile pair of the 30 models. 30
# members cannot support a narrower tail: the 2.5th percentile of 30 sorted
# values interpolates between the 1st and 2nd member, so one training subset
# would set the published bound, while the 5th sits between the 2nd and 3rd.
# The models are 70% resamples of one polygon set, so this is sensitivity to
# the training subset, not a confidence interval on the trend.
ENSEMBLE_QUANTILES: Final = (0.05, 0.95)
# Glaciers below this mean visible-crevasse area carry too little signal for the
# glacier-relative normalization behind the common-mode decomposition: dividing a
# near-zero annual value by a near-zero mean produces meaningless ratios.
COMMON_MODE_MIN_MEAN_AREA_KM2: Final = 0.5
HISTORICAL_TERMINATION_YEARS: Final = 50
PIXEL_SIZE_M: Final = 50
POSITIVE_THRESHOLD_U8: Final = 128
RASTER_CRS: Final = "EPSG:32633"

SURGE_DATABASE: Final = INPUT_DIR / "RGI_V7_Surge_Database.zip"
LAND_FILE: Final = INPUT_DIR / "S100_Land_f.gpkg"
OUTLINE_FILE: Final = INPUT_DIR / "glacier_outlines_mod.sqlite"
VALIDATION_HOLDOUT_2024_FILE: Final = INPUT_DIR / "crevasse_rf_ensemble_validation_holdout_2024.csv"

# Annual glacier mass-change estimates, used to test whether the visible
# crevasse decreases track mass loss. Cite the dataset as WGMS (2026) and the
# method as Dussaillant et al. (2025); see `src/fetch_inputs.py`.
MASS_CHANGE_URL: Final = "https://wgms.ch/downloads/wgms-amce-2026-02-10.zip"
MASS_CHANGE_FILE: Final = INPUT_DIR / "wgms-amce-2026-02-10.zip"
MASS_CHANGE_SERIES_MEMBER: Final = "glacier/SJM_mwe.csv"
MASS_CHANGE_METADATA_MEMBER: Final = "glacier/SJM_metadata.csv"
# RGI6 centroids are matched to RGI7 centroids; anything further apart than
# this is a genuine outline change rather than a version difference.
MASS_CHANGE_MATCH_DISTANCE_M: Final = 2_000

# Norwegian Polar Institute S0 terrain model at 50 m (NP_S0_DTM50), used to
# locate visible crevasse change by elevation. Cite as Norwegian Polar
# Institute (2014); see `src/fetch_inputs.py`.
DEM_URL: Final = (
    "https://next.api.npolar.no/dataset/dce53a47-c726-4845-85c3-a65b46fe2fea"
    "/attachment/838be1e3-58ef-4338-b80c-dec86ba07bd2/_blob"
)
DEM_FILE: Final = INPUT_DIR / "NP_S0_DTM50.zip"
DEM_MEMBER: Final = "NP_S0_DTM50/S0_DTM50.tif"
# Absolute elevation bands for locating visible crevasse change (m a.s.l.).
ELEVATION_BAND_M: Final = 100
# Relative bands split each glacier's own DEM elevation range into this many
# equal intervals, so glaciers of different elevation can be compared by
# position (terminus = 0, top = 1) rather than by absolute height.
RELATIVE_ELEVATION_BANDS: Final = 10
# Zone boundaries for summarizing where on a glacier visible crevasse change
# occurs. 400 m is the 100 m band edge just below the regional equilibrium-line
# altitude of about 450 m (Noel et al., 2020); relative band 4 splits the lower
# 40% of a glacier's elevation range from the rest.
ELEVATION_ZONE_SPLIT_M: Final = 400
RELATIVE_ZONE_SPLIT_BAND: Final = 4
# Absolute bands at or above this elevation are pooled for the interannual
# sensitivity, because the few glaciers reaching them give unstable ratios.
SENSITIVITY_POOL_ABOVE_M: Final = 1200

MODEL_RASTER_TEMPLATE: Final = "crevasse_model_probabilities_{year}.tif"
# The Sentinel-1 comparison run exports the same 30-band contract for the 2024
# holdout year, so the offline raster validation can score either predictor
# under one protocol. Only 2024 exists for Sentinel-1.
SENTINEL1_RASTER_TEMPLATE: Final = "crevasse_model_probabilities_{year}_sentinel1.tif"
SENTINEL1_YEARS: Final = (2024,)
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
