"""Read the RGI surge database and classify surge intervals."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from src import settings

SURGE_MEMBER = "RGI_V7_Surge_Database/RGI2000-v7.0-G-07_svalbard_jan_mayen_Surge_Database.shp"
SURGE_OVERRIDES = {
    "RGI2000-v7.0-G-07-00741": {"term_type": 1},
    "RGI2000-v7.0-G-07-00780": {"S_Term": 2017},
    "RGI2000-v7.0-G-07-01312": {"glac_name": "Lilliehöökbreen"},
    "RGI2000-v7.0-G-07-01379": {
        "S_Onset": "2015",
        "S_Term": "Ongoing",
        "glac_name": "Austfonna Basin-7",
    },
    "RGI2000-v7.0-G-07-01383": {"glac_name": "Storisstraumen", "S_Term": "1873; Ongoing"},
    "RGI2000-v7.0-G-07-01385": {"glac_name": "Austfonna Basin-2"},
    "RGI2000-v7.0-G-07-01381": {"glac_name": "Austfonna Basin-5"},
    "RGI2000-v7.0-G-07-01384": {"glac_name": "Bråsvellbreen"},
    "RGI2000-v7.0-G-07-00888": {"S_Onset": "2016", "S_Term": "2019"},
    "RGI2000-v7.0-G-07-00899": {"S_Onset": "1890; 2008", "S_Term": "1890; 2017"},
    "RGI2000-v7.0-G-07-00902": {"S_Onset": "1890; 2008", "S_Term": "1890; 2017"},
    "RGI2000-v7.0-G-07-00912": {"S_Onset": "1890; 2008", "S_Term": "1890; 2017"},
    "RGI2000-v7.0-G-07-01211": {"S_Onset": "n/a", "S_Term": "n/a"},
}


def _tokens(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return [part.strip() for part in str(value).split(";") if part.strip()]


def surge_overlaps(
    term_value: object,
    onset_value: object,
    min_year: int = settings.START_YEAR,
    max_year: int = settings.END_YEAR,
) -> bool:
    """Return whether any paired surge interval overlaps the inclusive year range."""
    terms, onsets = _tokens(term_value), _tokens(onset_value)
    for index, onset in enumerate(onsets):
        try:
            onset_year = int(onset)
        except ValueError:
            continue
        if onset_year > max_year:
            continue
        term = terms[index] if index < len(terms) else None
        if term is None or term.lower() in {"", "ongoing"}:
            term_year = np.inf
        elif term.lower() == "not observed":
            term_year = onset_year + 10
        else:
            try:
                term_year = int(term)
            except ValueError:
                term_year = np.inf
        if term_year > min_year:
            return True
    return False


def terminated_intervals(term_value: object, onset_value: object) -> list[tuple[int, int, bool]]:
    """Return paired completed surge intervals as ``(onset, termination, inferred)``.

    ``Not observed`` terminations are retained and marked as inferred because the
    project convention assigns them a ten-year duration. Ongoing and invalid
    terminations are not completed intervals.
    """
    terms, onsets = _tokens(term_value), _tokens(onset_value)
    intervals = []
    for index, onset in enumerate(onsets):
        try:
            onset_year = int(onset)
        except ValueError:
            continue
        if index >= len(terms):
            continue
        term = terms[index]
        inferred = term.lower() == "not observed"
        if inferred:
            term_year = onset_year + 10
        else:
            try:
                term_year = int(term)
            except ValueError:
                continue
        if term_year >= onset_year:
            intervals.append((onset_year, term_year, inferred))
    return intervals


def read_database(path: Path = settings.SURGE_DATABASE) -> gpd.GeoDataFrame:
    """Read, correct, and classify the Svalbard RGI surge database."""
    columns = [
        "rgi_id",
        "glims_id",
        "anlys_id",
        "glac_name",
        "area_km2",
        "S_Onset",
        "S_Term",
        "term_type",
        "geometry",
    ]
    source = f"zip://{path.resolve()}!{SURGE_MEMBER}"
    surge = gpd.read_file(source)[columns]
    for rgi_id, overrides in SURGE_OVERRIDES.items():
        mask = surge.rgi_id == rgi_id
        for key, value in overrides.items():
            if key in {"S_Onset", "S_Term"}:
                value = str(value)
            surge.loc[mask, key] = value
    for column in ("S_Onset", "S_Term"):
        surge[column] = surge[column].astype("string")
    surge["surging"] = [surge_overlaps(term, onset) for term, onset in zip(surge.S_Term, surge.S_Onset, strict=True)]
    return surge


def test_surge_intervals_and_special_terms() -> None:
    assert surge_overlaps("Ongoing", "2015", 2020, 2025)
    assert surge_overlaps("Not observed", "2015", 2024, 2025)
    assert not surge_overlaps("2019", "2015", 2020, 2025)
    assert surge_overlaps("1890; Ongoing", "1880; 2016", 2020, 2025)
    assert terminated_intervals("2019; Not observed; Ongoing", "2015; 2000; 2022") == [
        (2015, 2019, False),
        (2000, 2010, True),
    ]
