"""Validate the exported ensemble rasters against the hand-digitized labels.

This is a second, independent validation of the same models that
``gee/svalbard_crevasse_detection.js`` validates in Earth Engine, and
it answers a different question. The Earth Engine validation samples the
10 m embedding at points drawn inside label polygons, with a balanced
50/50 class ratio. This module scores the **50 m vote-fraction rasters
that are actually published**, over every labelled pixel, at the class
ratio the labels imply -- so its numbers describe the map product rather
than a balanced point sample, and they are not directly comparable to
``src/tables/validation.py``'s.

Two things make it worth having:

*Domain.* Negative label polygons were digitized generously and spill
onto rock, tundra and sea, which the ensemble rejects trivially (an audit
of the 2024 holdout put the false-positive rate at 0.05% off-glacier
against 10.19% on-glacier). Scoring those pixels inflates specificity by
roughly five points. The ``glacier`` domain restricts scoring to
glacierized area, which is also the only domain ``src/analysis.py``
reports over. The ``all_labelled`` domain keeps every labelled pixel and
is retained for comparison, not for publication.

*Size class.* ``big_crevasses`` marks positives containing large,
unambiguous crevasse fields. Recall computed per stratum shows whether
performance depends on crevasse size. Precision and F1 do **not** compare
across strata -- every stratum is scored against the same pool of
negatives, so a stratum holding fewer positives gets a lower precision
for that reason alone. Compare recall.

Runs entirely offline from ``input/crevasse_model_probabilities_*.tif``
and costs no Earth Engine quota.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.mask
import shapely
import tqdm

from src import label_audit, settings, surge

type Domain = Literal["glacier", "all_labelled"]

RASTER_VALIDATION_FILE: Final = settings.DATA_DIR / "raster-validation.csv"
SENTINEL1_VALIDATION_FILE: Final = settings.DATA_DIR / "raster-validation-sentinel1.csv"
ZONE_VALIDATION_FILE: Final = settings.DATA_DIR / "raster-validation-zones.csv"

DOMAINS: Final[tuple[Domain, ...]] = ("glacier", "all_labelled")

# Positive-class strata, keyed by the required value of `big_crevasses`.
# None keeps every positive.
STRATA: Final[dict[str, bool | None]] = {
    "all": None,
    "big_crevasses": True,
    "small_crevasses": False,
}

# Column names match `src/tables/validation.py`'s METRICS so its LaTeX
# renderer can consume this CSV unchanged.
METRIC_COLUMNS: Final = (
    "accuracy",
    "f1_crevasse",
    "false_negative_rate",
    "false_positive_rate",
    "precision_crevasse",
    "recall_crevasse",
    "specificity_noncrevasse",
)


@dataclass(frozen=True)
class PolygonVotes:
    """Per-model positive-vote counts for one label polygon.

    Attributes
    ----------
    polyg_id
        Label identifier from the Earth Engine export.
    year
        Label year; also selects the prediction raster.
    crevassed
        True for positive (crevassed) labels.
    big_crevasses
        True where the positive polygon contains large, unambiguous
        crevasse fields. Always False for negatives.
    n_pixels
        Raster pixels whose centre falls inside the polygon.
    positive_votes
        Shape ``(settings.N_MODELS,)``. Count of those pixels each model
        scored at or above ``settings.POSITIVE_THRESHOLD_U8``.
    """

    polyg_id: str
    year: int
    crevassed: bool
    big_crevasses: bool
    n_pixels: int
    positive_votes: np.ndarray


def read_glacier_domain(path: Path = settings.SURGE_DATABASE) -> shapely.Geometry:
    """Read the RGI7 glacier outlines as one geometry in the raster CRS.

    Parameters
    ----------
    path
        RGI7 surge-database archive; the same source `src.surge` reads.

    Returns
    -------
    Union of all glacier polygons, in `settings.RASTER_CRS`. Geometries
    are repaired first: the published outlines contain self-intersections
    that make a direct union raise a GEOS topology error.
    """
    outlines = gpd.read_file(f"zip://{path.resolve()}!{surge.SURGE_MEMBER}").to_crs(settings.RASTER_CRS)
    if len(outlines) != settings.EXPECTED_GLACIERS:
        raise ValueError(f"Expected {settings.EXPECTED_GLACIERS} glaciers in {path}, found {len(outlines)}")
    return shapely.union_all(shapely.make_valid(outlines.geometry.values))


def read_labels() -> gpd.GeoDataFrame:
    """Read Svalbard label polygons in the raster CRS, with repaired geometry.

    Labels outside `label_audit.SVALBARD_BOUNDS_32633` are dropped; they
    are known leftovers from an earlier pan-Arctic labeling attempt (see
    `label_audit.filter_to_svalbard`).
    """
    labels = label_audit.read_labels()
    if "big_crevasses" not in labels.columns:
        raise ValueError(
            f"{label_audit.LABELS_SOURCE} has no 'big_crevasses' column; re-export it from "
            "gee/svalbard_crevasse_labelling.js and run `just labels`"
        )
    in_bounds, _ = label_audit.filter_to_svalbard(labels)
    in_bounds["geometry"] = shapely.make_valid(in_bounds.geometry.values)
    return in_bounds


def polygon_votes(source: rasterio.DatasetReader, geometry: shapely.Geometry) -> tuple[int, np.ndarray]:
    """Count per-model positive votes among the pixels inside one geometry.

    Returns
    -------
    n_pixels, positive_votes
        ``(0, zeros)`` when the geometry covers no pixel centre, which
        happens for polygons smaller than one 50 m pixel and for negatives
        that clip away to nothing.
    """
    empty = (0, np.zeros(settings.N_MODELS, dtype=np.int64))
    if geometry.is_empty:
        return empty
    try:
        masked, _ = rasterio.mask.mask(source, [geometry], crop=True, filled=False)
    except ValueError:
        # rasterio raises when the geometry does not overlap the raster.
        return empty

    inside = ~masked.mask[0]
    if not inside.any():
        return empty
    pixels = np.asarray(masked)[:, inside]
    return int(inside.sum()), (pixels >= settings.POSITIVE_THRESHOLD_U8).sum(axis=1).astype(np.int64)


def collect_votes(
    labels: gpd.GeoDataFrame,
    domain: shapely.Geometry | None,
    years: tuple[int, ...],
    template: str = settings.MODEL_RASTER_TEMPLATE,
) -> list[PolygonVotes]:
    """Score every label polygon of the given years against its year's raster.

    Parameters
    ----------
    labels
        Output of `read_labels`.
    domain
        Restricts scoring to this geometry, or None to score every
        labelled pixel. Both classes are clipped: the reported claim is
        accuracy over the domain, so positives outside it are outside the
        claim too.
    years
        Label years to score. Each needs its prediction raster in
        `settings.INPUT_DIR`.
    template
        Prediction-raster filename template. Defaults to the embedding run;
        pass `settings.SENTINEL1_RASTER_TEMPLATE` to score the Sentinel-1
        comparison run under exactly the same protocol.
    """
    collected: list[PolygonVotes] = []

    for year in years:
        raster_path = settings.INPUT_DIR / template.format(year=year)
        selected = labels[labels.year == year]

        with rasterio.open(raster_path) as source:
            if source.count != settings.N_MODELS:
                raise ValueError(f"{raster_path} has {source.count} bands, expected {settings.N_MODELS}")

            for record in tqdm.tqdm(selected.itertuples(), total=len(selected), desc=f"{year} labels", leave=False):
                geometry = record.geometry if domain is None else record.geometry.intersection(domain)
                n_pixels, votes = polygon_votes(source, geometry)
                if n_pixels == 0:
                    continue
                collected.append(
                    PolygonVotes(
                        polyg_id=record.polyg_id,
                        year=year,
                        crevassed=bool(record.crevassed),
                        big_crevasses=bool(record.big_crevasses),
                        n_pixels=n_pixels,
                        positive_votes=votes,
                    )
                )

    return collected


def _metric_rows(votes: list[PolygonVotes], year: int, domain: Domain) -> Iterator[dict[str, object]]:
    """Yield one row per stratum and model for a single year and domain."""
    negatives = [vote for vote in votes if not vote.crevassed]
    false_positives = np.sum([vote.positive_votes for vote in negatives], axis=0)
    n_negative = sum(vote.n_pixels for vote in negatives)
    true_negatives = n_negative - false_positives

    for stratum, required in STRATA.items():
        positives = [vote for vote in votes if vote.crevassed and (required is None or vote.big_crevasses is required)]
        if not positives:
            continue

        true_positives = np.sum([vote.positive_votes for vote in positives], axis=0)
        n_positive = sum(vote.n_pixels for vote in positives)
        false_negatives = n_positive - true_positives

        # Denominators are counts of labelled pixels and cannot be zero
        # here: a stratum with no pixels was skipped above, and a year
        # with no negatives would have failed the label audit long before.
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / n_positive

        for model_id in range(settings.N_MODELS):
            yield {
                "model_id": model_id,
                "year": year,
                "domain": domain,
                "stratum": stratum,
                "n_positive_px": n_positive,
                "n_negative_px": n_negative,
                "tp": int(true_positives[model_id]),
                "fp": int(false_positives[model_id]),
                "fn": int(false_negatives[model_id]),
                "tn": int(true_negatives[model_id]),
                "accuracy": (true_positives[model_id] + true_negatives[model_id]) / (n_positive + n_negative),
                "precision_crevasse": precision[model_id],
                "recall_crevasse": recall[model_id],
                "specificity_noncrevasse": true_negatives[model_id] / n_negative,
                "false_positive_rate": false_positives[model_id] / n_negative,
                "false_negative_rate": false_negatives[model_id] / n_positive,
                "f1_crevasse": (
                    2 * precision[model_id] * recall[model_id] / (precision[model_id] + recall[model_id])
                    if precision[model_id] + recall[model_id] > 0
                    else 0.0
                ),
            }


def build_validation(
    years: tuple[int, ...] = settings.YEARS, template: str = settings.MODEL_RASTER_TEMPLATE
) -> pd.DataFrame:
    """Score every year, domain and stratum against the prediction rasters.

    Parameters
    ----------
    years
        Label years to score.
    template
        Prediction-raster filename template; see `collect_votes`.

    Returns
    -------
    One row per year, domain, stratum and model, with the metric columns
    named as `src/tables/validation.py` expects.
    """
    labels = read_labels()
    scored = sorted(set(years) & set(labels.year.unique()))
    if not scored:
        raise ValueError(f"No labels for any of {years}")

    glacier_domain = read_glacier_domain()

    rows: list[dict[str, object]] = []
    for domain in DOMAINS:
        votes = collect_votes(labels, glacier_domain if domain == "glacier" else None, tuple(scored), template)
        for year in scored:
            year_votes = [vote for vote in votes if vote.year == year]
            if year_votes:
                rows.extend(_metric_rows(year_votes, year, domain))

    return pd.DataFrame(rows)


def run(output_path: Path = RASTER_VALIDATION_FILE) -> pd.DataFrame:
    """Build the raster validation table and write it to CSV."""
    validation = build_validation()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    validation.to_csv(output_path, index=False)
    return validation


def run_sentinel1(output_path: Path = SENTINEL1_VALIDATION_FILE) -> pd.DataFrame:
    """Score the Sentinel-1 comparison run under the identical protocol.

    The Sentinel-1 experiment exports only the 2024 holdout year, but under
    the same 30-band contract, so scoring it here rather than against the
    Earth Engine tables gives the one comparison that is like for like: same
    labels, same withheld year, same 50 m rasters, same glacierized domain
    and the same natural class ratio.
    """
    validation = build_validation(years=settings.SENTINEL1_YEARS, template=settings.SENTINEL1_RASTER_TEMPLATE)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    validation.to_csv(output_path, index=False)
    return validation


def _summarise(validation: pd.DataFrame) -> pd.DataFrame:
    """Ensemble mean of each metric, as percentages, for console reporting."""
    grouped = validation.groupby(["year", "domain", "stratum"], as_index=False)[list(METRIC_COLUMNS)].mean()
    grouped[list(METRIC_COLUMNS)] = (grouped[list(METRIC_COLUMNS)] * 100).round(1)
    return grouped


if __name__ == "__main__":
    result = run()
    print(f"Wrote {len(result)} rows to {RASTER_VALIDATION_FILE}\n")
    summary = _summarise(result)
    holdout = summary[summary.year == 2024]
    print("2024 temporal holdout (no model has seen this year):")
    reported = ["domain", "stratum", "recall_crevasse", "specificity_noncrevasse", "accuracy"]
    print(holdout[reported].to_string(index=False))


def test_strata_cover_the_positive_class() -> None:
    assert STRATA["all"] is None
    assert {STRATA["big_crevasses"], STRATA["small_crevasses"]} == {True, False}


def test_metric_rows_are_consistent() -> None:
    votes = [
        PolygonVotes("neg", 2024, False, False, 100, np.full(settings.N_MODELS, 10)),
        PolygonVotes("pos_big", 2024, True, True, 50, np.full(settings.N_MODELS, 40)),
        PolygonVotes("pos_small", 2024, True, False, 50, np.full(settings.N_MODELS, 30)),
    ]
    rows = pd.DataFrame(_metric_rows(votes, 2024, "glacier"))

    assert set(rows.stratum) == set(STRATA)
    assert (rows.tp + rows.fn == rows.n_positive_px).all()
    assert (rows.tn + rows.fp == rows.n_negative_px).all()

    combined = rows[rows.stratum == "all"].iloc[0]
    assert combined.tp == 70  # 40 + 30
    assert combined.recall_crevasse == 70 / 100
    assert combined.specificity_noncrevasse == 90 / 100

    big = rows[rows.stratum == "big_crevasses"].iloc[0]
    assert big.recall_crevasse == 40 / 50


def zone_validation(year: int = 2024, output_path: Path = ZONE_VALIDATION_FILE) -> pd.DataFrame:
    """Score the holdout year separately within each named glacier zone.

    This reports how detection performance varies between regions of
    Svalbard. It is **not** a test of spatial generalization: every model was
    trained on polygons drawn from every zone, so each zone's score still
    benefits from that zone having been seen during training. Only a
    leave-one-zone-out retraining would isolate spatial leakage, and that
    cannot be done offline because the models are fitted in Earth Engine on
    10 m embeddings that are not held locally.

    Zones are also unevenly labelled: in 2024 only S Spitsbergen, NE
    Spitsbergen and NW Spitsbergen carry enough positive polygons for their
    recall to mean much. The polygon count, not the pixel count, is the
    effective sample size, because pixels inside one polygon are strongly
    correlated, and the two diverge sharply: Austfonna has the largest
    positive-pixel count of any zone from only three polygons. The manuscript
    therefore reports the three well-labelled zones in prose rather than
    tabulating all of them; this CSV goes in the data deposit so the rest
    remain available.

    Returns
    -------
    pandas.DataFrame
        One row per zone and model, over the glacierized domain and the
        combined positive class, with the metric columns of
        `build_validation` plus `zone_name` and polygon counts.
    """
    zones = gpd.read_file(settings.GLACIER_ZONES_FILE)
    labels = read_labels()
    zoned = label_audit.join_zones(labels.to_crs(zones.crs), zones)
    zone_of = dict(zip(zoned.polyg_id, zoned.zone_name, strict=True))

    votes = collect_votes(read_labels(), read_glacier_domain(), (year,))
    rows: list[dict[str, object]] = []
    for zone_name in sorted({name for name in zone_of.values() if isinstance(name, str)}):
        in_zone = [vote for vote in votes if zone_of.get(vote.polyg_id) == zone_name]
        positives = [vote for vote in in_zone if vote.crevassed]
        negatives = [vote for vote in in_zone if not vote.crevassed]
        if not positives or not negatives:
            continue
        for row in _metric_rows(in_zone, year, "glacier"):
            if row["stratum"] != "all":
                continue
            rows.append(
                row
                | {
                    "zone_name": zone_name,
                    "n_positive_polygons": len(positives),
                    "n_negative_polygons": len(negatives),
                }
            )

    validation = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    validation.to_csv(output_path, index=False)
    return validation
