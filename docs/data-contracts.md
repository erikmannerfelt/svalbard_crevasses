# Data Contracts

## Framing: Exposed Vs. Structural Crevassing

The prediction rasters and everything derived from them (vote fractions,
annual/trend products, figures) measure *detectable/exposed* crevasse
extent, not total structural crevassing. Two things drive the gap:

- **Snow/firn bridging.** A structurally real crevasse can be invisible to
  both the training labels and the predictor for an entire melt season if
  it stays snow-covered. It doesn't stop existing; it stops being
  detectable. A later low-snow year can re-expose it with no new fracturing
  having occurred.
- **Label/predictor timing mismatch.** Training labels are hand-digitized
  from specific (often late-summer) scenes, when snow/firn cover is at its
  seasonal minimum and crevasses are most visible. The predictor
  (`GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`) is an annual composite. A polygon
  correctly labeled positive from one late-summer scene can still carry an
  annual embedding signature dominated by firn/snow, because that's what the
  pixel looked like most of the year. This was identified as a likely
  contributor to model false positives in firn-covered terrain (see
  `docs/label-priorities.md` and the label-embedding diagnostics in
  `scripts/label_embedding_diagnostics.py`).

**Consequence for interpretation:** a change in predicted area/trend can
reflect real structural change, a change in regional snow/melt conditions
between years, or both, and the two are not separable from the predicted
area alone. This directly bears on the surge-period comparison this project
is built around (`surge-period-trend-summary.csv`): a Svalbard-wide
low-snow year would inflate predicted area across surging and non-surging
glaciers alike, which could dilute or fabricate an apparent trend
independent of surge dynamics. Treat predicted-area results as evidence
about *exposed* crevassing, and prefer "exposed crevasse(s)"/"exposed
crevasse extent" over bare "crevasse(s)" when describing the predicted or
trend quantity in new writing.

## Source Bundle

The source bundle is provisioned manually in `input/`. The workflow does not
download or refresh it.

The prediction rasters are named
`crevasse_model_probabilities_{year}.tif` for every year from 2017 through
2025. Each raster must contain:

- 30 bands of type `uint8`.
- Band descriptions `prob_m00` through `prob_m29` in order.
- CRS `EPSG:32633`.
- 50 m by 50 m pixels.

A model vote is positive when its uint8 value is at least 128. Derived vote
fractions contain one uint8 band named `vote_fraction`; decode a stored value
as `value / 255`.

## Identity And Geometry

Prediction rows are keyed by `(anlys_id, model_id, year)` and must be unique.
The expected data contain 1,583 glaciers, 30 models, and all nine years from
2017 through 2025.

Predictions join the RGI surge database through `anlys_id`. The prediction
`glac_id` and RGI `glims_id` are independent identity fields and must match
exactly after the join. Do not silently substitute one for the other.

Use `rgi_id` as the glacier identity. Glacier names are not unique in RGI7.
The largest-polygon rule is intentionally limited to the configured
name-based six-glacier timeseries selection.

RGI geometries must be transformed to `EPSG:32633` before rasterization,
sampling, or centroid calculations.

## Surge Data

`src/surge.py` reads the RGI surge archive, applies the project’s documented
manual overrides, and computes whether a glacier has a surge overlapping the
configured analysis period.

The interval rules are:

- `Ongoing` has no upper bound.
- `Not observed` is treated as ending ten years after onset.
- Semicolon-separated onset and termination values are paired by position.
- Invalid or non-numeric onset values do not create an interval.

The overrides are provenance data from the project interpretation of the
source archive. Preserve them unless the interpretation is intentionally
revised and documented.

## Canonical Tables

Canonical Arrow products are written under `output/data/`:

| Product | Grain |
| --- | --- |
| `annual-model.arrow` | one glacier, model, and year |
| `glacier-model-trends.arrow` | one glacier and model |
| `glacier-trends.arrow` | one glacier, with ensemble summaries |
| `regional-annual-model.arrow` | one region/group, model, and year |
| `regional-model-trends.arrow` | one region/group and model |
| `regional-trends.arrow` | one region/group |
| `representative-glaciers.arrow` | selected glacier examples |

Areas are expressed in square metres in the raw annual columns and square
kilometres in derived area and trend columns. Trend rates are square
kilometres per year unless a column explicitly says percentage points per
year.

## Surge-Period Summary

`surge-period-trend-summary.csv` is an analysis result, not map-only data. It
contains six rows for:

```text
pre/recent/none × negative/positive
```

Its fields are:

- `surge_period`
- `trend_sign`
- `median_rate_km2_yr`
- `rate_low_km2_yr`
- `rate_high_km2_yr`
- `rate_uncertainty_km2_yr`
- `significant_glacier_count`
- `glacier_count`
- `model_count`

The `pre`, `recent`, and `none` classifications are calculated in
`src/analysis.py`. The trend map reads this CSV and must not duplicate the
classification or rate aggregation.
