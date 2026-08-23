# Figures

Figure modules are renderers. They load canonical analysis products and own
visual presentation, but they should not recalculate scientific summaries that
already exist in `output/data/`.

## Trend Map

`src/figures/trend_map.py` renders the RGI glacier trend polygons over the S100
land layer. It also shows:

- Recent and earlier surge centroids.
- Fixed annotations for the six selected glaciers.
- A compact trend colorbar and legend.
- The colored uncertainty matrix in the lower-right corner.

The matrix reads `surge-period-trend-summary.csv`. Its cell colors use the
rate values, while its text displays the median rate, rate uncertainty, and
significant-glacier count. Matrix classifications and calculations belong to
the analysis pipeline, not the renderer.

## Selected Timeseries

`src/figures/timeseries.py` renders the six configured glaciers and annual
Arnesenbreen raster panels.

The timeseries uses sampled annual model data linked to RGI7. When multiple
RGI7 polygons share a configured glacier name, it selects the largest polygon
as an explicit presentation choice.

The Arnesenbreen map panels intentionally use outlines from
`glacier_outlines_mod.sqlite`, including the fallback outline years in
`src/settings.py`. They are not substituted with the RGI polygon used for the
timeseries. The figure also intentionally includes the original panel border,
connecting lines, ice-flow arrow, scale bar, and vote-fraction colorbar.

## Per-Surge Figures

`src/figures/per_surge.py` renders one figure for each configured surging
glacier above the fixed area threshold. Outputs are placed under
`output/figures/per-surge/` in directories named from the RGI identifier and a
filesystem-safe glacier name.

The per-surge chart combines the annual model envelope with masked annual
vote-fraction raster panels. RGI geometries are reprojected before sampling.

## Presentation Changes

The established layout and annotations are part of the project’s figure
behavior. Refactors should change data access or configuration deliberately,
not alter visual spacing, annotations, legends, colorbars, matrix styling, or
panel relationships incidentally.
