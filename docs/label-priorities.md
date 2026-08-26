# Crevasse Label Coverage Priorities

Snapshot as of 2026-08-25 (second pass, after the year-field fix), from
`just label-audit` / `just label-audit-map` (see
`output/data/label-zone-year-summary.csv` and
`output/figures/label-audit-map.png`, and "Label Coverage Audit" in
`AGENTS.md`). This is a point-in-time status note, not a fixed reference doc
-- regenerate the audit and rewrite this file after each labeling pass
rather than trusting the numbers below once new labels have been added.

Target training-year set: 2017, 2019, 2021, 2023, 2025 (every other year).
2024 is an intentional holdout and is not scored here.

683 raw labels total (up from 648), 131 dropped as expected pan-Arctic
remnants (same count as the prior pass -- confirms none of the new labels
are pan-Arctic leftovers), 1 dropped as outside all named zones
(`2024_neg_25`, on Hopen).

## All three priority gaps from the last pass are fixed

- **N Spitsbergen, 2025**: 0/5 -> **7/5**. This was the one 2025 gap that
  hadn't been touched in the first round of new labeling; now has a solid
  set of positives.
- **2023: N Spitsbergen (was blank) and C Spitsbergen (was positive-only)**:
  N Spitsbergen 0/0 -> 2/2, C Spitsbergen 1/0 -> 5/3. 2023's total also grew
  37 -> 50 and no longer has any single-class or blank cells in a
  known-crevassed zone.
- **Austfonna, 2021**: 0/0 -> **4/3**. This was flagged as pre-existing
  (2021 hadn't been touched at all in the prior pass); now filled in.

## Remaining items, ranked by severity

| Rank | Issue | Detail |
|---|---|---|
| 1 | **N Spitsbergen 2017 (0/1) and 2019 (1/0), single-class** | Same two minor flags from last time, still open. Low volume (1 label each) -- a couple of the missing-class examples would clear these, but low-stakes relative to what's already been fixed. |
| -- | Nothing else known-crevassed-zone-related is currently flagged. | All of NW/N/C/S/NE Spitsbergen and Austfonna have both classes in every target year except the two N Spitsbergen cells above. |

## Reading this table

- "Blank" (zero labels) and "single-class" (only positive or only negative)
  cells are not automatically defects. Cross-check against
  `output/figures/label-audit-map.png` and physical plausibility.
- **Standing pattern, not a gap**: Barentsøya/Edgeøya (blank 2021, 2023),
  Vestfonna (blank 2019, 2025), and Kvitøya (blank in every target year
  except 2017) remain thin/blank. This is consistent across the whole
  dataset and matches known low real crevassing in these zones -- not
  something to chase.
- Yearly totals across target years: 2017: 93, 2019: 95, 2021: 101, 2023:
  50, 2025: 92. 2023 is still the lowest-volume year but is no longer a
  structural outlier -- it's roughly half the others rather than a fifth.
- This is a proof-of-concept; the goal is defensible labeling, not perfect
  stratified balance across all 9 zones x 9 years x 2 classes.
