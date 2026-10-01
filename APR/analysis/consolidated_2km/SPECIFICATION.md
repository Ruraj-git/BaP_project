# Consolidated 2 km input-ladder analysis — frozen specification

Status: APPROVED by author 2026-09-24 (with §6a additions) and frozen. Any
deviation is logged in an addendum with its reason, before results are seen
where possible.

## 1. Purpose

Replace the historically grown model set (PM-XGB, FULL-ID, G, G+P, P+calendar,
M-AUX; two static supports) with one nested ladder of input groups at a single
2 km static support, evaluated on the same records, folds and scoring as the
current analysis. Aims: (a) isolate each input group's contribution in both the
temporal-gap and withheld-station tasks; (b) remove the two-support caveat;
(c) give M-AUX a defined role; (d) settle the weighting question on the
principal model.

## 2. Fixed elements (unchanged from the clean rebuild r02/r03)

- Records: 6462 observations, 21 stations, 2 June 2023 – 30 December 2025
  (`APR/results/clean_rebuild/r02/inputs/train_ready_permissive_2km_exact.csv`;
  strict-hour variant `train_ready_strict18_2km_exact.csv`). SHA-256 of both
  recorded in the run contract; target keys asserted identical to r02.
- Primary scoring: 2024–2025 (5089 obs; 4852 nonindustrial; 237 SK0018A).
  Nonindustrial is the primary population; full network always reported;
  industrial station descriptive and kept in training.
- Estimator: XGBoost, 1200 trees, learning rate 0.02, max depth 6, L1 0.1,
  L2 1.2, seed 42, one thread; initial prediction = unweighted training mean on
  the fitted scale. Target log(1+B[a]P), back-transform max(0, exp(x)−1).
  Monotone directions unchanged (traffic load +, heavy-duty load +, residential
  B[a]P/PM2.5 emissions +, distance to major road −, altitude − where present).
- Folds: 32 network-wide 30-day blocks from 2023-06-02; 21 LOSO folds;
  9 dispersed folds ((date − 2023-06-02) mod 9); year-out (2024, 2025),
  complete season-year-out, calendar-season-out.
- Uncertainty: paired percentile bootstrap, 2000 replicates, seed 20260921;
  pigeonhole (station × block) for 30-day daily scores, station clusters
  otherwise. "Supported" = 95% interval excludes zero; otherwise "unresolved".
- Simple comparators (PM ridge, seasonal harmonics, linear and log-linear
  interpolation) do not use static covariates; their existing r02 predictions
  are reused unchanged.

## 3. Static support

All tree models use exact-intersection 2 km static covariates, matching the
2 km ALADIN meteorological grid. Rationale stated in the paper: one support for
all inputs; finer static support is a separate question for the mapping study.
The existing r02 G+P results at 500 m are reported as a support sensitivity
(no refit).

## 4. Naming and input groups

Calendar terms (day-of-year sine/cosine, heating season, month, weekend; 5) are
in every tree model and are not part of the names.

| Letter | Group | Variables |
|---|---|---|
| PM | PM10, PM2.5: same-day, lag 1, 3-day mean | 6 |
| P  | PM (6) + NO2 same-day, lag 1, 3-day mean | 9 |
| E  | meteorology (22) + terrain (6) + traffic (3) + residential emissions (2) | 33 |
| M  | station metadata: area type, source type, station altitude | 3 |
| I  | one-hot training-station indicators | 20–21 |

## 5. Model ladder

| Model | Inputs (+ calendar) | Features excl. I | Old name |
|---|---|---|---|
| PM | PM | 11 | — (new) |
| P | P | 14 | P+calendar |
| E | E | 38 | G |
| E+P | E + P (principal model) | 47 | G+P |
| E+P+M | E + P + M | 50 | M-AUX |
| E+P+M+I | E + P + M + I | 50 + I | FULL-ID |
| PM+I | PM + I | 11 + I | PM-XGB |

Models with I are evaluated only in temporal tasks (every test station is in
training under temporal withholding).

## 6. Tasks and fits

| Block | Models / variants | Folds | Fits |
|---|---|---|---|
| 30-day gaps | all 7 models | 32 | 224 |
| LOSO | PM, P, E, E+P, E+P+M | 21 | 105 |
| Dispersed gaps | E+P | 9 | 9 |
| Stress tests | E, E+P | 2 + 9 + 4 | 30 |
| Sensitivity: inverse weights 1/(B[a]P+0.5) | E+P | 32 + 21 | 53 |
| Sensitivity: untransformed target | E+P | 32 + 21 | 53 |
| Sensitivity: ≥18 valid hours per pollutant day | E+P (strict input) | 32 + 21 | 53 |
| Sensitivity: 500 m static support | E+P | reuse r02 | 0 |
| **Total** | | | **527** |

## 6a. Monitoring-programme handling (added before any fit, 2026-09-24)

Stations differ in pollutant programmes (SK0006R no PM; SK0018A no NO2;
SK0076A large gaps in all three). Reference handling stays XGBoost's native
missing-value routing (no imputation, no indicators). Added:

1. Programme-matched models (30-day and LOSO, 2 km, uniform, log target):
   - E+PM: E + PM (6) + calendar = 44 features (for stations without NO2);
   - E+NO2: E + NO2 (3) + calendar = 41 features (for stations without PM).
   Fits: 2 x (32 + 21) = 106.
   Programme-matched composite ("E+P matched"): each target day is predicted by
   the ladder model matching the pollutant groups available on that day
   (all -> E+P; PM only -> E+PM; NO2 only -> E+NO2; none -> E). Scored against
   E+P on identical records, overall and on incomplete-pollutant days.
2. Masking-augmentation sensitivity (E+P, 30-day and LOSO): each training row
   is duplicated once with one pollutant group masked (PM, NO2 or all,
   chosen uniformly with seed 20260924); original and copy weight 1; test rows
   unchanged. Fits: 53. Scored on all days and on incomplete-pollutant days.
3. Reporting: complete-pollutant-day sensitivity for LOSO +E and +P (§7).

Revised total: 527 + 106 + 53 = 686 fits.

## 7. Pre-specified contrasts (candidate minus comparator)

Primary daily RMSE, nonindustrial and full network; station-year-mean RMSE for
LOSO contrasts.

| Label | Comparison | Tasks |
|---|---|---|
| +NO2 | P vs PM | 30-day, LOSO |
| +E | E+P vs P | 30-day, LOSO |
| +P | E+P vs E | 30-day, LOSO |
| +M | E+P+M vs E+P | 30-day, LOSO |
| +I | E+P+M+I vs E+P+M | 30-day |
| +I (PM level) | PM+I vs PM | 30-day |
| Bundle | E+P+M+I vs PM+I | 30-day (continuity with the current comparison i) |
| Weights | E+P inverse vs E+P uniform | 30-day, LOSO; also by station-year stratum (>1 vs ≤1 ng m⁻³) |
| Target | E+P untransformed vs E+P log | 30-day, LOSO |
| Strict hours | E+P strict vs E+P | 30-day, LOSO |
| Support | E+P 500 m vs E+P 2 km | 30-day, LOSO |
| Dispersed | E+P vs PM ridge, harmonics, linear, log-linear interpolation | dispersed (daily and reconstructed mean) |
| Stress | E+P vs E | year-, season-year-, calendar-season-out |
| Programme-matched | E+P matched vs E+P | 30-day, LOSO; all and incomplete-pollutant days |
| Masking | E+P masked vs E+P | 30-day, LOSO; all and incomplete-pollutant days |
| +NO2 given E | E+P vs E+PM | 30-day, LOSO |
| +PM given E | E+P vs E+NO2 | 30-day, LOSO |

Additional reporting: MAE, bias, R² (coefficient of determination); station
influence (omit one station from scoring) for +E and +P; per-station counts of
improvement; complete-pollutant-day sensitivity for LOSO +E and +P.

Reference choices fixed now: uniform weights, log target, permissive pollutant
days, 2 km support. They are not changed after results are seen.

## 8. Code, outputs and manuscript

- Code: `APR/analysis/consolidated_2km/` (contract with the seven feature lists
  and a task builder), reusing `fit_worker.py`/`scoring_core.py` logic.
  Station indicators applied for arms ending in +I.
- Outputs: `APR/results/consolidated_2km/` (contract, predictions, assembly,
  scores). r02/r03 remain untouched.
- Manuscript: new working copy `APR/manuscript_v3/` from `APR/manuscript`;
  generated assets in `generated_c2km/`; current main paper kept as fallback.
- Release: new public tag `apr-v1.1` after the manuscript is updated.

## 9. Stages (each Slurm stage needs author approval)

1. Code + local contract build + smoke test of a few tasks (Slurm, short queue).
2. Full fits (686 tasks, Slurm array, 1 CPU each).
3. Assembly, validation, scoring and bootstrap (Slurm).
4. Manuscript assets, text update, numeric and cross-reference audit.

## 10. Known consequences

- Headline numbers change (E and E+P move from 500 m to 2 km); conclusions are
  reported as found.
- Comparison labels change from (i)–(iii) to the +NO2/+E/+P/+M/+I scheme.
- The weighting evidence moves from M-AUX to the principal model.

## Addendum A (2026-09-24, after fits, before any scoring)

1. E+P vs simple comparators is additionally scored on 30-day gaps (common
   bracketable support), for continuity with the current paper. Dispersed
   remains the pre-specified comparison.
2. Availability-subset contrasts (complete / incomplete pollutant days) fall
   back from station x block to station-cluster resampling when a pigeonhole
   draw is empty; the resampling scheme is recorded per row.

## Addendum B (2026-09-24, AFTER the 2 km results were seen)

Deviation, declared as post-hoc. The 2 km ladder showed that the withheld-
station gain from environmental inputs (+E, LOSO) is unresolved at 2 km
(-0.037 [-0.230, 0.194] ng m-3) whereas the identical comparison at 500 m
static support (r02 G+P vs P+calendar) was supported (-0.172 [-0.275, -0.056]).
500 m was the primary static support of the previous manuscript version.
Therefore the entire ladder (same 686 tasks, features, folds, estimator, seeds,
scoring) is refitted with 500 m static covariates
(`train_ready_{permissive,strict18}_500m.csv`; meteorology remains 2 km).
Both supports are reported; the support contrast (E+P 500 m vs 2 km) uses the
two consolidated runs. Models without static inputs (PM, P, PM+I) are
support-independent and must reproduce the 2 km run exactly (check).
Outputs: `APR/results/consolidated_500m/`. The choice of primary support for the
manuscript is made with the author after both runs are scored, and the reason
is stated in the paper.

Decision (2026-09-24, author): 500 m static support is primary for the
manuscript (APR/manuscript_v3); the complete 2 km ladder is reported as the
support sensitivity. Reason stated in the paper (Methods 2.4, Discussion):
500 m was the principal model's support during development, and the
withheld-station contribution of environmental inputs depends on support.

## Addendum E (2026-09-30): environmental-bundle decomposition (reserve)

E+P with one environmental subgroup removed at a time: meteorology (22),
terrain (6), traffic (3), residential emissions (2); 32 x 30-day + 21 LOSO folds
each (212 fits, 500 m, reference settings). Scored against E+P. Kept as a
reserve analysis for review; not part of the manuscript unless requested.
Interpretation caveat fixed in advance: correlated groups can substitute for one
another, so a small drop-one effect does not show that a group is uninformative.
Status (2026-09-30): after the results, the author decided to include the
decomposition in the manuscript (Methods 2.4, Results 3.2-3.3, Discussion 4.1,
Table S6).

## Addendum F (2026-10-01, before fitting): inverse-weighted environmental model

Question: can inverse weighting 1/(B[a]P+0.5) compensate for the missing level
information of the environment-only model E at withheld stations, and at what
cost? Fits: E (500 m, log target, all pollutant hours, native missing handling)
with inverse weights; 32 x 30-day + 21 LOSO folds (53 fits).
Scored at the 20 nonindustrial stations against E (uniform) and E+P (uniform):
daily RMSE and R2 (30-day, LOSO); LOSO station-year-mean RMSE, R2 and bias,
overall and by observed stratum (> 1 vs <= 1 ng m-3); paired bootstrap as in
Section 7. Reported whatever the result; intended for one Results sentence, a
row in Table S3 and a supplementary figure in the layout of Fig. 5.
