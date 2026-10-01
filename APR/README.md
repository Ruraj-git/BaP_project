# APR manuscript: code and derived validation tables

Code and derived validation tables for the manuscript *"Benzo[a]pyrene
reconstruction from routine pollutant measurements and environmental data
across the Slovak monitoring network"* (submitted to *Atmospheric Pollution
Research*). Input data (B[a]P observations, ALADIN/SHMÚ meteorology, routine
pollutant measurements, traffic and emission inventories) are not included; see
the manuscript's data-availability statement. Scripts expect the repository
layout below and resolve paths relative to the repository root.

## Pipeline (in order)

| Step | Script(s) in `analysis/clean_rebuild/` |
|---|---|
| Hourly ALADIN extraction at station cells (`ALADIN_GRIB_DIR` = GRIB archive) | `extract_station_hourly.py` |
| Daily inputs: meteorology, pollutant features, static covariates (500 m / 2 km) | `build_daily_inputs.py` |
| Frozen model/feature contract and fit manifest | `model_contract.py`, `build_fit_preflight.py` |
| XGBoost fits (30-day blocks, LOSO, stress tests, sensitivities) | `fit_worker.py` |
| Assembly and validation of out-of-fold predictions | `assemble_validate.py` |
| Simple comparators and scoring, paired bootstrap intervals | `gap_baselines.py`, `scoring_core.py`, `score_evidence.py` |
| Review extension: P+calendar LOSO and G+P dispersed fits | `prepare_review_extension.py`, `review_fit_worker.py`, `validate_review_fits.py`, `score_review_extension.py` |
| Manuscript tables, macros and figures | `generate_manuscript_r02.py`, `generate_manuscript_r03.py`, `generate_manuscript_merge.py`, `review_reading_assets.py` |

Additional figure/table scripts: `analysis/observed_context/review_context_v1.py`
(B[a]P/PM10 context, Fig. 2), `analysis/manuscript_v2/network_typology_map.py`
(Fig. 1) and `analysis/manuscript_v2/contrasts_v2.py` (Fig. 3 and the
supplementary contrast table).

## Consolidated input ladder (release apr-v1.1)

The revised manuscript evaluates a nested ladder of input groups (PM, P = PM +
NO2, E = environmental inputs, M = station metadata, I = station indicators)
with identical settings and 500 m static covariates for all tree models; the
complete ladder with 2 km static covariates is reported as a support
sensitivity. Code: `analysis/consolidated_2km/` (see its README and
`SPECIFICATION.md`); manuscript tables, macros and figures: `analysis/manuscript_v3/assets_v3.py` (Fig. 1: `network_typology_map_v3.py`).
Derived tables: `results/consolidated_500m/scores/` (primary) and
`results/consolidated_2km/scores_v2/` (2 km, scored against 500 m). The drop-one decomposition of the environmental group is in
`results/consolidated_500m_decomp/scores/` (release apr-v1.2). The inverse-weighted environmental model is in
`results/consolidated_500m_einv/scores/` (release apr-v1.3).

Model names in the revised manuscript map to the earlier ones as follows:
PM+I = PM-XGB, E+P+M+I = FULL-ID, E+P+M = M-AUX, E = G, E+P = G+P,
P = P+calendar. E, E+P and P reproduce the earlier G, G+P and P+calendar
predictions exactly.

## Derived validation tables

`results/clean_rebuild/r02/scores/` (main fits) and
`results/clean_rebuild/r03_review/scores/` (review extension):

- `summary_metrics.csv` — RMSE, MAE, bias and R² by model, task, estimand and population;
- `paired_uncertainty.csv` — paired differences with 95% bootstrap intervals;
- `station_year_means.csv` — observed and predicted station-year means over sampled dates.

Units are ng m⁻³. Tested with the versions in `requirements.txt`.
