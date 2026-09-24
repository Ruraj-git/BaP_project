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

## Derived validation tables

`results/clean_rebuild/r02/scores/` (main fits) and
`results/clean_rebuild/r03_review/scores/` (review extension):

- `summary_metrics.csv` — RMSE, MAE, bias and R² by model, task, estimand and population;
- `paired_uncertainty.csv` — paired differences with 95% bootstrap intervals;
- `station_year_means.csv` — observed and predicted station-year means over sampled dates.

Units are ng m⁻³. Tested with the versions in `requirements.txt`.
