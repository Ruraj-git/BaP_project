# Consolidated input ladder (both static supports)

Despite the folder name, these scripts run the ladder at either static support
(`--support 500m` or `--support 2km_exact` in `contract_c2km.py`); the fit
worker and assembly read the support from the manifest. The manuscript uses
500 m as primary and reports the complete 2 km ladder as a support sensitivity.
The frozen specification, with its addenda and the support decision, is in
`SPECIFICATION.md`.

Pipeline: `contract_c2km.py` (frozen feature/fold contract and 686-task
manifest) -> `fit_worker_c2km.py` (one XGBoost fit per task) ->
`assemble_c2km.py` (re-validation of every fit, assembled out-of-fold
predictions) -> `score_c2km.py` (scores, paired bootstrap intervals,
input-group and sensitivity contrasts; `--alt-assembled` gives the other
support's run for the support contrast). Shared functions are imported from
`../clean_rebuild/` (`model_contract.py`, `build_fit_preflight.py`,
`fit_worker.py`, `scoring_core.py`). Manuscript tables, macros and figures:
`../manuscript_v3/assets_v3.py`.

Environmental-group decomposition (manuscript Table S6; SPECIFICATION.md addendum E):
`contract_decomp.py` builds the drop-one manifest (E+P without meteorology, terrain, traffic or
residential emissions; 212 fits), fitted and validated with the same worker and assembly, and scored
against E+P by `score_decomp.py`. Derived table: `../../results/consolidated_500m_decomp/scores/`.

