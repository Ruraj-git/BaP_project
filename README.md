# GAPFILL-BaP: Virtuálna monitorovacia sieť pre Benzo(a)pyrén

> **About this repository.** Analysis code for the manuscript *"A globally
> trained machine-learning model for daily benzo[a]pyrene gap-filling and the
> construction of a virtual monitoring network over Slovakia"* (SHMÚ). The
> input data (B[a]P observations, ALADIN meteorology, proxy pollutants and the
> static covariates) and the model outputs are **not** included here — see the
> data-availability statement in the paper. All paths are derived automatically
> from `config.py`; install dependencies with `pip install -r requirements.txt`.
> *(The documentation below is in Slovak.)*

Tento projekt implementuje globálny model strojového učenia na dopĺňanie chýbajúcich meraní $BaP$ a tvorbu virtuálnej monitorovacej siete pre celé Slovensko.

## Model a Metodika
- **Algoritmus:** XGBoost Regressor (Globálny model pre celú sieť).
- **Transformácia:** Logaritmická škála $\ln(x+1)$ pre stabilizáciu čistých lokalít.
- **Vstupy:**
  - Meteorológia: ALADIN SHMU (2km rozlíšenie).
  - Proxy dáta: Hodinové koncentrácie $PM_{10}$, $PM_{2.5}$ a $NO_2$ z API.
  - Typológia: Klasifikácia staníc (LOC, TYPE) z metadát NMSKO.

## Štruktúra projektu
- `scripts/prepare_stations.py`: Generovanie zoznamu aktívnych a virtuálnych staníc.
- `scripts/fetch_pollutants.py`: Sťahovanie hodinových proxy dát z API.
- `scripts/aggregate_daily.py`: Feature engineering a spájanie dátových zdrojov.
- `scripts/train_and_fill_bap.py`: Trénovanie modelu a generovanie gap-filled výsledkov.
- `scripts/evaluate_results.py`: Výpočet štatistík ($R^2$, MAE) pre jednotlivé stanice.

## Dátová štruktúra a behy (runs)
Cesty sú riadené centrálne v `config.py`. `PROJECT_ROOT` sa odvodzuje z polohy
súboru, takže projekt sa dá presunúť/skopírovať bez úprav ciest. Premenná
`RUN` v `config.py` vyberá aktívny beh:

```
config.py            # PROJECT_ROOT (auto) + RUN ("base", "extended", ...)
data/                # ZDIEĽANÉ vstupy (rovnaké pre všetky behy)
  ├── stations.csv, bap_obs.csv, pollutants/, other/
  ├── DMR3.5/, roads/, emis/, ghsl/, emep/   # zdroje statických kovariátov
  └── covariates/     # statické grid/station kovariáty (terén, doprava, emisie)
runs/<RUN>/          # ARTEFAKTY konkrétneho behu
  ├── features/hourly_extracted/, daily_rich_vectors/ (train_ready_bap_v4..v6)
  ├── output/validation/ (+ plots/), output/gap_filled/
  └── models/
runs/current -> base # symlink, ktorý určuje, z ktorého behu číta článok (paper/)
```

Prepnutie behu: uprav `RUN` v `config.py` (pipeline) a `ln -sfn <run> runs/current`
(figúry v článku). `base` = pôvodný dataset 2024–2025; `extended` = + 2023.

Verzie `v4/v5/v6` sú **fázy stavby príznakov** v rámci jedného behu
(v4 = meteo, v5 = +priestorové, v6 = +emisie), nie samostatné behy.

## Výkonnosť
Model dosahuje globálne $R^2 \approx 0.71$, pričom na kľúčových staniciach v údoliach presahuje hodnotu **0.95**.
