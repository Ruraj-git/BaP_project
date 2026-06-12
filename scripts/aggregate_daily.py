import pandas as pd
import numpy as np
import glob
import os
import sys

# Cesty riadené cez config.py (PROJECT_ROOT + RUN)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(REPO_ROOT)
import config  # noqa: E402
HEATING_THRESHOLD = 15.5
NIGHT_HOURS = [20, 21, 22, 23, 0, 1, 2, 3, 4, 5]
HEATING_MONTHS = {10, 11, 12, 1, 2, 3, 4}

# Prah pre detekciu stagnácie (slabý vietor + sucho -> hromadenie BaP)
STAGNATION_WS = 2.0      # m/s
STAGNATION_RAIN = 1.0    # mm/deň


def get_pollutant_features(eoi):
    poll_data = {}
    for poll in ['pm10', 'pm25', 'no2']:
        f_path = os.path.join(config.POLLUTANTS_DIR, poll, f'{eoi}_hourly.csv')
        if os.path.exists(f_path):
            df_p = pd.read_csv(f_path)
            df_p['time'] = pd.to_datetime(df_p['time'])
            df_p['datum'] = df_p['time'].dt.date
            daily = df_p.groupby('datum')['value'].mean().rename(f'{poll}_mean')
            poll_data[poll] = daily
    return pd.concat(poll_data.values(), axis=1).reset_index() if poll_data else pd.DataFrame()


def create_rich_features(df_hourly):
    """Denné príznaky z hodinových meteo dát (vrátane nových group-1 príznakov)."""
    df_hourly['time'] = pd.to_datetime(df_hourly['time'])
    df_hourly['datum'] = df_hourly['time'].dt.date
    daily_features = []
    for d, group in df_hourly.groupby('datum'):
        temp_c = group['2t'] - 273.15

        # Vektorový denný priemer vetra -> prevažujúci smer (sin/cos)
        u_mean, v_mean = group['10u'].mean(), group['10v'].mean()
        wdir_rad = np.arctan2(u_mean, v_mean)  # meteorologický smer odkiaľ fúka

        feat = {
            'datum': d,
            't_mean': temp_c.mean(),
            't_range': temp_c.max() - temp_c.min(),
            'heating_degree_hours': temp_c.apply(lambda x: max(0, HEATING_THRESHOLD - x)).sum(),
            'vrate_min': group['vrate'].min(),
            'hpbl_min': group['hpbl'].min(),
            'total_rain': group['rain'].sum(),
            'ws_max': np.sqrt(group['10u'] ** 2 + group['10v'] ** 2).max(),
            # --- NOVÉ (group-1) ---
            'wdir_sin': np.sin(wdir_rad),
            'wdir_cos': np.cos(wdir_rad),
            'ssrd_total': group['ssrd'].sum() if 'ssrd' in group else np.nan,   # slnečné žiarenie -> fotodegradácia
            'sp_mean': group['sp'].mean() if 'sp' in group else np.nan,          # tlak (stagnácia)
            'rh_mean': group['2r'].mean() if '2r' in group else np.nan,          # vlhkosť
            'is_weekend': 1 if d.weekday() >= 5 else 0,
            'month': d.month,
        }
        night = group[group['time'].dt.hour.isin(NIGHT_HOURS)]
        feat['night_vrate_avg'] = night['vrate'].mean() if not night.empty else np.nan
        feat['night_sshf_min'] = night['sshf'].min() if not night.empty else np.nan
        daily_features.append(feat)

    daily = pd.DataFrame(daily_features)

    # --- Cyklická sezónnosť + vykurovacia sezóna ---
    dt = pd.to_datetime(daily['datum'])
    doy = dt.dt.dayofyear
    daily['doy_sin'] = np.sin(2 * np.pi * doy / 365.25)
    daily['doy_cos'] = np.cos(2 * np.pi * doy / 365.25)
    daily['heating_season'] = daily['month'].isin(HEATING_MONTHS).astype(int)

    return daily


def add_temporal_features(daily):
    """Lag / rolling / stagnácia – počíta sa na SÚVISLOM dennom rade (pred join s obs)."""
    daily = daily.sort_values('datum').reset_index(drop=True)

    # Kĺzavé priemery disperzných premenných (atmosférická "pamäť" epizód)
    for col, win in [('hpbl_min', 3), ('hpbl_min', 7),
                     ('vrate_min', 3), ('t_mean', 3)]:
        daily[f'{col}_{win}d'] = daily[col].rolling(win, min_periods=1).mean()

    # Lag predošlého dňa
    daily['t_mean_lag1'] = daily['t_mean'].shift(1)
    daily['hpbl_min_lag1'] = daily['hpbl_min'].shift(1)

    # Lagy REÁLNYCH koncentrácií (PM10/PM2.5/NO2) – na súvislom kalendárnom rade.
    # BaP zámerne NIE (autoregresia BaP nie je dostupná v reálnych medzerách / na virt. staniciach).
    for p in ['pm10_mean', 'pm25_mean', 'no2_mean']:
        if p not in daily.columns:
            daily[p] = np.nan
        daily[f'{p}_lag1'] = daily[p].shift(1)
        daily[f'{p}_3d'] = daily[p].rolling(3, min_periods=1).mean()

    # Tendencia tlaku (deň-na-deň)
    daily['sp_tendency'] = daily['sp_mean'].diff()

    # Počet po sebe nasledujúcich stagnačných dní (slabý vietor + sucho)
    is_stag = ((daily['ws_max'] < STAGNATION_WS) & (daily['total_rain'] < STAGNATION_RAIN)).astype(int)
    run = is_stag.copy()
    for i in range(1, len(run)):
        if is_stag.iloc[i]:
            run.iloc[i] = run.iloc[i - 1] + 1
    daily['stagnation_run'] = run.values

    return daily


def prepare_training_data():
    hourly_dir = config.HOURLY_FEATURES_DIR
    hourly_files = glob.glob(f'{hourly_dir}/*_hourly.csv')
    stations_meta = pd.read_csv(config.STATIONS_CSV)
    stations_meta['eoi'] = stations_meta['eoi'].astype(str).str.strip()

    obs = pd.read_csv(config.OBSERVATIONS_CSV)
    obs['datum'] = pd.to_datetime(obs['datum']).dt.date
    obs['eoi'] = obs['eoi'].astype(str).str.strip()

    final_dataset = []
    for f in hourly_files:
        eoi = os.path.basename(f).replace('_hourly.csv', '').strip()
        s_meta = stations_meta[stations_meta['eoi'] == eoi]
        if s_meta.empty:
            continue

        df_met = create_rich_features(pd.read_csv(f))
        df_poll = get_pollutant_features(eoi)
        df_combined = pd.merge(df_met, df_poll, on='datum', how='left') if not df_poll.empty else df_met

        # Časové príznaky na súvislom rade (pred join s obs!)
        df_combined = add_temporal_features(df_combined)

        # Statické príznaky
        df_combined['eoi'] = eoi
        df_combined['typ_oblasti'] = s_meta['typ_oblasti'].values[0]
        df_combined['typ_zdroja'] = s_meta['typ_zdroja'].values[0]
        df_combined['lat'] = s_meta['lat'].values[0]
        df_combined['lon'] = s_meta['lon'].values[0]
        df_combined['altitude'] = s_meta['altitude'].values[0]

        station_obs = obs[obs['eoi'] == eoi]
        merged = pd.merge(station_obs, df_combined, on=['datum', 'eoi'], how='inner')
        if not merged.empty:
            final_dataset.append(merged)

    if final_dataset:
        full_train = pd.concat(final_dataset, ignore_index=True)
        os.makedirs(config.DAILY_DIR, exist_ok=True)
        out_path = config.train_ready("v4")
        full_train.to_csv(out_path, index=False)
        print(f"✅ Vytvorený dataset (group-1 príznaky): {out_path}")
        print(f"   {len(full_train)} riadkov, {full_train['eoi'].nunique()} staníc, "
              f"{full_train.shape[1]} stĺpcov")


if __name__ == "__main__":
    prepare_training_data()
