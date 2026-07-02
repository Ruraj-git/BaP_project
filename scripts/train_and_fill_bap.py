import pandas as pd
import numpy as np
import xgboost as xgb
import os
import glob
import sys

# --- IMPORT CONFIG ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    import config
except ImportError:
    print("❌ Chyba: config.py nebol nájdený!")
    sys.exit(1)

def get_pollutant_features(eoi):
    poll_data = {}
    for poll in ['pm10', 'pm25', 'no2']:
        f_path = os.path.join(config.DATA_DIR, f'pollutants/{poll}/{eoi}_hourly.csv')
        if os.path.exists(f_path):
            df_p = pd.read_csv(f_path)
            df_p['time'] = pd.to_datetime(df_p['time'])
            df_p['datum'] = df_p['time'].dt.date
            poll_data[poll] = df_p.groupby('datum')['value'].mean().rename(f'{poll}_mean')
    return pd.concat(poll_data.values(), axis=1).reset_index() if poll_data else pd.DataFrame()

def create_rich_features(df_hourly):
    df_hourly['time'] = pd.to_datetime(df_hourly['time'])
    df_hourly['datum'] = df_hourly['time'].dt.date
    daily = []
    for d, group in df_hourly.groupby('datum'):
        temp_c = group['2t'] - 273.15
        feat = {
            'datum': d,
            't_mean': temp_c.mean(),
            'heating_degree_hours': temp_c.apply(lambda x: max(0, config.HEATING_THRESHOLD - x)).sum(),
            'vrate_min': group['vrate'].min(),
            'hpbl_min': group['hpbl'].min(),
            'total_rain': group['rain'].sum(),
            'ws_max': np.sqrt(group['10u']**2 + group['10v']**2).max(),
            'is_weekend': 1 if d.weekday() >= 5 else 0,
            'month': d.month
        }
        night = group[group['time'].dt.hour.isin(config.NIGHT_HOURS)]
        feat['night_vrate_avg'] = night['vrate'].mean() if not night.empty else np.nan
        feat['night_sshf_min'] = night['sshf'].min() if not night.empty else np.nan
        daily.append(feat)
    return pd.DataFrame(daily)

def train_and_impute():
    print("🚀 Spúšťam stabilnú verziu BaP (váženie podľa koncentrácie)...")
    
    df_train = pd.read_csv(config.TRAIN_READY_CSV)
    df_train['typ_oblasti_code'] = df_train['typ_oblasti'].str.strip().map(config.AREA_MAP).fillna(0)
    df_train['typ_zdroja_code'] = df_train['typ_zdroja'].astype('category').cat.codes

    features = [
        't_mean', 'heating_degree_hours', 'vrate_min', 'hpbl_min', 
        'night_vrate_avg', 'night_sshf_min', 'total_rain', 'ws_max',
        'is_weekend', 'month', 'pm10_mean', 'pm25_mean', 'no2_mean',
        'typ_oblasti_code', 'typ_zdroja_code', 'altitude'
    ]
    
    train_clean = df_train.dropna(subset=['bap']).copy()
    X = train_clean[features]
    y = np.log1p(train_clean['bap'])
    
    # NOVÝ PRÍSTUP K VÁHAM: 
    # Čím nižšia je nameraná hodnota BaP, tým vyššiu váhu má vzorka. 
    # To pomôže pozaďovým staniciam (Kolonické sedlo) bez rozbitia kategórií.
    weights = 1.0 / (train_clean['bap'] + 0.5)

    params = dict(config.MODEL_PARAMS)
    params["monotone_constraints"] = config.monotone_for(features)
    model = xgb.XGBRegressor(**params)
    model.fit(X, y, sample_weight=weights)

    stations_meta = pd.read_csv(config.STATIONS_CSV)
    hourly_files = glob.glob(os.path.join(config.FEATURES_DIR, 'hourly_extracted/*_hourly.csv'))

    for f in hourly_files:
        eoi = os.path.basename(f).replace('_hourly.csv', '').strip()
        df_daily = create_rich_features(pd.read_csv(f))
        df_poll = get_pollutant_features(eoi)
        df_final = pd.merge(df_daily, df_poll, on='datum', how='left') if not df_poll.empty else df_daily

        for poll_col in ['pm10_mean', 'pm25_mean', 'no2_mean']:
            if poll_col not in df_final.columns: df_final[poll_col] = np.nan

        s_meta = stations_meta[stations_meta['eoi'] == eoi]
        if s_meta.empty: continue
        
        df_final['typ_oblasti_code'] = config.AREA_MAP.get(str(s_meta['typ_oblasti'].values[0]).strip(), 0)
        df_final['typ_zdroja_code'] = pd.Categorical([str(s_meta['typ_zdroja'].values[0]).strip()], 
                                                     categories=df_train['typ_zdroja'].astype('category').cat.categories).codes[0]
        df_final['altitude'] = s_meta['altitude'].values[0]
        
        df_final['bap_predicted'] = np.maximum(0, np.expm1(model.predict(df_final[features])))
        
        obs = pd.read_csv(config.OBSERVATIONS_CSV)
        obs['datum'] = pd.to_datetime(obs['datum']).dt.date
        station_obs = obs[obs['eoi'] == eoi]
        df_out = pd.merge(df_final, station_obs[['datum', 'bap']], on='datum', how='left')
        df_out['bap_filled'] = df_out['bap'].fillna(df_out['bap_predicted'])
        
        df_out.to_csv(os.path.join(config.OUTPUT_DIR, f"{eoi}_filled.csv"), index=False)

    print(f"✅ Hotovo. Výsledky sú v {config.OUTPUT_DIR}")

if __name__ == "__main__":
    train_and_impute()