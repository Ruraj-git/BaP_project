#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import time
import logging
import calendar
import os
import pandas as pd
import requests
from datetime import datetime
from pathlib import Path

# --- TRIK PRE IMPORT Z ROOTU ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import config
except ImportError:
    print("❌ Chyba: Súbor config.py nebol nájdený!")
    sys.exit(1)

# Nastavenie loggingu
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_api_response(response_json, eoi):
    """Prečistí JSON z API a priradí EOI kód."""
    if not isinstance(response_json, list):
        return []
    
    parsed = []
    for row in response_json:
        if row.get('value') is not None:
            parsed.append({
                'time': row.get('timestamp'),
                'eoi': eoi,
                'value': row.get('value')
            })
    return parsed

def main():
    # Cesty z configu (prevedieme na Path objekty pre lepšiu prácu)
    pollutants_dir = Path(config.DATA_DIR) / "pollutants"
    pollutants_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Načítanie metadát
    if not os.path.exists(config.STATIONS_CSV) or not os.path.exists(config.METADATA_CSV):
        logging.error("Chýbajú konfiguračné CSV súbory v data/ priečinku.")
        return
        
    df_my_stations = pd.read_csv(config.STATIONS_CSV)
    df_meta = pd.read_csv(config.METADATA_CSV, sep=';')
    
    # Spojenie, aby sme získali interné ID (II) pre API volania
    df_meta['EOI'] = df_meta['EOI'].str.strip()
    df_stations = pd.merge(df_my_stations[['eoi']], df_meta[['EOI', 'II']], left_on='eoi', right_on='EOI')
    
    current_year = datetime.now().year
    years_to_fetch = config.YEARS

    for pollutant in config.POLLUTANTS:
        poll_dir = pollutants_dir / pollutant.lower()
        poll_dir.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"🚀 Štartujem sťahovanie pre: {pollutant}")
        
        for index, row in df_stations.iterrows():
            eoi = row['eoi']
            station_ii = row['II']
            
            if pd.isna(station_ii):
                continue
                
            output_file = poll_dir / f"{eoi}_hourly.csv"
            station_data = []
            
            logging.info(f"  Sťahujem {eoi} (II: {int(station_ii)})")
            
            for year in years_to_fetch:
                # Ak sme v aktuálnom roku, sťahujeme len po aktuálny mesiac
                max_month = datetime.now().month if year == current_year else 12

                # Neúplné roky (napr. 2023 od júna) -> preskoč skoršie mesiace
                year_start = config.YEAR_START.get(year)
                start_month = pd.to_datetime(year_start).month if year_start else 1

                for month in range(start_month, max_month + 1):
                    last_day = calendar.monthrange(year, month)[1]
                    
                    # URL poskladaná z configu
                    url = f"{config.API_BASE_URL}/{int(station_ii)}/{pollutant.lower()}"
                    params = {
                        "time_from": f"{year}-{month:02d}-01",
                        "time_to": f"{year}-{month:02d}-{last_day:02d}",
                        "aggregation": "hourly"
                    }
                    
                    try:
                        response = requests.get(url, params=params, timeout=20)
                        response.raise_for_status()
                        data = parse_api_response(response.json(), eoi)
                        if data:
                            station_data.extend(data)
                    except Exception as e:
                        logging.debug(f"Žiadne dáta pre {eoi} v období {year}-{month:02d}")
                    
                    time.sleep(config.API_SLEEP_TIME)

            if station_data:
                df_new = pd.DataFrame(station_data)
                # Pripojenie k existujúcemu súboru (ak je) + dedup + zoradenie.
                # keep='last' -> novo stiahnuté hodnoty prepíšu prekrývajúce sa.
                if output_file.exists():
                    df_old = pd.read_csv(output_file)
                    df_new = pd.concat([df_old, df_new], ignore_index=True)
                df_new['_t'] = pd.to_datetime(df_new['time'])
                df_new = (df_new
                          .drop_duplicates(subset=['eoi', 'time'], keep='last')
                          .sort_values('_t')
                          .drop(columns='_t'))
                df_new.to_csv(output_file, index=False)
                logging.info(f"    ✅ {len(df_new)} riadkov spolu v {output_file.name}")

    logging.info("✨ Sťahovanie všetkých polutantov dokončené.")

if __name__ == "__main__":
    main()