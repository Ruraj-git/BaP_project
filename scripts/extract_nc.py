import xarray as xr
import pandas as pd
import os
import glob
import sys
from datetime import timedelta

# --- TRIK PRE IMPORT Z ROOTU ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import config
except ImportError:
    print("❌ Chyba: Súbor config.py nebol nájdený v koreňovom priečinku projektu!")
    sys.exit(1)

def run_extraction():
    # Použitie ciest z konfigurácie
    stations_path = config.STATIONS_CSV
    meteo_path = config.METEO_NC_PATH
    output_dir = config.HOURLY_FEATURES_DIR

    if not os.path.exists(stations_path):
        print(f"❌ Chyba: {stations_path} nebol nájdený.")
        return

    # Načítanie zoznamu staníc
    stations_df = pd.read_csv(stations_path)
    station_list = list(stations_df[['eoi', 'x', 'y']].itertuples(index=False, name=None))
    
    # Hľadanie súborov LEN pre vybrané roky (config.YEARS)
    nc_files = []
    for year in config.YEARS:
        year_dir = os.path.join(meteo_path, str(year))
        found = glob.glob(f'{year_dir}/**/*.nc', recursive=True)
        if not found:
            # fallback: súbory pomenované YYYY-*.nc kdekoľvek v strome
            found = glob.glob(f'{meteo_path}/**/{year}-*.nc', recursive=True)
        if not found:
            print(f"  [Upozornenie] Žiadne .nc súbory pre rok {year}")
        nc_files.extend(found)
    nc_files = sorted(nc_files)
    os.makedirs(output_dir, exist_ok=True)

    all_extracted_data = {s[0]: [] for s in station_list}
    total_files = len(nc_files)

    print(f"🚀 Extrakcia rokov {config.YEARS}: {len(station_list)} staníc z {total_files} súborov...")

    for i, nc_path in enumerate(nc_files):
        if i % 50 == 0:
            print(f"  Spracovávam súbor {i}/{total_files} ({(i/total_files)*100:.1f}%) ...")
            
        try:
            with xr.open_dataset(nc_path) as ds:
                # Získanie dátumu (atribút alebo názov súboru)
                file_date_str = ds.attrs.get('date')
                if not file_date_str:
                    file_date_str = os.path.basename(nc_path).split('.')[0]
                
                base_date = pd.to_datetime(file_date_str)

                for eoi, x_idx, y_idx in station_list:
                    # Výber konkrétneho pixelu z Aladina
                    subset = ds.isel(x=int(x_idx), y=int(y_idx))
                    df_day = subset.to_dataframe().reset_index()
                    
                    # Vyčistenie názvov (napr. odstránenie lomítok z NC premenných)
                    df_day.columns = [c.replace('\\', '') for c in df_day.columns]
                    
                    # Prepočet indexu hodín na reálny Timestamp
                    df_day['time'] = df_day['time'].apply(lambda h: base_date + timedelta(hours=int(h)))
                    
                    all_extracted_data[eoi].append(df_day)

        except Exception as e:
            print(f"  [Chyba] {os.path.basename(nc_path)}: {e}")
            continue

    print("\n💾 Ukladám/pripájam vyextrahované časové rady pre jednotlivé stanice...")
    for eoi, df_list in all_extracted_data.items():
        if not df_list:
            continue
        df_new = pd.concat(df_list, ignore_index=True)
        out_file = os.path.join(output_dir, f"{eoi}_hourly.csv")
        # Pripojenie k existujúcemu záznamu (ak je) + dedup + zoradenie podľa času.
        # keep='last' -> novo vyextrahované hodiny prepíšu prekrývajúce sa.
        if os.path.exists(out_file):
            df_old = pd.read_csv(out_file)
            df_new = pd.concat([df_old, df_new], ignore_index=True)
        df_new['time'] = pd.to_datetime(df_new['time'])
        df_new = df_new.drop_duplicates(subset='time', keep='last').sort_values('time')

        # Ventilačný index jednotne pre CELÝ záznam: vrate = hpbl × rýchlosť vetra.
        # 2023 ho v nc nemá natívne (až od 2024), preto prepočítame všetky roky
        # rovnakým vzorcom -> konzistentná definícia naprieč 2023-2026.
        wind = (df_new['10u'] ** 2 + df_new['10v'] ** 2) ** 0.5
        df_new['vrate'] = df_new['hpbl'] * wind

        df_new.to_csv(out_file, index=False)
        print(f"  Uložené ({len(df_new)} riadkov): {eoi}")

    print(f"\n✅ Extrakcia úspešne dokončená. Dáta sú v: {output_dir}")

if __name__ == "__main__":
    run_extraction()