import pandas as pd
import pyproj
import os
import sys

# --- TRIK PRE IMPORT Z ROOTU ---
# Pridá nadradený priečinok do cesty, aby Python našiel config.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import config
except ImportError:
    print("❌ Chyba: Súbor config.py nebol nájdený v koreňovom priečinku projektu!")
    sys.exit(1)

def main():
    # Inicializácia projekcie z konfigurácie
    p = pyproj.Proj(config.PROJ4)
    
    # Načítanie metadát
    if not os.path.exists(config.METADATA_CSV):
        print(f"❌ Chyba: Súbor metadát neexistuje na ceste: {config.METADATA_CSV}")
        return

    metadata_df = pd.read_csv(config.METADATA_CSV, sep=';')
    metadata_df['EOI'] = metadata_df['EOI'].str.strip()
    
    # Len slovenské stanice so súradnicami
    virtual_stations = metadata_df[
        (metadata_df['EOI'].str.startswith('SK', na=False)) & 
        (metadata_df['LAT'].notna()) & (metadata_df['LON'].notna())
    ].copy()
    
    station_list = []
    for _, row in virtual_stations.iterrows():
        # Prepočet súradníc na indexy gridu
        x_map, y_map = p(row['LON'], row['LAT'])
        x_idx = int(round((x_map - config.XORIG) / config.DX))
        y_idx = int(round((y_map - config.YORIG) / config.DY))
        
        # Kontrola, či je stanica v rámci nášho Aladin výrezu
        if 0 <= x_idx < config.NX and 0 <= y_idx < config.NY:
            station_list.append({
                'eoi': row['EOI'], 
                'name': row['NAME'],
                'lat': row['LAT'], 
                'lon': row['LON'],
                'x': x_idx, 
                'y': y_idx,
                'typ_oblasti': row['LOC'], 
                'typ_zdroja': row['TYPE'],
                'altitude': row['ALT']
            })
    
    # Uloženie prefiltrovaného zoznamu staníc
    df_out = pd.DataFrame(station_list)
    df_out.to_csv(config.STATIONS_CSV, index=False)
    print(f"✅ Hotovo. stations.csv bol úspešne vytvorený a obsahuje {len(station_list)} staníc.")

if __name__ == "__main__":
    main()