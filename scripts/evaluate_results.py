import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import sys
from sklearn.metrics import r2_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    import config
except ImportError:
    sys.exit(1)

def evaluate_network():
    results_files = glob.glob(os.path.join(config.OUTPUT_DIR, "*_filled.csv"))
    stations_df = pd.read_csv(config.STATIONS_CSV)
    name_map = dict(zip(stations_df['eoi'], stations_df['name']))

    plot_dir = os.path.join(config.OUTPUT_DIR, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    stats, g_obs, g_pred = [], [], []

    for f in results_files:
        eoi = os.path.basename(f).replace('_filled.csv', '')
        df = pd.read_csv(f)
        df['datum'] = pd.to_datetime(df['datum'])
        
        eval_df = df.dropna(subset=['bap'])
        has_obs = len(eval_df) >= 3
        
        res = {'EOI': eoi, 'Name': name_map.get(eoi, eoi), 'R2': np.nan, 'Bias_%': 0}
        
        if has_obs:
            res['R2'] = r2_score(eval_df['bap'], eval_df['bap_predicted'])
            res['Bias_%'] = ((eval_df['bap_predicted'].mean() - eval_df['bap'].mean()) / eval_df['bap'].mean()) * 100
            stats.append(res)
            g_obs.extend(eval_df['bap'])
            g_pred.extend(eval_df['bap_predicted'])

        # GRAF PRE KAŽDÚ STANICU
        plt.figure(figsize=(12, 5))
        plt.plot(df['datum'], df['bap_predicted'], color='red', label='Model')
        if has_obs:
            plt.scatter(df['datum'], df['bap'], color='black', s=10, label='Obs')
        plt.axhline(1.0, color='gray', linestyle='--', label='Limit')
        plt.title(f"{eoi} - {res['Name']} (R2: {res['R2']:.2f})")
        plt.legend()
        plt.savefig(os.path.join(plot_dir, f"{eoi}_plot.png"))
        plt.close()

    # Sumár do konzoly
    df_s = pd.DataFrame(stats).sort_values('R2', ascending=False)
    print(df_s[['EOI', 'Name', 'R2', 'Bias_%']])
    
    print(f"\nGLOBÁLNY PRIEMER: Obs={np.mean(g_obs):.3f} | Model={np.mean(g_pred):.3f}")
    print(f"Grafy sú v: {plot_dir}")

if __name__ == "__main__":
    evaluate_network()