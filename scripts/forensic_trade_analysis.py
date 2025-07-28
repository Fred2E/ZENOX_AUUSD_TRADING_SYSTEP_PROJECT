import pandas as pd
import os
import matplotlib.pyplot as plt

# --- CONFIG ---
LOG_DIR = r"C:\Users\open\Documents\ZENO_XAUUSD\logs"
OUTPUT_DIR = os.path.join(LOG_DIR, "forensic_analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Auto-discover all trade_log_*.csv files to get available timeframes
TIMEFRAMES = []
for fname in os.listdir(LOG_DIR):
    if fname.startswith("trade_log_") and fname.endswith(".csv"):
        tf = fname.replace("trade_log_", "").replace(".csv", "")
        TIMEFRAMES.append(tf)
TIMEFRAMES = sorted(TIMEFRAMES)

def analyze_trades(tf):
    csv_path = os.path.join(LOG_DIR, f"trade_log_{tf}.csv")
    if not os.path.exists(csv_path):
        print(f"Trade log for {tf} not found.")
        return None
    df = pd.read_csv(csv_path)
    print(f"\n=== {tf} Forensic Analysis ===")
    print(f"Total trades: {len(df)}")
    losers = df[df['reward_pips'] < 0]
    print(f"Losing trades: {len(losers)} ({100*len(losers)/len(df):.2f}%)")

    # --- Group loss clusters ---
    clusters = {}
    keys = [
        ('pattern_code', 'Pattern Code'), 
        ('bias_bull', 'Bias Bull'),
        ('duration', 'Trade Duration'),
        ('hour', 'Hour'),
        ('dow', 'Day of Week'),
        ('side', 'Side'),
        ('forced', 'Forced Close')
    ]
    for col, name in keys:
        if col in losers.columns:
            grp = losers.groupby(col)['reward_pips'].agg(['count', 'mean', 'min', 'sum']).sort_values('mean')
            clusters[col] = grp
            out_csv = os.path.join(OUTPUT_DIR, f"{tf}_loss_cluster_{col}.csv")
            grp.to_csv(out_csv)
            print(f"  Exported loss cluster by {name} to {out_csv}")

            # --- Plotting (matplotlib bar plot) ---
            plt.figure(figsize=(7,4))
            grp['sum'].plot(kind='bar', color='red')
            plt.title(f"{tf} Total Loss Pips by {name}")
            plt.xlabel(name)
            plt.ylabel("Total Loss Pips")
            plt.tight_layout()
            plot_path = os.path.join(OUTPUT_DIR, f"{tf}_loss_pips_by_{col}.png")
            plt.savefig(plot_path)
            plt.close()
            print(f"  Saved plot: {plot_path}")

    # --- Summary of worst trades ---
    worst = losers.sort_values('reward_pips').head(10)
    summary_path = os.path.join(OUTPUT_DIR, f"{tf}_worst_trades.csv")
    worst.to_csv(summary_path, index=False)
    print(f"  Top 10 worst trades exported to {summary_path}")

for tf in TIMEFRAMES:
    analyze_trades(tf)

print("\n=== Forensic analysis complete. Check logs/forensic_analysis/ for details. ===")
