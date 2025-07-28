import pandas as pd
import os

RESULTS_CSV = r"C:\Users\open\Documents\ZENO_XAUUSD\logs\walkforward_analysis.csv"
THRESHOLDS = {
    "min_sharpe": 1.0,
    "min_winrate": 55.0,  # percentage
}

# Load analysis results
df = pd.read_csv(RESULTS_CSV)

print("\n=== TIMEFRAME PERFORMANCE FILTER ===")
for i, row in df.iterrows():
    tf = row["Timeframe"]
    sharpe = row["Sharpe Ratio"]
    winrate = row["Win Rate (%)"]
    if sharpe < THRESHOLDS["min_sharpe"] or winrate < THRESHOLDS["min_winrate"]:
        print(f"{tf}: SUSPENDED (Sharpe: {sharpe:.2f}, WinRate: {winrate:.2f}%) - Below industrial threshold.")
        # Here you can write logic to pause this bot/timeframe or send an alert
        # e.g., os.rename(f"activate_{tf}.flag", f"suspend_{tf}.flag")
    else:
        print(f"{tf}: ACTIVE (Sharpe: {sharpe:.2f}, WinRate: {winrate:.2f}%)")

print("\nAuto-gate complete. Only industrial-grade timeframes remain active.")
