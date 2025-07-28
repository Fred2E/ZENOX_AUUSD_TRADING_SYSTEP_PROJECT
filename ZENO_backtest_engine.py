import pandas as pd
import os

# === FILES ===
signal_file = r"C:\Users\open\Documents\ZENO_XAUUSD\outputs\setups\ZENO_A+_signals_ALL.csv"
live_data_root = r"C:\Users\open\Documents\ZENO_XAUUSD\data\live\XAUUSD"
output_file = signal_file  # overwrite with corrected version

# === LOAD SIGNALS ===
df_signals = pd.read_csv(signal_file, parse_dates=['datetime'])
df_signals.columns = [c.lower() for c in df_signals.columns]
df_signals['datetime'] = pd.to_datetime(df_signals['datetime'])

# === ENRICH HIGH/LOW ===
highs = []
lows = []

for _, row in df_signals.iterrows():
    tf = row['timeframe']
    dt = row['datetime']
    live_path = os.path.join(live_data_root, tf, f"XAUUSD_{tf}_LIVE.csv")

    if not os.path.exists(live_path):
        highs.append(None)
        lows.append(None)
        continue

    df_live = pd.read_csv(live_path, parse_dates=['datetime'])
    df_live.columns = [c.lower() for c in df_live.columns]
    df_live.set_index('datetime', inplace=True)

    if dt in df_live.index:
        highs.append(df_live.loc[dt]['high'])
        lows.append(df_live.loc[dt]['low'])
    else:
        highs.append(None)
        lows.append(None)

df_signals['high'] = highs
df_signals['low'] = lows

# Drop rows missing price data
df_signals.dropna(subset=['high', 'low'], inplace=True)

# Save
df_signals.to_csv(output_file, index=False)
print(f"✅ Backtest file updated with high/low: {output_file}")
