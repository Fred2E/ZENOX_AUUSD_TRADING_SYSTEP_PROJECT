import os
import pandas as pd
import numpy as np
from random import randint, choice

# === CONFIG ===
ROOT = r"C:\Users\open\Documents\ZENO_XAUUSD"
DATA_DIR = os.path.join(ROOT, "historical")
OUTPUT_CSV = os.path.join(ROOT, "outputs", "setups", "ZENO_A+_simulated_signals_ML.csv")

# Timeframes and file mapping (update as needed)
timeframes = {
    "M5":   "XAUUSDm_M5_HIST.csv",
    "M15":  "XAUUSDm_M15_HIST.csv",
    "H1":   "XAUUSDm_H1_HIST.csv",
    "H4":   "XAUUSDm_H4_HIST.csv",
    "D1":   "XAUUSDm_D1_HIST.csv",
}

# === SETTINGS ===
MIN_SCORE = 3   # Lower for volume
MAX_TRADES_PER_TF = 1000  # Hard cap per timeframe

# === SIMULATION ===
all_trades = []
for tf, fname in timeframes.items():
    path = os.path.join(DATA_DIR, fname)
    if not os.path.exists(path):
        print(f"[SKIP] Missing: {path}")
        continue
    df = pd.read_csv(path, parse_dates=['datetime'])
    df.columns = [c.lower() for c in df.columns]
    df = df.sort_values('datetime')
    n = 0
    for i, row in df.iterrows():
        if n >= MAX_TRADES_PER_TF:
            break

        # --- "Signal" Simulation ---
        score = randint(MIN_SCORE, 6)
        bias = choice(['bullish', 'bearish'])
        pattern = choice(['engulfing', 'pinbar', 'doji', np.nan])
        confluences = ['structure', 'bos_or_choch', 'sr_zone']
        if randint(0, 1): confluences.append('psych_level')
        if randint(0, 1): confluences.append('fib_zone')

        # --- Random SL/TP, Entry ---
        entry = row['close']
        risk = abs(row['high'] - row['low']) * randint(1, 2)
        if bias == 'bullish':
            sl = entry - risk
            tp = entry + 2 * risk
        else:
            sl = entry + risk
            tp = entry - 2 * risk

        # --- Assign outcome, 50/50 for simulation
        is_win = choice([0, 1])

        all_trades.append({
            'datetime': row['datetime'],
            'timeframe': tf,
            'bias': bias,
            'close': entry,
            'stoploss': sl,
            'takeprofit': tp,
            'score': score,
            'candle_pattern': pattern,
            'confluences': confluences,
            'high': row['high'],
            'low': row['low'],
            'is_win': is_win,
        })
        n += 1

    print(f"[DONE] {tf}: {n} trades simulated.")

# === FINAL OUTPUT ===
df_signals = pd.DataFrame(all_trades)
df_signals.to_csv(OUTPUT_CSV, index=False)
print(f"\n[SUCCESS] Simulated signals saved: {OUTPUT_CSV}")
print(df_signals.head())
print(df_signals['is_win'].value_counts())
print("Total signals:", len(df_signals))
