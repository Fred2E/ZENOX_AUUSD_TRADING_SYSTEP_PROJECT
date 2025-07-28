# fill_all_price_gaps.py

import pandas as pd
import numpy as np
import os

BASE_DIR = r"C:\Users\open\Documents\ZENO_XAUUSD\data\live\XAUUSD"
TF_INFO = {
    "M5":  {"interval": "5min"},
    "M15": {"interval": "15min"},
    "H1":  {"interval": "1h"},
    "H4":  {"interval": "4h"},
    "D1":  {"interval": "1d"},
}

def fill_gaps(input_csv, output_csv, interval):
    print(f"\nGap-filling: {input_csv} -> {output_csv}")
    if not os.path.exists(input_csv):
        print(f"ERROR: File not found: {input_csv}")
        return

    df = pd.read_csv(input_csv)
    # --- Robust time column handling ---
    cols = [c.lower() for c in df.columns]
    df.columns = cols
    if 'datetime' not in df.columns:
        print(f"ERROR: No datetime column in {input_csv}. Skipping.")
        return
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').set_index('datetime')

    # Detect expected full range
    full_index = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq=interval
    )

    before = len(df)
    df = df.reindex(full_index)

    # Force all OHLCV columns to exist and fill (robust to case issues)
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col not in df.columns:
            print(f"ERROR: {col} column missing in {input_csv}, skipping this TF.")
            return
    df[['open','high','low','close','volume']] = df[['open','high','low','close','volume']].ffill()

    # Restore Datetime as column (title-case for pipeline)
    df = df.reset_index().rename(columns={'index': 'Datetime', 'datetime': 'Datetime'})

    # Null check after fill
    n_nulls = df[['open','high','low','close','volume']].isnull().sum().sum()
    if n_nulls > 0:
        print(f"WARNING: {n_nulls} nulls remain after fill in {output_csv}. Check your data!")

    df.to_csv(output_csv, index=False)
    print(f"Done. Rows before: {before}, after: {len(df)}, filled: {len(df)-before}, nulls remaining: {n_nulls}")

if __name__ == "__main__":
    for tf, tfcfg in TF_INFO.items():
        input_csv = os.path.join(BASE_DIR, tf, f"XAUUSD_{tf}_LIVE.csv")
        output_csv = os.path.join(BASE_DIR, tf, f"XAUUSD_{tf}_LIVE_CLEAN.csv")
        fill_gaps(input_csv, output_csv, tfcfg['interval'])

    print("\n=== All timeframes gap-filled. Manually inspect output for any warnings. ===")
