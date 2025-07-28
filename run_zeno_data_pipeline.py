import os
import pandas as pd
import numpy as np

DATA_PATHS = [
    r"data\live\XAUUSD\M5\XAUUSD_M5_LIVE.csv",
    r"data\live\XAUUSD\M5\XAUUSD_M5_LIVE_CLEAN.csv",
    r"data\live\XAUUSD\M15\XAUUSD_M15_LIVE.csv",
    r"data\live\XAUUSD\H1\XAUUSD_H1_LIVE.csv",
    r"data\live\XAUUSD\H4\XAUUSD_H4_LIVE.csv",
    r"data\live\XAUUSD\D1\XAUUSD_D1_LIVE.csv",
]

def zeno_clean_features(df):
    # Make all columns lower-case for safety
    df.columns = [c.lower() for c in df.columns]
    # Remove duplicates
    df = df.drop_duplicates(subset=['datetime'], keep='last')
    # Sort
    df = df.sort_values('datetime')
    # Set datetime index
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True, errors='coerce')
    df = df.set_index('datetime')
    # Drop missing bars
    df = df.dropna(subset=['open','high','low','close'])
    # Forward-fill volume if zero
    if 'volume' in df.columns:
        df['volume'] = df['volume'].replace(0, np.nan).ffill()
    # Add engineered features
    df['returns'] = df['close'].pct_change()
    df['logret'] = np.log(df['close'] / df['close'].shift(1))
    df['volatility'] = df['returns'].rolling(window=12, min_periods=1).std()
    df['ma_20'] = df['close'].rolling(window=20, min_periods=1).mean()
    df['hour'] = df.index.hour
    df['dow'] = df.index.dayofweek
    return df

def process_file(fpath):
    if not os.path.exists(fpath):
        print(f"[ERROR] Not found: {fpath}")
        return
    print(f"[INFO] Processing {fpath}")
    df = pd.read_csv(fpath)
    clean = zeno_clean_features(df)
    outpath = fpath.replace(".csv", "_CLEAN.csv")
    clean.to_csv(outpath)
    print(f"[SUCCESS] Saved cleaned file: {outpath}")
    print(f"         Rows: {len(clean)} | Columns: {list(clean.columns)}")

if __name__ == "__main__":
    os.chdir(r"C:\Users\open\Documents\ZENO_XAUUSD")
    for p in DATA_PATHS:
        process_file(p)
