import pandas as pd
import numpy as np
import os

# Correct paths and filenames (historical/raw not raw/)
RAW_DIR = r"C:\Users\open\Documents\ZENO_XAUUSD\historical\raw"
OUT_DIR = r"C:\Users\open\Documents\ZENO_XAUUSD\outputs\ml_data"
os.makedirs(OUT_DIR, exist_ok=True)

# Map: timeframe → best file for each TF
RAW_FILES = {
    "M15": "XAUUSD_M15_A_confluence_debug_fibpatch.csv",
    "H1":  "XAUUSD_H1_A_confluence_debug_fibpatch.csv",
    "H4":  "XAUUSD_H4_B_confluence_debug_fibpatch.csv"
}

def clean_data(df):
    # Basic cleaning: drop duplicates/nulls, check datatypes
    df = df.drop_duplicates().dropna(subset=['datetime', 'close', 'high', 'low', 'atr'])
    df['datetime'] = pd.to_datetime(df['datetime'])
    for col in ['close','high','low','atr','open','volume','spread']:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def add_features(df):
    # ML-selectable features only; add more as needed!
    df['returns'] = df['close'].pct_change()
    df['range'] = df['high'] - df['low']
    if 'hour' not in df.columns:
        df['hour'] = df['datetime'].dt.hour
    if 'dow' not in df.columns:
        df['dow'] = df['datetime'].dt.dayofweek
    df['atr_rolling'] = df['atr'].rolling(14).mean()
    # Add more context/volatility/session features here if wanted
    return df

for tf, fname in RAW_FILES.items():
    infile = os.path.join(RAW_DIR, fname)
    outfile = os.path.join(OUT_DIR, f"trade_events_{tf}_FULL.csv")
    if not os.path.exists(infile):
        print(f"[SKIP] {infile} not found.")
        continue
    df = pd.read_csv(infile)
    df = clean_data(df)
    df = add_features(df)
    df.to_csv(outfile, index=False)
    print(f"[OK] Saved clean features: {outfile}")
