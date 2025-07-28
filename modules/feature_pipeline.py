import pandas as pd
import numpy as np
import os
import sys

# Add modules to sys.path for import
modules_path = r"C:\Users\open\Documents\ZENO_XAUUSD\modules"
if modules_path not in sys.path:
    sys.path.insert(0, modules_path)

from confluence_scanner import evaluate_confluence  # MUST include ATR logic!

BASE_DIR = r"C:\Users\open\Documents\ZENO_XAUUSD\data\live\XAUUSD"
TIMEFRAMES = ['M5', 'M15', 'H1', 'H4', 'D1']

for tf in TIMEFRAMES:
    input_path = os.path.join(BASE_DIR, tf, f"XAUUSD_{tf}_LIVE_with_patterns.csv")
    features_path = os.path.join(BASE_DIR, tf, f"XAUUSD_{tf}_LIVE_FEATURES.csv")
    if not os.path.exists(input_path):
        print(f"[{tf}] MISSING INPUT FILE: {input_path}")
        continue

    # 1. Load structured+patterned data (structure + candle patterns)
    df = pd.read_csv(input_path)
    df.columns = [c.lower() for c in df.columns]

    # 2. Compute all confluence features (MUST include ATR!)
    df_features = evaluate_confluence(df, tf)

    # 3. Save as new features file (lowercase cols only)
    df_features.columns = [c.lower() for c in df_features.columns]
    df_features.to_csv(features_path, index=False)
    print(f"[{tf}] Feature engineering complete. Output: {features_path}")

    # 4. Quick audit: Ensure ATR is present
    if 'atr' not in df_features.columns:
        raise ValueError(f"[{tf}] ERROR: ATR column missing from features file!")

print("ALL TIMEFRAMES: Feature pipeline complete and ATR-verified.")
