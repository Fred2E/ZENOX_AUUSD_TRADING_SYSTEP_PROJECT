import pandas as pd
import numpy as np
import os
import sys

# Add modules to sys.path for import
modules_path = r"C:\Users\open\Documents\ZENO_XAUUSD\modules"
if modules_path not in sys.path:
    sys.path.insert(0, modules_path)

from confluence_scanner import evaluate_confluence
from zeno_config import CANDLE_THRESH, SWING_PARAMS, ENTRY_FILTERS

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

    # 2. Compute all confluence features (pass tf to ensure adaptive config)
    df_features = evaluate_confluence(df, timeframe=tf)

    # 3. Save as new features file (lowercase cols only)
    df_features.columns = [c.lower() for c in df_features.columns]
    df_features.to_csv(features_path, index=False)
    print(f"[{tf}] Feature engineering complete. Output: {features_path}")

    # 4. Audit: Ensure ATR and all primary/secondary confluences are present
    must_have = ['atr'] + ENTRY_FILTERS['min_primary_confs'].keys()
    for col in ['atr', 'conf_structure', 'conf_bos_or_choch', 'conf_candle', 'conf_sr_zone']:
        if col not in df_features.columns:
            raise ValueError(f"[{tf}] ERROR: {col} column missing from features file!")

    # 5. Log some confluence counts for diagnostics
    print(f"[{tf}] PRIMARY CONFLUENCE COUNTS:")
    for col in ['conf_structure', 'conf_bos_or_choch', 'conf_candle', 'conf_sr_zone']:
        print(f"  {col}: {df_features[col].sum()}")

print("ALL TIMEFRAMES: Feature pipeline complete, tf-adaptive, and ATR-verified.")
