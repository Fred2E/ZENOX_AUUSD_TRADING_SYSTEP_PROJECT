import pandas as pd
import numpy as np
import os
import sys

modules_path = r"C:\Users\open\Documents\ZENO_XAUUSD\modules"
if modules_path not in sys.path:
    sys.path.insert(0, modules_path)

from confluence_scanner import evaluate_confluence
from zeno_config import CANDLE_THRESH, SWING_PARAMS, ENTRY_FILTERS

BASE_DIR = r"C:\Users\open\Documents\ZENO_XAUUSD\data\live\XAUUSD"
TIMEFRAMES = ['M5', 'M15', 'H1', 'H4', 'D1']

MUST_HAVE = [
    'datetime', 'open', 'high', 'low', 'close', 'volume', 'atr',
    'conf_structure', 'conf_bos_or_choch', 'conf_candle', 'conf_sr_zone',
    'conf_psych_level', 'conf_fib_zone', 'conf_volume', 'conf_liquidity', 'conf_spread'
]

for tf in TIMEFRAMES:
    input_path = os.path.join(BASE_DIR, tf, f"XAUUSD_{tf}_LIVE_with_patterns.csv")
    features_path = os.path.join(BASE_DIR, tf, f"XAUUSD_{tf}_LIVE_FEATURES.csv")
    if not os.path.exists(input_path):
        print(f"[{tf}] MISSING INPUT FILE: {input_path}")
        continue

    # 1. Load data
    df = pd.read_csv(input_path)
    df.columns = [c.lower() for c in df.columns]

    # Defensive: Warn if windowed to small size
    if len(df) < 5000 and tf in ['M5', 'M15', 'H1', 'H4']:
        print(f"[{tf}] WARNING: Only {len(df)} rows loaded! Likely capped. Check upstream pipeline!")

    # 2. Compute confluence features
    df_features = evaluate_confluence(df, timeframe=tf)

    # 3. Save features
    df_features.columns = [c.lower() for c in df_features.columns]
    df_features.to_csv(features_path, index=False)
    print(f"[{tf}] Feature engineering complete. Output: {features_path}")

    # 4. Audit features
    missing = [col for col in MUST_HAVE if col not in df_features.columns]
    if missing:
        raise ValueError(f"[{tf}] ERROR: Missing columns in features file: {missing}")

    print(f"[{tf}] PRIMARY CONFLUENCE COUNTS:")
    for col in ['conf_structure', 'conf_bos_or_choch', 'conf_candle', 'conf_sr_zone']:
        print(f"  {col}: {df_features[col].sum()}")
    print(f"[{tf}] SECONDARY CONFS VALUE COUNTS:")
    for col in ['conf_psych_level', 'conf_fib_zone', 'conf_volume', 'conf_liquidity', 'conf_spread']:
        print(f"  {col}: {df_features[col].sum()}")

print("ALL TIMEFRAMES: Feature pipeline complete, tf-adaptive, and ATR-verified.")
