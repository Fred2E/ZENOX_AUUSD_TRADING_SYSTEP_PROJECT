# zeno_live_feature_pipeline.py

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from modules.structure_detector import detect_structure
from modules.confluence_scanner import evaluate_confluence
from modules.candle_patterns import detect_candle_patterns

LIVE_DATA_ROOT = os.path.join("data", "live", "XAUUSD")
TIMEFRAMES = ["M5", "M15", "H1", "H4", "D1"]

ML_FEATURES = [
    'close', 'score', 'num_confs', 'pattern_code', 'bias_bull', 'hour', 'dow', 'atr'
]

def enrich_features(df, tf):
    # Force lowercase on entry
    df.columns = [c.lower() for c in df.columns]

    # Robust time col detection
    time_cols = [c for c in df.columns if c in ['datetime', 'time', 'timestamp']]
    if not time_cols:
        raise ValueError("No datetime/time/timestamp column found in data!")
    time_col = time_cols[0]

    # Guarantee 'datetime' is preserved through modules
    datetime_series = df[time_col].copy()

    # Feature engineering with error handling and column normalization after every step
    try:
        df = detect_structure(df)
    except Exception as e:
        print(f"[ERROR] detect_structure failed for {tf}: {e}")
        raise
    df.columns = [c.lower() for c in df.columns]
    if 'datetime' not in df.columns:
        df['datetime'] = datetime_series

    try:
        df = detect_candle_patterns(df)
    except Exception as e:
        print(f"[ERROR] detect_candle_patterns failed for {tf}: {e}")
        raise
    df.columns = [c.lower() for c in df.columns]
    if 'datetime' not in df.columns:
        df['datetime'] = datetime_series

    try:
        df = evaluate_confluence(df, timeframe=tf)
    except Exception as e:
        print(f"[ERROR] evaluate_confluence failed for {tf}: {e}")
        raise
    df.columns = [c.lower() for c in df.columns]
    if 'datetime' not in df.columns:
        df['datetime'] = datetime_series

    # Derived features
    df['num_confs'] = df['confluences'].apply(lambda x: len(x) if isinstance(x, list) else 0) if 'confluences' in df.columns else 0
    df['pattern_code'] = df['candle_pattern'].astype('category').cat.codes if 'candle_pattern' in df.columns else -1
    df['bias_bull'] = (df['bias'].astype(str).str.lower() == 'bullish').astype(int) if 'bias' in df.columns else 0
    df['hour'] = pd.to_datetime(df[time_col]).dt.hour
    df['dow'] = pd.to_datetime(df[time_col]).dt.dayofweek

    # --- Guarantee ATR exists and is correct ---
    if 'atr' not in df.columns:
        high = df['high']
        low = df['low']
        close = df['close']
        prev_close = close.shift(1)
        tr = pd.concat([
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=14, min_periods=1).mean()

    # Guarantee all ML features exist
    for feat in ML_FEATURES:
        if feat not in df.columns:
            print(f"[WARN] Feature '{feat}' missing in {tf}, filling with 0.")
            df[feat] = 0

    # FINAL: Always keep 'datetime' as canonical lowercase for output
    df['datetime'] = datetime_series

    # === PATCH: Output 'Close' (uppercase) for ML model compatibility ===
    if 'close' in df.columns and 'Close' not in df.columns:
        df['Close'] = df['close']

    # Output: ALL required ML features, with 'Close' (uppercase) first, then rest, then 'datetime'
    out_cols = ['Close', 'score', 'num_confs', 'pattern_code', 'bias_bull', 'hour', 'dow', 'datetime']
    # Add 'close', 'atr' for future-proofing/data analysis
    for feat in ['close', 'atr']:
        if feat not in out_cols and feat in df.columns:
            out_cols.insert(1, feat)  # after 'Close'
    # Add missing required output columns as 0
    missing_out = [f for f in out_cols if f not in df.columns]
    for f in missing_out:
        print(f"[FATAL] Output feature '{f}' missing in {tf}—pipeline is still broken.")
        df[f] = 0

    df_ml = df[out_cols]
    return df_ml

def main():
    for tf in TIMEFRAMES:
        csv_file = f"XAUUSD_{tf}_LIVE.csv"
        path = os.path.join(LIVE_DATA_ROOT, tf, csv_file)
        if not os.path.exists(path):
            print(f"[SKIP] {tf}: File not found: {path}")
            continue
        print(f"[PROCESS] {tf}...")
        df = pd.read_csv(path)
        df_ml = enrich_features(df, tf)
        print(f"[INFO] {tf} final feature columns: {list(df_ml.columns)}")
        print(df_ml.tail(3))
        enriched_path = path.replace("_LIVE.csv", "_LIVE_FEATURES.csv")
        df_ml.to_csv(enriched_path, index=False)
        print(f"[SUCCESS] {tf} features saved to {enriched_path}")

if __name__ == "__main__":
    main()
