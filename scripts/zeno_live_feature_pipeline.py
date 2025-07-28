# zeno_live_feature_pipeline.py

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from modules.structure_detector import detect_structure
from modules.confluence_scanner import evaluate_confluence
from modules.candle_patterns import detect_candle_patterns

# --- Directory and file configuration ---
LIVE_DATA_ROOT = os.path.join("data", "live", "XAUUSD")
TIMEFRAMES = ["M5", "M15", "H1", "H4", "D1"]

# --- ML Features required by downstream model ---
ML_FEATURES = [
    'close', 'score', 'num_confs', 'pattern_code', 'bias_bull', 'hour', 'dow', 'atr'
]

# --- Confluence column names (must match those created in evaluate_confluence) ---
PRIMARY_CONFS = ['conf_structure', 'conf_bos_or_choch', 'conf_candle', 'conf_sr_zone']
SECONDARY_CONFS = ['conf_psych_level', 'conf_fib_zone', 'bias_bull']
ALL_CONFS = PRIMARY_CONFS + SECONDARY_CONFS

def enrich_features(df, tf):
    """
    Feature engineering pipeline for live XAUUSD candles.
    - Ensures consistent lowercased columns.
    - Derives structure, confluence, pattern, and volatility features.
    - Ensures compatibility for ML/RL model and audit.
    """
    df.columns = [c.lower() for c in df.columns]
    time_cols = [c for c in df.columns if c in ['datetime', 'time', 'timestamp']]
    if not time_cols:
        raise ValueError("No datetime/time/timestamp column found in data!")
    time_col = time_cols[0]
    datetime_series = df[time_col].copy()

    # === Structure detection ===
    try:
        df = detect_structure(df)
    except Exception as e:
        print(f"[ERROR] detect_structure failed for {tf}: {e}")
        raise
    df.columns = [c.lower() for c in df.columns]
    if 'datetime' not in df.columns:
        df['datetime'] = datetime_series

    # === Candle patterns ===
    try:
        df = detect_candle_patterns(df)
    except Exception as e:
        print(f"[ERROR] detect_candle_patterns failed for {tf}: {e}")
        raise
    df.columns = [c.lower() for c in df.columns]
    if 'datetime' not in df.columns:
        df['datetime'] = datetime_series

    # === Confluence evaluation ===
    try:
        df = evaluate_confluence(df, timeframe=tf)
    except Exception as e:
        print(f"[ERROR] evaluate_confluence failed for {tf}: {e}")
        raise
    df.columns = [c.lower() for c in df.columns]
    if 'datetime' not in df.columns:
        df['datetime'] = datetime_series

    # --- Ensure all confluence flag columns exist ---
    for conf in ALL_CONFS:
        if conf not in df.columns:
            print(f"[WARN] Confluence flag '{conf}' missing in {tf}, filling with 0.")
            df[conf] = 0

    # --- CORRECT: Sum the binary confluence columns ---
    df['num_confs'] = df[ALL_CONFS].sum(axis=1)

    # --- Pattern encoding ---
    df['pattern_code'] = df['candle_pattern'].astype('category').cat.codes if 'candle_pattern' in df.columns else -1

    # --- Bias as binary for ML ---
    if 'bias' in df.columns:
        if pd.api.types.is_numeric_dtype(df['bias']):
            df['bias_bull'] = (df['bias'] == 1).astype(int)
        else:
            df['bias_bull'] = (df['bias'].astype(str).str.lower() == 'bullish').astype(int)
    else:
        df['bias_bull'] = 0

    # --- Time features ---
    df['hour'] = pd.to_datetime(df[time_col]).dt.hour
    df['dow'] = pd.to_datetime(df[time_col]).dt.dayofweek

    # --- ATR (robust fallback) ---
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

    # --- Fill any missing ML features with 0 ---
    for feat in ML_FEATURES:
        if feat not in df.columns:
            print(f"[WARN] Feature '{feat}' missing in {tf}, filling with 0.")
            df[feat] = 0

    df['datetime'] = datetime_series

    out_cols = ML_FEATURES + ['datetime']
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
