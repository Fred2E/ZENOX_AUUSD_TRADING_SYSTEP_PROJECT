import os
import pandas as pd
import joblib

LIVE_DATA_ROOT = r"C:\Users\open\Documents\ZENO_XAUUSD\data\live\XAUUSD"
MODEL_PATH = r"C:\Users\open\Documents\ZENO_XAUUSD\outputs\ml_data\zeno_lgbm.pkl"
TIMEFRAMES = ["M5", "M15", "H1", "H4", "D1"]
ML_FEATURES = ['close', 'score', 'num_confs', 'pattern_code', 'bias_bull', 'hour', 'dow']
PRED_COLS = ['datetime', 'prob_win', 'rl_action', 'atr']

def calculate_atr(df, tf):
    if all(c in df.columns for c in ['high', 'low', 'close']):
        highs = df['high'].astype(float)
        lows = df['low'].astype(float)
        closes = df['close'].astype(float)
        prev_closes = closes.shift(1)
        tr = pd.concat([
            highs - lows,
            (highs - prev_closes).abs(),
            (lows - prev_closes).abs()
        ], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=14, min_periods=1).mean()
        print(f"[INFO] {tf}: ATR calculated.")
    else:
        print(f"[WARNING] {tf}: Missing columns for ATR calculation.")
        df['atr'] = None
    return df

def predict_for_tf(tf):
    feature_csv = os.path.join(LIVE_DATA_ROOT, tf, f"XAUUSD_{tf}_LIVE_FEATURES.csv")
    pred_csv = os.path.join(LIVE_DATA_ROOT, tf, f"XAUUSD_{tf}_LIVE_PRED.csv")
    merged_csv = os.path.join(LIVE_DATA_ROOT, tf, f"XAUUSD_{tf}_LIVE_PRED_MERGED.csv")
    print(f"\n[INFO] {tf}: Loading features from {feature_csv}")

    if not os.path.exists(feature_csv):
        print(f"[ERROR] {tf}: Features CSV not found.")
        return

    df = pd.read_csv(feature_csv, parse_dates=['datetime'])
    df.columns = [c.lower().strip() for c in df.columns]
    if 'datetime' not in df.columns:
        print(f"[ERROR] {tf}: 'datetime' column is missing.")
        return
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    missing = [col for col in ML_FEATURES if col not in df.columns]
    if missing:
        print(f"[ERROR] {tf}: Feature columns missing: {missing}")
        return
    df = calculate_atr(df, tf)

    if 'atr' not in df.columns or df['atr'].isnull().all():
        print(f"[ERROR] {tf}: ATR calculation failed (all NaN or missing)!")
    elif df['atr'].notnull().sum() < len(df) // 2:
        print(f"[WARNING] {tf}: Less than half ATR values calculated.")
    else:
        print(f"[INFO] {tf}: ATR looks healthy. Last 5:\n{df[['datetime','atr']].tail(5)}")

    print(f"[INFO] {tf}: Loading model from {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model file not found: {MODEL_PATH}")
        return
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return

    try:
        df["prob_win"] = model.predict_proba(df[ML_FEATURES])[:, 1]
    except Exception as e:
        print(f"[ERROR] Prediction failed for {tf}: {e}")
        return
    if 'prob_win' not in df.columns or df['prob_win'].isnull().all():
        print(f"[ERROR] {tf}: 'prob_win' missing or all NaN!")
        return
    df['rl_action'] = df['prob_win'].apply(lambda x: 'BUY' if x > 0.5 else 'HOLD')

    # Save just the prediction table for raw consumption
    out_df = df[PRED_COLS]
    out_df.to_csv(pred_csv, index=False)
    print(f"[SUCCESS] {tf}: Raw predictions saved to {pred_csv}")
    print(out_df.tail(3))

    # Now merge with the original features for full downstream compatibility
    features_for_merge = pd.read_csv(feature_csv, parse_dates=['datetime'])
    features_for_merge.columns = [c.lower().strip() for c in features_for_merge.columns]
    merged = pd.merge(features_for_merge, out_df, on='datetime', how='left')
    merged.to_csv(merged_csv, index=False)
    print(f"[SUCCESS] {tf}: Merged predictions saved to {merged_csv}")
    print(merged.tail(3))

def main():
    for tf in TIMEFRAMES:
        predict_for_tf(tf)

if __name__ == "__main__":
    main()
