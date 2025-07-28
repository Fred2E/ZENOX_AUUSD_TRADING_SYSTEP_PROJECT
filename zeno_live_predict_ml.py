# zeno_live_predict_ml.py

import os
import pandas as pd
import joblib

LIVE_DATA_ROOT = os.path.join("data", "live", "XAUUSD")
MODEL_PATH  = os.path.join("outputs", "ml_data", "zeno_lgbm.pkl")
TIMEFRAMES = ["M5", "M15", "H1", "H4", "D1"]

ML_FEATURES = ['close', 'score', 'num_confs', 'pattern_code', 'bias_bull', 'hour', 'dow']

def predict_for_tf(tf):
    feature_csv = os.path.join(LIVE_DATA_ROOT, tf, f"XAUUSD_{tf}_LIVE_FEATURES.csv")
    out_csv     = os.path.join(LIVE_DATA_ROOT, tf, f"XAUUSD_{tf}_LIVE_PRED.csv")

    print(f"[INFO] {tf}: Loading features from {feature_csv}")
    if not os.path.exists(feature_csv):
        print(f"[SKIP] {tf}: Features CSV not found.")
        return

    df = pd.read_csv(feature_csv, parse_dates=['datetime'])
    df.columns = [c.lower() for c in df.columns]

    print(f"[INFO] {tf}: Loading model from {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model file not found: {MODEL_PATH}")
        return
    model = joblib.load(MODEL_PATH)

    # Confirm features exist
    missing = [col for col in ML_FEATURES if col not in df.columns]
    if missing:
        print(f"[ERROR] {tf}: Feature columns missing: {missing}")
        return

    # Predict probability of win
    df["prob_win"] = model.predict_proba(df[ML_FEATURES])[:, 1]

    # Save output
    df.to_csv(out_csv, index=False)
    print(f"[SUCCESS] {tf}: Predictions saved to {out_csv}")
    print(df[["datetime", "prob_win"]].tail(3))

def main():
    for tf in TIMEFRAMES:
        predict_for_tf(tf)

if __name__ == "__main__":
    main()
