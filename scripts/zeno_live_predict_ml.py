import os
import pandas as pd
import joblib

# === Configuration: Directories, Model Path, and Features ===
LIVE_DATA_ROOT = os.path.join("data", "live", "XAUUSD")
MODEL_PATH = r"C:\Users\open\Documents\ZENO_XAUUSD\outputs\ml_data\zeno_lgbm.pkl"  # Path to your trained LightGBM model
TIMEFRAMES = ["M5", "M15", "H1", "H4", "D1"]

# === ML Feature Columns REQUIRED by the model (no other columns needed here) ===
ML_FEATURES = ['close', 'score', 'num_confs', 'pattern_code', 'bias_bull', 'hour', 'dow']

def predict_for_tf(tf):
    """
    For a given timeframe:
    - Loads the latest engineered feature CSV
    - Ensures all feature columns are present and lowercase
    - Loads trained ML model, generates 'prob_win' prediction for each row
    - Adds 'rl_action' based on 'prob_win' and saves predictions to a new CSV for downstream decision engine
    """
    # Construct input and output file paths
    feature_csv = os.path.join(LIVE_DATA_ROOT, tf, f"XAUUSD_{tf}_LIVE_FEATURES.csv")
    out_csv = os.path.join(LIVE_DATA_ROOT, tf, f"XAUUSD_{tf}_LIVE_PRED.csv")

    print(f"[INFO] {tf}: Loading features from {feature_csv}")
    if not os.path.exists(feature_csv):
        print(f"[SKIP] {tf}: Features CSV not found.")
        return

    # --- Load features, ensure columns lowercase for consistency ---
    df = pd.read_csv(feature_csv, parse_dates=['datetime'])
    df.columns = [c.lower() for c in df.columns]

    # --- Validate datetime column ---
    if 'datetime' not in df.columns:
        print(f"[ERROR] {tf}: 'datetime' column is missing from the features file.")
        return
    else:
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')

    # --- Confirm that all model-required ML columns are present ---
    missing = [col for col in ML_FEATURES if col not in df.columns]
    if missing:
        print(f"[ERROR] {tf}: Feature columns missing: {missing}")
        return

    # --- Load trained model ---
    print(f"[INFO] {tf}: Loading model from {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model file not found: {MODEL_PATH}")
        return

    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        print(f"[ERROR] Failed to load model from {MODEL_PATH}: {e}")
        return

    # --- Predict probability of a "win" (positive class) for each row ---
    try:
        df["prob_win"] = model.predict_proba(df[ML_FEATURES])[:, 1]
    except Exception as e:
        print(f"[ERROR] Failed to generate predictions for {tf}: {e}")
        return

    # --- Check if 'prob_win' column exists and is valid ---
    if 'prob_win' not in df.columns or df['prob_win'].isnull().any():
        print(f"[ERROR] 'prob_win' column missing or contains NaN values in {tf} predictions.")
        return

    # --- Add RL action based on 'prob_win' (if prob_win > 0.5, set RL action to 'BUY', else 'HOLD') ---
    df['rl_action'] = df['prob_win'].apply(lambda x: 'BUY' if x > 0.5 else 'HOLD')

    # --- Check if 'atr' column exists and is valid ---
    if 'atr' not in df.columns or df['atr'].isnull().any():
        print(f"[ERROR] 'atr' column missing or contains NaN values in {tf} predictions.")
        return

    # --- Save output (all features + prob_win + rl_action + datetime) for downstream engine or dashboard ---
    df.to_csv(out_csv, index=False)
    print(f"[SUCCESS] {tf}: Predictions saved to {out_csv}")
    
    # Print a quick preview for debugging
    print(df[["datetime", "prob_win", "rl_action", "atr"]].tail(3))

def main():
    """
    For each defined timeframe, load features, run prediction, and output results.
    Any errors, missing files, or missing columns are logged and that timeframe is skipped.
    """
    for tf in TIMEFRAMES:
        predict_for_tf(tf)

if __name__ == "__main__":
    main()

#roll back