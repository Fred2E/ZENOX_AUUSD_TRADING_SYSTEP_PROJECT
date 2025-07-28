# zeno_pipeline_validator.py

import os
import pandas as pd
from datetime import datetime, timedelta, timezone
import logging

# --- CONFIG ---
DATA_ROOT = os.path.join("data", "live", "XAUUSD")
TIMEFRAMES = ["M5", "M15", "H1", "H4", "D1"]
CSV_TEMPLATE = "XAUUSD_{}_LIVE.csv"
MAX_GAP_MINUTES = {"M5": 5, "M15": 15, "H1": 60, "H4": 240, "D1": 1440}
REQUIRED_COLS = ["datetime", "open", "high", "low", "close", "volume"]

# --- LOGGER ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()]
)

def load_and_validate(tf):
    path = os.path.join(DATA_ROOT, tf, CSV_TEMPLATE.format(tf))
    if not os.path.exists(path):
        logging.error(f"[{tf}] Missing file: {path}")
        return None
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    if not all(col in df.columns for col in REQUIRED_COLS):
        logging.error(f"[{tf}] Missing columns. Required: {REQUIRED_COLS}, Found: {df.columns.tolist()}")
        return None
    # Parse datetime
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    df = df.sort_values('datetime')
    # --- Continuity Check ---
    gaps = []
    for i in range(1, len(df)):
        delta = (df.iloc[i]['datetime'] - df.iloc[i-1]['datetime']).total_seconds() / 60
        if delta > MAX_GAP_MINUTES[tf] * 1.5:
            gaps.append((df.iloc[i-1]['datetime'], df.iloc[i]['datetime'], delta))
    if gaps:
        logging.error(f"[{tf}] GAP detected! {gaps}")
        return None
    # --- Staleness Check ---
    now = datetime.utcnow().replace(tzinfo=timezone.utc)
    last = df['datetime'].iloc[-1]
    min_since_last = (now - last).total_seconds() / 60
    if min_since_last > MAX_GAP_MINUTES[tf] * 2:
        logging.error(f"[{tf}] LAST BAR TOO OLD: {last} (now: {now})")
        return None
    logging.info(f"[{tf}] Data OK. Rows: {len(df)}, Last: {last}")
    return df

def main():
    logging.info("=== ZENO PIPELINE DATA VALIDATOR ===")
    dfs = {}
    for tf in TIMEFRAMES:
        df = load_and_validate(tf)
        if df is None:
            logging.critical(f"PIPELINE ABORTED: [{tf}] data not valid. FIX BEFORE PROCEEDING.")
            return
        dfs[tf] = df

    # --- Pipeline entry: ONLY if ALL DATA IS VALID ---
    from modules.feature_pipeline import engineer_features
    from modules.ml_model import predict_ml

    tf = "M5"
    df = dfs[tf]
    logging.info(f"[{tf}] Running feature engineering + ML prediction...")
    df, features = engineer_features(df)
    df = predict_ml(df)
    print(df.tail(5))

    logging.info("=== PIPELINE VALIDATION + EXECUTION SUCCESS ===")

if __name__ == "__main__":
    main()
