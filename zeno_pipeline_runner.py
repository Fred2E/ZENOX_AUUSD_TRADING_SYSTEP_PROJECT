# zeno_pipeline_runner.py

import os
import pandas as pd
from datetime import datetime, timedelta, timezone
import logging

DATA_ROOT = os.path.join("data", "live", "XAUUSD")
TIMEFRAMES = ["M5", "M15", "H1", "H4", "D1"]
CSV_TEMPLATE = "XAUUSD_{}_LIVE.csv"
OUTPUT_TEMPLATE = "ZENO_{}_ML.csv"
MAX_GAP_MINUTES = {"M5": 5, "M15": 15, "H1": 60, "H4": 240, "D1": 1440}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()]
)

def load_and_check(tf):
    path = os.path.join(DATA_ROOT, tf, CSV_TEMPLATE.format(tf))
    if not os.path.exists(path):
        logging.error(f"[{tf}] Missing file: {path}")
        return None
    df = pd.read_csv(path, parse_dates=['Datetime'])
    df = df.sort_values('Datetime')
    # Check for continuity
    gaps = []
    for i in range(1, len(df)):
        diff = (df.iloc[i]['Datetime'] - df.iloc[i-1]['Datetime']).total_seconds() / 60
        if diff > MAX_GAP_MINUTES[tf] * 1.5:
            gaps.append((df.iloc[i-1]['Datetime'], df.iloc[i]['Datetime'], diff))
    if gaps:
        logging.warning(f"[{tf}] Gaps detected: {gaps}")
        return None
    # Check for staleness
    now = datetime.utcnow().replace(tzinfo=timezone.utc)
    last = df['Datetime'].iloc[-1]
    minutes_since_last = (now - last).total_seconds() / 60
    if minutes_since_last > MAX_GAP_MINUTES[tf] * 2:
        logging.warning(f"[{tf}] Last bar too old: {last} (now: {now})")
        return None
    logging.info(f"[{tf}] Data OK: {len(df)} rows, last: {last}")
    return df

def main():
    logging.info("=== ZENO PIPELINE RUNNER: ALL TIMEFRAMES ===")
    from modules.feature_pipeline import engineer_features
    from modules.ml_model import predict_ml

    results = {}
    for tf in TIMEFRAMES:
        df = load_and_check(tf)
        if df is None:
            logging.error(f"[{tf}] Skipping ML pipeline due to bad data.")
            continue
        try:
            df, features = engineer_features(df)
            df = predict_ml(df)
            results[tf] = df
            # Save output for further inspection/use
            out_path = os.path.join(DATA_ROOT, tf, OUTPUT_TEMPLATE.format(tf))
            df.to_csv(out_path, index=False)
            logging.info(f"[{tf}] ML pipeline complete. Output saved: {out_path}")
            print(f"\n========== {tf} TAIL ==========")
            print(df.tail(5))
        except Exception as e:
            logging.error(f"[{tf}] ML pipeline error: {e}")

    logging.info("=== ZENO MULTI-TIMEFRAME PIPELINE COMPLETE ===")

if __name__ == "__main__":
    main()
