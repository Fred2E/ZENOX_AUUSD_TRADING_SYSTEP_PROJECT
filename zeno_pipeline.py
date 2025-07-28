# zeno_pipeline.py
import os
import pandas as pd
from datetime import datetime, timedelta, timezone
import logging

# ========== CONFIGURATION ==========
DATA_ROOT = os.path.join("data", "live", "XAUUSD")
TIMEFRAMES = ["M5", "M15", "H1", "H4", "D1"]
CSV_TEMPLATE = "XAUUSD_{}_LIVE.csv"
MAX_GAP_MINUTES = {"M5": 5, "M15": 15, "H1": 60, "H4": 240, "D1": 1440}
TF_TO_RUN = "M5"  # Change this as needed

# ========== LOGGING ==========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()]
)

def load_and_validate(tf):
    path = os.path.join(DATA_ROOT, tf, CSV_TEMPLATE.format(tf))
    if not os.path.exists(path):
        logging.error(f"[{tf}] File not found: {path}")
        return None
    df = pd.read_csv(path, parse_dates=["datetime"])
    df.columns = [c.lower() for c in df.columns]
    if df.empty:
        logging.error(f"[{tf}] CSV is empty.")
        return None
    df = df.sort_values("datetime")
    # --- Check for continuity ---
    expected_delta = MAX_GAP_MINUTES[tf]
    dt_col = df['datetime']
    gaps = dt_col.diff().dt.total_seconds().div(60).fillna(expected_delta).values
    for i, gap in enumerate(gaps[1:], start=1):
        if gap > expected_delta * 1.5:
            logging.error(f"[{tf}] Data gap between {dt_col.iloc[i-1]} and {dt_col.iloc[i]} ({gap:.1f}min)")
            return None
    # --- Check for staleness ---
    now = datetime.utcnow().replace(tzinfo=timezone.utc)
    last_bar = dt_col.iloc[-1]
    mins_old = (now - last_bar).total_seconds() / 60
    if mins_old > expected_delta * 2:
        logging.error(f"[{tf}] Last bar too old! Last: {last_bar} (age: {mins_old:.1f}min)")
        return None
    logging.info(f"[{tf}] Data OK ({len(df)} bars), last: {last_bar}")
    return df

def main():
    logging.info("=== ZENO PIPELINE START ===")
    # 1. Load and validate
    df = load_and_validate(TF_TO_RUN)
    if df is None:
        logging.error(f"[{TF_TO_RUN}] Data validation failed. STOP.")
        return
    # 2. Feature engineering
    logging.info(f"[{TF_TO_RUN}] Running feature engineering...")
    from modules.feature_pipeline import engineer_features
    df, features = engineer_features(df)
    logging.info(f"[{TF_TO_RUN}] Features: {features}")
    # 3. ML prediction
    logging.info(f"[{TF_TO_RUN}] Running ML prediction...")
    from modules.ml_model import predict_ml
    df = predict_ml(df)
    # 4. Show sample results
    logging.info(f"[{TF_TO_RUN}] Prediction sample:\n{df[['datetime','close','score','prob_win']].tail(5)}")
    logging.info("=== ZENO PIPELINE FINISHED ===")

if __name__ == "__main__":
    main()
