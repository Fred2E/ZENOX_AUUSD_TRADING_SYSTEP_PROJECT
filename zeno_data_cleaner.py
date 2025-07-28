# zeno_data_cleaner.py

import os
import pandas as pd
from datetime import timedelta
import logging

M5_PATH = os.path.join("data", "live", "XAUUSD", "M5", "XAUUSD_M5_LIVE.csv")
OUT_PATH = os.path.join("data", "live", "XAUUSD", "M5", "XAUUSD_M5_LIVE_CLEAN.csv")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()]
)

def clean_m5():
    if not os.path.exists(M5_PATH):
        logging.error(f"File not found: {M5_PATH}")
        return

    df = pd.read_csv(M5_PATH, parse_dates=['datetime'])
    df.columns = [c.lower() for c in df.columns]
    df = df.sort_values('datetime')
    # Use GMT: force to tz-aware if not already
    if df['datetime'].dt.tz is None:
        df['datetime'] = df['datetime'].dt.tz_localize('Etc/GMT')
    else:
        df['datetime'] = df['datetime'].dt.tz_convert('Etc/GMT')

    # Build full M5 range in GMT
    idx = pd.date_range(df['datetime'].iloc[0], df['datetime'].iloc[-1], freq="5min", tz='Etc/GMT')
    df = df.set_index('datetime').reindex(idx)
    missing = df[df['close'].isna()]
    if not missing.empty:
        logging.warning(f"MISSING BARS DETECTED: {len(missing)} bars will be filled as NaN.")
        print(missing.index)
    else:
        logging.info("No missing bars detected.")

    # Save cleaned file
    df.reset_index().rename(columns={'index':'datetime'}).to_csv(OUT_PATH, index=False)
    logging.info(f"Saved cleaned file: {OUT_PATH}")

if __name__ == "__main__":
    clean_m5()
