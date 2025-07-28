import os
import pandas as pd
from datetime import datetime

# Config
DATA_ROOT = os.path.join("data", "live", "XAUUSD")
TIMEFRAMES = ["M5", "M15", "H1", "H4", "D1"]
CSV_TEMPLATE = "XAUUSD_{}_LIVE.csv"
FREQ_MAP = {"M5": "5min", "M15": "15min", "H1": "1H", "H4": "4H", "D1": "1D"}

def print_gaps(tf):
    path = os.path.join(DATA_ROOT, tf, CSV_TEMPLATE.format(tf))
    if not os.path.exists(path):
        print(f"[{tf}] MISSING: {path}")
        return
    df = pd.read_csv(path, parse_dates=['Datetime'])
    df = df.sort_values('Datetime')
    # Set Datetime as index for reindexing
    df = df.set_index('Datetime')
    expected_idx = pd.date_range(df.index[0], df.index[-1], freq=FREQ_MAP[tf])
    missing = expected_idx.difference(df.index)
    if not missing.empty:
        print(f"[{tf}] {len(missing)} gap(s):")
        print(missing)
    else:
        print(f"[{tf}] ✅ No gaps. Data is continuous.")

def main():
    for tf in TIMEFRAMES:
        print_gaps(tf)

if __name__ == "__main__":
    main()
