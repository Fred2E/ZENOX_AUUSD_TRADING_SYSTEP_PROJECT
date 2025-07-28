# gap_neutralizer.py
import os
import pandas as pd
import numpy as np
from datetime import timedelta

# --- CONFIG ---
DATA_ROOT = os.path.join("data", "live", "XAUUSD")
TIMEFRAMES = {
    "M5": 5,
    "M15": 15,
    "H1": 60,
    "H4": 240,
    "D1": 1440,
}
CSV_TEMPLATE = "XAUUSD_{}_LIVE.csv"

def get_expected_times(df, tf_minutes):
    # Always use lowercase for datetime
    df.columns = [c.lower() for c in df.columns]
    df = df.sort_values('datetime')
    start = df['datetime'].iloc[0]
    end = df['datetime'].iloc[-1]
    expected_times = pd.date_range(start, end, freq=f"{tf_minutes}min", tz='UTC')
    return expected_times

def neutralize_gaps(tf):
    path = os.path.join(DATA_ROOT, tf, CSV_TEMPLATE.format(tf))
    if not os.path.exists(path):
        print(f"[{tf}] File missing: {path}")
        return
    df = pd.read_csv(path, parse_dates=['datetime'])
    df.columns = [c.lower() for c in df.columns]
    expected = get_expected_times(df, TIMEFRAMES[tf])
    df.set_index('datetime', inplace=True)
    # Mark missing
    missing = expected.difference(df.index)
    if len(missing) == 0:
        print(f"[{tf}] No gaps found.")
        return
    print(f"[{tf}] Neutralizing {len(missing)} gaps...")
    for ts in missing:
        gap_row = pd.Series({
            'open': np.nan, 'high': np.nan, 'low': np.nan, 'close': np.nan, 'volume': np.nan, 'gap_bar': True
        })
        df.loc[ts] = gap_row
    if 'gap_bar' not in df.columns:
        df['gap_bar'] = False
    df = df.sort_index()
    df.reset_index(inplace=True)
    # Always save lowercase columns
    df.columns = [c.lower() for c in df.columns]
    df.to_csv(path, index=False)
    print(f"[{tf}] Gaps neutralized and saved.")

def main():
    for tf in TIMEFRAMES:
        neutralize_gaps(tf)
    print("=== All gaps neutralized. Re-run your pipeline gap check. ===")

if __name__ == "__main__":
    main()
