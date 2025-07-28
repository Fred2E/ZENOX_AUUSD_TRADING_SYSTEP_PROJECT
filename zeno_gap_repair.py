# zeno_gap_repair.py

import os
import pandas as pd
from datetime import datetime, timedelta, timezone

DATA_ROOT = os.path.join("data", "live", "XAUUSD")
TIMEFRAMES = {"M5": 5, "M15": 15, "H1": 60, "H4": 240, "D1": 1440}
CSV_TEMPLATE = "XAUUSD_{}_LIVE.csv"

def expected_times(start, end, freq_minutes):
    times = []
    now = start
    while now <= end:
        times.append(now)
        now += timedelta(minutes=freq_minutes)
    return times

def find_and_report_gaps(tf):
    path = os.path.join(DATA_ROOT, tf, CSV_TEMPLATE.format(tf))
    if not os.path.exists(path):
        print(f"[{tf}] Missing file: {path}")
        return None
    df = pd.read_csv(path, parse_dates=['Datetime'])
    df = df.sort_values('Datetime').reset_index(drop=True)
    freq = TIMEFRAMES[tf]

    # Get expected times from first to last
    start = df['Datetime'].iloc[0]
    end = df['Datetime'].iloc[-1]
    expected = pd.Series(expected_times(start, end, freq), name='Datetime')
    df_full = pd.DataFrame({'Datetime': expected}).merge(df, on='Datetime', how='left')

    # Find gaps
    missing = df_full[df_full.isnull().any(axis=1)]
    if missing.shape[0] > 0:
        print(f"[{tf}] GAP DETECTED: {missing.shape[0]} missing bar(s)")
        print(missing[['Datetime']])
        # Save gap report
        report_path = os.path.join(DATA_ROOT, tf, f"GAP_REPORT_{tf}.csv")
        missing.to_csv(report_path, index=False)
        print(f"[{tf}] Gap report saved: {report_path}")

        # Uncomment to auto-fill (optional, advanced):
        # df_full.to_csv(path.replace(".csv", "_FILLED.csv"), index=False)
        # print(f"[{tf}] Gaps filled with NaN rows. Review before using.")
    else:
        print(f"[{tf}] OK - No gaps.")

    return df_full

def main():
    print("=== ZENO GAP SCAN ACROSS ALL TIMEFRAMES ===")
    for tf in TIMEFRAMES:
        find_and_report_gaps(tf)

if __name__ == "__main__":
    main()
