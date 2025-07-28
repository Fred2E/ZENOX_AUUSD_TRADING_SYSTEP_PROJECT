import os
import pandas as pd
from datetime import timedelta

DATA_ROOT = os.path.join("data", "live", "XAUUSD")
TIMEFRAMES = {"M5": 5, "M15": 15, "H1": 60, "H4": 240, "D1": 1440}
CSV_TEMPLATE = "XAUUSD_{}_LIVE.csv"

for tf, step_min in TIMEFRAMES.items():
    path = os.path.join(DATA_ROOT, tf, CSV_TEMPLATE.format(tf))
    if not os.path.exists(path):
        print(f"[{tf}] MISSING FILE: {path}")
        continue
    df = pd.read_csv(path, parse_dates=["Datetime"])
    df = df.sort_values("Datetime").reset_index(drop=True)
    gaps = []
    for i in range(1, len(df)):
        diff = (df.loc[i, "Datetime"] - df.loc[i-1, "Datetime"]).total_seconds() / 60
        if diff > step_min * 1.5:
            gaps.append((df.loc[i-1, "Datetime"], df.loc[i, "Datetime"], diff))
    if not gaps:
        print(f"[{tf}] No gaps detected ({len(df)} rows)")
    else:
        print(f"[{tf}] GAPS FOUND:")
        for g in gaps:
            print(f"  Between {g[0]} and {g[1]}: {g[2]} min gap")
