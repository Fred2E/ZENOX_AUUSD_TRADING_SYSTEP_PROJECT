import os
import pandas as pd
from datetime import datetime

ROOT = r"C:\Users\open\Documents\ZENO_XAUUSD\data\live\XAUUSD"
tfs = ["M5", "M15", "H1", "H4", "D1"]

for tf in tfs:
    fpath = os.path.join(ROOT, tf, f"XAUUSD_{tf}_LIVE.csv")
    print(f"\nChecking: {fpath}")
    if os.path.exists(fpath):
        mtime = os.path.getmtime(fpath)
        mod_dt = datetime.fromtimestamp(mtime)
        df = pd.read_csv(fpath)
        df.columns = [c.lower() for c in df.columns]  # Force lowercase for safety
        print(f"  Last Modified: {mod_dt}")
        print(f"  Rows: {len(df)}")
        if len(df) > 0:
            last = df.iloc[-1]
            dt = last['datetime'] if 'datetime' in last else "[MISSING]"
            close = last['close'] if 'close' in last else "[MISSING]"
            print(f"  Last Bar: {dt} | Close: {close}")
        else:
            print("  File exists but has NO data.")
    else:
        print("  [MISSING]")
