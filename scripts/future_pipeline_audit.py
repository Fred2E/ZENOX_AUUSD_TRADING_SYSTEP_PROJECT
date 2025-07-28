# future_pipeline_audit.py

import os
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
LIVE_DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "live", "XAUUSD")
TIMEFRAMES = ["M5", "M15", "H1", "H4", "D1"]
FEATURE_SUFFIX = "_LIVE_FEATURES.csv"
REQUIRED_FEATURES = ['datetime', 'close', 'score', 'num_confs', 'pattern_code', 'bias_bull', 'hour', 'dow']

print("=== FUTURE PIPELINE AUDIT ===")

failed = []

for tf in TIMEFRAMES:
    path = os.path.join(LIVE_DATA_ROOT, tf, f"XAUUSD_{tf}{FEATURE_SUFFIX}")
    print(f"\n[{tf}] Checking: {path}")
    if not os.path.exists(path):
        print(f"[FAIL] Feature file missing for {tf}: {path}")
        failed.append(tf)
        continue
    try:
        df = pd.read_csv(path)
        df.columns = [c.lower() for c in df.columns]
        missing = [col for col in REQUIRED_FEATURES if col not in df.columns]
        if missing:
            print(f"[FAIL] {tf} missing features: {missing}")
            failed.append(tf)
        else:
            print(f"[PASS] {tf} all required features present. Showing first row:")
            print(df[REQUIRED_FEATURES].head(1))
        if df['datetime'].isnull().any() or df['datetime'].eq('').any():
            print(f"[FAIL] {tf} has NULL/EMPTY datetime values.")
            failed.append(tf)
    except Exception as e:
        print(f"[FAIL] {tf} error: {e}")
        failed.append(tf)

print("\n=== PIPELINE AUDIT COMPLETE ===")
if failed:
    print(f"FAILED TIMEFRAMES: {failed}")
    raise RuntimeError(f"Pipeline audit failed for: {failed}")
else:
    print("ALL TIMEFRAMES PASSED AUDIT.")
