import os
import pandas as pd
import sys

# Path configs
RAW_DIR = r"C:\Users\open\Documents\ZENO_XAUUSD\historical\raw"
OUT_DIR = RAW_DIR  # Overwrite in-place or set to another dir
TIMEFRAMES = ['M5', 'M15', 'H1', 'H4']  # Add D1 if needed

modules_path = r"C:\Users\open\Documents\ZENO_XAUUSD\modules"
if modules_path not in sys.path:
    sys.path.insert(0, modules_path)

# Import your structure logic
from structure_detector import inject_structure_features  # Must output ALL 4 cols

for tf in TIMEFRAMES:
    path = os.path.join(RAW_DIR, f"XAUUSD_{tf}_CLEAN.csv")
    if not os.path.exists(path):
        print(f"[{tf}] File not found: {path}")
        continue
    print(f"[{tf}] Injecting structure: {path}")
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    # Must inject swing_high, swing_low, bos, choch for all rows
    df_struct = inject_structure_features(df, tf)
    # Save
    df_struct.to_csv(path, index=False)
    print(f"[{tf}] Structure injected and saved: {df_struct.shape}")

print("\n✅ Structure features now injected into ALL full-history CLEAN files. Proceed to pattern/confluence export next.")
