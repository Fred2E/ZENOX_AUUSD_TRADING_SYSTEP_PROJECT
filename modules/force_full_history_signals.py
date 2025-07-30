import os
import sys
import pandas as pd

# === Setup sys.path to include /modules and /scripts for direct imports ===
modules_dir = r"C:\Users\open\Documents\ZENO_XAUUSD\modules"
scripts_dir = r"C:\Users\open\Documents\ZENO_XAUUSD\scripts"
for p in [modules_dir, scripts_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from structure_detector import inject_structure_features      # modules/
from candle_patterns import detect_candle_patterns           # modules/
from evaluate_confluence import evaluate_confluence          # scripts/

RAW_DIR = r"C:\Users\open\Documents\ZENO_XAUUSD\historical\raw"
OUTPUT_DIR = r"C:\Users\open\Documents\ZENO_XAUUSD\historical\processed"
TIMEFRAMES = ["M5", "M15", "H1", "H4"]

def tag_setup(primary_score):
    if primary_score == 4:
        return "A+"
    elif primary_score == 3:
        return "A"
    elif primary_score == 2:
        return "B"
    else:
        return "C"

for tf in TIMEFRAMES:
    raw_path = os.path.join(RAW_DIR, f"XAUUSD_{tf}_CLEAN.csv")
    if not os.path.exists(raw_path):
        print(f"[{tf}] Missing: {raw_path}")
        continue

    print(f"\n[{tf}] Loading: {raw_path}")
    df = pd.read_csv(raw_path)
    df.columns = [c.lower() for c in df.columns]

    # Inject structure features
    df = inject_structure_features(df, tf=tf)
    print(f"[{tf}] Structure injected. Shape: {df.shape}")

    # Detect candle patterns
    df = detect_candle_patterns(df, tf=tf)
    print(f"[{tf}] Candle patterns injected. Shape: {df.shape}")

    # Compute confluences
    df = evaluate_confluence(df, tf)
    print(f"[{tf}] Confluence features injected. Shape: {df.shape}")

    # PATCH: Ensure all structure columns have no nulls or empty strings
    structure_cols = ['swing_high', 'swing_low', 'bos', 'choch']
    for col in structure_cols:
        if col in df.columns:
            if col in ['bos', 'choch']:
                df[col] = df[col].replace('', '0')
            df[col] = df[col].fillna(0)

    # CRITICAL PATCH: Inject regime_trend (if missing, use bias_bull as proxy)
    if 'regime_trend' not in df.columns:
        if 'bias_bull' in df.columns:
            df['regime_trend'] = df['bias_bull']
            print(f"[{tf}] PATCH: regime_trend injected as proxy from bias_bull.")
        else:
            df['regime_trend'] = 0
            print(f"[{tf}] WARNING: regime_trend not found and bias_bull missing. Defaulting to 0.")

    # === NEW: Inject setup_grade ===
    if 'primary_score' in df.columns:
        df['setup_grade'] = df['primary_score'].apply(tag_setup)
        print(f"[{tf}] setup_grade column injected.")
    else:
        df['setup_grade'] = "NA"
        print(f"[{tf}] WARNING: primary_score missing—setup_grade set to NA.")

    out_path = os.path.join(OUTPUT_DIR, f"signals_{tf}_FULL.csv")
    df.to_csv(out_path, index=False)
    print(f"[{tf}] Saved: {out_path}")

print("\n✅ FULL HISTORY SIGNALS COMPLETE: All timeframes processed, null-free structure features, fixed imports, regime_trend and setup_grade injected.")
