# grade_setups.py
import os, sys, numpy as np, pandas as pd

MODULES_DIR = r"C:\Users\open\Documents\ZENO_XAUUSD\modules"
if MODULES_DIR not in sys.path:
    sys.path.append(MODULES_DIR)

from io_utils import DATA_DIR, TIMEFRAMES, read_events_csv, write_csv

def grade_tf(tf: str):
    infile = os.path.join(DATA_DIR, f"trade_events_{tf}_FULL_enriched.csv")
    if not os.path.exists(infile):
        print(f"[SKIP] {tf}: {infile} not found.")
        return
    df = read_events_csv(infile)

    # Use total_confluence as base; fallback to (primary+secondary) if needed
    if "total_confluence" not in df.columns:
        df["total_confluence"] = df.get("primary_score", 0) + df.get("secondary_score", 0)

    # Robust: drop NaNs on the scoring column
    s = pd.to_numeric(df["total_confluence"], errors="coerce").fillna(-1)

    # Per‑TF adaptive cutoffs (prevents 0 A+ on H4)
    aplus_cut = np.percentile(s, 95)  # top 5%
    a_cut     = np.percentile(s, 70)  # top 30% after that

    def tag(v):
        if v >= aplus_cut: return "A+"
        if v >= a_cut:     return "A"
        return "B"

    df["setup_grade"] = s.map(tag)

    outp = os.path.join(DATA_DIR, f"trade_events_{tf}_graded.csv")
    write_csv(df, outp)
    print(f"[OK] {tf}: graded -> {outp} | A+={int((df['setup_grade']=='A+').sum())}")

def main():
    for tf in TIMEFRAMES:
        grade_tf(tf)

if __name__ == "__main__":
    main()
