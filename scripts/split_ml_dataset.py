import os
import pandas as pd
import numpy as np
import hashlib
from sklearn.model_selection import train_test_split

# =========== CONFIG ============
PROCESSED_DIR = r"C:\Users\open\Documents\ZENO_XAUUSD\historical\processed"
OUTPUT_DIR    = r"C:\Users\open\Documents\ZENO_XAUUSD\outputs\ml_data"
ML_FEATURES   = [
    "close", "score", "num_confs", "pattern_code", "bias_bull", "hour", "dow"
]
TARGET_COL    = "is_win"  # <-- CHANGE if your target is different!
TIMEFRAMES    = ["M5", "M15", "H1", "H4"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

def hash_dataframe(df):
    """Return SHA256 hash of DataFrame contents for audit."""
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values).hexdigest()

def audit_and_split(tf):
    fname = f"signals_{tf}_FULL.csv"
    fpath = os.path.join(PROCESSED_DIR, fname)
    if not os.path.exists(fpath):
        print(f"[{tf}] [FATAL] File not found: {fpath}")
        return None, None

    df = pd.read_csv(fpath)
    df.columns = [c.lower() for c in df.columns]

    # Audit required features
    missing = [f for f in ML_FEATURES if f not in df.columns]
    if missing:
        print(f"[{tf}] [FATAL] Missing ML features: {missing}")
        return None, None

    if TARGET_COL not in df.columns:
        print(f"[{tf}] [FATAL] Target column '{TARGET_COL}' not found in {fname}. You must generate it upstream!")
        return None, None

    # Remove nulls, assert dtypes
    df = df.dropna(subset=ML_FEATURES + [TARGET_COL])
    for f in ML_FEATURES:
        if not np.issubdtype(df[f].dtype, np.number):
            print(f"[{tf}] [WARN] Forcing numeric dtype for {f}")
            df[f] = pd.to_numeric(df[f], errors="coerce")
    df = df.dropna(subset=ML_FEATURES + [TARGET_COL])
    assert not df.isnull().any().any(), f"[{tf}] [FATAL] Nulls detected after cleaning!"

    # Stratify by setup_grade if available, else use target
    strat_col = 'setup_grade' if 'setup_grade' in df.columns else TARGET_COL
    train, test = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df[strat_col]
    )
    print(f"\n[{tf}] Train shape: {train.shape}, Test shape: {test.shape}")
    print(f"[{tf}] Train target counts:\n{train[TARGET_COL].value_counts(dropna=False)}")
    print(f"[{tf}] Test target counts:\n{test[TARGET_COL].value_counts(dropna=False)}")
    if 'setup_grade' in df.columns:
        print(f"[{tf}] Train grade counts:\n{train['setup_grade'].value_counts(dropna=False)}")
        print(f"[{tf}] Test grade counts:\n{test['setup_grade'].value_counts(dropna=False)}")

    # Hash splits for integrity
    train_hash = hash_dataframe(train)
    test_hash  = hash_dataframe(test)
    print(f"[{tf}] Train HASH: {train_hash}")
    print(f"[{tf}] Test  HASH: {test_hash}")

    # Save splits
    train_out = os.path.join(OUTPUT_DIR, f"train_ml_{tf}.pkl")
    test_out  = os.path.join(OUTPUT_DIR, f"test_ml_{tf}.pkl")
    train.to_pickle(train_out)
    test.to_pickle(test_out)
    print(f"[{tf}] [SAVED] train: {train_out} | test: {test_out}")

    return train_hash, test_hash

if __name__ == "__main__":
    results = []
    for tf in TIMEFRAMES:
        hashes = audit_and_split(tf)
        results.append((tf, hashes))
    print("\n=== ML DATASET SPLIT & AUDIT COMPLETE ===")
    for tf, hashes in results:
        print(f"{tf}: train_hash={hashes[0]} | test_hash={hashes[1]}")
