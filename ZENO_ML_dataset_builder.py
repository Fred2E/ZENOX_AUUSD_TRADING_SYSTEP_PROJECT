import os, sys, pathlib, ast
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split

ROOT = pathlib.Path(r"C:\Users\open\Documents\ZENO_XAUUSD")
MODULE_DIR = ROOT / "modules"
sys.path.append(str(MODULE_DIR))

DATA_DIR      = ROOT / "outputs" / "setups"
HIST_PROC_DIR = ROOT / "historical" / "processed"
TRADE_LOG     = ROOT / "outputs" / "performance_logs" / "ZENO_trade_log.csv"
ML_OUT        = ROOT / "outputs" / "ml_data"
ML_OUT.mkdir(parents=True, exist_ok=True)
np.random.seed(42)

ML_FEATURES = [
    "close", "score", "num_confs", "pattern_code", "bias_bull", "hour", "dow"
]

def parse_confs(x):
    # Defensive: handles stringified lists, real lists, empty, float NaN, etc.
    if isinstance(x, str):
        try: return len(ast.literal_eval(x))
        except: return 0
    try: return len(x)
    except: return 0

candidate_paths = list(DATA_DIR.glob("*.csv")) + list(HIST_PROC_DIR.glob("*.csv"))
signal_frames   = []
for p in candidate_paths:
    try:
        df = pd.read_csv(p)
        df.columns = [c.lower() for c in df.columns]
        # Force datetime parse as UTC
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
        if {"datetime", "close", "bias", "score"}.issubset(df.columns):
            # Defensive handling for confluences
            if "confluences" in df.columns:
                df["num_confs"] = df["confluences"].apply(parse_confs)
            else:
                df["num_confs"] = 0
            # Defensive: is_win present
            if "is_win" not in df.columns:
                df["is_win"] = 1
            signal_frames.append(df)
            print(f"✓ loaded {p.name:35s}  ({len(df):,} rows)")
    except Exception as e:
        print(f"⚠️  skipped {p.name}: {e}")

if not signal_frames:
    sys.exit("❌ No usable signal CSVs found – aborting.")

df_pos = pd.concat(signal_frames, ignore_index=True)
print("DEBUG: df_pos is_win counts:\n", df_pos['is_win'].value_counts(dropna=False))

# --- REAL LOSSES FROM TRADE LOG (optional, if any) --- #
df_neg_real = pd.DataFrame()
if TRADE_LOG.exists():
    log = pd.read_csv(TRADE_LOG)
    log.columns = [c.lower() for c in log.columns]
    log["datetime"] = pd.to_datetime(log["datetime"], utc=True, errors="coerce")
    if "tradeoutcome" in log.columns:
        df_neg_real = log[log["tradeoutcome"].str.lower() == "loss"].copy()
        if not df_neg_real.empty:
            df_neg_real["is_win"] = 0
            print(f"✓ pulled {len(df_neg_real)} real losses from trade log")
if not df_neg_real.empty:
    print("DEBUG: df_neg_real is_win counts:\n", df_neg_real['is_win'].value_counts())
else:
    print("DEBUG: No real losses found; all negatives will be synthetic.")

# --- SYNTHETIC LOSSES IF NEEDED (balanced, random time offset) --- #
need = max(0, len(df_pos) - len(df_neg_real))
if need:
    synth = df_pos.sample(need, replace=True).copy()
    synth["is_win"] = 0
    synth["datetime"] = pd.to_datetime(synth["datetime"], utc=True, errors="coerce")
    synth["datetime"] += pd.to_timedelta(
        np.random.randint(60, 3600, need), unit="s"
    )
    df_neg_real = pd.concat([df_neg_real, synth], ignore_index=True)
    print(f"✓ fabricated {need} synthetic negatives")
print("DEBUG: df_neg_real (after synth) is_win counts:\n", df_neg_real['is_win'].value_counts(dropna=False))

# --- COMBINE, FORCE ALL FEATURES --- #
df_full = pd.concat([df_pos, df_neg_real], ignore_index=True)
print("DEBUG: df_full is_win counts:\n", df_full['is_win'].value_counts(dropna=False))
print("DEBUG: df_full sample:\n", df_full[['datetime','is_win']].sample(10))

# --- FORCE BOTH CLASSES --- #
if set(df_full['is_win'].dropna().unique()) != {0, 1}:
    print("CRITICAL FAILURE: Not both classes present in final dataset!")
    print("is_win value_counts:\n", df_full['is_win'].value_counts())
    sys.exit(1)

# --- Feature Engineering --- #
df_full["pattern_code"] = df_full["candle_pattern"].astype("category").cat.codes if "candle_pattern" in df_full.columns else -1
df_full["bias_bull"] = df_full["bias"].str.lower().map({"bullish": 1, "bearish": 0}).fillna(0).astype(int) if "bias" in df_full.columns else 0
df_full["hour"] = df_full["datetime"].dt.hour
df_full["dow"]  = df_full["datetime"].dt.dayofweek
if "score" not in df_full.columns: df_full["score"] = 0
if "close" not in df_full.columns: df_full["close"] = 0

# --- ENFORCE COLUMN ORDER AND VALIDITY --- #
missing_feats = [feat for feat in ML_FEATURES if feat not in df_full.columns]
for m in missing_feats:
    df_full[m] = 0
df_ml = df_full[["datetime"] + ML_FEATURES + ["is_win"]].sort_values("datetime")

# --- CHRONOLOGICAL SPLIT --- #
train, test = train_test_split(df_ml, test_size=0.20, shuffle=False)

train.to_pickle(ML_OUT / "train_ml.pkl")
test.to_pickle (ML_OUT / "test_ml.pkl")

print("\n✅  ML datasets saved:")
print(f"   • train_ml.pkl : {len(train):,} rows")
print(f"   • test_ml.pkl  : {len(test ):,} rows\n")
print("Class balance:")
print(train["is_win"].value_counts(normalize=True).rename("train").to_frame().T)
print(test ["is_win"].value_counts(normalize=True).rename("test" ).to_frame().T)
