# ===== 1) io_utils (Jupyter cell) =====
import os, json, hashlib, platform
from datetime import datetime
import numpy as np
import pandas as pd

# --- Root & folders (tailored) ---
ROOT_DIR   = r"C:\Users\open\Documents\ZENO_XAUUSD"
DATA_DIR   = os.path.join(ROOT_DIR, "outputs", "ml_data")
SIGNAL_DIR = os.path.join(ROOT_DIR, "outputs", "ml_signals")
LOGS_DIR   = os.path.join(ROOT_DIR, "logs")
for _d in (DATA_DIR, SIGNAL_DIR, LOGS_DIR):
    os.makedirs(_d, exist_ok=True)

TIMEFRAMES = ["M15", "H1", "H4"]

# --- Required columns (baseline + ML) ---
REQUIRED_BASE = ["datetime", "open", "high", "low", "close", "volume", "atr"]
ML_FEATURES_ALL = [
    "close","high","low","atr","score","num_confs","pattern_code","bias_bull",
    "hour","dow","primary_score","secondary_score","total_confluence","regime_trend",
    "conf_sr_zone","conf_bos_or_choch","conf_psych_level","conf_fib_zone",
    "conf_volume","conf_liquidity","conf_spread"   # if this makes 21, we’ll drop if missing
]

# dtypes to suppress DtypeWarning
READ_DTYPES = {
    "open":"float64","high":"float64","low":"float64","close":"float64","volume":"float64","atr":"float64",
    "score":"float64","num_confs":"float64","pattern_code":"float64","bias_bull":"float64",
    "hour":"float64","dow":"float64","primary_score":"float64","secondary_score":"float64",
    "total_confluence":"float64","regime_trend":"float64","conf_sr_zone":"float64",
    "conf_bos_or_choch":"float64","conf_psych_level":"float64","conf_fib_zone":"float64",
    "conf_volume":"float64","conf_liquidity":"float64","conf_spread":"float64",
    "is_win":"Int64"
}

def ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    if "datetime" not in df.columns:
        raise ValueError("[ERR] missing 'datetime' column")
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=False)
    df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    return df

def read_events_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path, dtype=READ_DTYPES, parse_dates=["datetime"], low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    df = ensure_datetime(df)
    return df

def write_csv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[WRITE] {path} | shape={df.shape}")

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def safe_feature_list(df: pd.DataFrame) -> list:
    # Use intersection so we never fail if a column is missing
    feats = [c for c in ML_FEATURES_ALL if c in df.columns]
    return feats

def labeled_path(tf: str) -> str:
    # Your confirmed inputs (labeled files only)
    return os.path.join(DATA_DIR, f"trade_events_{tf}_FULL_labeled.csv")
