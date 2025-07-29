import pandas as pd
import numpy as np
import os
import sys

# === Inject module path ===
MODULES_PATH = 'C:/Users/open/Documents/ZENO_XAUUSD/modules'
if MODULES_PATH not in sys.path:
    sys.path.append(MODULES_PATH)

from evaluate_confluence import evaluate_confluence
from structure_detection import detect_structure_logic
from trend_bias import compute_bias_bull
from candle_patterns import detect_candle_patterns

# === Config ===
TIMEFRAMES = ['M5', 'M15', 'H1', 'H4', 'D1']
BASE_PATH = 'C:/Users/open/Documents/ZENO_XAUUSD/historical'
OUTPUT_DIR = os.path.join(BASE_PATH, 'processed')
CANDLE_LIMITS = {'M5': 720, 'M15': 480, 'H1': 360, 'H4': 200, 'D1': 100}
HTF_MAP = {'M5': 'H1', 'M15': 'H4', 'H1': 'D1', 'H4': 'D1', 'D1': None}  # Map each tf to its regime HTF

REQUIRED_CONFS = [
    'conf_structure', 'conf_bos_or_choch', 'conf_candle', 'conf_sr_zone',
    'conf_psych_level', 'conf_fib_zone', 'conf_volume', 'conf_liquidity', 'conf_spread'
]
PRIMARY_CONFS = ['conf_structure', 'conf_bos_or_choch', 'conf_candle', 'conf_sr_zone']

def load_and_prepare_data(csv_path, tf):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"[ERROR] Missing file: {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = [c.lower() for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]

    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce').dt.tz_localize(None)
    df = df.dropna(subset=['datetime']).reset_index(drop=True)

    candle_limit = CANDLE_LIMITS.get(tf, 500)
    if len(df) > candle_limit:
        df = df.tail(candle_limit).reset_index(drop=True)

    for col in ['bias_bull', 'atr', 'volatility_flag', 'noise_index']:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    df = detect_structure_logic(df)
    df = detect_candle_patterns(df)
    df['bias_bull'] = compute_bias_bull(df, tf=tf)  # ADAPTIVE PER TF

    tr = np.maximum(df['high'] - df['low'], 
                    np.abs(df['high'] - df['close'].shift(1)),
                    np.abs(df['low'] - df['close'].shift(1)))
    df['atr'] = tr.rolling(14, min_periods=1).mean()

    if tf == 'H4':
        df['volatility_flag'] = (df['atr'] > 10).astype(int)
    elif tf == 'M5':
        df['noise_index'] = df['close'].pct_change().abs().rolling(5, min_periods=1).mean()

    if 'volume' not in df.columns:
        print(f"[WARN] 'volume' missing in {tf}, faking with ones (FIX AT DATA SOURCE)")
        df['volume'] = 1

    df['conf_volume'] = (df['volume'] > df['volume'].rolling(20, min_periods=1).mean()).astype(int)
    df['conf_liquidity'] = (df['volume'].rolling(10, min_periods=1).sum() >
                            df['volume'].rolling(100, min_periods=1).sum().mean()).astype(int)

    if 'spread' in df.columns:
        df['conf_spread'] = (df['spread'] < df['spread'].rolling(10, min_periods=1).mean()).astype(int)
    else:
        print(f"[WARN] 'spread' missing in {tf}, faking conf_spread=1 (FIX AT DATA SOURCE)")
        df['conf_spread'] = 1

    for col in REQUIRED_CONFS:
        if col not in df.columns:
            print(f"[WARN] '{col}' missing after pipeline for {tf}, patching with 0.")
            df[col] = 0

    df = evaluate_confluence(df, tf)
    df = df.loc[:, ~df.columns.duplicated()]

    # === Inject regime trend from correct HTF ===
    htf_tf = HTF_MAP.get(tf)
    if htf_tf is not None:
        htf_file = os.path.join(OUTPUT_DIR, f'signals_{htf_tf}.csv')
        if os.path.exists(htf_file):
            htf = pd.read_csv(htf_file)
            htf['datetime'] = pd.to_datetime(htf['datetime'], errors='coerce').dt.tz_localize(None)
            htf = htf.sort_values('datetime').reset_index(drop=True)
            htf['regime_trend'] = compute_bias_bull(htf, tf=htf_tf)
            df = df.sort_values('datetime').reset_index(drop=True)
            df['regime_trend'] = df['datetime'].apply(
                lambda dt: htf[htf['datetime'] <= dt]['regime_trend'].iloc[-1] if not htf[htf['datetime'] <= dt].empty else 1
            )
        else:
            print(f"[WARN] No HTF file found for {tf}, defaulting regime_trend=1")
            df['regime_trend'] = 1
    else:
        df['regime_trend'] = 1  # Highest tf, use 1 as dummy

    # --- Must-have columns check ---
    must_have = [
        'datetime', 'open', 'high', 'low', 'close', 'volume', 'pattern_code',
        'score', 'num_confs', 'primary_score', 'secondary_score', 'total_confluence',
        'bias_bull', 'atr', 'regime_trend'
    ] + REQUIRED_CONFS
    missing_cols = [col for col in must_have if col not in df.columns]
    if missing_cols:
        raise ValueError(f"[FATAL] Missing columns: {missing_cols}")

    df = df.dropna(subset=must_have).reset_index(drop=True)
    print(f"\n[INFO] {tf} data loaded. Shape: {df.shape}. Columns: {list(df.columns)}")
    print("bias_bull value counts:", df['bias_bull'].value_counts().to_dict())
    print("regime_trend value counts:", df['regime_trend'].value_counts().to_dict())
    return df

def filter_primary_confs(df, n_required=3):
    """Return mask of rows where at least n_required of the 4 primary confluences are met."""
    return df.apply(lambda row: sum([row[c]==1 for c in PRIMARY_CONFS]) >= n_required, axis=1)

def evaluate_and_save_signals(tf):
    filename = f"XAUUSDm_{tf}_HIST.csv" if tf != 'D1' else f"XAUUSD_{tf}_HIST.csv"
    path = os.path.join(BASE_PATH, filename)
    df = load_and_prepare_data(path, tf)

    # SIGNAL DEFINITION: at least 3/4 primary confluences + regime alignment
    signals = df[
        filter_primary_confs(df, n_required=3) &
        (df['regime_trend'] == df['bias_bull'])
    ].copy()

    bos_count = df['bos'].isin(['↑', '↓']).sum() if 'bos' in df.columns else 0
    choch_count = df['choch'].isin(['↑', '↓']).sum() if 'choch' in df.columns else 0

    print(f"\n=== SIGNAL SNAPSHOT [{tf}] ===")
    print(signals[['datetime', 'primary_score', 'secondary_score', 'total_confluence', 'bias_bull', 'pattern_code']].tail(3))
    print(f"[STRUCTURE] BOS: {bos_count} | CHoCH: {choch_count}")
    print(f"[bias_bull] Nulls: {df['bias_bull'].isnull().sum()}")
    print(f"[SAVE] Signals: {signals.shape}, All: {df.shape}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"signals_{tf}.csv")
    signals.to_csv(out_path, index=False)
    df.to_csv(os.path.join(OUTPUT_DIR, f"full_{tf}_processed.csv"), index=False)
    print(f"[SAVED] {out_path}")

if __name__ == "__main__":
    for tf in TIMEFRAMES:
        try:
            evaluate_and_save_signals(tf)
        except Exception as e:
            print(f"[ERROR] Failed on {tf}: {e}")
