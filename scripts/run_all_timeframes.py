import pandas as pd
import numpy as np
import os
from evaluate_confluence import evaluate_confluence
from structure_detection import detect_structure_logic

TIMEFRAMES = ['M5', 'M15', 'H1', 'H4', 'D1']
CANDLE_LIMIT = 720
SEED = 42
DATA_FOLDER = "C:/Users/open/Documents/ZENO_XAUUSD/historical"

def load_candle_data(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"[ERROR] Historical data not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    df.columns = [col.lower() for col in df.columns]
    required = ['datetime', 'open', 'high', 'low', 'close', 'volume']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"[ERROR] Missing column: {col}")

    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    df = df.dropna(subset=['datetime'])
    df = df.tail(CANDLE_LIMIT).reset_index(drop=True)

    np.random.seed(SEED)
    N = len(df)

    if 'pattern_code' not in df.columns:
        df['pattern_code'] = np.random.choice([0, 1, 2, 3, 4, 5], size=N, p=[0.8, 0.04, 0.04, 0.04, 0.04, 0.04])
        print("[FAKE] Injected synthetic 'pattern_code'.")

    if 'bias' not in df.columns:
        df['bias'] = np.random.choice(['bullish', 'bearish', 'neutral'], size=N, p=[0.4, 0.4, 0.2])
        print("[FAKE] Injected synthetic 'bias'.")

    if 'trend_state' not in df.columns:
        df['trend_state'] = None

    # === Structure injection ===
    df = detect_structure_logic(df)

    # === TEMPORARY CHoCH injection ===
    if len(df) > 601:
        df.at[600, 'swing_high'] = df.loc[600, 'high']
        df.at[599, 'trend_state'] = 'bear'
        df.at[601, 'close'] = df.loc[600, 'high'] + 10
        print("[TEST] Injected synthetic CHoCH test case at index 600-601.")

    return df

def run_signal_engine(timeframe):
    print(f"\n=== Running for {timeframe} ===")
    csv_path = f"{DATA_FOLDER}/XAUUSDm_{timeframe}_HIST.csv"
    df = load_candle_data(csv_path)
    df = evaluate_confluence(df, timeframe)

    # Filter signals
    signals = df[df['primary_score'] >= 2].copy()

    # Print summary
    print(df[['datetime', 'swing_high', 'swing_low', 'bos', 'choch']].tail(10))
    print(f"[INFO] BOS count: {df['bos'].isin(['↑','↓']).sum()}, CHoCH count: {df['choch'].isin(['↑','↓']).sum()}")
    print(f"[INFO] {len(signals)} signals found from {len(df)} candles.")

    return signals

if __name__ == "__main__":
    for tf in TIMEFRAMES:
        run_signal_engine(tf)
