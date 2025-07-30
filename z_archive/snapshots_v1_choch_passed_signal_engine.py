import pandas as pd
import numpy as np
import os
from evaluate_confluence import evaluate_confluence
from structure_detection import detect_structure_logic  # REAL structure logic

# === PARAMETERS ===
TIMEFRAME = 'M15'
CSV_PATH = f'C:/Users/open/Documents/ZENO_XAUUSD/historical/XAUUSDm_{TIMEFRAME}_HIST.csv'
CANDLE_LIMIT = 720
SEED = 42

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

    # === Inject dummy non-structural fields ===
    np.random.seed(SEED)
    N = len(df)

    if 'pattern_code' not in df.columns:
        df['pattern_code'] = np.random.choice([0, 1, 2, 3, 4, 5], size=N, p=[0.8, 0.04, 0.04, 0.04, 0.04, 0.04])
        print("[FAKE] Injected synthetic 'pattern_code'.")

    if 'bias' not in df.columns:
        df['bias'] = np.random.choice(['bullish', 'bearish', 'neutral'], size=N, p=[0.4, 0.4, 0.2])
        print("[FAKE] Injected synthetic 'bias'.")

    # === Initialize trend_state early ===
    if 'trend_state' not in df.columns:
        df['trend_state'] = None

    # === Inject real structure fields ===
    df = detect_structure_logic(df)

    # === TEMPORARY TEST CHoCH INJECTION ===
    if len(df) > 601:
        df.at[600, 'swing_high'] = df.loc[600, 'high']
        df.at[599, 'trend_state'] = 'bear'
        df.at[601, 'close'] = df.loc[600, 'high'] + 10
        print("[TEST] Forced synthetic CHoCH at index 600-601")
        print("[TEST] Injected synthetic CHoCH test case at index 600-601.")

    return df

def run_signal_engine():
    df = load_candle_data(CSV_PATH)
    df = evaluate_confluence(df, TIMEFRAME)

    # Filter trades: at least 2 primary confluences
    signals = df[df['primary_score'] >= 2].copy()

    # === DEBUG OUTPUT ===
    print("\n=== SIGNAL SNAPSHOT ===")
    cols_to_show = ['datetime', 'primary_score', 'secondary_score', 'total_confluence', 'bias_bull']
    if 'pattern_code' in df.columns:
        cols_to_show.append('pattern_code')

    print(signals[cols_to_show].tail(5))
    print(f"\n[INFO] {len(signals)} signals found from {len(df)} candles.")

    return signals

# === RUN ENTRY POINT ===
if __name__ == "__main__":
    signals = run_signal_engine()
