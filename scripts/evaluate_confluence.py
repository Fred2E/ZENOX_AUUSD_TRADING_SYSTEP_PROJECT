import numpy as np
import pandas as pd
import os

PRIMARY_CONFS = ['conf_structure', 'conf_bos_or_choch', 'conf_candle', 'conf_sr_zone']
SECONDARY_CONFS = ['conf_psych_level', 'conf_fib_zone', 'bias_bull']

def _sanitize_direction(col):
    # Convert arrows to int: ↑=1, ↓=-1, else 0
    return col.fillna('').apply(lambda x: 1 if '↑' in str(x) else -1 if '↓' in str(x) else 0).astype(int)

def compute_atr(df, period=14):
    high = df['high'].astype(float)
    low = df['low'].astype(float)
    close = df['close'].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=period, min_periods=1).mean()
    return df

def evaluate_confluence(df, timeframe='M15'):
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    N = len(df)

    # == Validate essential columns ==
    required_cols = ['high', 'low', 'close']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # == Structure flags ==
    swing_high = df.get('swing_high', pd.Series([np.nan]*N))
    swing_low = df.get('swing_low', pd.Series([np.nan]*N))
    structure = (~swing_high.isna()) | (~swing_low.isna())

    bos_flag = _sanitize_direction(df.get('bos', pd.Series(['']*N)))
    choch_flag = _sanitize_direction(df.get('choch', pd.Series(['']*N)))
    bos_or_choch = ((bos_flag != 0) | (choch_flag != 0)).astype(int)

    # == Pattern code handling ==
    if 'pattern_code' not in df.columns:
        print("[WARN] 'pattern_code' missing, setting default 0.")
        df['pattern_code'] = 0
    else:
        df['pattern_code'] = df['pattern_code'].fillna(0).astype(int)

    candle = (df['pattern_code'] > 0)

    # == SR zone (5-candle window) ==
    price = df['close']
    sr_zone = pd.Series(False, index=df.index)
    for i in range(N):
        window = price[max(0, i-5):i+1]
        if not window.isnull().all():
            sr_zone.iat[i] = (window.min() < price.iat[i] < window.max())

    # == Psych level (nearest to 250s) ==
    psych_level = ((price.round() % 250 < 2) | (price.round() % 250 > 248))

    # == Fib zone (near recent swing points) ==
    fib_zone = pd.Series(False, index=df.index)
    if not swing_high.dropna().empty and not swing_low.dropna().empty:
        recent_sh = swing_high.dropna().values[-20:]
        recent_sl = swing_low.dropna().values[-20:]
        for i in range(N):
            p = price.iat[i]
            if not np.isnan(p):
                near_sh = np.any(np.abs(p - recent_sh) / p < 0.01)
                near_sl = np.any(np.abs(p - recent_sl) / p < 0.01)
                fib_zone.iat[i] = near_sh or near_sl

    # == Bias bull (from 'bias' string, else fallback zeros) ==
    if 'bias' in df.columns:
        bull_logic = df['bias'].astype(str).str.lower() == 'bullish'
        bear_logic = df['bias'].astype(str).str.lower() == 'bearish'
        bias_bull = np.where(bull_logic, 1, np.where(bear_logic, 0, 0))
    else:
        bias_bull = np.zeros(N, dtype=int)

    # == Confluence flags ==
    flags = {
        "conf_structure": structure.astype(int),
        "conf_bos_or_choch": bos_or_choch,
        "conf_candle": candle.astype(int),
        "conf_sr_zone": sr_zone.astype(int),
        "conf_psych_level": psych_level.astype(int),
        "conf_fib_zone": fib_zone.astype(int),
        "bias_bull": bias_bull
    }
    # Drop any existing to avoid duplication bugs
    for col in flags:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
    conf_df = pd.DataFrame(flags, index=df.index)
    df = pd.concat([df, conf_df], axis=1)

    # == Core confluence scores ==
    df['primary_score'] = df[PRIMARY_CONFS].sum(axis=1)
    df['secondary_score'] = df[SECONDARY_CONFS].sum(axis=1)
    df['total_confluence'] = df['primary_score'] + df['secondary_score']
    df['score'] = df['primary_score']
    df['num_confs'] = df[PRIMARY_CONFS + SECONDARY_CONFS].sum(axis=1)

    # == ATR and Time Features ==
    df = compute_atr(df)
    if 'datetime' in df.columns:
        dt_series = pd.to_datetime(df['datetime'], errors='coerce')
        df['hour'] = dt_series.dt.hour
        df['dow'] = dt_series.dt.dayofweek

    return df

# === MAIN EXECUTION ===
if __name__ == "__main__":
    TIMEFRAMES = ['M5', 'M15', 'H1', 'H4']  # D1 skipped until data exists
    BASE_PATH = 'C:/Users/open/Documents/ZENO_XAUUSD/historical'
    OUTPUT_DIR = os.path.join(BASE_PATH, 'processed')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for tf in TIMEFRAMES:
        try:
            filename = f"XAUUSDm_{tf}_HIST.csv"
            path = os.path.join(BASE_PATH, filename)

            print(f"Processing {tf} signals from {path} ...")
            df_raw = pd.read_csv(path)
            df_processed = evaluate_confluence(df_raw, timeframe=tf)

            out_path = os.path.join(OUTPUT_DIR, f"signals_{tf}.csv")
            df_processed.to_csv(out_path, index=False)

            print(f"Saved signals for {tf} to {out_path}\n")
        except Exception as e:
            print(f"[ERROR] Processing {tf}: {e}\n")
