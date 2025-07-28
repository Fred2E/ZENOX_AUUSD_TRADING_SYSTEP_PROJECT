import numpy as np
import pandas as pd

PRIMARY_CONFS = ['conf_structure', 'conf_bos_or_choch', 'conf_candle', 'conf_sr_zone']
SECONDARY_CONFS = ['conf_psych_level', 'conf_fib_zone', 'bias_bull']

def _sanitize_direction(col):
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

def evaluate_confluence(df, timeframe):
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    N = len(df)

    # --- PRIMARY CONFLUENCES ---
    structure = (~df.get('swing_high', pd.Series([np.nan]*N)).isna()) | \
                (~df.get('swing_low', pd.Series([np.nan]*N)).isna())
    bos_flag = _sanitize_direction(df.get('bos', pd.Series(['']*N)))
    choch_flag = _sanitize_direction(df.get('choch', pd.Series(['']*N)))
    bos_or_choch = ((bos_flag != 0) | (choch_flag != 0)).astype(int)
    candle = df.get('pattern_code', pd.Series([0]*N)).fillna(0).astype(int) > 0

    # SR Zone Detection: Price near local high/low range within N bars
    sr_zone = pd.Series([0]*N, index=df.index)
    price = df.get('close', pd.Series([np.nan]*N))
    for i in range(N):
        window = price[max(0, i-10):i+1]  # lookback = 10
        if not window.isnull().all():
            zone_low, zone_high = window.min(), window.max()
            sr_zone.iloc[i] = int(zone_low < price.iloc[i] < zone_high)

    # --- SECONDARY CONFLUENCES ---
    psych_level = ((price % 50 < 1.5) | (price % 50 > 48.5)).astype(int)  # rounder psych logic
    fib_zone = pd.Series([0]*N, index=df.index)
    if 'swing_high' in df.columns and 'swing_low' in df.columns:
        for i in range(N):
            p = price.iloc[i]
            if not np.isnan(p):
                ref_high = df['swing_high'].iloc[max(i-10, 0):i+1].dropna()
                ref_low = df['swing_low'].iloc[max(i-10, 0):i+1].dropna()
                if not ref_high.empty and not ref_low.empty:
                    fib_levels = [0.382, 0.5, 0.618]
                    for fh in ref_high:
                        for fl in ref_low:
                            diff = fh - fl
                            levels = [fl + fib * diff for fib in fib_levels]
                            if any(abs(p - lvl) / p < 0.01 for lvl in levels):
                                fib_zone.iloc[i] = 1
                                break

    # Bias Bull Logic
    if 'bias' in df.columns:
        bias_bull = df['bias'].astype(str).str.lower().map({'bullish': 1, 'bearish': 0}).fillna(np.nan)
    else:
        bias_bull = pd.Series([np.nan]*N)

    # --- ASSIGN CONFLUENCE FLAGS ---
    df['conf_structure'] = structure.astype(int)
    df['conf_bos_or_choch'] = bos_or_choch.astype(int)
    df['conf_candle'] = candle.astype(int)
    df['conf_sr_zone'] = sr_zone.astype(int)
    df['conf_psych_level'] = psych_level.astype(int)
    df['conf_fib_zone'] = fib_zone.astype(int)
    df['bias_bull'] = bias_bull.astype(float)

    # --- SCORE & ENGINEER ---
    df['primary_score'] = df[PRIMARY_CONFS].sum(axis=1)
    df['secondary_score'] = df[SECONDARY_CONFS].sum(axis=1)
    df['total_confluence'] = df['primary_score'] + df['secondary_score']
    df['score'] = df['primary_score']
    df['num_confs'] = df[PRIMARY_CONFS + SECONDARY_CONFS].sum(axis=1)

    # --- ADD ATR ---
    df = compute_atr(df)

    # --- ADD TIME FEATURES ---
    if 'datetime' in df.columns:
        df['hour'] = pd.to_datetime(df['datetime'], errors='coerce').dt.hour
        df['dow'] = pd.to_datetime(df['datetime'], errors='coerce').dt.dayofweek
    else:
        df['hour'], df['dow'] = -1, -1

    return df
