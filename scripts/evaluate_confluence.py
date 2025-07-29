import numpy as np
import pandas as pd
import os

# === Configurable params by timeframe ===
SR_LOOKBACK = {'M5': 10, 'M15': 20, 'H1': 30, 'H4': 50, 'D1': 100}
SR_THRESHOLD = {'M5': 0.0015, 'M15': 0.0015, 'H1': 0.001, 'H4': 0.001, 'D1': 0.001}
PSYCH_LEVELS = [50, 100, 250]
PSYCH_THRESH = {'M5': 1.5, 'M15': 2, 'H1': 2, 'H4': 2.5, 'D1': 3}

PRIMARY_CONFS = ['conf_structure', 'conf_bos_or_choch', 'conf_candle', 'conf_sr_zone']
SECONDARY_CONFS = ['conf_psych_level', 'conf_fib_zone', 'conf_volume', 'conf_liquidity', 'conf_spread']

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

def evaluate_confluence(df, timeframe='M15'):
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    N = len(df)

    # === Structure flags (strict: only if swing + BOS/CHoCH) ===
    swing_high = df.get('swing_high', pd.Series([np.nan]*N))
    swing_low = df.get('swing_low', pd.Series([np.nan]*N))
    bos_flag = _sanitize_direction(df.get('bos', pd.Series(['']*N)))
    choch_flag = _sanitize_direction(df.get('choch', pd.Series(['']*N)))
    conf_bos_or_choch = ((bos_flag != 0) | (choch_flag != 0)).astype(int)
    conf_structure = ((~swing_high.isna()) | (~swing_low.isna())) & (conf_bos_or_choch == 1)
    conf_structure = conf_structure.astype(int)

    # === Candle pattern ===
    candle = df.get('pattern_code', pd.Series([0]*N)).fillna(0).astype(int)
    conf_candle = (candle > 0).astype(int)

    # === SR zone (adaptive by tf) ===
    lookback = SR_LOOKBACK.get(timeframe, 20)
    threshold = SR_THRESHOLD.get(timeframe, 0.0015)
    price = df['close']
    conf_sr_zone = []
    for i in range(N):
        idx = max(0, i - lookback)
        window = df.iloc[idx:i+1]
        hi = window['high'].max()
        lo = window['low'].min()
        p = price.iloc[i]
        near_high = abs(p - hi) / p < threshold
        near_low = abs(p - lo) / p < threshold
        conf_sr_zone.append(int(near_high or near_low))
    conf_sr_zone = pd.Series(conf_sr_zone, index=df.index)

    # === Psych level (adaptive by tf, near 50/100/250s) ===
    psych_thresh = PSYCH_THRESH.get(timeframe, 2)
    conf_psych_level = pd.Series(0, index=df.index)
    for lvl in PSYCH_LEVELS:
        conf_psych_level |= ((price % lvl < psych_thresh) | (price % lvl > (lvl - psych_thresh))).astype(int)

    # === Fib zone (approx, near 0.382, 0.5, 0.618 of recent swings) ===
    fib_zone = []
    for i, row in df.iterrows():
        idx = max(0, i - lookback)
        sw_high = swing_high.iloc[idx:i+1].dropna()
        sw_low = swing_low.iloc[idx:i+1].dropna()
        close = row['close']
        found = 0
        if not sw_high.empty and not sw_low.empty:
            h, l = sw_high.max(), sw_low.min()
            diff = h - l
            fibs = [l + 0.382*diff, l + 0.5*diff, l + 0.618*diff]
            found = int(any(abs(close-f)/close < 0.01 for f in fibs))
        fib_zone.append(found)
    conf_fib_zone = pd.Series(fib_zone, index=df.index)

    # === Volume, Liquidity, Spread (already computed in pipeline or patch with 0) ===
    conf_volume = df.get('conf_volume', pd.Series([0]*N)).astype(int)
    conf_liquidity = df.get('conf_liquidity', pd.Series([0]*N)).astype(int)
    conf_spread = df.get('conf_spread', pd.Series([1]*N)).astype(int)

    # === Bias bull: DO NOT overwrite if already robustly computed ===
    if 'bias_bull' not in df.columns:
        # Only fallback to string bias if not already computed (should never happen in final pipeline)
        if 'bias' in df.columns:
            bull_logic = df['bias'].astype(str).str.lower() == 'bullish'
            bear_logic = df['bias'].astype(str).str.lower() == 'bearish'
            bias_bull = np.where(bull_logic, 1, np.where(bear_logic, 0, 0))
        else:
            bias_bull = np.zeros(N, dtype=int)
        df['bias_bull'] = bias_bull

    # === Confluence flags assignment ===
    df['conf_structure'] = conf_structure
    df['conf_bos_or_choch'] = conf_bos_or_choch
    df['conf_candle'] = conf_candle
    df['conf_sr_zone'] = conf_sr_zone
    df['conf_psych_level'] = conf_psych_level
    df['conf_fib_zone'] = conf_fib_zone
    df['conf_volume'] = conf_volume
    df['conf_liquidity'] = conf_liquidity
    df['conf_spread'] = conf_spread

    # === Confluence scores ===
    df['primary_score'] = df[PRIMARY_CONFS].sum(axis=1)
    df['secondary_score'] = df[SECONDARY_CONFS].sum(axis=1)
    df['total_confluence'] = df['primary_score'] + df['secondary_score']
    df['score'] = df['primary_score']
    df['num_confs'] = df[PRIMARY_CONFS + SECONDARY_CONFS].sum(axis=1)

    # === ATR and Time Features ===
    df = compute_atr(df)
    if 'datetime' in df.columns:
        dt_series = pd.to_datetime(df['datetime'], errors='coerce')
        df['hour'] = dt_series.dt.hour
        df['dow'] = dt_series.dt.dayofweek

    # === Debug print (per tf) ===
    print(f"\n[{timeframe}] PRIMARY CONFLUENCE COUNTS:")
    for conf in PRIMARY_CONFS:
        print(f"{conf}: {df[conf].sum()}")
    print(f"Total signals (3/4 primaries): {(df[PRIMARY_CONFS].sum(axis=1) >= 3).sum()}")

    return df

# No main block—meant for import into pipeline!
