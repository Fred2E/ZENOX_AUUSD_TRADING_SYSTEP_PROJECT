# === evaluate_confluence.py ===

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

def evaluate_confluence(df, timeframe='M15'):
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    N = len(df)

    # === PRIMARY CONFLUENCES ===
    structure = (~df.get('swing_high', pd.Series([np.nan]*N)).isna()) | \
                (~df.get('swing_low', pd.Series([np.nan]*N)).isna())

    bos_flag = _sanitize_direction(df.get('bos', pd.Series(['']*N)))
    choch_flag = _sanitize_direction(df.get('choch', pd.Series(['']*N)))
    bos_or_choch = ((bos_flag != 0) | (choch_flag != 0)).astype(int)

    pattern_code = df.get('pattern_code', pd.Series([0]*N)).fillna(0).astype(int)
    candle = (pattern_code > 0)

    price = df.get('close', pd.Series([np.nan]*N))
    sr_zone = pd.Series([False] * N, index=df.index)
    for i in range(N):
        window = price[max(0, i-5):i+1]
        if not window.isnull().all():
            sr_zone.iloc[i] = window.min() < price.iloc[i] < window.max()

    # === SECONDARY CONFLUENCES ===
    psych_level = ((price.round() % 250 < 2) | (price.round() % 250 > 248))

    fib_zone = pd.Series(False, index=df.index)
    if 'swing_high' in df.columns and 'swing_low' in df.columns:
        sh = df['swing_high'].dropna()
        sl = df['swing_low'].dropna()
        for i in range(N):
            p = price.iloc[i]
            if not np.isnan(p):
                near_sh = (not sh.empty) and (abs(p - sh.iloc[-1]) / p < 0.01)
                near_sl = (not sl.empty) and (abs(p - sl.iloc[-1]) / p < 0.01)
                fib_zone.iloc[i] = near_sh or near_sl

    # === Bias Conversion ===
    if 'bias' in df.columns:
        bull_logic = df['bias'].astype(str).str.lower() == 'bullish'
        bear_logic = df['bias'].astype(str).str.lower() == 'bearish'
        bias_bull = np.where(bull_logic, 1, np.where(bear_logic, 0, np.nan))
    else:
        bias_bull = np.zeros(N)

    # === Merge Flags ===
    flags = {
        "conf_structure": structure.astype(int),
        "conf_bos_or_choch": bos_or_choch,
        "conf_candle": candle.astype(int),
        "conf_sr_zone": sr_zone.astype(int),
        "conf_psych_level": psych_level.astype(int),
        "conf_fib_zone": fib_zone.astype(int),
        "bias_bull": bias_bull
    }
    conf_df = pd.DataFrame(flags, index=df.index)
    df = pd.concat([df, conf_df], axis=1)

    # === SCORING ===
    df['primary_score'] = df[PRIMARY_CONFS].sum(axis=1)
    df['secondary_score'] = df[SECONDARY_CONFS].sum(axis=1)
    df['total_confluence'] = df['primary_score'] + df['secondary_score']
    df['score'] = df['primary_score']
    df['num_confs'] = df[PRIMARY_CONFS + SECONDARY_CONFS].sum(axis=1)

    # === FEATURES ===
    df = compute_atr(df)
    if 'datetime' in df.columns:
        df['hour'] = pd.to_datetime(df['datetime'], errors='coerce').dt.hour
        df['dow'] = pd.to_datetime(df['datetime'], errors='coerce').dt.dayofweek

    return df
