import numpy as np
import pandas as pd

SWING_PARAMS = {
    'M5':  {'swing_window': 3,  'min_swing_dist': 3},
    'M15': {'swing_window': 4,  'min_swing_dist': 4},
    'H1':  {'swing_window': 5,  'min_swing_dist': 5},
    'H4':  {'swing_window': 7,  'min_swing_dist': 7},
    'D1':  {'swing_window': 10, 'min_swing_dist': 10}
}

def inject_structure_features(df, tf='M5'):
    """
    Inject swing_high, swing_low, BOS, CHoCH, bias, bias_label.
    Forces zero/null-safety on all outputs for downstream ML/RL.
    """
    params = SWING_PARAMS.get(tf, {'swing_window': 5, 'min_swing_dist': 5})
    swing_window = params['swing_window']
    min_swing_dist = params['min_swing_dist']

    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    n = len(df)

    df['swing_high'] = np.nan
    df['swing_low'] = np.nan
    df['bos'] = ''
    df['choch'] = ''
    df['bias'] = 1
    df['bias_label'] = 'bullish'

    swing_highs, swing_lows = [], []

    for i in range(swing_window, n - swing_window):
        window = df.iloc[i - swing_window:i + swing_window + 1]
        hi = df['high'].iloc[i]
        lo = df['low'].iloc[i]
        if hi == window['high'].max() and (not swing_highs or i - swing_highs[-1] >= min_swing_dist):
            df.at[df.index[i], 'swing_high'] = hi
            swing_highs.append(i)
        if lo == window['low'].min() and (not swing_lows or i - swing_lows[-1] >= min_swing_dist):
            df.at[df.index[i], 'swing_low'] = lo
            swing_lows.append(i)

    bias = 1  # 1 = bull, 0 = bear
    last_swing_high_idx, last_swing_low_idx = None, None

    for i in range(n):
        if not np.isnan(df['swing_high'].iloc[i]):
            if last_swing_high_idx is not None:
                prev_high = df['swing_high'].iloc[last_swing_high_idx]
                if bias == 0 and df['swing_high'].iloc[i] > prev_high:
                    df.at[df.index[i], 'choch'] = '↑'
                    bias = 1
                    df.at[df.index[i], 'bias_label'] = 'bullish'
                elif bias == 1 and df['swing_high'].iloc[i] > prev_high:
                    df.at[df.index[i], 'bos'] = '↑'
            last_swing_high_idx = i
        if not np.isnan(df['swing_low'].iloc[i]):
            if last_swing_low_idx is not None:
                prev_low = df['swing_low'].iloc[last_swing_low_idx]
                if bias == 1 and df['swing_low'].iloc[i] < prev_low:
                    df.at[df.index[i], 'choch'] = '↓'
                    bias = 0
                    df.at[df.index[i], 'bias_label'] = 'bearish'
                elif bias == 0 and df['swing_low'].iloc[i] < prev_low:
                    df.at[df.index[i], 'bos'] = '↓'
            last_swing_low_idx = i
        df.at[df.index[i], 'bias'] = bias

    # === FINAL PATCH: Null/empty fix (critical for audit pass!) ===
    df['swing_high'] = df['swing_high'].fillna(0)
    df['swing_low'] = df['swing_low'].fillna(0)
    df['bos'] = df['bos'].replace('', '0').fillna('0')
    df['choch'] = df['choch'].replace('', '0').fillna('0')
    df['bias'] = df['bias'].astype(int)
    df['bias_label'] = df['bias_label'].astype(str)
    df.columns = [c.lower() for c in df.columns]
    return df
