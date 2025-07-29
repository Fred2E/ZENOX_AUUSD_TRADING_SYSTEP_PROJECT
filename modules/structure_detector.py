import numpy as np
import pandas as pd

SWING_PARAMS = {
    'M5':  {'swing_window': 3,  'min_swing_dist': 3},
    'M15': {'swing_window': 4,  'min_swing_dist': 4},
    'H1':  {'swing_window': 5,  'min_swing_dist': 5},
    'H4':  {'swing_window': 7,  'min_swing_dist': 7},
    'D1':  {'swing_window': 10, 'min_swing_dist': 10}
}

def detect_structure_logic(df, tf='M5'):
    """
    Adaptive institutional swing structure, bias, BOS, and CHoCH detection for XAUUSD.
    Applies different swing/structure logic by timeframe as per institutional best practice.
    """
    params = SWING_PARAMS.get(tf, {'swing_window': 5, 'min_swing_dist': 5})
    swing_window = params['swing_window']
    min_swing_dist = params['min_swing_dist']

    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    n = len(df)

    # Initialize structure columns
    df['swing_high'] = np.nan
    df['swing_low'] = np.nan
    df['bos'] = ''
    df['choch'] = ''
    df['bias'] = 1
    df['bias_label'] = 'bullish'

    swing_highs, swing_lows = [], []

    # === Find adaptive swing highs/lows ===
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

    # === BOS/CHoCH and rolling bias ===
    bias = 1  # 1=bull, 0=bear
    last_swing_high_idx, last_swing_low_idx = None, None

    for i in range(n):
        # New swing high
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
        # New swing low
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

    # Final cleanup for types
    df['bos'] = df['bos'].astype(str)
    df['choch'] = df['choch'].astype(str)
    df['bias'] = df['bias'].astype(int)
    df['bias_label'] = df['bias_label'].astype(str)

    # Lowercase columns for pipeline compatibility
    df.columns = [c.lower() for c in df.columns]
    return df

# Usage:
# df = detect_structure_logic(df, tf='M5')
