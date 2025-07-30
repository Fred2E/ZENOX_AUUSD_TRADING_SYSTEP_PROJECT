import pandas as pd
import numpy as np

def detect_structure_logic(df, window=3):
    df = df.copy()
    N = len(df)

    swing_highs = [np.nan] * N
    swing_lows = [np.nan] * N

    atr = df['high'] - df['low']
    wick_threshold = atr.rolling(14, min_periods=1).mean() * 0.25

    for i in range(window, N - window):
        high = df.loc[i, 'high']
        low = df.loc[i, 'low']
        wick = wick_threshold.iloc[i]

        left = df.loc[i - window:i - 1]
        right = df.loc[i + 1:i + window]

        if high >= left['high'].max() and high >= right['high'].max():
            if (high - max(left['high'].max(), right['high'].max())) > wick:
                swing_highs[i] = high

        if low <= left['low'].min() and low <= right['low'].min():
            if (min(left['low'].min(), right['low'].min()) - low) > wick:
                swing_lows[i] = low

    df['swing_high'] = swing_highs
    df['swing_low'] = swing_lows
    df['bos'] = ''
    df['choch'] = ''
    df['trend_state'] = pd.Series([None] * N, dtype='object')

    last_swing_high = None
    last_swing_low = None
    trend = None

    # === TEMPORARY TEST CHoCH INJECTION ===
    df.at[600, 'swing_high'] = df.loc[600, 'high']
    df.at[601, 'close'] = df.loc[600, 'high'] + 10
    df.at[599, 'trend_state'] = 'bear'
    print("[TEST] Injected synthetic CHoCH test case at index 600-601.")

    for i in range(N - 1):
        close = df.loc[i + 1, 'close']
        swing_high = df.loc[i, 'swing_high']
        swing_low = df.loc[i, 'swing_low']

        # === RESPECT EXPLICIT TREND STATE ===
        if pd.notna(df.at[i, 'trend_state']):
            trend = df.at[i, 'trend_state']

        # === INIT Trend ===
        if trend is None:
            if not pd.isna(swing_low):
                trend = 'bull'
                last_swing_low = swing_low
            elif not pd.isna(swing_high):
                trend = 'bear'
                last_swing_high = swing_high

        # === CHoCH Logic ===
        if trend == 'bear' and not pd.isna(swing_high) and close > swing_high:
            df.at[i + 1, 'choch'] = '↑'
            trend = 'bull'
            last_swing_low = swing_low
        elif trend == 'bull' and not pd.isna(swing_low) and close < swing_low:
            df.at[i + 1, 'choch'] = '↓'
            trend = 'bear'
            last_swing_high = swing_high

        # === BOS Logic ===
        if trend == 'bull' and not pd.isna(swing_high):
            if last_swing_high is not None and close > last_swing_high:
                df.at[i + 1, 'bos'] = '↑'
            last_swing_high = swing_high

        if trend == 'bear' and not pd.isna(swing_low):
            if last_swing_low is not None and close < last_swing_low:
                df.at[i + 1, 'bos'] = '↓'
            last_swing_low = swing_low

        df.at[i + 1, 'trend_state'] = trend

    # === Final Audit ===
    print("[STRUCTURE] swing_high/swing_low/bos/choch injected.")
    print("[STRUCTURE] Totals: SwHigh={}, SwLow={}, BOS={}, CHoCH={}".format(
        df['swing_high'].notna().sum(),
        df['swing_low'].notna().sum(),
        df['bos'].isin(['↑', '↓']).sum(),
        df['choch'].isin(['↑', '↓']).sum()
    ))
    print(df[['datetime', 'swing_high', 'swing_low', 'bos', 'choch']].iloc[595:605])

    return df
