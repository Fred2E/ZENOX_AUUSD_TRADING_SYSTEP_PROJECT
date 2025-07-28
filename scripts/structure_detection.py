import pandas as pd
import numpy as np

def detect_structure_logic(df, window=3):
    df = df.copy()
    N = len(df)

    df['swing_high'] = np.nan
    df['swing_low'] = np.nan
    df['bos'] = ''
    df['choch'] = ''
    df['trend_state'] = None

    # === Step 1: Swing detection ===
    for i in range(window, N - window):
        if df['high'][i] > df['high'][i - window:i].max() and df['high'][i] > df['high'][i + 1:i + window + 1].max():
            df.at[i, 'swing_high'] = df['high'][i]
        if df['low'][i] < df['low'][i - window:i].min() and df['low'][i] < df['low'][i + 1:i + window + 1].min():
            df.at[i, 'swing_low'] = df['low'][i]

    # === Step 2: Structure tracking ===
    trend = None
    last_swing_high = None
    last_swing_low = None
    last_bull_swing_low = None
    last_bear_swing_high = None

    for i in range(1, N):
        close = df.loc[i, 'close']
        sh = df.loc[i - 1, 'swing_high']
        sl = df.loc[i - 1, 'swing_low']

        # Initialize trend based on first valid swing
        if trend is None:
            if not pd.isna(sh):
                trend = 'bear'
                last_bear_swing_high = sh
            elif not pd.isna(sl):
                trend = 'bull'
                last_bull_swing_low = sl

        # === CHoCH Logic ===
        if trend == 'bear' and last_bear_swing_high is not None and close > last_bear_swing_high:
            df.at[i, 'choch'] = '↑'
            trend = 'bull'
            last_bull_swing_low = sl

        elif trend == 'bull' and last_bull_swing_low is not None and close < last_bull_swing_low:
            df.at[i, 'choch'] = '↓'
            trend = 'bear'
            last_bear_swing_high = sh

        # === BOS Logic ===
        if trend == 'bull' and last_bear_swing_high is not None and close > last_bear_swing_high:
            df.at[i, 'bos'] = '↑'

        elif trend == 'bear' and last_bull_swing_low is not None and close < last_bull_swing_low:
            df.at[i, 'bos'] = '↓'

        # === Update memory with confirmed swings ===
        if not pd.isna(sh):
            last_swing_high = sh
            if trend == 'bear':
                last_bear_swing_high = sh

        if not pd.isna(sl):
            last_swing_low = sl
            if trend == 'bull':
                last_bull_swing_low = sl

        df.at[i, 'trend_state'] = trend

    # === Final audit ===
    print("[STRUCTURE] swing_high/swing_low/bos/choch injected.")
    print("[STRUCTURE] Totals: SwHigh={}, SwLow={}, BOS={}, CHoCH={}".format(
        df['swing_high'].notna().sum(),
        df['swing_low'].notna().sum(),
        df['bos'].isin(['↑', '↓']).sum(),
        df['choch'].isin(['↑', '↓']).sum()
    ))

    return df
