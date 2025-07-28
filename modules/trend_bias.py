import pandas as pd
import numpy as np

def compute_bias_bull(df: pd.DataFrame, ema_period=21) -> pd.Series:
    """
    Computes directional bias (1 = bullish, 0 = bearish) based on EMA trend slope.
    Uses EMA to determine if market is trending up/down.
    Returns a Series of bias_bull flags (0 or 1), no NaNs.
    """
    if 'close' not in df.columns:
        raise ValueError("Missing 'close' column for bias computation.")

    # Compute EMA
    df['ema'] = df['close'].ewm(span=ema_period, adjust=False).mean()

    # Compute EMA slope (simple diff over 3 bars)
    df['ema_slope'] = df['ema'].diff(3)

    # Define bias: 1 = bullish, 0 = bearish
    bias = np.where(df['ema_slope'] >= 0, 1, 0)

    # Drop helper columns after use
    df.drop(columns=['ema', 'ema_slope'], inplace=True)

    return pd.Series(bias, index=df.index)
