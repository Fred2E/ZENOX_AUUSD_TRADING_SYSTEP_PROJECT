import pandas as pd
import numpy as np

TF_BIAS_PARAMS = {
    'M5':  {'short_ema': 13,  'long_ema': 34},     # Fast regime for scalping
    'M15': {'short_ema': 21,  'long_ema': 50},
    'H1':  {'short_ema': 21,  'long_ema': 100},
    'H4':  {'short_ema': 50,  'long_ema': 200},    # Slow regime for high tf bias
    'D1':  {'short_ema': 100, 'long_ema': 200},
}

def compute_bias_bull(df: pd.DataFrame, tf='M5') -> pd.Series:
    """
    Computes directional bias (1 = bullish regime, 0 = bearish regime) using EMA crossover.
    Parameters adapt by timeframe for realistic, institutional confluence logic.
    """
    if 'close' not in df.columns:
        raise ValueError("Missing 'close' column for bias computation.")

    # Get params for this timeframe
    params = TF_BIAS_PARAMS.get(tf, {'short_ema': 21, 'long_ema': 50})
    short_ema = params['short_ema']
    long_ema = params['long_ema']

    df = df.copy()
    df['ema_fast'] = df['close'].ewm(span=short_ema, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=long_ema, adjust=False).mean()
    bias_bull = (df['ema_fast'] > df['ema_slow']).astype(int)

    # For future: plug in structure-based bias for specific tfs if needed

    df.drop(columns=['ema_fast', 'ema_slow'], inplace=True)
    return pd.Series(bias_bull, index=df.index)

# Example usage in pipeline:
# df['bias_bull'] = compute_bias_bull(df, tf='M5')
