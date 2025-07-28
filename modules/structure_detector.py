import numpy as np
import pandas as pd

def detect_structure(df, swing_window=2):
    """
    Institutional-level swing structure, bias, BOS, and CHoCH detection for XAUUSD.
    Returns a DataFrame with columns: [swing_high, swing_low, bos, choch, bias, bias_label].
    Output is fully lowercase for pipeline consistency.

    Args:
        df (pd.DataFrame): DataFrame with at least 'high', 'low', and (optional) 'datetime' columns.
        swing_window (int): Number of bars to look back/forward for swing detection.

    Returns:
        pd.DataFrame: With new structure/logic columns added.
    """

    # --- Preprocess and preserve datetime ---
    df.columns = [c.lower() for c in df.columns]
    datetime_series = df['datetime'].copy() if 'datetime' in df.columns else None

    # --- Initialize structure/logic columns ---
    df['swing_high'] = np.nan      # Price at confirmed swing high
    df['swing_low'] = np.nan       # Price at confirmed swing low
    df['bos'] = 0                  # Break of structure: 1 (bull), -1 (bear), 0 (none)
    df['choch'] = 0                # Change of character: 1 (bullish reversal), -1 (bearish reversal), 0 (none)
    df['bias'] = 1                 # Market regime: 1=bullish, -1=bearish (tracks real flow, not just local swing)
    df['bias_label'] = 'bullish'   # Human-readable bias

    # --- State variables for tracking last swings and regime ---
    last_high = last_low = None
    bias = 1  # Start with bullish context by default

    # --- Main loop: Institutional swing logic ---
    for i in range(swing_window, len(df) - swing_window):
        # Get local window of highs/lows for swing detection
        window = df.iloc[i - swing_window:i + swing_window + 1]
        high = df['high'].iloc[i]
        low = df['low'].iloc[i]

        # --- Identify new swing high ---
        if high == window['high'].max():
            df.at[df.index[i], 'swing_high'] = high

        # --- Identify new swing low ---
        if low == window['low'].min():
            df.at[df.index[i], 'swing_low'] = low

        # --- BOS/CHoCH strict logic ---
        # On new swing high:
        if not pd.isna(df.at[df.index[i], 'swing_high']):
            if bias == -1 and last_high is not None and high > last_high:
                # Bearish → Bullish reversal (CHoCH up)
                df.at[df.index[i], 'choch'] = 1
                bias = 1
                df.at[df.index[i], 'bias_label'] = 'bullish'
            elif bias == 1 and last_high is not None and high > last_high:
                # Bullish continuation (BOS up)
                df.at[df.index[i], 'bos'] = 1
            last_high = high

        # On new swing low:
        if not pd.isna(df.at[df.index[i], 'swing_low']):
            if bias == 1 and last_low is not None and low < last_low:
                # Bullish → Bearish reversal (CHoCH down)
                df.at[df.index[i], 'choch'] = -1
                bias = -1
                df.at[df.index[i], 'bias_label'] = 'bearish'
            elif bias == -1 and last_low is not None and low < last_low:
                # Bearish continuation (BOS down)
                df.at[df.index[i], 'bos'] = -1
            last_low = low

        # Track current bias in main column for use in other modules
        df.at[df.index[i], 'bias'] = bias

    # --- Ensure all columns are correct datatype for downstream ML/RL/analysis ---
    df['bos'] = df['bos'].astype(int)
    df['choch'] = df['choch'].astype(int)
    df['bias'] = df['bias'].astype(int)
    df['bias_label'] = df['bias_label'].astype(str)

    # --- Restore datetime column if previously present and missing (robust) ---
    if datetime_series is not None and 'datetime' not in df.columns:
        df['datetime'] = datetime_series

    # --- Lowercase all output columns for perfect pipeline compatibility ---
    df.columns = [c.lower() for c in df.columns]
    return df

# Usage:
# df = detect_structure(df, swing_window=2)
