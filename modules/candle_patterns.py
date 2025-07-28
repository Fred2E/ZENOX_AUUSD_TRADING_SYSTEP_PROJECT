import numpy as np
import pandas as pd

PATTERN_CODES = {
    'none': 0,
    'bullish_engulfing': 1,
    'bearish_engulfing': 2,
    'hammer': 3,
    'inverted_hammer': 4,
    'doji': 5,
}

def detect_candle_patterns(df):
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    # Validate presence of OHLC columns
    required = ['open', 'high', 'low', 'close']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing OHLC column(s): {missing}")

    ohlc = df[required].astype(float)
    n = len(ohlc)
    patterns = np.zeros(n, dtype=int)  # Default pattern_code: 0 (none)

    prev_open = ohlc['open'].shift(1)
    prev_close = ohlc['close'].shift(1)

    # Bullish Engulfing: previous candle down, current up, engulf prev body
    bullish_engulf = (
        (prev_close < prev_open) &
        (ohlc['close'] > ohlc['open']) &
        (ohlc['open'] < prev_close) &
        (ohlc['close'] > prev_open)
    ).fillna(False)
    patterns[bullish_engulf.values] = PATTERN_CODES['bullish_engulfing']

    # Bearish Engulfing: previous candle up, current down, engulf prev body
    bearish_engulf = (
        (prev_close > prev_open) &
        (ohlc['close'] < ohlc['open']) &
        (ohlc['open'] > prev_close) &
        (ohlc['close'] < prev_open)
    ).fillna(False)
    patterns[bearish_engulf.values] = PATTERN_CODES['bearish_engulfing']

    # Hammer: green candle with long lower wick, ignoring already matched patterns
    real_body = np.abs(ohlc['close'] - ohlc['open'])
    lower_wick = ohlc['open'] - ohlc['low']
    upper_wick = ohlc['high'] - ohlc['close']
    hammer = (
        (ohlc['close'] > ohlc['open']) &
        (lower_wick > 2 * real_body) &
        (real_body > 0.0001) &
        (patterns == PATTERN_CODES['none'])
    ).fillna(False)
    patterns[hammer.values] = PATTERN_CODES['hammer']

    # Inverted Hammer: green candle with long upper wick, ignoring already matched patterns
    inv_hammer = (
        (ohlc['close'] > ohlc['open']) &
        (upper_wick > 2 * real_body) &
        (real_body > 0.0001) &
        (patterns == PATTERN_CODES['none'])
    ).fillna(False)
    patterns[inv_hammer.values] = PATTERN_CODES['inverted_hammer']

    # Doji: very small real body relative to candle length, ignoring already matched patterns
    doji = (
        (real_body < 0.1 * (ohlc['high'] - ohlc['low'])) &
        (patterns == PATTERN_CODES['none'])
    ).fillna(False)
    patterns[doji.values] = PATTERN_CODES['doji']

    df['pattern_code'] = patterns.astype(int)
    df['candle_pattern'] = [
        next((k for k, v in PATTERN_CODES.items() if v == code), 'none')
        for code in patterns
    ]

    return df

# Usage example:
# df = detect_candle_patterns(df)
