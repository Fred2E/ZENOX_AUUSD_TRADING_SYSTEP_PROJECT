import numpy as np
import pandas as pd

PATTERN_CODES = {
    'none': 0,
    'bullish_engulfing': 1,
    'bearish_engulfing': 2,
    'hammer': 3,
    'inverted_hammer': 4,
    'doji': 5,
    'shooting_star': 6,
    'morning_star': 7,
    'evening_star': 8,
}

# Adaptive thresholds by timeframe
CANDLE_THRESH = {
    'M5':  {'wick_mult': 1.0, 'body_pct': 0.08, 'doji_body': 0.18},
    'M15': {'wick_mult': 1.2, 'body_pct': 0.10, 'doji_body': 0.16},
    'H1':  {'wick_mult': 1.5, 'body_pct': 0.12, 'doji_body': 0.14},
    'H4':  {'wick_mult': 1.7, 'body_pct': 0.13, 'doji_body': 0.12},
    'D1':  {'wick_mult': 2.0, 'body_pct': 0.15, 'doji_body': 0.10},
}

def detect_candle_patterns(df, tf='M5'):
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    params = CANDLE_THRESH.get(tf, CANDLE_THRESH['M15'])

    required = ['open', 'high', 'low', 'close']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing OHLC column(s): {missing}")

    ohlc = df[required].astype(float)
    n = len(ohlc)
    patterns = np.zeros(n, dtype=int)  # Default: none

    prev_open = ohlc['open'].shift(1)
    prev_close = ohlc['close'].shift(1)
    prev_low = ohlc['low'].shift(1)
    prev_high = ohlc['high'].shift(1)
    next_open = ohlc['open'].shift(-1)
    next_close = ohlc['close'].shift(-1)

    real_body = np.abs(ohlc['close'] - ohlc['open'])
    candle_length = ohlc['high'] - ohlc['low']
    upper_wick = ohlc['high'] - ohlc[['close', 'open']].max(axis=1)
    lower_wick = ohlc[['close', 'open']].min(axis=1) - ohlc['low']

    # === Bullish Engulfing ===
    bullish_engulf = (
        (prev_close < prev_open) &
        (ohlc['close'] > ohlc['open']) &
        (ohlc['close'] > prev_open) &
        (ohlc['open'] < prev_close)
    ).fillna(False)
    patterns[bullish_engulf.values] = PATTERN_CODES['bullish_engulfing']

    # === Bearish Engulfing ===
    bearish_engulf = (
        (prev_close > prev_open) &
        (ohlc['close'] < ohlc['open']) &
        (ohlc['close'] < prev_open) &
        (ohlc['open'] > prev_close)
    ).fillna(False)
    patterns[bearish_engulf.values] = PATTERN_CODES['bearish_engulfing']

    # === Hammer (bullish) ===
    hammer = (
        (ohlc['close'] > ohlc['open']) &
        (lower_wick > params['wick_mult'] * real_body) &
        (upper_wick < real_body) &
        (real_body > params['body_pct'] * candle_length) &
        (patterns == PATTERN_CODES['none'])
    ).fillna(False)
    patterns[hammer.values] = PATTERN_CODES['hammer']

    # === Inverted Hammer (bullish reversal) ===
    inv_hammer = (
        (ohlc['close'] > ohlc['open']) &
        (upper_wick > params['wick_mult'] * real_body) &
        (lower_wick < real_body) &
        (real_body > params['body_pct'] * candle_length) &
        (patterns == PATTERN_CODES['none'])
    ).fillna(False)
    patterns[inv_hammer.values] = PATTERN_CODES['inverted_hammer']

    # === Shooting Star (bearish hammer) ===
    shooting_star = (
        (ohlc['close'] < ohlc['open']) &
        (upper_wick > params['wick_mult'] * real_body) &
        (lower_wick < real_body) &
        (real_body > params['body_pct'] * candle_length) &
        (patterns == PATTERN_CODES['none'])
    ).fillna(False)
    patterns[shooting_star.values] = PATTERN_CODES['shooting_star']

    # === Doji ===
    doji = (
        (real_body < params['doji_body'] * candle_length) &
        (upper_wick > 0.2 * candle_length) &
        (lower_wick > 0.2 * candle_length) &
        (patterns == PATTERN_CODES['none'])
    ).fillna(False)
    patterns[doji.values] = PATTERN_CODES['doji']

    # === Morning Star (bullish reversal: 3-bar) ===
    morning_star = (
        (prev_close < prev_open) &
        (real_body > 0.1 * candle_length) &
        (ohlc['close'] > ohlc['open']) &
        (next_close > next_open) &
        (ohlc['close'] > prev_open) &
        (patterns == PATTERN_CODES['none'])
    ).fillna(False)
    patterns[morning_star.values] = PATTERN_CODES['morning_star']

    # === Evening Star (bearish reversal: 3-bar) ===
    evening_star = (
        (prev_close > prev_open) &
        (real_body > 0.1 * candle_length) &
        (ohlc['close'] < ohlc['open']) &
        (next_close < next_open) &
        (ohlc['close'] < prev_open) &
        (patterns == PATTERN_CODES['none'])
    ).fillna(False)
    patterns[evening_star.values] = PATTERN_CODES['evening_star']

    # Assign pattern labels
    df['pattern_code'] = patterns.astype(int)
    df['candle_pattern'] = [
        next((k for k, v in PATTERN_CODES.items() if v == code), 'none')
        for code in patterns
    ]

    # === Debug output
    print(f"[{tf}] Candle patterns value counts:", pd.Series(patterns).value_counts().to_dict())

    return df

# Usage:
# df = detect_candle_patterns(df, tf='M15')
