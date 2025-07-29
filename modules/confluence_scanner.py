import numpy as np
import pandas as pd

PRIMARY_CONFS = ['conf_structure', 'conf_bos_or_choch', 'conf_candle', 'conf_sr_zone']
SECONDARY_CONFS = ['conf_psych_level', 'conf_fib_zone', 'conf_volume', 'conf_liquidity', 'conf_spread']

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

def robust_sr_zone(df, lookback=20, threshold=0.0015):
    """Return 1 if price is within a tight % of rolling high/low (proxy for real S/R zone)."""
    sr_zone = []
    for i, row in df.iterrows():
        idx = max(0, i - lookback)
        window = df.iloc[idx:i+1]
        local_high = window['high'].max()
        local_low = window['low'].min()
        close = row['close']
        # Within 0.15% of recent high/low (tune as needed)
        near_high = abs(close - local_high) / close < threshold
        near_low = abs(close - local_low) / close < threshold
        sr_zone.append(int(near_high or near_low))
    return pd.Series(sr_zone, index=df.index)

def robust_candle_pattern(df):
    """Return 1 if a strong pattern is detected (engulfing, pinbar, etc.)—expand as needed."""
    # pattern_code already in df? 1+ is valid, else 0
    if 'pattern_code' in df.columns:
        return (df['pattern_code'].fillna(0).astype(int) > 0).astype(int)
    return pd.Series([0]*len(df), index=df.index)

def evaluate_confluence(df, timeframe):
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    N = len(df)

    # --- PRIMARY CONFLUENCES (REAL MARKET LOGIC) ---
    # Market structure: BOS/CHoCH detection
    bos_flag = _sanitize_direction(df.get('bos', pd.Series(['']*N)))
    choch_flag = _sanitize_direction(df.get('choch', pd.Series(['']*N)))
    conf_bos_or_choch = ((bos_flag != 0) | (choch_flag != 0)).astype(int)
    
    # Structure: require swing_high/low and BOS/CHoCH
    conf_structure = ((~df.get('swing_high', pd.Series([np.nan]*N)).isna()) | \
                      (~df.get('swing_low', pd.Series([np.nan]*N)).isna())) & (conf_bos_or_choch == 1)
    conf_structure = conf_structure.astype(int)

    # Candle pattern robust detection
    conf_candle = robust_candle_pattern(df)

    # S/R: is close near a recent local high/low (proxy for S/R, can replace with real OB/FVG detector)
    conf_sr_zone = robust_sr_zone(df, lookback=20, threshold=0.0015)

    # --- SECONDARY CONFLUENCES ---
    # Psych level: close near any round 50/100/250
    price = df['close']
    psych_level = ((price % 50 < 1.5) | (price % 50 > 48.5) |
                   (price % 100 < 2) | (price % 100 > 98) |
                   (price % 250 < 2.5) | (price % 250 > 247.5)).astype(int)

    # Fibonacci zone (approximate): If price is near 0.5, 0.618, or 0.382 between rolling swing_high and swing_low
    fib_zone = []
    if 'swing_high' in df.columns and 'swing_low' in df.columns:
        for i, row in df.iterrows():
            idx = max(0, i-20)
            sw_high = df.iloc[idx:i+1]['swing_high'].dropna()
            sw_low = df.iloc[idx:i+1]['swing_low'].dropna()
            close = row['close']
            found = 0
            if not sw_high.empty and not sw_low.empty:
                h, l = sw_high.max(), sw_low.min()
                diff = h - l
                fibs = [l + 0.382*diff, l + 0.5*diff, l + 0.618*diff]
                found = int(any(abs(close-f)/close < 0.01 for f in fibs))
            fib_zone.append(found)
    else:
        fib_zone = [0]*N
    fib_zone = pd.Series(fib_zone, index=df.index)

    # Volume spike: rolling mean
    conf_volume = (df['volume'] > df['volume'].rolling(20, min_periods=1).mean()).astype(int)
    conf_liquidity = (df['volume'].rolling(10, min_periods=1).sum() >
                      df['volume'].rolling(100, min_periods=1).sum().mean()).astype(int)

    # Spread: must already be present
    if 'spread' in df.columns:
        conf_spread = (df['spread'] < df['spread'].rolling(10, min_periods=1).mean()).astype(int)
    else:
        conf_spread = pd.Series([1]*N, index=df.index)

    # --- ASSIGN CONFLUENCE FLAGS ---
    df['conf_structure'] = conf_structure
    df['conf_bos_or_choch'] = conf_bos_or_choch
    df['conf_candle'] = conf_candle
    df['conf_sr_zone'] = conf_sr_zone
    df['conf_psych_level'] = psych_level
    df['conf_fib_zone'] = fib_zone
    df['conf_volume'] = conf_volume
    df['conf_liquidity'] = conf_liquidity
    df['conf_spread'] = conf_spread

    # === PRIMARY CONFLUENCE SCORING ===
    df['primary_score'] = df[PRIMARY_CONFS].sum(axis=1)
    df['secondary_score'] = df[SECONDARY_CONFS].sum(axis=1)
    df['total_confluence'] = df['primary_score'] + df['secondary_score']
    df['score'] = df['primary_score']
    df['num_confs'] = df[PRIMARY_CONFS + SECONDARY_CONFS].sum(axis=1)

    # --- ADD ATR ---
    df = compute_atr(df)

    # --- ADD TIME FEATURES ---
    if 'datetime' in df.columns:
        df['hour'] = pd.to_datetime(df['datetime'], errors='coerce').dt.hour
        df['dow'] = pd.to_datetime(df['datetime'], errors='coerce').dt.dayofweek
    else:
        df['hour'], df['dow'] = -1, -1

    # DEBUG: Print value counts for all confluences
    print("PRIMARY CONFS VALUE COUNTS:")
    print({c: df[c].sum() for c in PRIMARY_CONFS})
    print("SECONDARY CONFS VALUE COUNTS:")
    print({c: df[c].sum() for c in SECONDARY_CONFS})

    return df
