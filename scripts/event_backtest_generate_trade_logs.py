import os
import pandas as pd
import numpy as np

# =========== CONFIG =============
ROOT = r"C:\Users\open\Documents\ZENO_XAUUSD"
RAW_SIGNAL_DIR = os.path.join(ROOT, "outputs", "ml_data")  # Adjust if your signal CSVs are elsewhere
OUT_DIR = os.path.join(ROOT, "outputs", "trade_logs")
os.makedirs(OUT_DIR, exist_ok=True)
TIMEFRAMES = ["M5", "M15", "H1", "H4"]

# You must use your actual event-driven signal CSVs.
# For this template, assumes the format: trade_events_{TF}_FULL.csv

REQUIRED_COLS = [
    'datetime', 'open', 'high', 'low', 'close', 'volume', 'spread',
    'swing_high', 'swing_low', 'bos', 'choch', 'bias', 'bias_label', 'pattern_code',
    'candle_pattern', 'bias_bull', 'conf_structure', 'conf_bos_or_choch', 'conf_candle', 'conf_sr_zone',
    'conf_psych_level', 'conf_fib_zone', 'conf_volume', 'conf_liquidity', 'conf_spread',
    'primary_score', 'secondary_score', 'total_confluence', 'score', 'num_confs',
    'atr', 'hour', 'dow', 'regime_trend', 'setup_grade'
]

# These will be added per trade
TRADE_FIELDS = [
    'entry_time', 'entry_index', 'entry_price', 'direction', 'stop_loss', 'take_profit',
    'exit_time', 'exit_price', 'reward', 'outcome'
]

def simulate_trades(df, tf):
    """Minimal event-driven backtest. Replace with your advanced logic for live trading."""
    trades = []
    position = 0
    entry_idx = None
    entry_price = None
    stop_loss = None
    take_profit = None
    setup_grade = None
    atr_mult = 1.5  # Just an example

    for i, row in df.iterrows():
        # Simple entry condition: confluence + not in trade
        if position == 0 and row['score'] >= 2 and row['num_confs'] >= 3:
            position = 1 if row.get('bias_bull', 1) else -1
            entry_idx = i
            entry_price = row['close']
            stop_loss = entry_price - atr_mult * row['atr'] if position == 1 else entry_price + atr_mult * row['atr']
            take_profit = entry_price + 2 * (entry_price - stop_loss) if position == 1 else entry_price - 2 * (stop_loss - entry_price)
            setup_grade = row.get('setup_grade', 'A+')
            entry_row = row

        # Simple exit condition: TP/SL hit
        if position != 0:
            # Check SL/TP
            price = row['close']
            sl_hit = (price <= stop_loss) if position == 1 else (price >= stop_loss)
            tp_hit = (price >= take_profit) if position == 1 else (price <= take_profit)
            last = (i == len(df) - 1)

            if sl_hit or tp_hit or last:
                exit_price = price
                exit_time = row['datetime']
                reward = (exit_price - entry_price) * position  # Not pips, just price diff. Adjust as needed.
                outcome = 'win' if (tp_hit and not sl_hit) else ('loss' if sl_hit else 'breakeven')
                trade = {**{col: entry_row[col] for col in REQUIRED_COLS}, **{
                    'entry_time': entry_row['datetime'],
                    'entry_index': entry_idx,
                    'entry_price': entry_price,
                    'direction': 'long' if position == 1 else 'short',
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'exit_time': exit_time,
                    'exit_price': exit_price,
                    'reward': reward,
                    'outcome': outcome,
                }}
                trades.append(trade)
                position = 0

    return pd.DataFrame(trades, columns=REQUIRED_COLS + TRADE_FIELDS)

# ==== MAIN LOOP ====
for tf in TIMEFRAMES:
    raw_path = os.path.join(RAW_SIGNAL_DIR, f"trade_events_{tf}_FULL.csv")
    out_path = os.path.join(OUT_DIR, f"trade_log_{tf}.csv")
    if not os.path.exists(raw_path):
        print(f"[SKIP] {tf}: Raw event file not found: {raw_path}")
        continue
    df = pd.read_csv(raw_path)
    # Defensive: Lowercase cols
    df.columns = [c.lower() for c in df.columns]
    missing = set(REQUIRED_COLS) - set(df.columns)
    if missing:
        print(f"[FAIL] {tf}: Missing columns: {missing}")
        continue
    # Filter to A+/A/B only (if setup_grade present)
    if 'setup_grade' in df.columns:
        df = df[df['setup_grade'].isin(['A+', 'A', 'B'])]
    trade_log = simulate_trades(df, tf)
    # Save with header and assert shape
    trade_log.to_csv(out_path, index=False)
    print(f"[OK] {tf}: Trade log saved. Shape={trade_log.shape} → {out_path}")

print("\n✅ ALL DONE. Check outputs/trade_logs for trade logs for RL/backtest/statistics.")
