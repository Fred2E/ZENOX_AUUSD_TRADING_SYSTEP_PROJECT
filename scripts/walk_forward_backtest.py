import os
import pandas as pd

DATA_DIR = r"C:/Users/open/Documents/ZENO_XAUUSD/historical/processed"
LOGS_DIR = r"C:/Users/open/Documents/ZENO_XAUUSD/logs"
TIMEFRAMES = ["M5", "M15", "H1", "H4"]
SETUP_GRADES = ["A+", "A", "B"]  # <-- Choose what you want

PRIMARY_CONFS = ['conf_structure', 'conf_bos_or_choch', 'conf_candle', 'conf_sr_zone']
SECONDARY_CONFS = ['conf_psych_level', 'conf_fib_zone', 'conf_volume', 'conf_liquidity', 'conf_spread']

def confluence_scoring(row):
    primary_score = sum(int(row.get(c, 0)) for c in PRIMARY_CONFS)
    secondary_score = sum(int(row.get(c, 0)) for c in SECONDARY_CONFS)
    total_conf = primary_score + secondary_score
    return primary_score, secondary_score, total_conf

def get_regime_trend(row):
    if 'regime_trend' in row and not pd.isnull(row['regime_trend']):
        return row['regime_trend']
    elif 'bias_bull' in row and not pd.isnull(row['bias_bull']):
        return row['bias_bull']
    else:
        return None

def simulate_rule_based_trades(df):
    df = df.reset_index(drop=True)  # CRITICAL: Ensure index is 0...N-1
    trade_logs = []
    for i, row in df.iterrows():
        # Use only after setup_grade filtering
        primary_score, secondary_score, total_conf = confluence_scoring(row)
        entry = dict(row)
        entry['primary_score'] = primary_score
        entry['secondary_score'] = secondary_score
        entry['total_confluence'] = total_conf
        entry['total_conf'] = total_conf
        entry['regime_trend'] = get_regime_trend(row)
        entry['entry_time'] = row['datetime']
        entry['entry_index'] = i
        entry['entry_price'] = row['close']
        entry['direction'] = 'long' if row.get('bias_bull', 1) == 1 else 'short'
        atr = row['atr']
        if entry['direction'] == 'long':
            entry['stop_loss'] = entry['entry_price'] - atr
            entry['take_profit'] = entry['entry_price'] + 2 * atr
        else:
            entry['stop_loss'] = entry['entry_price'] + atr
            entry['take_profit'] = entry['entry_price'] - 2 * atr
        outcome, exit_price, reward, exit_time = 'breakeven', entry['entry_price'], 0, None
        for j in range(i+1, min(i+150, len(df))):
            high = df.iloc[j]['high']
            low = df.iloc[j]['low']
            if entry['direction'] == 'long':
                if low <= entry['stop_loss']:
                    outcome, exit_price, reward, exit_time = 'loss', entry['stop_loss'], entry['stop_loss'] - entry['entry_price'], df.iloc[j]['datetime']
                    break
                if high >= entry['take_profit']:
                    outcome, exit_price, reward, exit_time = 'win', entry['take_profit'], entry['take_profit'] - entry['entry_price'], df.iloc[j]['datetime']
                    break
            else:
                if high >= entry['stop_loss']:
                    outcome, exit_price, reward, exit_time = 'loss', entry['stop_loss'], entry['entry_price'] - entry['stop_loss'], df.iloc[j]['datetime']
                    break
                if low <= entry['take_profit']:
                    outcome, exit_price, reward, exit_time = 'win', entry['take_profit'], entry['entry_price'] - entry['take_profit'], df.iloc[j]['datetime']
                    break
        entry['exit_time'] = exit_time or row['datetime']
        entry['exit_price'] = exit_price
        entry['reward'] = reward
        entry['outcome'] = outcome
        trade_logs.append(entry)
    return trade_logs

def main():
    os.makedirs(LOGS_DIR, exist_ok=True)
    for tf in TIMEFRAMES:
        signal_file = os.path.join(DATA_DIR, f"signals_{tf}_FULL.csv")
        if not os.path.exists(signal_file):
            print(f"[SKIP] {tf}: Signal file missing ({signal_file})")
            continue
        df = pd.read_csv(signal_file)
        df.columns = [c.lower() for c in df.columns]
        if 'setup_grade' not in df.columns:
            print(f"[WARN] {tf}: 'setup_grade' missing; cannot filter for A+/A/B setups. Skipping.")
            continue
        df = df[df['setup_grade'].isin(SETUP_GRADES)]
        print(f"[{tf}] Filtering for setups: {SETUP_GRADES} — {len(df)} bars")
        if df.empty:
            print(f"[WARN] {tf}: No rows after setup_grade filter. Skipping.")
            continue
        trades = simulate_rule_based_trades(df)
        trade_log_df = pd.DataFrame(trades)
        trade_log_file = os.path.join(LOGS_DIR, f"trade_log_{tf}.csv")
        trade_log_df.to_csv(trade_log_file, index=False)
        print(f"[{tf}] Saved {len(trade_log_df)} trades to {trade_log_file}")

if __name__ == "__main__":
    main()
