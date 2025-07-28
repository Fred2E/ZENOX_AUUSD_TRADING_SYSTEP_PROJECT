import MetaTrader5 as mt5
import pandas as pd
import os
from datetime import datetime, timedelta, timezone

# === CONFIG ===
ACCOUNT = 76847754
PASSWORD = 'Lovetr!ck2000'
SERVER = 'Exness-MT5Trial5'
MT5_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe"
SYMBOL = 'XAUUSDm'
TIMEFRAMES = {
    'M5': mt5.TIMEFRAME_M5,
    'M15': mt5.TIMEFRAME_M15,
    'H1': mt5.TIMEFRAME_H1,
    'H4': mt5.TIMEFRAME_H4,
    'D1': mt5.TIMEFRAME_D1,
}
DATA_ROOT = r"C:\Users\open\Documents\ZENO_XAUUSD\data\live\XAUUSD"
CSV_TEMPLATE = "XAUUSD_{}_LIVE.csv"
UTC = timezone.utc

def login():
    mt5.shutdown()
    if not mt5.initialize(MT5_PATH, login=ACCOUNT, password=PASSWORD, server=SERVER):
        raise RuntimeError(f"MT5 login failed: {mt5.last_error()}")
    print("[INFO] MT5 login OK")

def fetch_and_merge(tf, gaps):
    csv_path = os.path.join(DATA_ROOT, tf, CSV_TEMPLATE.format(tf))
    df = pd.read_csv(csv_path, parse_dates=['datetime'])
    df.columns = [c.lower() for c in df.columns]
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    for gap_start, gap_end in gaps:
        # Get all missing bar times
        delta = {
            'M5': timedelta(minutes=5), 'M15': timedelta(minutes=15),
            'H1': timedelta(hours=1), 'H4': timedelta(hours=4),
            'D1': timedelta(days=1)
        }[tf]
        t = gap_start + delta
        bars_to_fetch = []
        while t <= gap_end - delta:
            bars_to_fetch.append(t)
            t += delta
        if not bars_to_fetch:
            continue
        print(f"[{tf}] Repairing {len(bars_to_fetch)} missing bars between {gap_start} and {gap_end}")
        # Download missing bars in one chunk if possible
        rates = mt5.copy_rates_range(SYMBOL, TIMEFRAMES[tf], bars_to_fetch[0], bars_to_fetch[-1] + delta)
        if rates is None or len(rates) == 0:
            print(f"[{tf}] Failed to fetch missing bars: {bars_to_fetch}")
            continue
        # Convert to DataFrame, UTC-localize
        fix_df = pd.DataFrame(rates)
        fix_df['datetime'] = pd.to_datetime(fix_df['time'], unit='s').dt.tz_localize('UTC')
        # Rename columns to lowercase for consistency
        fix_df.rename(columns={
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'tick_volume': 'volume'
        }, inplace=True)
        # Drop duplicates, align columns
        cols = [c for c in df.columns if c in fix_df.columns]
        fix_df = fix_df[cols]
        # Merge into original DataFrame, drop duplicates
        df = pd.concat([df, fix_df]).drop_duplicates('datetime').sort_values('datetime')
    # Save repaired CSV (lowercase columns only)
    df.to_csv(csv_path, index=False)
    print(f"[{tf}] Gap repair complete. Data now has {len(df)} rows. Last: {df['datetime'].iloc[-1]}")
    return df

def parse_gaps(gap_report):
    gaps_dict = {}
    for tf in gap_report:
        if gap_report[tf]:
            gaps = []
            for pair in gap_report[tf]:
                start, end = pair
                if isinstance(start, str):  # Convert to Timestamp
                    start = pd.Timestamp(start).tz_localize('UTC')
                    end = pd.Timestamp(end).tz_localize('UTC')
                gaps.append((start, end))
            gaps_dict[tf] = gaps
    return gaps_dict

def main():
    # --- Paste your gap report here (example) ---
    GAP_REPORT = {
        'M5': [(pd.Timestamp('2025-07-16 21:00:00+0000', tz='UTC'), pd.Timestamp('2025-07-16 21:55:00+0000', tz='UTC')),
               (pd.Timestamp('2025-07-17 21:00:00+0000', tz='UTC'), pd.Timestamp('2025-07-17 21:55:00+0000', tz='UTC'))],
        'M15': [...],  # Paste rest from your actual output!
        'H1':   [...],
        'H4':   [...],
        'D1':   [...],
    }
    # ----
    login()
    for tf, gaps in GAP_REPORT.items():
        if not gaps: continue
        fetch_and_merge(tf, gaps)
    mt5.shutdown()
    print("[DONE] Gap repair finished for all timeframes.")

if __name__ == "__main__":
    main()
