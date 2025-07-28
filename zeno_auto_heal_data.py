# zeno_auto_heal_data.py

import os
import pandas as pd
from datetime import datetime, timedelta, timezone
import logging
import MetaTrader5 as mt5

# --- CONFIG ---
DATA_ROOT = os.path.join("data", "live", "XAUUSD")
TIMEFRAMES = ["M5", "M15", "H1", "H4", "D1"]
CSV_TEMPLATE = "XAUUSD_{}_LIVE.csv"
MT5_SYMBOL = "XAUUSDm"
MT5_TIMEFRAMES = {
    "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15, "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4, "D1": mt5.TIMEFRAME_D1
}
MAX_GAP_MINUTES = {"M5": 5, "M15": 15, "H1": 60, "H4": 240, "D1": 1440}
BROKER_LOGIN = 76847754
BROKER_PASSWORD = "Lovetr!ck2000"
BROKER_SERVER = "Exness-MT5Trial5"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def fetch_bars(tf, start, end):
    tf_mt5 = MT5_TIMEFRAMES[tf]
    rates = mt5.copy_rates_range(MT5_SYMBOL, tf_mt5, start, end)
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df.rename(columns={
        'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'tick_volume': 'volume'
    }, inplace=True)
    return df[['datetime', 'open', 'high', 'low', 'close', 'volume']]

def fill_gaps(tf):
    path = os.path.join(DATA_ROOT, tf, CSV_TEMPLATE.format(tf))
    if not os.path.exists(path):
        logging.warning(f"[{tf}] File missing: {path}. Skipping.")
        return
    df = pd.read_csv(path, parse_dates=['datetime'])
    df.columns = [c.lower() for c in df.columns]
    df = df.sort_values('datetime')
    all_ok = True
    for i in range(1, len(df)):
        prev = df.iloc[i-1]['datetime']
        curr = df.iloc[i]['datetime']
        expected = prev + timedelta(minutes=MAX_GAP_MINUTES[tf])
        if curr > expected:
            # Gap detected
            logging.warning(f"[{tf}] Gap detected: {prev} -> {curr} (expected {expected})")
            # Attempt to fetch missing bars
            mt5.initialize(login=BROKER_LOGIN, password=BROKER_PASSWORD, server=BROKER_SERVER)
            patch = fetch_bars(tf, expected, curr - timedelta(minutes=MAX_GAP_MINUTES[tf]))
            mt5.shutdown()
            if patch is not None and not patch.empty:
                logging.info(f"[{tf}] Filling gap with {len(patch)} bars from MT5...")
                df = pd.concat([df.iloc[:i], patch, df.iloc[i:]], ignore_index=True)
                df = df.drop_duplicates('datetime').sort_values('datetime').reset_index(drop=True)
            else:
                logging.error(f"[{tf}] Could not fetch/fill missing bars. Manual fix required.")
                all_ok = False
    # Check staleness
    now = datetime.utcnow().replace(tzinfo=timezone.utc)
    last = df['datetime'].iloc[-1]
    minutes_since_last = (now - last).total_seconds() / 60
    if minutes_since_last > MAX_GAP_MINUTES[tf] * 2:
        logging.warning(f"[{tf}] Last bar too old: {last} (now: {now})")
        all_ok = False
    # Always save lowercase columns
    df.columns = [c.lower() for c in df.columns]
    df.to_csv(path, index=False)
    if all_ok:
        logging.info(f"[{tf}] Final continuity OK. Data patched and saved.")
    else:
        logging.error(f"[{tf}] Data is NOT fully valid after patch attempt.")

def main():
    for tf in TIMEFRAMES:
        fill_gaps(tf)

if __name__ == "__main__":
    main()
