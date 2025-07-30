# mt5_live_export.py

import MetaTrader5 as mt5
import pandas as pd
import os
from datetime import datetime

# ---- CONFIG ----
LOGIN    = 76847754
PASSWORD = 'Lovetr!ck2000'
SERVER   = 'Exness-MT5Trial5'
MT5_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe"
SYMBOL   = 'XAUUSDm'
TIMEFRAMES = {
    'M5':  mt5.TIMEFRAME_M5,
    'M15': mt5.TIMEFRAME_M15,
    'H1':  mt5.TIMEFRAME_H1,
    'H4':  mt5.TIMEFRAME_H4,
    'D1':  mt5.TIMEFRAME_D1,
}
N_BARS = 500

def ensure_dirs():
    for tf in TIMEFRAMES:
        outdir = os.path.join("data", "live", "XAUUSD", tf)
        os.makedirs(outdir, exist_ok=True)

def fetch_and_save():
    ensure_dirs()
    if not mt5.initialize(MT5_PATH, login=LOGIN, password=PASSWORD, server=SERVER):
        print("[CRITICAL] MT5 init failed:", mt5.last_error())
        return
    print("[INFO] Connected to MT5")

    for tf_name, tf_code in TIMEFRAMES.items():
        print(f"[INFO] Fetching {N_BARS} bars: {SYMBOL} {tf_name}")
        bars = mt5.copy_rates_from_pos(SYMBOL, tf_code, 0, N_BARS)
        if bars is None or len(bars) == 0:
            print(f"[ERROR] No data for {SYMBOL} {tf_name}. Skipping.")
            continue

        df = pd.DataFrame(bars)
        df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
        df.rename(columns={
            'open':'open', 'high':'high', 'low':'low', 'close':'close', 'tick_volume':'volume'
        }, inplace=True)
        # Always use lowercase column order for pipeline compatibility
        df = df[['datetime','open','high','low','close','volume']]

        outfile = os.path.join("data", "live", "XAUUSD", tf_name, f"XAUUSD_{tf_name}_LIVE.csv")
        df.to_csv(outfile, index=False)
        print(f"[SUCCESS] {tf_name}: Saved {len(df)} rows to {outfile} | Last: {df.iloc[-1].datetime}")

    mt5.shutdown()

if __name__ == "__main__":
    fetch_and_save()
