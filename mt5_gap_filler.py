import MetaTrader5 as mt5
import pandas as pd
import os
from datetime import datetime, timedelta, timezone

# === CONFIG ===
LOGIN = 76847754
PASSWORD = "Lovetr!ck2000"
SERVER = "Exness-MT5Trial5"
SYMBOL = "XAUUSDm"
DATA_ROOT = os.path.join("data", "live", "XAUUSD")
TIMEFRAMES = {
    "M5": (mt5.TIMEFRAME_M5, 5),
    "M15": (mt5.TIMEFRAME_M15, 15),
    "H1": (mt5.TIMEFRAME_H1, 60),
    "H4": (mt5.TIMEFRAME_H4, 240),
    "D1": (mt5.TIMEFRAME_D1, 1440)
}
CSV_TEMPLATE = "XAUUSD_{}_LIVE.csv"

def fetch_and_patch(tf, gaps):
    tf_code, minutes = TIMEFRAMES[tf]
    path = os.path.join(DATA_ROOT, tf, CSV_TEMPLATE.format(tf))
    print(f"[{tf}] Patching {len(gaps)} gaps...")
    df = pd.read_csv(path, parse_dates=['datetime'])
    df.columns = [c.lower() for c in df.columns]
    for start, end, _ in gaps:
        dt = start + timedelta(minutes=minutes)
        while dt < end:
            rates = mt5.copy_rates_range(SYMBOL, tf_code, dt, dt + timedelta(minutes=minutes))
            if rates is not None and len(rates) > 0:
                bar = rates[0]
                bar_dt = datetime.utcfromtimestamp(bar['time']).replace(tzinfo=timezone.utc)
                data = {
                    "datetime": [bar_dt],
                    "open": [bar['open']],
                    "high": [bar['high']],
                    "low": [bar['low']],
                    "close": [bar['close']],
                    "volume": [bar['tick_volume']]
                }
                new_row = pd.DataFrame(data)
                df = pd.concat([df, new_row], ignore_index=True)
                print(f"  Patched {bar_dt}")
            else:
                print(f"  [WARNING] Could not fetch bar at {dt} for {tf}.")
            dt += timedelta(minutes=minutes)
    df = df.sort_values("datetime").drop_duplicates("datetime")
    df.to_csv(path, index=False)
    print(f"[{tf}] Patch complete. Saved {path}")

def detect_gaps(tf):
    path = os.path.join(DATA_ROOT, tf, CSV_TEMPLATE.format(tf))
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path, parse_dates=['datetime'])
    df.columns = [c.lower() for c in df.columns]
    df = df.sort_values('datetime')
    minutes = TIMEFRAMES[tf][1]
    gaps = []
    for i in range(1, len(df)):
        diff = (df.iloc[i]['datetime'] - df.iloc[i-1]['datetime']).total_seconds() / 60
        if diff > minutes * 1.5:
            gaps.append((df.iloc[i-1]['datetime'], df.iloc[i]['datetime'], diff))
    return gaps

def main():
    print("=== MT5 GAP FILLER ===")
    if not mt5.initialize(login=LOGIN, password=PASSWORD, server=SERVER):
        print("MT5 initialize() failed:", mt5.last_error())
        return
    for tf in TIMEFRAMES:
        gaps = detect_gaps(tf)
        if gaps:
            fetch_and_patch(tf, gaps)
        else:
            print(f"[{tf}] No gaps found.")
    mt5.shutdown()
    print("=== GAP FILLING COMPLETE ===")
    # Final check:
    print("=== POST-PATCH GAP CHECK ===")
    for tf in TIMEFRAMES:
        gaps = detect_gaps(tf)
        if not gaps:
            print(f"[{tf}] ✅ NO GAPS. DATA GOOD.")
        else:
            print(f"[{tf}] ❌ STILL HAS GAPS: {gaps}")

if __name__ == "__main__":
    main()
