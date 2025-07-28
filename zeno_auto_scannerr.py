import logging
import time
import datetime
import pandas as pd
import os

from modules.mt5_bridge import MT5Bridge
from modules.structure_detector import detect_structure
from modules.candle_patterns import detect_candle_patterns
from modules.confluence_scanner import evaluate_confluence
from modules.ml_model import predict_ml
from modules.rl_model import rl_decision

# ─── CONFIG ──────────────────────────────────────────────────────────────────
ROOT = r"C:\Users\open\Documents\ZENO_XAUUSD"
DATA_ROOT = os.path.join(ROOT, "data", "live", "XAUUSD")
TIMEFRAMES = {"M5": 5, "M15": 15, "H1": 60, "H4": 240, "D1": 1440}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ─── SETUP ──────────────────────────────────────────────────────────────────
mt5 = MT5Bridge(max_spread=0.2)
mt5.initialize()

def next_bar_close(minutes_interval):
    now = datetime.datetime.now()
    delta = (now.minute % minutes_interval, now.second, now.microsecond)
    mb = minutes_interval - delta[0] - (delta[1] > 0)
    next_time = (now + datetime.timedelta(minutes=mb)).replace(second=1, microsecond=0)
    return next_time

def run_scan(tf):
    filepath = os.path.join(DATA_ROOT, tf, f"XAUUSD_{tf}_LIVE.csv")
    if not os.path.exists(filepath):
        logging.warning(f"❌ [{tf}] File missing: {filepath!r}, skipping")
        return

    df = pd.read_csv(filepath, parse_dates=["datetime"])
    df.columns = [c.lower() for c in df.columns]
    df.set_index("datetime", inplace=True)
    df = detect_structure(df)
    df = detect_candle_patterns(df)
    df = evaluate_confluence(df, tf)
    df = predict_ml(df)
    df = df[df["prob_win"] >= 0.39]

    if df.empty:
        logging.info(f"⏳ [{tf}] No high-prob signals.")
        return

    latest = df.iloc[-1]
    action = rl_decision(latest["close"], latest["score"])
    logging.info(f"⚙️ [{tf}] RL action={action} (0=hold,1=buy,2=sell)")

    if action != 0:
        price = latest['close']
        stoploss = price - 0.002 * price if action == 1 else price + 0.002 * price
        takeprofit = price + 0.004 * price if action == 1 else price - 0.004 * price
        ok, msg = mt5.place_order('XAUUSDm', 'buy' if action == 1 else 'sell', 1.0, price, stoploss, takeprofit)
        if ok:
            logging.info(f"✅ [{tf}] Order placed: {msg}")
        else:
            logging.error(f"❌ [{tf}] Order failed: {msg}")

# ─── MAIN LOOP ────────────────────────────────────────────────────────────────
def main():
    try:
        while True:
            schedule = {tf: next_bar_close(int(mins)) for tf, mins in TIMEFRAMES.items()}
            tf, wakeup = min(schedule.items(), key=lambda kv: kv[1])
            wait = (wakeup - datetime.datetime.now()).total_seconds()
            logging.info(f"Sleeping {wait:.1f}s until next {tf} bar-close at {wakeup}")
            if wait > 0:
                time.sleep(wait)
            run_scan(tf)
    except KeyboardInterrupt:
        logging.info("👋 Interrupted by user. Exiting.")
    finally:
        mt5.shutdown()

if __name__ == "__main__":
    main()
