import os
import pandas as pd
import requests
from modules.structure_detector import detect_structure
from modules.candle_patterns import detect_candle_patterns
from modules.confluence_scanner import evaluate_confluence

# === CONFIG ===
BOT_TOKEN = "7282917429:AAG66-KnJd7lMilPoxoi1thiZPnDFyez9aU"
CHAT_ID = "7598848875"
PROXY = {
    "http": "http://127.0.0.1:7890",
    "https": "http://127.0.0.1:7890"
}
SCORE_THRESHOLD = 5
DATA_ROOT = r"C:\Users\open\Documents\ZENO_XAUUSD\data\live\XAUUSD"
TIMEFRAMES = ["M5", "M15", "H1", "H4", "D1"]

# === ESCAPE FOR TELEGRAM MARKDOWN V2 ===
def escape_md(text):
    if text is None or pd.isna(text): return "N/A"
    text = str(text)
    escape_chars = r"_*[]()~`>#+-=|{}.!"
    return ''.join(['\\' + c if c in escape_chars else c for c in text])

# === TELEGRAM SEND ===
def send_signal(latest, tf):
    conf = latest.get("confluences", [])
    if isinstance(conf, str):
        try:
            conf = eval(conf)
        except:
            conf = [conf]
    if not isinstance(conf, list):
        conf = [str(conf)]
    conf_str = ", ".join(conf)

    try:
        msg = (
            f"*🔥 ZENO SIGNAL ACTIVE 🔥*\n\n"
            f"*Timeframe:* `{escape_md(tf)}`\n"
            f"*Datetime:* `{escape_md(str(latest.name))}`\n"
            f"*Direction:* `{escape_md(latest.get('direction'))}`\n"
            f"*Pattern:* `{escape_md(latest.get('candle_pattern'))}`\n"
            f"*Score:* `{escape_md(latest.get('score'))}`\n"
            f"*Entry:* `{escape_md(f'{latest.get('close', 0):.2f}')}`\n"
            f"*Stop Loss:* `{escape_md(f'{latest.get('stoploss', 0):.2f}')}`\n"
            f"*Take Profit:* `{escape_md(f'{latest.get('takeprofit', 0):.2f}')}`\n"
            f"*Confluences:* `{escape_md(conf_str)}`\n\n"
            f"_ZENO Auto Scanner \\(Alex\\)_"
        )

        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": CHAT_ID,
            "text": msg,
            "parse_mode": "MarkdownV2"
        }

        response = requests.post(url, data=payload, proxies=PROXY)
        if response.status_code == 200:
            print(f"✅ [{tf}] Signal sent to Telegram")
        else:
            print(f"❌ [{tf}] Failed to send:", response.text)

    except Exception as e:
        print(f"❌ [{tf}] Telegram error:", e)

# === MAIN LOOP ===
for tf in TIMEFRAMES:
    print(f"🔍 Scanning {tf}...")
    path = os.path.join(DATA_ROOT, tf, f"XAUUSD_{tf}_LIVE.csv")
    if not os.path.exists(path):
        print(f"❌ [{tf}] Data not found.")
        continue

    try:
        df = pd.read_csv(path, parse_dates=['datetime'])
        df.columns = [c.lower() for c in df.columns]
        df.set_index('datetime', inplace=True)
        df = detect_structure(df)
        df = detect_candle_patterns(df)
        df = evaluate_confluence(df, tf)

        if 'direction' not in df.columns:
            df['direction'] = df['bias']

        def compute_stop_loss(row):
            idx = df.index.get_loc(row.name)
            lookback = df.iloc[max(0, idx - 10):idx]
            if row['direction'] == 'bullish':
                lows = lookback['swing_low'].dropna()
                return lows.iloc[-1] if not lows.empty else None
            else:
                highs = lookback['swing_high'].dropna()
                return highs.iloc[-1] if not highs.empty else None

        def compute_take_profit(row):
            entry = row['close']
            sl = row['stoploss']
            if pd.isna(sl):
                return None
            risk = abs(entry - sl)
            return entry + 2 * risk if row['direction'] == 'bullish' else entry - 2 * risk

        df['stoploss'] = df.apply(compute_stop_loss, axis=1)
        df['takeprofit'] = df.apply(compute_take_profit, axis=1)
        df.dropna(subset=['stoploss', 'takeprofit'], inplace=True)
        df = df[df['score'] >= SCORE_THRESHOLD]

        if df.empty:
            print(f"⏳ [{tf}] No valid signals.")
            continue

        latest = df.iloc[-1]
        send_signal(latest, tf)

    except Exception as e:
        print(f"❌ [{tf}] Error:", e)
