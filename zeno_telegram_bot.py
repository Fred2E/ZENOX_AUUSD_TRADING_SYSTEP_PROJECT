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

# === ESCAPE MARKDOWN FOR TELEGRAM ===
def escape_md(text):
    if text is None or pd.isna(text):
        return "N/A"
    text = str(text)
    chars_to_escape = r"_*[]()~`>#+-=|{}.!"
    return ''.join(['\\' + c if c in chars_to_escape else c for c in text])

def escape_code(val):
    return f"`{escape_md(val)}`"

# === TELEGRAM SEND ===
def send_signal(latest, tf):
    confluences = latest.get("confluences", [])
    if isinstance(confluences, str):
        try:
            confluences = eval(confluences)
        except:
            confluences = [confluences]
    if not isinstance(confluences, list):
        confluences = [str(confluences)]
    conf_str = ', '.join([escape_md(str(c)) for c in confluences])

    # Extract everything with safe fallback for missing keys
    def safe_get(col, default="N/A"):
        return latest.get(col, latest.get(col.lower(), default))

    try:
        msg = (
            f"*🔥 ZENO SIGNAL ACTIVE 🔥*\n\n"
            f"*Timeframe:* {escape_code(tf)}\n"
            f"*Datetime:* {escape_code(str(latest.name))}\n"
            f"*Direction:* {escape_code(safe_get('direction'))}\n"
            f"*Bias:* {escape_code(safe_get('bias'))}\n"
            f"*Entry:* {escape_code(f'{safe_get('close', 0):.2f}')}\n"
            f"*Stop Loss:* {escape_code(f'{safe_get('stoploss', 0):.2f}')}\n"
            f"*Take Profit:* {escape_code(f'{safe_get('takeprofit', 0):.2f}')}\n"
            f"*Pattern:* {escape_code(safe_get('candle_pattern'))}\n"
            f"*Score:* {escape_code(safe_get('score'))}\n"
            f"*Confluences:* {escape_code(conf_str)}\n\n"
            f"_ZENO Auto Scanner (Alex)_"
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
            print(f"❌ [{tf}] Failed to send signal:", response.text)
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
        df = pd.read_csv(path, parse_dates=['Datetime'])
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
