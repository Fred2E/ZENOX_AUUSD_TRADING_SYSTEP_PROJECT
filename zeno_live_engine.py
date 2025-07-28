import os
import pandas as pd
from structure_detector import detect_structure
from confluence_scanner import evaluate_confluence
from candle_patterns import detect_candle_patterns

base_path = r"C:\Users\open\Documents\ZENO_XAUUSD\data\live\XAUUSD"
timeframes = ["M5", "M15", "H1", "H4", "D1"]

def run_zeno_live_engine():
    all_signals = []

    for tf in timeframes:
        file = os.path.join(base_path, tf, f"XAUUSD_{tf}_LIVE.csv")
        print(f"⚙️  Processing {tf}: {file}")

        if not os.path.exists(file):
            print(f"❌ Missing file: {file}")
            continue

        try:
            df = pd.read_csv(file, parse_dates=["datetime"])
            df.columns = [c.lower() for c in df.columns]
            df = df.dropna(subset=["open", "high", "low", "close", "volume"])

            df = detect_structure(df)
            df = detect_candle_patterns(df)
            df = evaluate_confluence(df, tf)

            df['timeframe'] = tf
            a_plus = df[df['score'] >= 5]

            if not a_plus.empty:
                print(a_plus[['datetime', 'timeframe', 'close', 'confluences', 'score', 'candle_pattern']].tail(5))
                all_signals.append(a_plus)
            else:
                print(f"ℹ️ No A+ setups in {tf}")

        except Exception as e:
            print(f"❌ Error processing {tf}: {e}")

    if all_signals:
        result = pd.concat(all_signals)
        result.sort_values("datetime", ascending=False, inplace=True)
        print("\n✅ LIVE A+ SIGNALS (Top 10):")
        print(result[['datetime', 'timeframe', 'close', 'confluences', 'score', 'candle_pattern']].head(10))
    else:
        print("⚠️ No signals found across any timeframe.")

if __name__ == "__main__":
    run_zeno_live_engine()
