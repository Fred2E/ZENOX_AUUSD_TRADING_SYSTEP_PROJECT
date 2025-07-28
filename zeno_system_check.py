import os
import pandas as pd
import logging
from datetime import datetime, timezone
from modules.feature_pipeline import engineer_features
from modules.ml_model import predict_ml
from modules.rl_model import rl_decision

# CONFIG
DATA_ROOT = os.path.join("data", "live", "XAUUSD")
TIMEFRAMES = ["M5", "M15", "H1", "H4", "D1"]
CSV_TEMPLATE = "XAUUSD_{}_LIVE.csv"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def load_latest_bar(tf):
    path = os.path.join(DATA_ROOT, tf, CSV_TEMPLATE.format(tf))
    if not os.path.exists(path):
        return None, f"[{tf}] MISSING: {path}"
    try:
        df = pd.read_csv(path, parse_dates=['datetime'])
        df.columns = [c.lower() for c in df.columns]
        if df.empty:
            return None, f"[{tf}] EMPTY FILE"
        bar = df.iloc[-1].copy()
        return bar, None
    except Exception as e:
        return None, f"[{tf}] LOAD ERROR: {str(e)}"

def check_and_process(tf):
    bar, err = load_latest_bar(tf)
    if err:
        return None, err
    df = pd.DataFrame([bar])
    df.columns = [c.lower() for c in df.columns]
    df, features = engineer_features(df)
    try:
        df = predict_ml(df)
        prob_win = float(df["prob_win"].iloc[-1])
        score = float(df["score"].iloc[-1]) if "score" in df.columns else None
        close = float(df["close"].iloc[-1]) if "close" in df.columns else None
        action = rl_decision(close, score)
        return {
            "tf": tf,
            "datetime": str(df["datetime"].iloc[-1]),
            "close": close,
            "prob_win": prob_win,
            "score": score,
            "action": int(action),
        }, None
    except Exception as e:
        return None, f"[{tf}] PROCESSING ERROR: {str(e)}"

def main():
    logging.info("==== ZENO SYSTEM CHECK & ACTION ====")
    issues = []
    outputs = []
    for tf in TIMEFRAMES:
        out, err = check_and_process(tf)
        if err:
            issues.append(err)
        else:
            outputs.append(out)
    if issues:
        logging.error("❌ SYSTEM CHECK FAILED:")
        for i in issues:
            logging.error(i)
        logging.error("Aborting execution. Fix the above and rerun.")
        return
    # Print/Log All Results
    for out in outputs:
        logging.info(
            f"[{out['tf']}] {out['datetime']} | Close={out['close']} | prob_win={out['prob_win']:.3f} | score={out['score']} | RL action={out['action']}"
        )
    # Optionally: send Telegram alert, log to DB, trigger dashboard, etc.
    logging.info("✅ SYSTEM CHECK PASSED: All timeframes processed.")
    # If you want: Insert trading logic, dashboard refresh, or Telegram signal here

if __name__ == "__main__":
    main()
