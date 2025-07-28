import os
import pandas as pd

# === PATH CONFIG ===
log_path = r"C:\Users\open\Documents\ZENO_XAUUSD\outputs\performance_logs\ZENO_trade_log.csv"

# === REQUIRED STRUCTURE ===
columns = [
    "Datetime", "Timeframe", "Direction", "Entry", "StopLoss", "TakeProfit",
    "Exit", "ExitReason", "Result", "R_Multiple", "Duration",
    "bias", "candle_pattern", "score", "confluences"
]

# === CREATE IF NOT EXISTS ===
if not os.path.exists(log_path):
    df = pd.DataFrame(columns=columns)
    df.to_csv(log_path, index=False)
    print(f"✅ Created empty trade journal at:\n{log_path}")
else:
    df = pd.read_csv(log_path)
    missing_cols = [c for c in columns if c not in df.columns]
    if missing_cols:
        for col in missing_cols:
            df[col] = None
        df.to_csv(log_path, index=False)
        print(f"⚠️ Existing log found, but missing columns were added:\n{missing_cols}")
    else:
        print(f"✅ Trade journal already exists and is complete:\n{log_path}")
