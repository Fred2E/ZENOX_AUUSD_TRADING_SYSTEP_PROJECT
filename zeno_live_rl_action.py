# zeno_live_rl_action.py

import os
import pandas as pd

LIVE_DATA_ROOT = os.path.join("data", "live", "XAUUSD")
TIMEFRAMES = ["M5", "M15", "H1", "H4", "D1"]

def decide_action(row):
    # Replace with your real RL policy logic
    if 'prob_win' in row and row['prob_win'] > 0.5:
        return 'BUY'
    else:
        return 'HOLD'

def main():
    for tf in TIMEFRAMES:
        pred_path = os.path.join(LIVE_DATA_ROOT, tf, f"XAUUSD_{tf}_LIVE_PRED.csv")
        rl_out_path = os.path.join(LIVE_DATA_ROOT, tf, f"XAUUSD_{tf}_LIVE_RL_ACTION.csv")
        if not os.path.exists(pred_path):
            print(f"[SKIP] {tf}: Prediction file not found: {pred_path}")
            continue
        df = pd.read_csv(pred_path)
        df['rl_action'] = df.apply(decide_action, axis=1)
        print(f"[{tf}] RL action samples:")
        print(df[['Datetime', 'prob_win', 'rl_action']].tail(5))
        df.to_csv(rl_out_path, index=False)
        print(f"[SUCCESS] {tf}: RL actions saved to {rl_out_path}")

if __name__ == "__main__":
    main()
