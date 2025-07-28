# zeno_live_rl_action.py

import os
import pandas as pd

# --- Directory and timeframes setup ---
LIVE_DATA_ROOT = os.path.join("data", "live", "XAUUSD")
TIMEFRAMES = ["M5", "M15", "H1", "H4", "D1"]

def decide_action(row):
    """
    Simple RL/Policy decision logic:
    - Example: If model probability of win > 0.5, issue BUY, else HOLD.
    - Replace with your real RL or policy function as needed.
    """
    if 'prob_win' in row and row['prob_win'] > 0.5:
        return 'BUY'
    else:
        return 'HOLD'

def main():
    """
    For each timeframe:
    - Load prediction CSV from prior ML step.
    - Standardize columns to lowercase (robust to casing bugs).
    - Apply RL/policy logic and add 'rl_action' column.
    - Save new CSV with RL actions and print sample output for audit.
    """
    for tf in TIMEFRAMES:
        pred_path = os.path.join(LIVE_DATA_ROOT, tf, f"XAUUSD_{tf}_LIVE_PRED.csv")
        rl_out_path = os.path.join(LIVE_DATA_ROOT, tf, f"XAUUSD_{tf}_LIVE_RL_ACTION.csv")
        if not os.path.exists(pred_path):
            print(f"[SKIP] {tf}: Prediction file not found: {pred_path}")
            continue

        # --- Load prediction CSV and force all columns to lowercase for consistency ---
        df = pd.read_csv(pred_path)
        df.columns = [c.lower() for c in df.columns]

        # --- Apply RL action/policy logic to each row ---
        df['rl_action'] = df.apply(decide_action, axis=1)

        # --- Print sample of output for quick audit, always use lowercase column names ---
        print(f"[{tf}] RL action samples:")
        sample_cols = ['datetime', 'prob_win', 'rl_action']
        missing = [col for col in sample_cols if col not in df.columns]
        if missing:
            print(f"[WARN] {tf}: Missing columns in output: {missing}")
        else:
            print(df[sample_cols].tail(5))

        # --- Save RL actions CSV for downstream trading/analytics ---
        df.to_csv(rl_out_path, index=False)
        print(f"[SUCCESS] {tf}: RL actions saved to {rl_out_path}")

if __name__ == "__main__":
    main()
