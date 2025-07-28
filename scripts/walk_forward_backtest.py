import os
import sys
import numpy as np
import pandas as pd
import json
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Inject module path
MODULES_PATH = 'C:/Users/open/Documents/ZENO_XAUUSD/modules'
if MODULES_PATH not in sys.path:
    sys.path.append(MODULES_PATH)

from zeno_rl_env import ZenoRLTradingEnv

TIMEFRAMES = ["M5", "M15", "H1", "H4"]

DATA_DIR = r"C:/Users/open/Documents/ZENO_XAUUSD/historical/processed"
MODEL_DIR = r"C:/Users/open/Documents/ZENO_XAUUSD/models"
LOGS_DIR = r"C:/Users/open/Documents/ZENO_XAUUSD/logs"

def load_features_from_json(tf):
    feat_json = os.path.join(MODEL_DIR, f"rl_policy_{tf}_features.json")
    if not os.path.exists(feat_json):
        raise FileNotFoundError(f"Feature file not found for {tf}: {feat_json}")
    with open(feat_json, 'r') as f:
        RL_FEATURES = json.load(f)
    return RL_FEATURES

def print_metrics(trade_log, tf):
    if trade_log.empty:
        print(f"{tf}: No trades to analyze.")
        return
    cum_rewards = trade_log['reward'].cumsum()
    max_dd = np.min(cum_rewards - np.maximum.accumulate(cum_rewards))
    sharpe = trade_log['reward'].mean() / (trade_log['reward'].std() + 1e-8)
    win_rate = (trade_log['reward'] > 0).mean() * 100

    print(f"\n=== Trade Metrics for {tf} ===")
    print(f"Total pips: {trade_log['reward'].sum():.2f}")
    print(f"Mean pips/trade: {trade_log['reward'].mean():.2f}")
    print(f"Win rate: {win_rate:.2f}%")
    print(f"Sharpe: {sharpe:.2f}")
    print(f"Max Drawdown: {max_dd:.2f}")
    print(f"Trade count: {len(trade_log)}")
    # Print a snapshot for audit
    cols_to_show = [c for c in ['entry_time', 'exit_time', 'side', 'reward', 'score', 'pattern_code'] if c in trade_log.columns]
    print(trade_log[cols_to_show].head())

def main():
    print("STARTING WALK-FORWARD BACKTEST...")
    os.makedirs(LOGS_DIR, exist_ok=True)
    results = []

    for tf in TIMEFRAMES:
        data_csv = os.path.join(DATA_DIR, f"signals_{tf}.csv")
        model_path = os.path.join(MODEL_DIR, f"rl_policy_{tf}_latest.zip")
        features_json = os.path.join(MODEL_DIR, f"rl_policy_{tf}_features.json")

        if not os.path.exists(data_csv):
            print(f"{tf}: Data file missing ({data_csv}), skipping")
            continue
        if not os.path.exists(model_path):
            print(f"{tf}: Model file missing ({model_path}), skipping")
            continue
        if not os.path.exists(features_json):
            print(f"{tf}: Features JSON missing ({features_json}), skipping")
            continue

        RL_FEATURES = load_features_from_json(tf)
        print(f"\n[{tf}] RL_FEATURES used: {RL_FEATURES}")

        df = pd.read_csv(data_csv)
        df.columns = [c.lower() for c in df.columns]

        # Check all RL_FEATURES exist in signals
        missing = [col for col in RL_FEATURES + ['close', 'datetime'] if col not in df.columns]
        if missing:
            print(f"{tf}: Missing columns {missing}, skipping")
            continue

        df = df.dropna(subset=RL_FEATURES + ['close', 'datetime']).reset_index(drop=True)
        env = ZenoRLTradingEnv(df.copy(), timeframe=tf, features=RL_FEATURES)
        vec_env = DummyVecEnv([lambda: env])
        model = PPO.load(model_path, env=vec_env)

        obs = vec_env.reset()
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = vec_env.step(action)

        if not hasattr(env, 'trades') or len(env.trades) == 0:
            print(f"{tf}: No trades detected.")
            continue

        trade_log = pd.DataFrame(env.trades)

        # Map bar indices to datetime for entry and exit
        if 'entry_step' in trade_log.columns and 'exit_step' in trade_log.columns:
            trade_log['entry_time'] = trade_log['entry_step'].apply(lambda x: df['datetime'].iloc[int(x)] if int(x) < len(df) else "NA")
            trade_log['exit_time'] = trade_log['exit_step'].apply(lambda x: df['datetime'].iloc[int(x)] if int(x) < len(df) else "NA")
        else:
            trade_log['entry_time'] = "NA"
            trade_log['exit_time'] = "NA"

        log_path = os.path.join(LOGS_DIR, f"trade_log_{tf}.csv")
        trade_log.to_csv(log_path, index=False)
        print(f"{tf}: Saved trade log with {len(trade_log)} trades to {log_path}")

        print_metrics(trade_log, tf)

        results.append({
            "Timeframe": tf,
            "Total Reward": trade_log['reward'].sum(),
            "Trade Count": len(trade_log)
        })

    summary_df = pd.DataFrame(results)
    summary_path = os.path.join(LOGS_DIR, "walkforward_results.csv")
    summary_df.to_csv(summary_path, index=False)
    print("\nSUMMARY:")
    print(summary_df)
    print(f"\nWalk-forward backtest complete. Results saved to {summary_path}")

if __name__ == "__main__":
    main()
