import os 
import sys
import numpy as np
import pandas as pd
import json
import hashlib
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

MODULES_PATH = 'C:/Users/open/Documents/ZENO_XAUUSD/modules'
if MODULES_PATH not in sys.path:
    sys.path.append(MODULES_PATH)
from zeno_rl_env import ZenoRLTradingEnv
import zeno_config

TIMEFRAMES = ["M5", "M15", "H1", "H4"]
DATA_DIR = r"C:/Users/open/Documents/ZENO_XAUUSD/historical/processed"
MODEL_DIR = r"C:/Users/open/Documents/ZENO_XAUUSD/models"
LOGS_DIR = r"C:/Users/open/Documents/ZENO_XAUUSD/logs"

def hash_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def hash_json(path):
    with open(path, 'r') as f:
        j = json.load(f)
    return hashlib.sha256(json.dumps(j, sort_keys=True).encode()).hexdigest()

def load_features_from_json(tf):
    feat_json = os.path.join(MODEL_DIR, f"rl_policy_{tf}_features.json")
    if not os.path.exists(feat_json):
        raise FileNotFoundError(f"Feature file not found for {tf}: {feat_json}")
    with open(feat_json, 'r') as f:
        RL_FEATURES = json.load(f)
    return RL_FEATURES

def get_tf_param(tf, key, default):
    """Helper to pull tf-specific config from zeno_config or fallback."""
    val = zeno_config.WALKFORWARD_PARAMS.get(tf, {}).get(key, None)
    if val is not None:
        return val
    # fallback to global config
    return zeno_config.WALKFORWARD_PARAMS.get("default", {}).get(key, default)

def print_metrics(trade_log, tf):
    if trade_log.empty:
        print(f"{tf}: No trades to analyze.")
        return {
            "Total Reward": 0.0,
            "Mean Reward": 0.0,
            "Trade Count": 0,
            "Win Rate (%)": 0.0,
            "Sharpe": 0.0,
            "Max Drawdown": 0.0,
            "Status": "NO_TRADES"
        }
    cum_rewards = trade_log['reward'].cumsum()
    max_dd = np.min(cum_rewards - np.maximum.accumulate(cum_rewards))
    sharpe = trade_log['reward'].mean() / (trade_log['reward'].std() + 1e-8)
    win_rate = (trade_log['reward'] > 0).mean() * 100
    avg_reward = trade_log['reward'].mean()
    total_reward = trade_log['reward'].sum()
    trade_count = len(trade_log)
    print(f"\n=== Trade Metrics for {tf} ===")
    print(f"Total pips: {total_reward:.2f}")
    print(f"Mean pips/trade: {avg_reward:.2f}")
    print(f"Win rate: {win_rate:.2f}%")
    print(f"Sharpe: {sharpe:.2f}")
    print(f"Max Drawdown: {max_dd:.2f}")
    print(f"Trade count: {trade_count}")
    return {
        "Total Reward": total_reward,
        "Mean Reward": avg_reward,
        "Trade Count": trade_count,
        "Win Rate (%)": win_rate,
        "Sharpe": sharpe,
        "Max Drawdown": max_dd,
        "Status": "OK"
    }

def main():
    print("STARTING WALK-FORWARD BACKTEST...")
    os.makedirs(LOGS_DIR, exist_ok=True)
    results = []
    blocked = []
    passed = []

    for tf in TIMEFRAMES:
        try:
            data_csv = os.path.join(DATA_DIR, f"signals_{tf}.csv")
            model_path = os.path.join(MODEL_DIR, f"rl_policy_{tf}_latest.zip")
            features_json = os.path.join(MODEL_DIR, f"rl_policy_{tf}_features.json")

            hashes = {
                "DataCSV_Hash": hash_file(data_csv) if os.path.exists(data_csv) else "",
                "Model_Hash": hash_file(model_path) if os.path.exists(model_path) else "",
                "Features_Hash": hash_json(features_json) if os.path.exists(features_json) else ""
            }

            if not all(os.path.exists(p) for p in [data_csv, model_path, features_json]):
                print(f"{tf}: Missing required files, BLOCKED")
                blocked.append({"Timeframe": tf, "Reason": "MISSING_FILES"})
                results.append({"Timeframe": tf, "Status": "MISSING_FILES", **hashes})
                continue

            RL_FEATURES = load_features_from_json(tf)
            df = pd.read_csv(data_csv)
            df.columns = [c.lower() for c in df.columns]
            df = df.dropna(subset=RL_FEATURES + ['close', 'datetime']).reset_index(drop=True)

            # Pull tf-specific params
            min_trades = get_tf_param(tf, "min_trades", 30)
            min_winrate = get_tf_param(tf, "min_winrate", 50.0)

            if len(df) < min_trades:
                print(f"{tf}: Only {len(df)} trades. BLOCKED: <{min_trades} trades.")
                blocked.append({"Timeframe": tf, "Reason": f"TOO_FEW_TRADES ({len(df)})"})
                results.append({"Timeframe": tf, "Status": "TOO_FEW_TRADES", **hashes})
                continue

            env = ZenoRLTradingEnv(df.copy(), timeframe=tf, features=RL_FEATURES)
            vec_env = DummyVecEnv([lambda: env])
            model = PPO.load(model_path, env=vec_env)
            obs = vec_env.reset()
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _ = vec_env.step(action)

            trade_log = pd.DataFrame(env.trades) if hasattr(env, 'trades') else pd.DataFrame()
            if not trade_log.empty:
                trade_log['entry_time'] = trade_log['entry_step'].apply(lambda x: df['datetime'].iloc[min(int(x), len(df)-1)] if pd.notna(x) else "NA")
                trade_log['exit_time'] = trade_log['exit_step'].apply(lambda x: df['datetime'].iloc[min(int(x), len(df)-1)] if pd.notna(x) else "NA")
                log_path = os.path.join(LOGS_DIR, f"trade_log_{tf}.csv")
                trade_log.to_csv(log_path, index=False)
                print(f"{tf}: Saved trade log with {len(trade_log)} trades to {log_path}")

            metrics = print_metrics(trade_log, tf)
            metrics.update({"Timeframe": tf, **hashes})

            if metrics["Trade Count"] < min_trades:
                metrics["Status"] = "BLOCKED_TOO_FEW_TRADES"
                blocked.append({"Timeframe": tf, "Reason": f"TOO_FEW_TRADES ({metrics['Trade Count']})"})
            elif metrics["Win Rate (%)"] < min_winrate:
                metrics["Status"] = "BLOCKED_WINRATE"
                blocked.append({"Timeframe": tf, "Reason": f"LOW_WINRATE ({metrics['Win Rate (%)']:.2f}%)"})
            else:
                metrics["Status"] = "PASSED"
                passed.append({"Timeframe": tf, "Winrate": metrics["Win Rate (%)"], "TradeCount": metrics["Trade Count"]})

            results.append(metrics)

        except Exception as e:
            print(f"{tf}: Exception - {str(e)}")
            blocked.append({"Timeframe": tf, "Reason": f"EXCEPTION: {e}"})
            results.append({"Timeframe": tf, "Status": f"EXCEPTION_{str(e)}", "DataCSV_Hash": "", "Model_Hash": "", "Features_Hash": ""})

    summary_df = pd.DataFrame(results)
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(LOGS_DIR, f"walkforward_results_{now_str}.csv")
    summary_df.to_csv(summary_path, index=False)
    print("\nSUMMARY:")
    print(summary_df)
    print(f"Results saved to {summary_path}")

    print("\n==== PASSED REGIMES ====")
    if passed:
        for r in passed:
            print(f"✅ {r['Timeframe']}: Winrate={r['Winrate']:.2f}%, Trades={r['TradeCount']}")
    else:
        print("❌ NONE. No viable regime found.")

    print("\n==== BLOCKED REGIMES ====")
    for r in blocked:
        print(f"⛔ {r['Timeframe']}: {r['Reason']}")

    # Log phase with summary hash
    with open(os.path.join(LOGS_DIR, f"PHASE_WALKFORWARD_COMPLETE_{now_str}.txt"), 'w') as f:
        f.write(f"walkforward_complete | {now_str} | summary_hash: {hash_file(summary_path)}\n")

if __name__ == "__main__":
    main()
