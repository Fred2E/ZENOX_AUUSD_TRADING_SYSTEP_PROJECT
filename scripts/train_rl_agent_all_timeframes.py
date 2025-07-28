import os
import sys
import pandas as pd
import json
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# === Add module path ===
MODULES_PATH = 'C:/Users/open/Documents/ZENO_XAUUSD/modules'
if MODULES_PATH not in sys.path:
    sys.path.append(MODULES_PATH)

from zeno_rl_env import ZenoRLTradingEnv  # Make sure your env is up to date

DATA_PATH = 'C:/Users/open/Documents/ZENO_XAUUSD/historical/processed'
MODEL_PATH = 'C:/Users/open/Documents/ZENO_XAUUSD/models'
LOG_PATH = 'C:/Users/open/Documents/ZENO_XAUUSD/logs'
TIMEFRAMES = ['M5', 'M15', 'H1', 'H4']

ALL_POSSIBLE_FEATURES = [
    'score', 'num_confs', 'pattern_code', 'bias_bull', 'hour', 'dow', 'atr',
    'conf_structure', 'conf_bos_or_choch', 'conf_candle', 'conf_sr_zone',
    'conf_psych_level', 'conf_fib_zone', 'conf_volume', 'conf_liquidity', 'conf_spread'
]

def train_rl_agent(tf):
    signal_file = os.path.join(DATA_PATH, f"signals_{tf}.csv")
    if not os.path.exists(signal_file):
        print(f"[SKIP] {tf} → Signal file missing at {signal_file}.")
        return

    df = pd.read_csv(signal_file)
    df.columns = [c.lower() for c in df.columns]

    # === Auto-detect features from CSV ===
    RL_FEATURES = [f for f in ALL_POSSIBLE_FEATURES if f in df.columns]
    missing_crit = [c for c in ['close', 'datetime'] if c not in df.columns]
    if missing_crit:
        print(f"[FATAL] {tf}: Missing critical columns: {missing_crit}")
        return

    # === Audit: print features ===
    print(f"\n=== Training RL Agent [{tf}] ===")
    print(f"RL_FEATURES: {RL_FEATURES} | Total: {len(RL_FEATURES)}")
    print(f"Shape: {df.shape}, Columns: {df.columns.tolist()}")

    # === Save RL_FEATURES for audit before training ===
    feature_json = os.path.join(MODEL_PATH, f"rl_policy_{tf}_features.json")
    with open(feature_json, 'w') as f:
        json.dump(RL_FEATURES, f, indent=2)
    print(f"[INFO] RL_FEATURES saved to {feature_json}")

    # === Save training config hash for reproducibility ===
    config = {
        "RL_FEATURES": RL_FEATURES,
        "env_timeframe": tf,
        "n_steps": 256,
        "batch_size": 64,
        "learning_rate": 3e-4,
        "total_timesteps": 20000
    }
    config_json = os.path.join(MODEL_PATH, f"rl_policy_{tf}_config.json")
    with open(config_json, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"[INFO] RL training config saved to {config_json}")

    # === Drop NaNs in only the selected RL_FEATURES + critical cols ===
    df = df.dropna(subset=RL_FEATURES + ['close', 'datetime']).reset_index(drop=True)

    # === Train RL agent ===
    env = DummyVecEnv([lambda: ZenoRLTradingEnv(df.copy(), timeframe=tf, features=RL_FEATURES)])

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=256,
        batch_size=64,
        learning_rate=3e-4,
        gamma=0.99,
        ent_coef=0.01,
        tensorboard_log=os.path.join(LOG_PATH, f"tensorboard_{tf}")
    )
    model.learn(total_timesteps=20000)
    model_path = os.path.join(MODEL_PATH, f"rl_policy_{tf}_latest.zip")
    model.save(model_path)
    print(f"[SAVED] {model_path}")

if __name__ == "__main__":
    os.makedirs(MODEL_PATH, exist_ok=True)
    os.makedirs(LOG_PATH, exist_ok=True)
    for tf in TIMEFRAMES:
        try:
            train_rl_agent(tf)
        except Exception as e:
            print(f"[ERROR] Failed training {tf}: {e}")
