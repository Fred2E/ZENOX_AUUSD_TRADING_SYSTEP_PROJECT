import os
import sys
import pandas as pd
import json
import hashlib
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# --- Path setup
MODULES_PATH = 'C:/Users/open/Documents/ZENO_XAUUSD/modules'
if MODULES_PATH not in sys.path:
    sys.path.append(MODULES_PATH)

from zeno_rl_env import ZenoRLTradingEnv

DATA_PATH = 'C:/Users/open/Documents/ZENO_XAUUSD/historical/processed'
MODEL_PATH = 'C:/Users/open/Documents/ZENO_XAUUSD/models'
LOG_PATH = 'C:/Users/open/Documents/ZENO_XAUUSD/logs'
TIMEFRAMES = ['M5', 'M15', 'H1', 'H4']

ALL_POSSIBLE_FEATURES = [
    'score', 'num_confs', 'pattern_code', 'bias_bull', 'hour', 'dow', 'atr',
    'conf_structure', 'conf_bos_or_choch', 'conf_candle', 'conf_sr_zone',
    'conf_psych_level', 'conf_fib_zone', 'conf_volume', 'conf_liquidity', 'conf_spread'
]

# --- Special regime inputs, add any filtered log path here
SPECIAL_TF_INPUTS = {
    "M15": os.path.join(LOG_PATH, "trade_log_M15_Aplus.csv"),
    # Add more like: "M5": "trade_log_M5_Aplus.csv", etc if needed
}

def hash_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def train_rl_agent(tf):
    # Select raw or regime-specific log
    signal_file = SPECIAL_TF_INPUTS.get(tf, os.path.join(DATA_PATH, f"signals_{tf}.csv"))
    if not os.path.exists(signal_file):
        print(f"[SKIP] {tf}: Data/log missing at {signal_file}.")
        return

    df = pd.read_csv(signal_file)
    df.columns = [c.lower() for c in df.columns]

    # Use only features present in the dataset
    RL_FEATURES = [f for f in ALL_POSSIBLE_FEATURES if f in df.columns]
    missing_crit = [c for c in ['close', 'datetime'] if c not in df.columns]
    if missing_crit:
        print(f"[FATAL] {tf}: Missing critical columns: {missing_crit}")
        return

    # Drop NaNs in selected features and critical cols
    df = df.dropna(subset=RL_FEATURES + ['close', 'datetime']).reset_index(drop=True)

    # Audit/skip if dataset is too small
    if len(df) < 15:
        print(f"[SKIP] {tf}: Not enough rows to train RL agent. ({len(df)} rows)")
        return

    # Save RL_FEATURES to JSON before training
    feature_json = os.path.join(MODEL_PATH, f"rl_policy_{tf}_features.json")
    with open(feature_json, 'w') as f:
        json.dump(RL_FEATURES, f, indent=2)
    print(f"[INFO] RL_FEATURES saved to {feature_json}")

    # Save config and hash
    config = {
        "RL_FEATURES": RL_FEATURES,
        "env_timeframe": tf,
        "n_steps": 256,
        "batch_size": 64,
        "learning_rate": 3e-4,
        "total_timesteps": 20000,
        "feature_json_hash": hash_file(feature_json),
        "data_hash": hash_file(signal_file)
    }
    config_json = os.path.join(MODEL_PATH, f"rl_policy_{tf}_config.json")
    with open(config_json, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"[INFO] RL training config saved to {config_json}")

    # --- Train RL agent
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

    # Hash/Log checkpoint for reproducibility
    checkpoint_log = {
        "model_hash": hash_file(model_path),
        "features_hash": hash_file(feature_json),
        "data_hash": hash_file(signal_file),
        "config_hash": hash_file(config_json),
        "n_rows": len(df),
        "timeframe": tf
    }
    log_json = os.path.join(LOG_PATH, f"checkpoint_rl_{tf}_{len(df)}.json")
    with open(log_json, 'w') as f:
        json.dump(checkpoint_log, f, indent=2)
    print(f"[CHECKPOINT] {log_json}")

if __name__ == "__main__":
    os.makedirs(MODEL_PATH, exist_ok=True)
    os.makedirs(LOG_PATH, exist_ok=True)
    for tf in TIMEFRAMES:
        try:
            train_rl_agent(tf)
        except Exception as e:
            print(f"[ERROR] Failed training {tf}: {e}")
