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
import zeno_config

DATA_PATH = 'C:/Users/open/Documents/ZENO_XAUUSD/historical/processed'
MODEL_PATH = 'C:/Users/open/Documents/ZENO_XAUUSD/models'
LOG_PATH = 'C:/Users/open/Documents/ZENO_XAUUSD/logs'
TIMEFRAMES = ['M5', 'M15', 'H1', 'H4']

# --- TF-specific RL params (should be in zeno_config) ---
RL_TRAIN_PARAMS = {
    'M5':  {'n_steps': 256, 'batch_size': 64,  'total_timesteps': 40000, 'learning_rate': 3e-4},
    'M15': {'n_steps': 256, 'batch_size': 64,  'total_timesteps': 40000, 'learning_rate': 3e-4},
    'H1':  {'n_steps': 256, 'batch_size': 32,  'total_timesteps': 10000, 'learning_rate': 1e-4},
    'H4':  {'n_steps': 128, 'batch_size': 16,  'total_timesteps': 5000,  'learning_rate': 1e-4},
}

# Features to use for RL agent per tf (should match pipeline output)
TF_FEATURES = {
    tf: [f for f in zeno_config.PRIMARY_CONFS + zeno_config.SECONDARY_CONFS
         if f in pd.read_csv(os.path.join(DATA_PATH, f'signals_{tf}.csv')).columns]
    for tf in TIMEFRAMES
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
    # Only proceed for M5/M15, or if enough signals
    signal_file = os.path.join(DATA_PATH, f"signals_{tf}.csv")
    if not os.path.exists(signal_file):
        print(f"[SKIP] {tf}: Data/log missing at {signal_file}.")
        return

    df = pd.read_csv(signal_file)
    df.columns = [c.lower() for c in df.columns]
    n_signals = len(df)
    if tf in ['H1', 'H4'] and n_signals < 30:
        print(f"[SKIP] {tf}: Not enough signals for RL training ({n_signals} rows).")
        return
    if n_signals < 15:
        print(f"[SKIP] {tf}: Not enough rows to train RL agent. ({n_signals} rows)")
        return

    RL_FEATURES = [f for f in zeno_config.PRIMARY_CONFS + zeno_config.SECONDARY_CONFS if f in df.columns]
    missing_crit = [c for c in ['close', 'datetime'] if c not in df.columns]
    if missing_crit:
        print(f"[FATAL] {tf}: Missing critical columns: {missing_crit}")
        return

    df = df.dropna(subset=RL_FEATURES + ['close', 'datetime']).reset_index(drop=True)

    # Save RL_FEATURES to JSON before training
    feature_json = os.path.join(MODEL_PATH, f"rl_policy_{tf}_features.json")
    with open(feature_json, 'w') as f:
        json.dump(RL_FEATURES, f, indent=2)
    print(f"[INFO] RL_FEATURES saved to {feature_json}")

    # Fetch tf-specific params
    params = RL_TRAIN_PARAMS[tf]
    config = {
        "RL_FEATURES": RL_FEATURES,
        "env_timeframe": tf,
        **params,
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
        n_steps=params['n_steps'],
        batch_size=params['batch_size'],
        learning_rate=params['learning_rate'],
        gamma=0.99,
        ent_coef=0.01,
        tensorboard_log=os.path.join(LOG_PATH, f"tensorboard_{tf}")
    )
    model.learn(total_timesteps=params['total_timesteps'])
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
