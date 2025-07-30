import os
import pandas as pd
import json
import hashlib
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from zeno_rl_env import ZenoRLTradingEnv
import zeno_config

# === CONFIG ===
TF = "M5"  # Change as needed ("M15", "H1", "H4")
BASE = "C:/Users/open/Documents/ZENO_XAUUSD"
SIGNALS_PATH = f"{BASE}/historical/processed/signals_{TF}.csv"
FEATURES_PATH = f"{BASE}/models/rl_policy_{TF}_features.json"
MODEL_OUT = f"{BASE}/models/rl_policy_{TF}_trained_on_grade.zip"
LOG_PATH = f"{BASE}/logs/rl_train_{TF}.json"

# --- Load signals and features ---
signals = pd.read_csv(SIGNALS_PATH)
signals['datetime'] = pd.to_datetime(signals['datetime'])
with open(FEATURES_PATH, 'r') as f:
    features = json.load(f)

# --- TF-adaptive filter logic (must be centralized, but can override here for debug) ---
def tf_filter(row):
    # Example: for M5, at least 3/4 primary confluences and regime match; can pull from zeno_config
    min_score = zeno_config.ENTRY_FILTERS['min_score'][TF]
    min_confs = zeno_config.ENTRY_FILTERS['min_confs'][TF]
    # Add any tf-specific filters here
    return row['score'] >= min_score and row['num_confs'] >= min_confs

signals_filtered = signals[signals.apply(tf_filter, axis=1)].copy()
signals_filtered = signals_filtered.dropna(subset=features + ['close', 'datetime']).reset_index(drop=True)
print(f"✅ {TF}: {len(signals_filtered)} bars for RL training (config-grade filtered)")

# --- TF-specific RL params (should be centralized in zeno_config.py) ---
RL_TRAIN_PARAMS = zeno_config.RL_TRAIN_PARAMS.get(TF, {
    'n_steps': 256,
    'batch_size': 64,
    'learning_rate': 3e-4,
    'total_timesteps': max(1000, 10 * len(signals_filtered))
})

# --- RL Training ---
env = DummyVecEnv([lambda: ZenoRLTradingEnv(signals_filtered.copy(), timeframe=TF, features=features)])
model = PPO(
    'MlpPolicy',
    env,
    verbose=1,
    n_steps=RL_TRAIN_PARAMS['n_steps'],
    batch_size=RL_TRAIN_PARAMS['batch_size'],
    learning_rate=RL_TRAIN_PARAMS['learning_rate'],
    gamma=0.99,
    ent_coef=0.01,
    tensorboard_log=f"{BASE}/logs/tensorboard_{TF}"
)
model.learn(total_timesteps=RL_TRAIN_PARAMS['total_timesteps'])
model.save(MODEL_OUT)
print(f"✅ RL model trained and saved for {TF}: {MODEL_OUT}")

# --- Audit-trail/log the training session ---
def hash_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

log = {
    'timeframe': TF,
    'signals_rows': len(signals_filtered),
    'features': features,
    'train_params': RL_TRAIN_PARAMS,
    'signals_hash': hash_file(SIGNALS_PATH),
    'features_hash': hash_file(FEATURES_PATH),
    'model_hash': hash_file(MODEL_OUT)
}
with open(LOG_PATH, 'w') as f:
    json.dump(log, f, indent=2)
print(f"✅ RL training session log saved: {LOG_PATH}")
