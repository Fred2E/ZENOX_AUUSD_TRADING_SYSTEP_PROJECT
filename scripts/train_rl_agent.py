import pandas as pd
import json
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from zeno_rl_env import ZenoRLTradingEnv

# === CONFIG ===
TF = "M5"  # Change to "M15", "H1", "H4" as needed
SIGNALS_PATH = f"C:/Users/open/Documents/ZENO_XAUUSD/historical/processed/signals_{TF}.csv"
FEATURES_PATH = f"C:/Users/open/Documents/ZENO_XAUUSD/models/rl_policy_{TF}_features.json"
MODEL_OUT = f"C:/Users/open/Documents/ZENO_XAUUSD/models/rl_policy_{TF}_trained_on_grade.zip"

# --- Load and Filter Full Signals (example: B-grade logic, adjust as needed) ---
signals = pd.read_csv(SIGNALS_PATH)
signals['datetime'] = pd.to_datetime(signals['datetime'])
with open(FEATURES_PATH, 'r') as f:
    features = json.load(f)

def is_B_grade(row):
    # Adjust as needed for your regime logic
    return row['score'] >= 2 and row['num_confs'] >= 3

signals_filtered = signals[signals.apply(is_B_grade, axis=1)].copy()
signals_filtered = signals_filtered.dropna(subset=features + ['close', 'datetime']).reset_index(drop=True)
print(f"✅ {TF}: {len(signals_filtered)} bars for RL training (grade-filtered)")

# --- RL Training ---
env = DummyVecEnv([lambda: ZenoRLTradingEnv(signals_filtered.copy(), timeframe=TF, features=features)])
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=max(1000, 10 * len(signals_filtered)))  # Adjust for sample size
model.save(MODEL_OUT)
print(f"✅ RL model trained and saved for {TF}: {MODEL_OUT}")
