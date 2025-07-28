print("STARTING RL TRAINING SCRIPT...")  # Entry log

# --- Imports ---
import sys
import argparse
from pathlib import Path

# --- Custom module path ---
sys.path.append(r'C:\Users\open\Documents\ZENO_XAUUSD\modules')
print("Sys path set.")

# --- Parse command line arguments ---
parser = argparse.ArgumentParser()
parser.add_argument('--timeframe', type=str, default='M15', help='Choose timeframe: M5, M15, H1, H4, D1')
args = parser.parse_args()
timeframe = args.timeframe.upper()

# --- Validate timeframe ---
valid_timeframes = ['M5', 'M15', 'H1', 'H4', 'D1']
if timeframe not in valid_timeframes:
    raise ValueError(f"❌ Invalid timeframe: {timeframe}. Must be one of {valid_timeframes}")

# --- Dynamic file paths ---
DATA_CSV = fr'C:\Users\open\Documents\ZENO_XAUUSD\data\live\XAUUSD\{timeframe}\XAUUSD_{timeframe}_LIVE_FEATURES.csv'
MODEL_PATH = fr'C:\Users\open\Documents\ZENO_XAUUSD\models\rl_policy_{timeframe}_latest.zip'

# --- Confirm file loading ---
print(f"*** RL ENV loading file: {DATA_CSV} ***")

# --- Imports ---
try:
    from stable_baselines3 import PPO
    print("Imported PPO.")
except Exception as e:
    print(f"❌ Failed at PPO import: {e}")
    sys.exit(1)

try:
    from zeno_env import ZenoTradingEnv
    print("Imported ZenoTradingEnv.")
except Exception as e:
    print(f"❌ Failed at ZenoTradingEnv import: {e}")
    sys.exit(1)

# --- Environment Initialization ---
try:
    env = ZenoTradingEnv(DATA_CSV)
    print("✅ Initialized environment.")
except Exception as e:
    print(f"❌ Failed to initialize environment: {e}")
    sys.exit(1)

# --- PPO Model Initialization ---
try:
    model = PPO('MlpPolicy', env, verbose=1)
    print("✅ Initialized PPO model.")
except Exception as e:
    print(f"❌ Failed to initialize PPO model: {e}")
    sys.exit(1)

# --- Training Phase ---
try:
    model.learn(total_timesteps=10000)
    print("✅ Finished learning.")
    model.save(MODEL_PATH)
    print(f"✅ Trained PPO RL policy saved to: {MODEL_PATH}")
except Exception as e:
    print(f"❌ Training failed: {e}")
    sys.exit(1)
