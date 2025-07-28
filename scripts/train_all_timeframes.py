print("STARTING FULL RL RETRAIN...")

import sys, os, time, traceback, shutil
sys.path.append(r'C:\Users\open\Documents\ZENO_XAUUSD\modules')
print("Sys path set.")

from stable_baselines3 import PPO
from zeno_env import ZenoTradingEnv

MODEL_DIR = r"C:\Users\open\Documents\ZENO_XAUUSD\models"
DATA_DIR = r"C:\Users\open\Documents\ZENO_XAUUSD\data\live\XAUUSD"
LOG_PATH = r"C:\Users\open\Documents\ZENO_XAUUSD\models\training_audit.log"
QA_DIR = r"C:\Users\open\Documents\ZENO_XAUUSD\models\data_preview"

RL_FEATURES = ['score', 'num_confs', 'pattern_code', 'bias_bull', 'hour', 'dow', 'atr']
TIMEFRAMES = ["M5", "M15", "H1", "H4", "D1"]
TRAIN_STEPS = 50000
VERBOSE = 1

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(QA_DIR, exist_ok=True)

def log(msg):
    print(msg)
    with open(LOG_PATH, "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")

for tf in TIMEFRAMES:
    try:
        data_csv = os.path.join(DATA_DIR, tf, f"XAUUSD_{tf}_LIVE_FEATURES.csv")
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        model_path = os.path.join(MODEL_DIR, f"rl_policy_{tf}_{timestamp}.zip")
        model_latest = os.path.join(MODEL_DIR, f"rl_policy_{tf}.zip")

        if not os.path.exists(data_csv):
            log(f"{tf}: Data missing, skipping.")
            continue

        # --- Dataset health check ---
        import pandas as pd
        df = pd.read_csv(data_csv)
        log(f"{tf}: Dataset shape: {df.shape}, Columns: {list(df.columns)}")

        # Enforce datetime as first column
        if 'datetime' not in df.columns:
            log(f"{tf}: FATAL - MISSING 'datetime' column!")
            log(df.head(3).to_string())
            continue
        if df.columns[0] != 'datetime':
            cols = df.columns.tolist()
            cols.insert(0, cols.pop(cols.index('datetime')))
            df = df[cols]
            df.to_csv(data_csv, index=False)
            log(f"{tf}: Reordered columns to make 'datetime' first.")

        if df['datetime'].isnull().any() or df['datetime'].eq('').any():
            log(f"{tf}: FATAL - NULL/EMPTY values in 'datetime' column!")
            log(df.head(3).to_string())
            continue

        # Save a QA preview
        preview_path = os.path.join(QA_DIR, f"{tf}_features_preview.csv")
        df.head(10).to_csv(preview_path, index=False)
        log(f"{tf}: Saved preview to {preview_path}")

        # Check RL features
        missing = [col for col in RL_FEATURES if col not in df.columns]
        if missing:
            log(f"{tf}: FATAL - Missing RL features: {missing}")
            continue
        for f in RL_FEATURES:
            if not pd.api.types.is_numeric_dtype(df[f]):
                log(f"{tf}: FATAL - RL feature {f} is NOT numeric!")
                continue
        if df.isnull().any().any():
            log(f"{tf}: FATAL - NaNs detected!")
            continue

        if len(df) < 500:
            log(f"{tf}: FATAL - Data too short ({len(df)} rows). Need at least 500.")
            continue

        log(f"{tf}: Feature summary:\n{df.describe(include='all').loc[:, RL_FEATURES]}")

        log(f"\n--- Training {tf} ---")
        log(f"Features: {RL_FEATURES}")
        log(f"Train steps: {TRAIN_STEPS}, Balance: 10000, Data: {data_csv}")

        env = ZenoTradingEnv(data_csv, features=RL_FEATURES)
        model = PPO("MlpPolicy", env, verbose=VERBOSE)
        model.learn(total_timesteps=TRAIN_STEPS)
        model.save(model_path)
        shutil.copyfile(model_path, model_latest)
        log(f"{tf}: Training finished and model saved to {model_path} and {model_latest}")

        last_rewards = env.trades[-10:] if hasattr(env, "trades") and env.trades else []
        log(f"{tf}: Last 10 trade logs (post-train): {last_rewards}")

    except Exception as e:
        log(f"{tf} - FATAL ERROR:\n{traceback.format_exc()}")

print("\nALL TIMEFRAMES TRAINED. MODELS UPDATED.")
