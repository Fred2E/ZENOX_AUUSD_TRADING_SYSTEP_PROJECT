import sys
import datetime
import logging
sys.path.append(r'C:\Users\open\Documents\ZENO_XAUUSD\modules')
from zeno_env import ZenoTradingEnv
from stable_baselines3 import PPO

logging.basicConfig(filename=r'C:\Users\open\Documents\ZENO_XAUUSD\logs\rl_training.log',
                    level=logging.INFO, format='%(asctime)s %(message)s')

timeframes = [
    ('M5',  r'C:\Users\open\Documents\ZENO_XAUUSD\data\live\XAUUSD\M5\XAUUSD_M5_LIVE_FEATURES.csv'),
    ('M15', r'C:\Users\open\Documents\ZENO_XAUUSD\data\live\XAUUSD\M15\XAUUSD_M15_LIVE_FEATURES.csv'),
    ('H1',  r'C:\Users\open\Documents\ZENO_XAUUSD\data\live\XAUUSD\H1\XAUUSD_H1_LIVE_FEATURES.csv'),
    ('H4',  r'C:\Users\open\Documents\ZENO_XAUUSD\data\live\XAUUSD\H4\XAUUSD_H4_LIVE_FEATURES.csv'),
    ('D1',  r'C:\Users\open\Documents\ZENO_XAUUSD\data\live\XAUUSD\D1\XAUUSD_D1_LIVE_FEATURES.csv'),
]

for tf, csv in timeframes:
    try:
        print(f"[{datetime.datetime.now()}] Training PPO for {tf} ...")
        env = ZenoTradingEnv(csv)
        model = PPO('MlpPolicy', env, verbose=1)
        model.learn(total_timesteps=10000)
        model_path = rf"C:\Users\open\Documents\ZENO_XAUUSD\models\rl_policy_{tf}_latest.zip"
        model.save(model_path)
        print(f"[{datetime.datetime.now()}] Saved: {model_path}")
        logging.info(f"{tf} - Training SUCCESS: Saved to {model_path}")
    except Exception as e:
        logging.error(f"{tf} - Training FAILED: {e}")
        print(f"Training FAILED for {tf}: {e}")
