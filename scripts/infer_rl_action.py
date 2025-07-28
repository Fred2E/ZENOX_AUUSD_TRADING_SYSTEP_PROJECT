import sys
sys.path.append(r'C:\Users\open\Documents\ZENO_XAUUSD\modules')

from stable_baselines3 import PPO
from zeno_env import ZenoTradingEnv

def run_inference(timeframe, state_df):
    model_path = fr"C:\Users\open\Documents\ZENO_XAUUSD\models\rl_policy_{timeframe}_latest.zip"
    model = PPO.load(model_path)
    
    # Env should accept state injection
    env = ZenoTradingEnv(data_df=state_df)  # Must support live data injection
    obs = env.reset()
    
    action, _ = model.predict(obs, deterministic=True)
    return action
