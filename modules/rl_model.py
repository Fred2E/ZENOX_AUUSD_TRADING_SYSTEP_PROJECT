import os
from stable_baselines3 import PPO

# Path to your latest RL model for the timeframe (edit as needed)
MODEL_PATH = r"C:\Users\open\Documents\ZENO_XAUUSD\models\rl_policy_H1_latest.zip"

# Load the model ONCE (not every call)
_model = None

def load_rl_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"RL model not found at {MODEL_PATH}")
        _model = PPO.load(MODEL_PATH)
    return _model

def rl_decision(obs):
    """
    obs: 1D numpy array of engineered features (float32)
    Returns: action (int) 0=hold, 1=buy, 2=sell
    """
    model = load_rl_model()
    action, _ = model.predict(obs, deterministic=True)
    return int(action)
