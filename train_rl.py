# train_rl.py

import os
import gymnasium as gym
import numpy as np
import pandas as pd

# ─── FORCE SB3 → TENSORBOARDX ─────────────────────────────────────────────────
# (must run before any stable_baselines3 imports)
try:
    import tensorboardX
    import torch.utils.tensorboard as torch_tb
    torch_tb.SummaryWriter = tensorboardX.SummaryWriter
    print("✅ Patched SB3 to use tensorboardX")
except ImportError:
    print("⚠️ tensorboardX not installed—skipping TensorBoard patch")

# ─── NOW IMPORT SB3 ─────────────────────────────────────────────────────────────
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium import spaces

# ─── CONFIG ────────────────────────────────────────────────────────────────────
ROOT         = r"C:\Users\open\Documents\ZENO_XAUUSD"
HIST_FILE    = os.path.join(ROOT, "historical",   "XAUUSDm_H1_HIST.csv")
SIG_FILE     = os.path.join(ROOT, "outputs",      "setups", "ZENO_A+_signals_H1.csv")
MODEL_OUT    = os.path.join(ROOT, "outputs",      "ml_data", "zeno_rl_agent.zip")
INITIAL_CASH = 100_000.0
POSITION_SZ  = 1.0

# ─── ENVIRONMENT ────────────────────────────────────────────────────────────────
class ZenoTradingEnv(gym.Env):
    def __init__(self, price_csv, sig_csv):
        super().__init__()
        # load and align
        self.df_price = pd.read_csv(price_csv, parse_dates=["Datetime"]).set_index("Datetime")
        self.df_sig   = pd.read_csv(sig_csv,   parse_dates=["Datetime"]).set_index("Datetime")
        self.df       = (
            self.df_price
              .join(self.df_sig[["score"]], how="left")
              .fillna(0)
        )
        self.n_steps = len(self.df)
        # spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)  # 0=hold,1=buy,2=sell
        self.reset()

    def reset(self, **kwargs):
        self.step_idx  = 0
        self.cash      = INITIAL_CASH
        self.position  = 0.0
        self.avg_price = 0.0
        obs = self._get_obs()
        return obs, {}

    def _get_obs(self):
        row = self.df.iloc[self.step_idx]
        return np.array([row.Close, row.score], dtype=np.float32)

    def step(self, action):
        # get current price & reward container
        price  = self.df.iloc[self.step_idx].Close
        reward = 0.0

        # EXECUTE
        if action == 1 and self.position == 0:
            self.position  = POSITION_SZ
            self.avg_price = price
        elif action == 2 and self.position == 0:
            self.position  = -POSITION_SZ
            self.avg_price = price
        elif action == 0 and self.position != 0:
            pnl           = (price - self.avg_price) * self.position
            reward       += pnl
            self.cash    += pnl
            self.position = 0.0

        # advance index
        self.step_idx += 1
        done = self.step_idx >= self.n_steps

        # force‐close PnL at end
        if done and self.position != 0:
            final_price  = self.df.iloc[-1].Close
            pnl          = (final_price - self.avg_price) * self.position
            reward      += pnl
            self.cash   += pnl
            self.position = 0.0

        # next_obs: if we're out of data, just repeat last valid row
        if not done:
            next_obs = self._get_obs()
        else:
            # use the final row for obs to keep shapes consistent
            last = self.df.iloc[-1]
            next_obs = np.array([last.Close, last.score], dtype=np.float32)

        return next_obs, reward, done, False, {}

# ─── TRAIN SCRIPT ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # wrap environment
    def make_env():
        return Monitor(ZenoTradingEnv(HIST_FILE, SIG_FILE))
    vecenv = DummyVecEnv([make_env])

    # init PPO on CPU, no tensorboard_log so it won't try TF
    model = PPO("MlpPolicy", vecenv, verbose=1, device="cpu")

    print("⏳ Starting training for 200k timesteps…")
    model.learn(total_timesteps=200_000)
    print("✅ Training done, saving model to:\n   ", MODEL_OUT)

    model.save(MODEL_OUT)
    print("✅ RL agent saved as ZIP.")
