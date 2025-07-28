import gym
from gym import spaces
import numpy as np
import pandas as pd

RL_FEATURES = ['score', 'num_confs', 'pattern_code', 'bias_bull', 'hour', 'dow', 'atr']

class ZenoTradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df, timeframe='M5', initial_balance=10000, features=None, max_steps=500, log_path=None, verbose=True):
        super(ZenoTradingEnv, self).__init__()
        self.df = df.copy().reset_index(drop=True)
        self.df.columns = [c.lower() for c in self.df.columns]

        self.features = features or RL_FEATURES
        self.df = self.df.dropna(subset=self.features + ['close', 'datetime']).reset_index(drop=True)

        self.timeframe = timeframe
        self.initial_balance = initial_balance
        self.max_steps = max_steps
        self.log_path = log_path
        self.verbose = verbose

        self.action_space = spaces.Discrete(3)  # 0=hold, 1=long, 2=short
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.features),), dtype=np.float32)

        self._reset_internal()

    def _reset_internal(self):
        self.balance = self.initial_balance
        self.position = 0
        self.current_step = 0
        self.pnl = 0
        self.trades = []
        self.entry_price = None
        self.sl = None
        self.tp = None
        self.partial_tp_hit = False
        self.trade_duration = 0
        self.breakeven_applied = False

    def reset(self):
        self._reset_internal()
        return self._next_observation()

    def _next_observation(self):
        if self.current_step >= len(self.df):
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        return self.df.loc[self.current_step, self.features].astype(np.float32).values

    def _detect_regime(self, row):
        if row['atr'] < 5 and row['bias_bull'] == 0:
            return 'flat'
        elif row['atr'] > 12:
            return 'breakout'
        else:
            return 'normal'

    def _is_valid_signal(self, row):
        score = row['score']
        confs = row['num_confs']
        pattern_code = row['pattern_code']
        regime = self._detect_regime(row)
        if self.timeframe == 'M5':
            return regime != 'flat' and score >= 2 and confs >= 2 and pattern_code > 0
        elif self.timeframe == 'M15':
            return score >= 3 and confs >= 3 and pattern_code > 0
        elif self.timeframe == 'H1':
            return score >= 2.5 and confs >= 3 and pattern_code in [1, 2, 3]
        elif self.timeframe == 'H4':
            return score >= 3 and confs >= 3 and pattern_code in [1, 2, 3, 4] and row['bias_bull'] != 0 and row['atr'] >= 8
        return False

    def step(self, action):
        if self.current_step >= len(self.df):
            return np.zeros(self.observation_space.shape, dtype=np.float32), -1.0, True, {'reason': 'Out of data'}

        row = self.df.iloc[self.current_step]
        price = row['close']
        reward = 0.0

        # Regime filter penalty
        if row['bias_bull'] == 0 and row['atr'] < 5:
            reward -= 0.3
            self.current_step += 1
            return self._next_observation(), reward, False, {'reason': 'Regime filtered'}

        valid_signal = self._is_valid_signal(row)

        # Enter new position
        if self.position == 0 and action in [1, 2] and valid_signal:
            self.position = 1 if action == 1 else -1
            self.entry_price = price
            atr = row['atr']

            # Timeframe-specific SL/TP multipliers and breakeven thresholds
            if self.timeframe == 'M5':
                sl_mult, tp_mult, be_threshold = 1.5, 2.0, 5
            elif self.timeframe == 'M15':
                sl_mult, tp_mult, be_threshold = 2.0, 2.0, 8
            elif self.timeframe == 'H1':
                sl_mult, tp_mult, be_threshold = 2.0, 2.0, 12
            else:  # H4
                sl_mult, tp_mult, be_threshold = 2.5, 2.5, 15

            self.sl_pips = np.clip(atr * sl_mult, 10, 100)
            self.tp_pips = self.sl_pips * tp_mult
            self.be_threshold = be_threshold

            self.sl = price - self.sl_pips if self.position == 1 else price + self.sl_pips
            self.tp = price + self.tp_pips if self.position == 1 else price - self.tp_pips
            self.partial_tp_level = price + self.sl_pips if self.position == 1 else price - self.sl_pips
            self.trade_start_step = self.current_step

            if self.verbose:
                print(f"[OPEN] {row['datetime']} | {'LONG' if self.position == 1 else 'SHORT'} @ {price:.2f} | SL={self.sl:.2f} TP={self.tp:.2f}")

            reward += 0.2

        # Manage open trade
        if self.position != 0:
            self.trade_duration += 1
            reward += self._manage_open_trade(price, row)

        # Encourage holding trades longer than 5 steps
        if self.trade_duration > 5:
            reward += 0.05

        self.pnl += reward
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1 or self.current_step >= self.max_steps

        if done:
            self.save_trades()

        return self._next_observation(), reward, done, {'pnl': self.pnl, 'balance': self.balance}

    def _manage_open_trade(self, price, row):
        reward = 0.0
        gain = (price - self.entry_price) * self.position

        # Emergency loss cap
        if abs(gain) > 500:
            reward = -2.0
            self._close_trade(price, row, reward)
            return reward

        # Breakeven trigger
        if not self.breakeven_applied and gain >= self.be_threshold:
            self.sl = self.entry_price
            self.breakeven_applied = True
            if self.verbose:
                print(f"[BE] SL moved to breakeven @ {self.sl:.2f}")

        # Trailing stop loss
        if self.breakeven_applied and gain >= self.sl_pips * 1.2:
            trailing_buffer = 0.5 * self.sl_pips
            if self.position == 1:
                self.sl = max(self.sl, price - trailing_buffer)
            else:
                self.sl = min(self.sl, price + trailing_buffer)

        # Partial take profit
        if not self.partial_tp_hit:
            if (self.position == 1 and price >= self.partial_tp_level) or (self.position == -1 and price <= self.partial_tp_level):
                self.partial_tp_hit = True
                partial_reward = 0.5 * (self.partial_tp_level - self.entry_price) * self.position
                self.balance += partial_reward
                reward += partial_reward / self.sl_pips
                if self.verbose:
                    print(f"[TP1] Partial TP hit @ {self.partial_tp_level:.2f} | Partial Reward: {partial_reward:.2f}")

        # Check for full TP or SL hit
        hit_tp = price >= self.tp if self.position == 1 else price <= self.tp
        hit_sl = price <= self.sl if self.position == 1 else price >= self.sl

        if hit_tp or hit_sl:
            pip_reward = (price - self.entry_price) * self.position / self.sl_pips
            reward += pip_reward
            if self.verbose:
                print(f"[EXIT] {'TP' if hit_tp else 'SL'} @ {price:.2f} | PnL: {pip_reward:.2f}")
            self._close_trade(price, row, reward)
        elif self.trade_duration >= 15:
            # Timeout exit, penalty included
            pip_reward = (price - self.entry_price) * self.position / self.sl_pips - 0.2
            reward += pip_reward
            if self.verbose:
                print(f"[EXIT] Duration maxed out @ {price:.2f} | PnL: {pip_reward:.2f}")
            self._close_trade(price, row, reward)

        return reward

    def _close_trade(self, price, row, reward):
        if self.verbose:
            print(f"[CLOSE] {row['datetime']} | Exit @ {price:.2f} | Reward: {reward:.2f}")
        self.trades.append({
            'entry_step': self.trade_start_step,
            'exit_step': self.current_step,
            'entry_price': self.entry_price,
            'exit_price': price,
            'sl_pips': self.sl_pips,
            'tp_pips': self.tp_pips,
            'side': 'long' if self.position == 1 else 'short',
            'reward': reward,
            'score': row['score'],
            'pattern_code': row['pattern_code'],
            'datetime': row['datetime'],
        })
        self.position = 0
        self.trade_duration = 0
        self.sl = None
        self.tp = None
        self.entry_price = None
        self.breakeven_applied = False
        self.partial_tp_hit = False

    def render(self, mode='human'):
        print(f"[Step {self.current_step}] Pos={self.position}, Balance={self.balance:.2f}, PnL={self.pnl:.2f}")

    def save_trades(self):
        if self.log_path and self.trades:
            pd.DataFrame(self.trades).to_csv(self.log_path, index=False)
            print(f"✅ Trade log saved to {self.log_path}")
