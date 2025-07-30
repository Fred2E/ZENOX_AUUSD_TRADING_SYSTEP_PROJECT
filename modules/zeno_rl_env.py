import gym
from gym import spaces
import numpy as np
import pandas as pd
import zeno_config  # Centralized rules

class ZenoRLTradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df, timeframe='M5', initial_balance=100000, features=None,
                 max_steps=500, log_path=None, verbose=True, pip_value=100, spread=0.5):
        super().__init__()
        self.df = df.copy().reset_index(drop=True)
        self.df.columns = [c.lower() for c in self.df.columns]
        self.timeframe = timeframe

        # --- Features and tf-specific config ---
        assert features and len(features) > 0, "Must pass list of RL_FEATURES."
        self.features = features

        self.tf_entry = zeno_config.ENTRY_FILTERS
        self.tf_bias = zeno_config.TF_BIAS_PARAMS.get(self.timeframe, zeno_config.TF_BIAS_PARAMS['M15'])
        self.tf_swing = zeno_config.SWING_PARAMS.get(self.timeframe, zeno_config.SWING_PARAMS['M15'])

        # --- Required columns
        required = self.features + [
            'close', 'datetime', 'atr', 'score', 'num_confs', 'trend_state', 'bias_bull',
            'conf_structure', 'conf_bos_or_choch', 'conf_candle', 'conf_sr_zone'
        ]
        missing = set(required) - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        self.df = self.df.dropna(subset=required).reset_index(drop=True)

        self.initial_balance = initial_balance
        self.max_steps = max_steps
        self.log_path = log_path
        self.verbose = verbose
        self.pip_value = pip_value
        self.spread = spread

        self.action_space = spaces.Discrete(3)  # [0: hold, 1: long, 2: short]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.features),), dtype=np.float32)

        self._reset_internal()
        self.trades = []

    def _reset_internal(self):
        self.balance = self.initial_balance
        self.position = 0
        self.current_step = 0
        self.pnl = 0.0
        self.entry_price = None
        self.sl = None
        self.tp = None
        self.lot_size = 0
        self.sl_pips = None
        self.tp_pips = None
        self.trade_start_step = None
        self.last_trade_step = -999
        self.last_direction = None

    def reset(self):
        self._reset_internal()
        return self._next_observation()

    def _next_observation(self):
        if self.current_step >= len(self.df):
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        return self.df.loc[self.current_step, self.features].astype(np.float32).values

    def _is_trade_allowed(self, row):
        tf = self.timeframe
        min_score = self.tf_entry['min_score'][tf]
        min_confs = self.tf_entry['min_confs'][tf]
        min_primary = self.tf_entry['min_primary_confs'][tf]
        atr_min = self.tf_entry['atr_threshold'][tf]
        bias_required = self.tf_entry['bias_match_required'][tf]

        # --- Adaptive entry checks ---
        if self.current_step - self.last_trade_step < 10:
            return False
        if row['score'] < min_score:
            return False
        if row['num_confs'] < min_confs:
            return False
        if (
            row['conf_structure'] + row['conf_bos_or_choch'] +
            row['conf_candle'] + row['conf_sr_zone']
        ) < min_primary:
            return False
        if row['atr'] < atr_min:
            return False
        if 'trend_state' in row and self.tf_entry.get('trend_required') and row['trend_state'] not in self.tf_entry['trend_required']:
            return False
        if bias_required:
            expected = 1 if row.get('trend_state', 'bull') == 'bull' else 0
            if row['bias_bull'] != expected:
                return False
        return True

    def _position_size(self, sl_distance):
        # --- Use MAX_LOT_RISK and pip_value from config ---
        risk_amount = self.balance * zeno_config.MAX_LOT_RISK
        effective_sl = sl_distance + self.spread
        lot = risk_amount / (effective_sl * self.pip_value)
        return np.round(max(lot, 0.01), 2)

    def _sl_tp_calc(self, entry_price, atr, direction):
        # --- Use tf-adaptive RR from config if available ---
        rr = 2.0  # Default R:R
        sl = entry_price - atr if direction == 1 else entry_price + atr
        tp = entry_price + rr * (entry_price - sl) if direction == 1 else entry_price - rr * (sl - entry_price)
        return round(sl, 2), round(tp, 2), abs(entry_price - sl)

    def step(self, action):
        done = False
        reward = 0.0

        if self.current_step >= len(self.df):
            if self.position != 0:
                self._close_trade(self.df.iloc[-1]['close'], 'forced_close')
            return self._next_observation(), reward, True, {}

        row = self.df.iloc[self.current_step]
        price = row['close']

        if self.position == 0 and action in [1, 2] and self._is_trade_allowed(row):
            self.position = 1 if action == 1 else -1
            self.entry_price = price
            self.trade_start_step = self.current_step
            self.sl, self.tp, sl_pips = self._sl_tp_calc(price, row['atr'], self.position)
            self.sl_pips = sl_pips
            self.tp_pips = 2 * sl_pips
            self.lot_size = self._position_size(sl_pips)
            self.last_trade_step = self.current_step
            self.last_direction = self.position

        if self.position != 0:
            reward += self._manage_trade(price, row)

        self.pnl += reward
        self.current_step += 1
        done = self.current_step >= len(self.df) or self.current_step >= self.max_steps
        if done and self.position != 0:
            reward += self._close_trade(price, 'forced_close')

        return self._next_observation(), reward, done, {'pnl': self.pnl, 'balance': self.balance}

    def _manage_trade(self, price, row):
        hit_tp = price >= self.tp if self.position == 1 else price <= self.tp
        hit_sl = price <= self.sl if self.position == 1 else price >= self.sl
        if hit_tp:
            return self._close_trade(price, 'TP')
        elif hit_sl:
            return self._close_trade(price, 'SL')
        elif self.current_step - self.trade_start_step >= 15:
            return self._close_trade(price, 'timeout')
        return 0.0

    def _close_trade(self, price, reason):
        direction = 1 if self.position == 1 else -1
        base_pips = (price - self.entry_price) * direction
        reward = base_pips * self.lot_size * self.pip_value
        reward -= 0.1 * self.pip_value  # slippage/fee

        if reason == 'SL': reward -= 2.0
        if reason == 'TP': reward += 1.0

        self.balance += reward
        safe_index = min(self.current_step, len(self.df) - 1)

        self.trades.append({
            'entry_step': self.trade_start_step,
            'exit_step': self.current_step,
            'entry_price': self.entry_price,
            'exit_price': price,
            'lot_size': self.lot_size,
            'sl_pips': self.sl_pips,
            'tp_pips': self.tp_pips,
            'reason': reason,
            'reward': reward,
            'balance': self.balance,
            'datetime': self.df['datetime'].iloc[safe_index],
        })
        self.position = 0
        return reward

    def render(self, mode='human'):
        print(f"Step={self.current_step} | Balance={self.balance:.2f} | PnL={self.pnl:.2f}")

    def save_trades(self):
        if self.log_path and self.trades:
            pd.DataFrame(self.trades).to_csv(self.log_path, index=False)
            if self.verbose:
                print(f"✅ Trade log saved to {self.log_path}")

