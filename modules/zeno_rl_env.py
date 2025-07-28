import gym
from gym import spaces
import numpy as np
import pandas as pd

class ZenoRLTradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df, timeframe='M5', initial_balance=100000, features=None,
                 max_steps=500, log_path=None, verbose=True, pip_value=100, spread=0.5):
        super().__init__()
        self.df = df.copy().reset_index(drop=True)
        self.df.columns = [c.lower() for c in self.df.columns]

        assert features is not None and len(features) > 0, "You must pass a list of RL_FEATURES."
        self.features = features

        self.PRIMARY_CONFS = [c for c in ['conf_structure', 'conf_bos_or_choch', 'conf_candle', 'conf_sr_zone'] if c in self.features]
        self.SECONDARY_CONFS = [c for c in ['conf_psych_level', 'conf_fib_zone', 'conf_volume', 'conf_liquidity', 'conf_spread'] if c in self.features]

        missing = set(self.features + ['close', 'datetime']) - set(self.df.columns)
        if missing:
            raise ValueError(f"Env Init: Missing columns: {missing}")
        self.df = self.df.dropna(subset=self.features + ['close', 'datetime']).reset_index(drop=True)

        self.timeframe = timeframe
        self.initial_balance = initial_balance
        self.max_steps = max_steps
        self.log_path = log_path
        self.verbose = verbose
        self.pip_value = pip_value
        self.spread = spread

        self.action_space = spaces.Discrete(3)  # 0=hold, 1=long, 2=short
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.features),), dtype=np.float32)

        self._reset_internal()

    def _reset_internal(self):
        self.balance = self.initial_balance
        self.position = 0
        self.current_step = 0
        self.pnl = 0.0
        self.trades = []
        self.entry_price = None
        self.sl = None
        self.tp = None
        self.partial_tp_hit = False
        self.trade_duration = 0
        self.breakeven_applied = False
        self.lot_size = 0
        self.risk_pct = 0
        self.sl_pips = None
        self.tp_pips = None
        self.trade_start_step = None

    def reset(self):
        self._reset_internal()
        return self._next_observation()

    def _next_observation(self):
        if self.current_step >= len(self.df):
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        return self.df.loc[self.current_step, self.features].astype(np.float32).values

    def _confluence_score(self, row):
        primary = sum(int(row.get(c, 0)) for c in self.PRIMARY_CONFS)
        secondary = sum(int(row.get(c, 0)) for c in self.SECONDARY_CONFS)
        return primary, secondary

    def _get_risk_pct(self, primary_count, secondary_count, drawdown=0.0, equity=None):
        # NEW: Loosen threshold, use your risk management logic
        if primary_count < 2:    # Instead of requiring ALL primary, allow at least 2
            return 0.0
        risk_pct = 0.003         # Slightly more conservative base risk
        if secondary_count >= 2:
            risk_pct = 0.007
        if secondary_count >= 3:
            risk_pct = 0.01
        if drawdown > 0.2:
            risk_pct *= 0.8
        if drawdown > 0.5:
            risk_pct = 0.002
        return risk_pct

    def _sl_tp_calc(self, entry_price, atr, direction):
        sl = entry_price - atr if direction == 1 else entry_price + atr
        tp = entry_price + 2 * (entry_price - sl) if direction == 1 else entry_price - 2 * (sl - entry_price)
        return round(sl, 2), round(tp, 2), abs(entry_price - sl)

    def _position_size(self, risk_pct, entry_price, sl, direction):
        position_risk = self.balance * risk_pct
        stop_loss_distance = abs(entry_price - sl)
        effective_stop = stop_loss_distance + self.spread
        if effective_stop == 0:
            return 0.0
        lot_size = position_risk / (effective_stop * self.pip_value)
        return np.round(lot_size, 2)

    def _is_trade_allowed(self, row):
        primary, secondary = self._confluence_score(row)
        risk_pct = self._get_risk_pct(primary, secondary)
        if self.verbose:
            print(f"Step {self.current_step}: Primary={primary}, Secondary={secondary}, risk_pct={risk_pct}, Score={row.get('score', 'NA')}, NumConfs={row.get('num_confs', 'NA')}")
        if risk_pct == 0.0:
            return False, risk_pct
        return True, risk_pct

    def step(self, action):
        done = False
        reward = 0.0

        if self.current_step >= len(self.df):
            done = True
            return np.zeros(self.observation_space.shape, dtype=np.float32), reward, done, {'reason': 'end_of_data'}

        row = self.df.iloc[self.current_step]
        price = row['close']

        trade_allowed, risk_pct = self._is_trade_allowed(row)

        if self.position == 0 and action in [1, 2] and trade_allowed:
            self.position = 1 if action == 1 else -1
            self.entry_price = price
            self.trade_start_step = self.current_step
            self.risk_pct = risk_pct

            atr = row['atr']
            self.sl, self.tp, sl_pips = self._sl_tp_calc(price, atr, self.position)
            self.sl_pips = sl_pips
            self.tp_pips = 2 * sl_pips
            self.lot_size = self._position_size(risk_pct, price, self.sl, self.position)
            self.partial_tp_level = price + sl_pips if self.position == 1 else price - sl_pips

            if self.verbose:
                print(f"[OPEN] {row['datetime']} | {'LONG' if self.position == 1 else 'SHORT'} @ {price:.2f} | "
                      f"SL={self.sl:.2f} TP={self.tp:.2f} Lot={self.lot_size:.2f}")

            reward += 0.2

        if self.position != 0:
            self.trade_duration += 1
            reward += self._manage_open_trade(price, row)

        self.pnl += reward
        self.current_step += 1

        done = self.current_step >= len(self.df) or self.current_step >= self.max_steps
        if done:
            self.save_trades()

        return self._next_observation(), reward, done, {'pnl': self.pnl, 'balance': self.balance}

    def _manage_open_trade(self, price, row):
        reward = 0.0
        be_trigger = self.entry_price + 0.7 * (self.tp - self.entry_price) if self.position == 1 else \
                     self.entry_price - 0.7 * (self.entry_price - self.tp)
        if not self.breakeven_applied and \
                ((self.position == 1 and price >= be_trigger) or (self.position == -1 and price <= be_trigger)):
            self.sl = self.entry_price
            self.breakeven_applied = True
            if self.verbose:
                print(f"[BE] SL moved to breakeven @ {self.sl:.2f}")

        if not self.partial_tp_hit:
            partial_tp_level = self.partial_tp_level
            if (self.position == 1 and price >= partial_tp_level) or (self.position == -1 and price <= partial_tp_level):
                self.partial_tp_hit = True
                partial_reward = 0.5 * abs(partial_tp_level - self.entry_price) * self.lot_size * self.pip_value
                self.balance += partial_reward
                reward += partial_reward / (self.lot_size * self.sl_pips * self.pip_value)
                if self.verbose:
                    print(f"[TP1] Partial TP hit @ {partial_tp_level:.2f} | Partial Reward: {partial_reward:.2f}")

        hit_tp = price >= self.tp if self.position == 1 else price <= self.tp
        hit_sl = price <= self.sl if self.position == 1 else price >= self.sl

        if hit_tp or hit_sl:
            pip_reward = (price - self.entry_price) * self.position * self.lot_size * self.pip_value
            reward += pip_reward / (self.lot_size * self.sl_pips * self.pip_value)
            if self.verbose:
                print(f"[EXIT] {'TP' if hit_tp else 'SL'} @ {price:.2f} | PnL: {pip_reward:.2f}")
            self._close_trade(price, row, reward)
        elif self.trade_duration >= 15:
            pip_reward = (price - self.entry_price) * self.position * self.lot_size * self.pip_value - 0.2
            reward += pip_reward / (self.lot_size * self.sl_pips * self.pip_value)
            if self.verbose:
                print(f"[EXIT] Duration maxed out @ {price:.2f} | PnL: {pip_reward:.2f}")
            self._close_trade(price, row, reward)
        return reward

    def _close_trade(self, price, row, reward):
        if self.verbose:
            print(f"[CLOSE] {row['datetime']} | Exit @ {price:.2f} | Reward: {reward:.2f}")

        trade_record = {
            'entry_step': self.trade_start_step,
            'exit_step': self.current_step,
            'entry_price': self.entry_price,
            'exit_price': price,
            'sl_pips': self.sl_pips,
            'tp_pips': self.tp_pips,
            'side': 'long' if self.position == 1 else 'short',
            'lot_size': self.lot_size,
            'risk_pct': self.risk_pct,
            'reward': reward,
            'score': row.get('score', None),
            'pattern_code': row.get('pattern_code', None),
            'datetime': row.get('datetime', None),
            'balance': self.balance
        }
        self.trades.append(trade_record)
        self.position = 0
        self.trade_duration = 0
        self.sl = None
        self.tp = None
        self.entry_price = None
        self.breakeven_applied = False
        self.partial_tp_hit = False
        self.lot_size = 0
        self.risk_pct = 0

    def render(self, mode='human'):
        print(f"[Step {self.current_step}] Pos={self.position}, Balance={self.balance:.2f}, PnL={self.pnl:.2f}")

    def save_trades(self):
        if self.log_path and self.trades:
            pd.DataFrame(self.trades).to_csv(self.log_path, index=False)
            if self.verbose:
                print(f"✅ Trade log saved to {self.log_path}")
