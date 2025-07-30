# --- INSTITUTIONAL-GRADE ZENO CONFIG ---

ENTRY_FILTERS = {
    'min_primary_confs': {
        'M5': 2,
        'M15': 3,
        'H1': 3,
        'H4': 3,
        'D1': 3,
    },
    'min_score': {
        'M5': 2,
        'M15': 2,
        'H1': 2,
        'H4': 2,
        'D1': 2,
    },
    'min_confs': {
        'M5': 3,
        'M15': 3,
        'H1': 3,
        'H4': 3,
        'D1': 3,
    },
    'atr_threshold': {
        'M5': 1.0,
        'M15': 1.0,
        'H1': 1.5,
        'H4': 2.0,
        'D1': 2.0,
    },
    'trend_required': ['bull', 'bear'],
    'bias_match_required': {
        'M5': True,
        'M15': True,
        'H1': True,
        'H4': True,
        'D1': True,
    }
}

# --- Trend/bias parameters per timeframe ---
TF_BIAS_PARAMS = {
    'M5':  {'short_ema': 13,  'long_ema': 34},
    'M15': {'short_ema': 21,  'long_ema': 50},
    'H1':  {'short_ema': 21,  'long_ema': 100},
    'H4':  {'short_ema': 50,  'long_ema': 200},
    'D1':  {'short_ema': 100, 'long_ema': 200},
}

# --- Structure detection params per tf ---
SWING_PARAMS = {
    'M5':  {'swing_window': 3,  'min_swing_dist': 3},
    'M15': {'swing_window': 4,  'min_swing_dist': 4},
    'H1':  {'swing_window': 5,  'min_swing_dist': 5},
    'H4':  {'swing_window': 7,  'min_swing_dist': 7},
    'D1':  {'swing_window': 10, 'min_swing_dist': 10}
}

# --- SR and psych level params per tf ---
SR_LOOKBACK = {
    'M5': 10, 'M15': 20, 'H1': 30, 'H4': 50, 'D1': 100
}
SR_THRESHOLD = {
    'M5': 0.0015, 'M15': 0.0015, 'H1': 0.001, 'H4': 0.001, 'D1': 0.001
}
PSYCH_LEVELS = [50, 100, 250]
PSYCH_THRESH = {
    'M5': 1.5, 'M15': 2, 'H1': 2, 'H4': 2.5, 'D1': 3
}

# --- Candle pattern thresholds per tf ---
CANDLE_THRESH = {
    'M5':  {'wick_mult': 1.0, 'body_pct': 0.08, 'doji_body': 0.18},
    'M15': {'wick_mult': 1.2, 'body_pct': 0.10, 'doji_body': 0.16},
    'H1':  {'wick_mult': 1.5, 'body_pct': 0.12, 'doji_body': 0.14},
    'H4':  {'wick_mult': 1.7, 'body_pct': 0.13, 'doji_body': 0.12},
    'D1':  {'wick_mult': 2.0, 'body_pct': 0.15, 'doji_body': 0.10},
}

# --- Max risk per trade (always enforced) ---
MAX_LOT_RISK = 0.02

# --- Main tf/HTF mapping for regime logic ---
HTF_MAP = {
    'M5': 'H1', 'M15': 'H4', 'H1': 'D1', 'H4': 'D1', 'D1': None
}

# --- Centralize primary and secondary confluences for all modules ---
PRIMARY_CONFS = ['conf_structure', 'conf_bos_or_choch', 'conf_candle', 'conf_sr_zone']
SECONDARY_CONFS = ['conf_psych_level', 'conf_fib_zone', 'conf_volume', 'conf_liquidity', 'conf_spread']

# --- RL training params (tf-specific, use in RL agent scripts) ---
RL_TRAIN_PARAMS = {
    'M5':  {'n_steps': 256, 'batch_size': 64,  'total_timesteps': 40000, 'learning_rate': 3e-4},
    'M15': {'n_steps': 256, 'batch_size': 64,  'total_timesteps': 40000, 'learning_rate': 3e-4},
    'H1':  {'n_steps': 256, 'batch_size': 32,  'total_timesteps': 10000, 'learning_rate': 1e-4},
    'H4':  {'n_steps': 128, 'batch_size': 16,  'total_timesteps': 5000,  'learning_rate': 1e-4},
}

# --- Walkforward thresholds per tf (for walk_forward_backtest.py) ---
WALKFORWARD_PARAMS = {
    'M5':  {'min_trades': 30, 'min_winrate': 50.0},
    'M15': {'min_trades': 30, 'min_winrate': 50.0},
    'H1':  {'min_trades': 30, 'min_winrate': 50.0},
    'H4':  {'min_trades': 30, 'min_winrate': 50.0},
    'default': {'min_trades': 30, 'min_winrate': 50.0}
}

# --- Usage example in other scripts ---
# from zeno_config import ENTRY_FILTERS, TF_BIAS_PARAMS, SWING_PARAMS, SR_LOOKBACK, CANDLE_THRESH, MAX_LOT_RISK, PRIMARY_CONFS, RL_TRAIN_PARAMS, WALKFORWARD_PARAMS

