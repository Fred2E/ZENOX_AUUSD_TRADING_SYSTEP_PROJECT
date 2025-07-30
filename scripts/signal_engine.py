import os
import pandas as pd

# === Canonical Data Source ===
PROCESSED_DIR = r"C:\Users\open\Documents\ZENO_XAUUSD\historical\processed"
TIMEFRAMES = ["M5", "M15", "H1", "H4"]  # Remove D1 if you don’t maintain a D1 FULL file

REQUIRED_COLS = [
    'datetime', 'open', 'high', 'low', 'close', 'volume', 'pattern_code', 'bias_bull', 'atr',
    'conf_structure', 'conf_bos_or_choch', 'conf_candle', 'conf_sr_zone',
    'conf_psych_level', 'conf_fib_zone', 'conf_volume', 'conf_liquidity', 'conf_spread'
]
PRIMARY_CONFS = ['conf_structure', 'conf_bos_or_choch', 'conf_candle', 'conf_sr_zone']

def filter_primary_confs(df, n_required=3):
    # Returns True if row has at least n_required primary confluences
    return df[PRIMARY_CONFS].sum(axis=1) >= n_required

def load_signals_file(tf):
    path = os.path.join(PROCESSED_DIR, f"signals_{tf}_FULL.csv")
    if not os.path.exists(path):
        print(f"[{tf}] MISSING: {path}")
        return None
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    missing = [col for col in REQUIRED_COLS if col not in df.columns]
    if missing:
        print(f"[{tf}] ERROR: Missing columns: {missing}")
        return None
    return df

def audit_and_output_signals(tf):
    df = load_signals_file(tf)
    if df is None:
        return

    # Filter: At least 3/4 primary confluences, optional regime logic (add your own rules)
    mask = filter_primary_confs(df, n_required=3)  # & (df['bias_bull'] == 1)   # Uncomment for bullish regime only
    signals = df[mask].copy()
    print(f"\n=== {tf} ===")
    print(f"Total bars: {df.shape[0]}, Signals: {signals.shape[0]}")
    print(signals[['datetime', 'primary_score', 'secondary_score', 'total_confluence', 'bias_bull', 'pattern_code']].tail(5))

    # Save signals (optional, or adapt for your output needs)
    out_path = os.path.join(PROCESSED_DIR, f"trade_signals_{tf}.csv")
    signals.to_csv(out_path, index=False)
    print(f"[{tf}] Trade signals saved: {out_path}")

if __name__ == "__main__":
    for tf in TIMEFRAMES:
        audit_and_output_signals(tf)

print("\n✅ SIGNAL ENGINE: Now fully decoupled from feature pipeline. No legacy patching, no windowing, no cap. All outputs from canonical full-history files only.")
