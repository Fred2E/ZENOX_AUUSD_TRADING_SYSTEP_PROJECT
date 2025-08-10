import os
import pandas as pd
import numpy as np
import joblib

# === CONFIG ===
DATA_DIR = r"C:\Users\open\Documents\ZENO_XAUUSD\outputs\ml_data"
MODEL_DIR = r"C:\Users\open\Documents\ZENO_XAUUSD\outputs\ml_data"
TIMEFRAMES = ["M15", "H1", "H4"]

# Model and features
ML_FEATURES = [
    "close", "high", "low", "atr", "score", "num_confs", "pattern_code",
    "bias_bull", "hour", "dow", "primary_score", "secondary_score", "total_confluence",
    "regime_trend", "conf_sr_zone", "conf_bos_or_choch", "conf_psych_level",
    "conf_fib_zone", "conf_volume", "conf_liquidity", "conf_spread"
]

def load_model(tf):
    model_path = os.path.join(MODEL_DIR, f"zeno_lgbm_{tf}.pkl")
    if not os.path.exists(model_path):
        print(f"[SKIP] ML model for {tf} not found: {model_path}")
        return None
    return joblib.load(model_path)

def apply_confluence_grades(df):
    # --- Institutional A+/A/B logic; tweak as needed ---
    a_plus = (df['primary_score'] >= 6) & (df['score'] >= 6)
    a = (df['primary_score'] >= 4) & (df['score'] >= 4)
    b = (df['primary_score'] >= 2)
    df['setup_grade'] = np.select([a_plus, a, b], ['A+', 'A', 'B'], default='NONE')
    return df

def apply_ml_signals(df, model, threshold=0.5):
    X = df[ML_FEATURES].astype(float)
    prob_win = model.predict_proba(X)[:, 1]
    df['ml_prob_win'] = prob_win
    df['ml_signal'] = (prob_win >= threshold).astype(int)
    return df

for tf in TIMEFRAMES:
    infile = os.path.join(DATA_DIR, f"trade_events_{tf}_FULL.csv")
    if not os.path.exists(infile):
        print(f"[SKIP] {infile} not found.")
        continue
    df = pd.read_csv(infile)
    print(f"\n[{tf}] Loaded event file: {df.shape}")

    # 1. Confluence scoring/classic signals
    df = apply_confluence_grades(df)

    # 2. ML signals
    model = load_model(tf)
    if model is not None:
        df = apply_ml_signals(df, model, threshold=0.5)

        # Save ML signals
        ml_signals = df[df['ml_signal'] == 1].copy()
        out_ml = os.path.join(DATA_DIR, f"trade_signals_{tf}_ML.csv")
        ml_signals.to_csv(out_ml, index=False)
        print(f"[OK] ML signals saved: {out_ml} ({len(ml_signals)})")

    # Save confluence-based setups
    for grade in ["A+", "A", "B"]:
        out_conf = os.path.join(DATA_DIR, f"trade_signals_{tf}_{grade}.csv")
        conf_signals = df[df['setup_grade'] == grade].copy()
        conf_signals.to_csv(out_conf, index=False)
        print(f"[OK] Confluence signals ({grade}) saved: {out_conf} ({len(conf_signals)})")

    # (Optional) Save hybrid file: Only A+ setups that also pass ML
    if model is not None:
        hybrid = df[(df['setup_grade'] == 'A+') & (df['ml_signal'] == 1)].copy()
        out_hybrid = os.path.join(DATA_DIR, f"trade_signals_{tf}_A+_ML.csv")
        hybrid.to_csv(out_hybrid, index=False)
        print(f"[OK] Hybrid A+∩ML signals saved: {out_hybrid} ({len(hybrid)})")

print("\n✅ SIGNAL ENGINE COMPLETE: Confluence & ML. Compare both for walkforward.")
