import os
import pandas as pd
import lightgbm as lgb
import joblib

# === CONFIG ===
DATA_DIR = r"C:\Users\open\Documents\ZENO_XAUUSD\data\live\XAUUSD"
OUTPUT_MODEL_PATH = r"C:\Users\open\Documents\ZENO_XAUUSD\outputs\ml_data\zeno_lgbm.pkl"
TIMEFRAMES = ["M5", "M15", "H1", "H4", "D1"]  # Use whichever you want for training (e.g., M15+H1+H4+D1, avoid only M5 for now)
ML_FEATURES = ['close', 'score', 'num_confs', 'pattern_code', 'bias_bull', 'hour', 'dow']
TARGET_COL = 'score'  # Use your actual target here (change as needed)

# Confluence logic
PRIMARY_CONFS = ['conf_structure', 'conf_bos_or_choch', 'conf_candle', 'conf_sr_zone']
SECONDARY_CONFS = ['conf_psych_level', 'conf_fib_zone', 'bias_bull']

def compute_confluence_scores(df):
    """
    Compute confluence scores based on primary and secondary factors.

    Args:
    df (pandas.DataFrame): DataFrame containing the confluence data.

    Returns:
    pandas.DataFrame: DataFrame with computed confluence scores.
    """
    # Primary score is the sum of all primary confs
    df['primary_score'] = df[PRIMARY_CONFS].sum(axis=1)

    # Secondary score is the sum of all secondary confs
    df['secondary_score'] = df[SECONDARY_CONFS].sum(axis=1)

    # Total confluence score is the sum of primary and secondary scores
    df['total_confluence'] = df['primary_score'] + df['secondary_score']

    return df

def load_feature_data(timeframes):
    dfs = []
    for tf in timeframes:
        path = os.path.join(DATA_DIR, tf, f"XAUUSD_{tf}_LIVE_FEATURES.csv")
        if not os.path.exists(path):
            print(f"[WARN] {tf}: File not found: {path}")
            continue
        df = pd.read_csv(path)
        # --- Enforce lowercase --- 
        df.columns = [c.lower() for c in df.columns]
        
        # Apply confluence scoring logic
        df = compute_confluence_scores(df)

        dfs.append(df)
    if not dfs:
        raise FileNotFoundError("No feature files found for training!")
    return pd.concat(dfs, ignore_index=True)

def train_and_save_model(df, features, target, model_path):
    # === Check for feature compliance ===
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Training data is missing columns: {missing}")
    X = df[features]
    y = df[target]

    print(f"Training shape: {X.shape}, Features: {features}")

    # --- Train LightGBM Model ---
    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=32,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)

    # Save the trained model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)  # Ensure the directory exists
    joblib.dump(model, model_path)
    print(f"[SUCCESS] Model trained and saved to {model_path}")

def main():
    print("=== ML Model Training (LOWERCASE FEATURES, RL-ALIGNED) ===")
    # 1. Load and stack all feature CSVs
    df = load_feature_data(TIMEFRAMES)

    # 2. Drop rows with missing required features (final sanity)
    df = df.dropna(subset=ML_FEATURES + [TARGET_COL])

    # Optional: filter rows with 'score'==0 or whatever you consider as no-signal
    # df = df[df['score'] > 0]  # Uncomment if needed

    # 3. Train & Save
    train_and_save_model(df, ML_FEATURES, TARGET_COL, OUTPUT_MODEL_PATH)

    # 4. Output quick validation
    print("Feature columns in model:", ML_FEATURES)
    print(df[ML_FEATURES + [TARGET_COL]].tail(5))

if __name__ == "__main__":
    main()
