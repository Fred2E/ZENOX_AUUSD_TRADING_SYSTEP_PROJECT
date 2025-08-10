import os
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = r"C:\Users\open\Documents\ZENO_XAUUSD"
EVENT_DIR = os.path.join(ROOT, "outputs", "ml_data")
ML_OUT = os.path.join(ROOT, "outputs", "ml_data")
os.makedirs(ML_OUT, exist_ok=True)

TIMEFRAMES = ["M5", "M15", "H1", "H4"]

ML_FEATURES = [
    "close", "score", "num_confs", "pattern_code", "bias_bull", "hour", "dow",
    "primary_score", "secondary_score", "total_confluence", "regime_trend"
]
ML_LABEL = "is_win"

def compute_label(row):
    if 'outcome' in row:
        if row['outcome'] == "win":
            return 1
        elif row['outcome'] == "loss":
            return 0
        elif row['outcome'] == "breakeven":
            return -1
    if 'reward' in row:
        return int(row['reward'] > 0)
    return None

def build_dataset_per_tf():
    for tf in TIMEFRAMES:
        event_path = os.path.join(EVENT_DIR, f"trade_events_{tf}_FULL.csv")
        if not os.path.exists(event_path):
            print(f"[WARN] Missing event file: {event_path}")
            continue
        df = pd.read_csv(event_path)
        df.columns = [c.lower() for c in df.columns]
        df['is_win'] = df.apply(compute_label, axis=1)
        cols_keep = [f for f in ML_FEATURES if f in df.columns] + [ML_LABEL]
        df = df[cols_keep].dropna(subset=[ML_LABEL])
        df = df[df[ML_LABEL] != -1]
        pos = (df[ML_LABEL] == 1).sum()
        neg = (df[ML_LABEL] == 0).sum()
        print(f"\n[{tf}] ML Samples: {len(df)} | Pos: {pos} | Neg: {neg}")

        # Small data patch: skip stratified split if not enough data for both classes
        if min(pos, neg) < 2:
            print(f"[WARN] {tf}: Not enough samples for stratified split. Saving all data as both train and test.")
            train = df.copy()
            test = df.copy()
        else:
            X = df[ML_FEATURES]
            y = df[ML_LABEL]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=42
            )
            train = X_train.copy()
            train[ML_LABEL] = y_train
            test = X_test.copy()
            test[ML_LABEL] = y_test

        train_path = os.path.join(ML_OUT, f"train_ml_{tf}.pkl")
        test_path  = os.path.join(ML_OUT, f"test_ml_{tf}.pkl")
        train.to_pickle(train_path)
        test.to_pickle(test_path)
        print(f"[SAVED] {tf}: Train: {train.shape} | Test: {test.shape}")
        print(f"Train label dist:\n{train[ML_LABEL].value_counts()}")
        print(f"Test label dist:\n{test[ML_LABEL].value_counts()}")

if __name__ == "__main__":
    build_dataset_per_tf()
