# modules/ml_model.py

import joblib
import os
import logging
import pandas as pd
from modules.feature_pipeline import engineer_features  # <- Use the same feature engineering as RL!

ROOT = r"C:\Users\open\Documents\ZENO_XAUUSD"
MODEL_PATH = os.path.join(ROOT, "outputs", "ml_data", "zeno_lgbm.pkl")

def load_model(model_path=MODEL_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"ML model file not found at {model_path}")
    return joblib.load(model_path)

def predict_ml(df):
    # Feature engineering (robust, centralized)
    df_features, features = engineer_features(df)
    # Model loading and prediction
    model = load_model()
    # Model/feature alignment check
    model_features = getattr(model, 'feature_names_in_', features)
    if list(model_features) != list(features):
        logging.warning(f"Feature misalignment: model expects {model_features}, input {features}")

    # NaN/inf checks
    if df_features[features].isnull().any().any():
        logging.error("Input features have NaN, replacing with 0")
        df_features[features] = df_features[features].fillna(0)
    # Model inference
    prob_col = "prob_win"
    df_features[prob_col] = model.predict_proba(df_features[features])[:, 1]
    return df_features

# Usage:
# df_pred = predict_ml(df)
