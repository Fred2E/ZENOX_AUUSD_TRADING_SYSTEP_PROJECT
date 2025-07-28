# zeno_live_features_build.py

import os
import pandas as pd
import logging

# Set up logger
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

TF = "M5"  # Change to desired timeframe if needed
RAW_PATH = os.path.join("data", "live", "XAUUSD", TF, f"XAUUSD_{TF}_LIVE.csv")
OUT_PATH = os.path.join("data", "live", "XAUUSD", TF, f"XAUUSD_{TF}_LIVE_FEATURES.csv")

logging.info(f"Loading raw data from {RAW_PATH}")
df = pd.read_csv(RAW_PATH, parse_dates=["Datetime"])

# === FEATURE ENGINEERING ===
try:
    from modules.structure_detector import detect_structure
    from modules.candle_patterns import detect_candle_patterns
    from modules.confluence_scanner import evaluate_confluence
except Exception as e:
    logging.error(f"Failed to import feature modules: {e}")
    exit(1)

# 1. Structure
df = detect_structure(df)
if 'bias' not in df:
    logging.warning("Feature 'bias' missing after structure detection. Patching as 'neutral'.")
    df['bias'] = "neutral"

# 2. Candle Patterns
df = detect_candle_patterns(df)
if 'candle_pattern' not in df:
    logging.warning("Feature 'candle_pattern' missing after pattern detection. Patching as ''.")
    df['candle_pattern'] = ""

# 3. Confluence & Score
df = evaluate_confluence(df, TF)
if 'confluences' not in df:
    logging.warning("Feature 'confluences' missing after confluence detection. Patching as [].")
    df['confluences'] = [[] for _ in range(len(df))]
if 'score' not in df:
    logging.warning("Feature 'score' missing after confluence detection. Patching as 0.")
    df['score'] = 0

# 4. ML Features
from modules.feature_pipeline import engineer_features
df, features = engineer_features(df)

# 5. Save to file
df.to_csv(OUT_PATH, index=False)
logging.info(f"[SUCCESS] Saved engineered features to {OUT_PATH}")
logging.info(f"[COLUMNS] {list(df.columns)}")
print(df.tail(3))

