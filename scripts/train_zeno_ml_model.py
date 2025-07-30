import os
import joblib
import pandas as pd
from lightgbm import LGBMClassifier

# --- Paths
ROOT = r"C:\Users\open\Documents\ZENO_XAUUSD"
ML_OUT = os.path.join(ROOT, "outputs", "ml_data")
train_path = os.path.join(ML_OUT, "train_ml.pkl")
test_path  = os.path.join(ML_OUT, "test_ml.pkl")
model_path = os.path.join(ML_OUT, "zeno_lgbm.pkl")

# === Features & Labels (MUST be derived from event-driven trade logs, not raw signals!)
ML_FEATURES = [
    "close", "score", "num_confs", "pattern_code", "bias_bull", "hour", "dow",
    "primary_score", "secondary_score", "total_confluence", "regime_trend"
]
ML_LABEL   = "is_win"

# --- Load & Integrity Check
train = pd.read_pickle(train_path)
test  = pd.read_pickle(test_path)

# Force all columns to lowercase
train.columns = [c.lower() for c in train.columns]
test.columns = [c.lower() for c in test.columns]

for feature in ML_FEATURES + [ML_LABEL]:
    if feature not in train.columns or feature not in test.columns:
        raise ValueError(f"Missing critical feature '{feature}' in ML dataset.")

# === Data Leak Protection: No future info
assert "outcome" not in ML_FEATURES, "DO NOT use any future or post-trade info as input features!"

X_train, y_train = train[ML_FEATURES], train[ML_LABEL]
X_test,  y_test  = test[ML_FEATURES],  test[ML_LABEL]

# === Model Training
model = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.03,
    random_state=42,
    class_weight='balanced'
)
model.fit(X_train, y_train)

score = model.score(X_test, y_test)
print(f"[RESULT] Test accuracy: {score:.3%}")

joblib.dump(model, model_path)
print(f"[SUCCESS] Model saved: {model_path}")

# === Output Feature Importance
importances = pd.Series(model.feature_importances_, index=ML_FEATURES)
print("\nFeature Importances:")
print(importances.sort_values(ascending=False))

# === Save audit info for reproducibility
audit_info = {
    "features": ML_FEATURES,
    "label": ML_LABEL,
    "train_shape": X_train.shape,
    "test_shape": X_test.shape,
    "model_path": model_path,
    "score": float(score)
}
audit_path = os.path.join(ML_OUT, "ml_training_audit.json")
import json
with open(audit_path, "w") as f:
    json.dump(audit_info, f, indent=2)
print(f"[AUDIT] Training metadata saved: {audit_path}")
