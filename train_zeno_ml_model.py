import os
import joblib
import pandas as pd
from lightgbm import LGBMClassifier

ROOT = r"C:\Users\open\Documents\ZENO_XAUUSD"
ML_OUT = os.path.join(ROOT, "outputs", "ml_data")
train_path = os.path.join(ML_OUT, "train_ml.pkl")
test_path  = os.path.join(ML_OUT, "test_ml.pkl")
model_path = os.path.join(ML_OUT, "zeno_lgbm.pkl")

ML_FEATURES = ["close", "score", "num_confs", "pattern_code", "bias_bull", "hour", "dow"]

train = pd.read_pickle(train_path)
test  = pd.read_pickle(test_path)

# Force all columns to lowercase for safety
train.columns = [c.lower() for c in train.columns]
test.columns = [c.lower() for c in test.columns]

X_train, y_train = train[ML_FEATURES], train["is_win"]
X_test,  y_test  = test[ML_FEATURES],  test["is_win"]

model = LGBMClassifier(n_estimators=300, learning_rate=0.03, random_state=42)
model.fit(X_train, y_train)

score = model.score(X_test, y_test)
print(f"[RESULT] Test accuracy: {score:.3%}")

joblib.dump(model, model_path)
print(f"[SUCCESS] Model saved: {model_path}")
