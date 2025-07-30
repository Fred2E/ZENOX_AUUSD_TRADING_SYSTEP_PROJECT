import os
import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import json

ROOT = r"C:\Users\open\Documents\ZENO_XAUUSD"
ML_OUT = os.path.join(ROOT, "outputs", "ml_data")
TIMEFRAMES = ["M5", "M15", "H1", "H4"]

ML_FEATURES = [
    "close", "score", "num_confs", "pattern_code", "bias_bull", "hour", "dow",
    "primary_score", "secondary_score", "total_confluence", "regime_trend"
]
ML_LABEL = "is_win"

for tf in TIMEFRAMES:
    train_path = os.path.join(ML_OUT, f"train_ml_{tf}.pkl")
    test_path  = os.path.join(ML_OUT, f"test_ml_{tf}.pkl")
    model_path = os.path.join(ML_OUT, f"zeno_lgbm_{tf}.pkl")
    audit_path = os.path.join(ML_OUT, f"ml_training_audit_{tf}.json")

    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        print(f"[SKIP] {tf}: Missing train/test splits.")
        continue

    train = pd.read_pickle(train_path)
    test = pd.read_pickle(test_path)
    train.columns = [c.lower() for c in train.columns]
    test.columns = [c.lower() for c in test.columns]

    for feature in ML_FEATURES + [ML_LABEL]:
        if feature not in train.columns or feature not in test.columns:
            raise ValueError(f"[{tf}] Missing critical feature '{feature}' in ML dataset.")

    # Defensive: Remove rows with nulls in features or labels
    train = train.dropna(subset=ML_FEATURES + [ML_LABEL])
    test = test.dropna(subset=ML_FEATURES + [ML_LABEL])

    X_train, y_train = train[ML_FEATURES], train[ML_LABEL]
    X_test,  y_test  = test[ML_FEATURES],  test[ML_LABEL]

    # Skip training if <10 samples (e.g. H4 with tiny samples)
    if len(X_train) < 10 or len(X_test) < 2:
        print(f"[SKIP] {tf}: Not enough data to train a model (train: {len(X_train)}, test: {len(X_test)})")
        continue

    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.03,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    try:
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
    except Exception:
        auc = None

    acc = model.score(X_test, y_test)

    print(f"\n=== [{tf}] ML RESULTS ===")
    print("Classification Report:\n", classification_report(y_test, y_pred, digits=2))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print(f"ROC-AUC: {auc:.3f}" if auc is not None else "ROC-AUC: N/A")
    print(f"Test accuracy: {acc:.3%}")

    joblib.dump(model, model_path)
    print(f"[SAVED] Model: {model_path}")

    importances = pd.Series(model.feature_importances_, index=ML_FEATURES)
    print("\nFeature Importances:\n", importances.sort_values(ascending=False))

    # Save audit for reproducibility
    audit_info = {
        "features": ML_FEATURES,
        "label": ML_LABEL,
        "train_shape": X_train.shape,
        "test_shape": X_test.shape,
        "model_path": model_path,
        "score": float(acc),
        "roc_auc": float(auc) if auc is not None else None,
        "classification_report": classification_report(y_test, y_pred, digits=2, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "feature_importances": importances.sort_values(ascending=False).to_dict(),
    }
    with open(audit_path, "w") as f:
        json.dump(audit_info, f, indent=2)
    print(f"[AUDIT] Training metadata saved: {audit_path}")

print("\n✅ ML training run complete for all timeframes.")
