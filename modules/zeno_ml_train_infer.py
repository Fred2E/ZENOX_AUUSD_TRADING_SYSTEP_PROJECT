# ===== 3) TRAIN / INFER LGBM (Jupyter cell) =====
import json
import lightgbm as lgb
import joblib
from sklearn.metrics import roc_auc_score

CUTOFF = pd.Timestamp("2024-01-01 00:00:00")  # time-based split

for tf in TIMEFRAMES:
    clean_path = os.path.join(DATA_DIR, f"trade_events_{tf}_FULL_clean.csv")
    if not os.path.exists(clean_path):
        print(f"[SKIP] {tf}: {clean_path} not found.")
        continue

    df = read_events_csv(clean_path)
    feats = safe_feature_list(df)
    if "is_win" not in df.columns:
        print(f"[FAIL] {tf}: missing is_win.")
        continue

    # Split
    train_df = df[df["datetime"] < CUTOFF].copy()
    test_df  = df[df["datetime"] >= CUTOFF].copy()
    if train_df.empty or test_df.empty:
        print(f"[WARN] {tf}: empty split (train or test). Skipping.")
        continue

    X_train = train_df[feats].astype(float)
    y_train = train_df["is_win"].astype(int)
    X_test  = test_df[feats].astype(float)
    y_test  = test_df["is_win"].astype(int)

    # Model
    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Save model + features
    model_path = os.path.join(DATA_DIR, f"zeno_lgbm_{tf}.pkl")
    joblib.dump(model, model_path)
    with open(os.path.join(DATA_DIR, f"features_{tf}.txt"), "w") as f:
        f.write("\n".join(feats))

    # Metadata (fixes the NameError you hit)
    meta = {
        "tf": tf,
        "model_path": model_path,
        "model_sha256": sha256_file(model_path),
        "features": feats,
        "cutoff": CUTOFF.isoformat(),
        "random_state": 42,
        "lightgbm": lgb.__version__,
        "python": platform.python_version(),
        "created_at": datetime.utcnow().isoformat()+"Z",
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test))
    }
    with open(os.path.join(DATA_DIR, f"zeno_lgbm_{tf}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Inference on test, include labels for calibration/walkforward
    test_df = test_df[["datetime","is_win"] + feats].copy()
    test_df["prob_win"] = model.predict_proba(X_test)[:,1]
    out_sig = os.path.join(SIGNAL_DIR, f"ML_signals_{tf}.csv")
    write_csv(test_df, out_sig)

    # quick AUC
    try:
        auc = roc_auc_score(y_test, test_df["prob_win"])
        print(f"[{tf}] AUC={auc:.4f} | model: {model_path}")
    except Exception as e:
        print(f"[{tf}] AUC failed: {e}")
