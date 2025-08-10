# ===== 5) THRESHOLD CALIBRATION (Jupyter cell) =====
def calibrate_for_tf(tf: str, thresholds=(0.60,0.65,0.70,0.75,0.80)):
    ml_path = os.path.join(SIGNAL_DIR, f"ML_signals_{tf}.csv")
    if not os.path.exists(ml_path):
        print(f"[SKIP] {tf}: {ml_path} not found.")
        return
    ml = read_events_csv(ml_path)[["datetime","is_win","prob_win"]].copy()
    ml["is_win"] = ml["is_win"].astype(int)

    # ALL
    rows = []
    for thr in thresholds:
        sub = ml[ml["prob_win"] >= thr]
        n = len(sub)
        wins = int(sub["is_win"].sum())
        winrate = (wins / n) if n > 0 else np.nan
        rows.append({"threshold":thr,"n_trades":n,"wins":wins,"winrate":winrate})
    all_df = pd.DataFrame(rows)
    out_all = os.path.join(LOGS_DIR, f"calibration_{tf}_ALL.csv")
    write_csv(all_df, out_all)

    # A+ only (merge with graded)
    graded_path = os.path.join(DATA_DIR, f"trade_events_{tf}_graded.csv")
    if os.path.exists(graded_path):
        g = read_events_csv(graded_path)[["datetime","setup_grade"]]
        mix = ml.merge(g, on="datetime", how="left")
        rows2 = []
        for thr in thresholds:
            sub = mix[(mix["prob_win"] >= thr) & (mix["setup_grade"]=="A+")]
            n = len(sub)
            wins = int(sub["is_win"].sum())
            winrate = (wins / n) if n > 0 else np.nan
            rows2.append({"threshold":thr,"n_trades":n,"wins":wins,"winrate":winrate})
        aplus_df = pd.DataFrame(rows2)
        out_ap = os.path.join(LOGS_DIR, f"calibration_{tf}_Aplus.csv")
        write_csv(aplus_df, out_ap)
    else:
        print(f"[WARN] {tf}: graded file not found; skipping A+ calibration.")

for tf in TIMEFRAMES:
    calibrate_for_tf(tf)
