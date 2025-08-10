# ===== 6) WALKFORWARD EVAL + AUDIT (Jupyter cell) =====
# Simple walkforward using a per-TF default threshold.
# You can change thresholds after checking the calibration CSVs.
DEFAULT_THR = {"M15":0.65, "H1":0.70, "H4":0.72}
R_POS, R_NEG = 100.0, -100.0  # per-trade P&L step (tweak as desired)

def walkforward_tf(tf: str, thr: float):
    path = os.path.join(SIGNAL_DIR, f"ML_signals_{tf}.csv")
    if not os.path.exists(path):
        print(f"[SKIP] {tf}: signals not found -> {path}")
        return
    df = read_events_csv(path)[["datetime","is_win","prob_win"]].copy()
    df["is_win"] = df["is_win"].astype(int)
    df["signal"] = (df["prob_win"] >= thr).astype(int)
    df = df[df["signal"] == 1].copy()
    if df.empty:
        print(f"[{tf}] No trades at threshold {thr}.")
        return

    # reward & balance
    df["reward"] = np.where(df["is_win"]==1, R_POS, R_NEG)
    df["balance"] = 10000 + df["reward"].cumsum()

    # quick metrics
    wins = int(df["is_win"].sum())
    n = int(len(df))
    winrate = wins / n if n>0 else np.nan
    end_balance = float(df["balance"].iloc[-1])

    # save log
    out_log = os.path.join(LOGS_DIR, f"walkforward_{tf}_log.csv")
    write_csv(df, out_log)
    print(f"[{tf}] Walkforward: thr={thr:.2f} | winrate={winrate:.2%} | trades={n} | end_balance={end_balance:,.2f}")
    print(f"[{tf}] Trade log saved: {out_log}")

    # audits
    print(f"\n[{tf}] AUDIT")
    print("  - dup datetimes:", int(df["datetime"].duplicated().sum()))
    print("  - strictly increasing:", bool(df["datetime"].is_monotonic_increasing))
    leak_cols = [c for c in df.columns if "future" in c.lower() or "target" in c.lower()]
    print("  - obvious future/target cols in log:", leak_cols if leak_cols else "none")
    return df

for tf in TIMEFRAMES:
    thr = DEFAULT_THR.get(tf, 0.70)
    walkforward_tf(tf, thr)
