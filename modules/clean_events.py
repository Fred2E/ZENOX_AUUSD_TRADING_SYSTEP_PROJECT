# ===== 2) CLEAN LABELED EVENTS (Jupyter cell) =====
from collections import defaultdict

summary = defaultdict(dict)

for tf in TIMEFRAMES:
    path = labeled_path(tf)
    if not os.path.exists(path):
        print(f"[SKIP] {tf}: {path} not found.")
        continue

    df = read_events_csv(path)

    # keep only columns we can use safely
    keep_cols = list(set(["datetime", "is_win"] + REQUIRED_BASE + ML_FEATURES_ALL) & set(df.columns))
    df = df[keep_cols].copy()

    # drop NAs on essential modeling columns (datetime handled in read)
    essential = ["is_win"] + [c for c in safe_feature_list(df)]
    df = df.dropna(subset=essential).copy()

    # ensure no duplicate datetimes
    before = len(df)
    df = df.drop_duplicates(subset=["datetime"], keep="first")
    dropped = before - len(df)

    # final checks
    assert df["datetime"].is_monotonic_increasing, "[ERR] datetime not sorted"
    assert df["datetime"].duplicated().sum() == 0, "[ERR] duplicated datetimes after clean"

    out_clean = os.path.join(DATA_DIR, f"trade_events_{tf}_FULL_clean.csv")
    write_csv(df, out_clean)

    summary[tf] = dict(rows=int(len(df)), dropped=int(dropped), out=out_clean)

print("\n[SUMMARY]")
for tf, s in summary.items():
    print(f"  {tf}: rows={s.get('rows',0)} dropped={s.get('dropped',0)} -> {s.get('out','N/A')}")
