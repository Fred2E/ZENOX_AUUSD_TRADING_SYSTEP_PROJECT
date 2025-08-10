# ===== 4) CONFLUENCE GRADING (Jupyter cell) =====
# Creates trade_signals_{tf}_{grade}.csv and a combined graded file
def dynamic_grades(df: pd.DataFrame) -> pd.DataFrame:
    # choose available scoring columns
    c1 = "primary_score" if "primary_score" in df.columns else None
    c2 = "score" if "score" in df.columns else None
    c3 = "total_confluence" if "total_confluence" in df.columns else None

    # build a composite score that exists
    cols = [c for c in [c1,c2,c3] if c and c in df.columns]
    if not cols:
        df["setup_score"] = 0.0
    else:
        df["setup_score"] = df[cols].mean(axis=1)

    # quantile thresholds to avoid 0 A+ on any TF
    qAplus = df["setup_score"].quantile(0.90)  # top 10%
    qA     = df["setup_score"].quantile(0.60)  # 60-90%
    df["setup_grade"] = np.where(df["setup_score"] >= qAplus, "A+",
                          np.where(df["setup_score"] >= qA, "A", "B"))
    return df

for tf in TIMEFRAMES:
    src = os.path.join(DATA_DIR, f"trade_events_{tf}_FULL_clean.csv")
    if not os.path.exists(src):
        print(f"[SKIP] {tf}: {src} not found.")
        continue
    df = read_events_csv(src)
    df = dynamic_grades(df)

    # write per-grade
    for g in ["A+","A","B"]:
        outp = os.path.join(DATA_DIR, f"trade_signals_{tf}_{g}.csv")
        write_csv(df[df["setup_grade"]==g], outp)

    # write combined graded file
    out_all = os.path.join(DATA_DIR, f"trade_events_{tf}_graded.csv")
    write_csv(df, out_all)

    print(f"[{tf}] graded counts:", df["setup_grade"].value_counts(dropna=False).to_dict())
