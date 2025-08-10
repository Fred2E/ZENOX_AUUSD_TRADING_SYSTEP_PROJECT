@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ============================
REM ZENO ML FULL REBUILD (CMD)
REM ============================
REM This script:
REM  1) Creates a Python venv (.\.venv_zeno)
REM  2) Installs deps
REM  3) Writes modules\zeno_pipeline.py (single-file pipeline)
REM  4) Runs full ML pipeline end-to-end
REM  5) Outputs to your existing directories under ZENO_XAUUSD
REM NOTE: RL is NOT included here.

REM ---- CONFIG (tailored to your project) ----
set "ROOT=C:\Users\open\Documents\ZENO_XAUUSD"
set "DATA_DIR=%ROOT%\outputs\ml_data"
set "SIGNAL_DIR=%ROOT%\outputs\ml_signals"
set "LOGS_DIR=%ROOT%\logs"
set "OUTBOX_DIR=%ROOT%\outputs\bridge_outbox"

REM ---- Check inputs exist (exact files) ----
for %%F in ("%DATA_DIR%\trade_events_M15_FULL_labeled.csv" "%DATA_DIR%\trade_events_H1_FULL_labeled.csv" "%DATA_DIR%\trade_events_H4_FULL_labeled.csv") do (
  if not exist "%%~F" (
    echo [FATAL] Missing required input: %%~F
    echo Aborting.
    exit /b 1
  )
)

REM ---- Create folders if missing ----
if not exist "%SIGNAL_DIR%" mkdir "%SIGNAL_DIR%"
if not exist "%LOGS_DIR%"   mkdir "%LOGS_DIR%"
if not exist "%OUTBOX_DIR%" mkdir "%OUTBOX_DIR%"
if not exist "%ROOT%\modules" mkdir "%ROOT%\modules"

REM ---- Create venv if missing ----
if not exist "%ROOT%\.venv_zeno" (
  echo [SETUP] Creating virtualenv...
  py -3 -m venv "%ROOT%\.venv_zeno"
)
call "%ROOT%\.venv_zeno\Scripts\activate.bat"

REM ---- Upgrade pip + install deps ----
python -m pip install --upgrade pip >nul
python -m pip install pandas numpy scikit-learn lightgbm joblib >nul

REM ---- Emit the pipeline python (single file) ----
powershell -NoProfile -Command ^
  "$code = @'
import os, json, hashlib, platform, math, zipfile
from datetime import datetime
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

ROOT       = r\"%ROOT%\"
DATA_DIR   = r\"%DATA_DIR%\"
SIGNAL_DIR = r\"%SIGNAL_DIR%\"
LOGS_DIR   = r\"%LOGS_DIR%\"
OUTBOX_DIR = r\"%OUTBOX_DIR%\"
TIMEFRAMES = [\"M15\",\"H1\",\"H4\"]
LABELED = {
  \"M15\": os.path.join(DATA_DIR, \"trade_events_M15_FULL_labeled.csv\"),
  \"H1\" : os.path.join(DATA_DIR, \"trade_events_H1_FULL_labeled.csv\"),
  \"H4\" : os.path.join(DATA_DIR, \"trade_events_H4_FULL_labeled.csv\"),
}

REQUIRED_BASE = [\"datetime\",\"open\",\"high\",\"low\",\"close\",\"volume\",\"atr\"]
ML_FEATURES_ALL = [
    \"close\",\"high\",\"low\",\"atr\",\"score\",\"num_confs\",\"pattern_code\",\"bias_bull\",
    \"hour\",\"dow\",\"primary_score\",\"secondary_score\",\"total_confluence\",\"regime_trend\",
    \"conf_sr_zone\",\"conf_bos_or_choch\",\"conf_psych_level\",\"conf_fib_zone\",
    \"conf_volume\",\"conf_liquidity\",\"conf_spread\"
]
READ_DTYPES = {
  \"open\":\"float64\",\"high\":\"float64\",\"low\":\"float64\",\"close\":\"float64\",\"volume\":\"float64\",\"atr\":\"float64\",
  \"score\":\"float64\",\"num_confs\":\"float64\",\"pattern_code\":\"float64\",\"bias_bull\":\"float64\",
  \"hour\":\"float64\",\"dow\":\"float64\",\"primary_score\":\"float64\",\"secondary_score\":\"float64\",
  \"total_confluence\":\"float64\",\"regime_trend\":\"float64\",\"conf_sr_zone\":\"float64\",
  \"conf_bos_or_choch\":\"float64\",\"conf_psych_level\":\"float64\",\"conf_fib_zone\":\"float64\",
  \"conf_volume\":\"float64\",\"conf_liquidity\":\"float64\",\"conf_spread\":\"float64\",
  \"is_win\":\"Int64\"
}

def _ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def read_events_csv(path:str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=READ_DTYPES, parse_dates=[\"datetime\"], low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    df = df.dropna(subset=[\"datetime\"]).sort_values(\"datetime\").reset_index(drop=True)
    return df

def write_csv(df:pd.DataFrame, path:str):
    _ensure_dir(os.path.dirname(path))
    df.to_csv(path, index=False)
    print(f\"[WRITE] {path} | shape={df.shape}\")

def sha256_file(path:str)->str:
    h=hashlib.sha256()
    with open(path,\"rb\") as f:
        for ch in iter(lambda: f.read(1<<20), b\"\"):
            h.update(ch)
    return h.hexdigest()

def safe_feats(df):
    return [c for c in ML_FEATURES_ALL if c in df.columns]

# ---------- 1) CLEAN ----------
def step_clean():
    for tf in TIMEFRAMES:
        src = LABELED[tf]
        df  = read_events_csv(src)
        keep = list(set([\"datetime\",\"is_win\"] + REQUIRED_BASE + ML_FEATURES_ALL) & set(df.columns))
        df = df[keep].copy()
        essential = [\"is_win\"] + safe_feats(df)
        df = df.dropna(subset=essential).copy()
        df = df.drop_duplicates(subset=[\"datetime\"], keep=\"first\").reset_index(drop=True)
        assert df[\"datetime\"].is_monotonic_increasing
        out = os.path.join(DATA_DIR, f\"trade_events_{tf}_FULL_clean.csv\")
        write_csv(df, out)

# ---------- 2) FEATURE POLICY (keep/drop) ----------
def step_feature_policy():
    # Simple hard-coded keepers based on your last pass; adjust anytime
    keep = ['atr','bias_bull','close','conf_liquidity','conf_sr_zone','dow','high','hour','low','num_confs','pattern_code','score']
    drop = ['conf_bos_or_choch','secondary_score']
    neutral = ['conf_fib_zone','primary_score','total_confluence']
    pol = {\"keep\": keep, \"drop\": drop, \"neutral\": neutral}
    outp = os.path.join(LOGS_DIR, \"feature_policy.json\")
    with open(outp, \"w\") as f: json.dump(pol, f, indent=2)
    print(f\"[OK] feature_policy.json -> {outp}\")
    return pol

# ---------- 3) TRAIN + INFER (base) ----------
CUTOFF = pd.Timestamp(\"2024-01-01 00:00:00\")
def step_train_infer(pol):
    for tf in TIMEFRAMES:
        path = os.path.join(DATA_DIR, f\"trade_events_{tf}_FULL_clean.csv\")
        df = read_events_csv(path)
        feats = [c for c in pol[\"keep\"] if c in df.columns]
        if \"is_win\" not in df.columns:
            print(f\"[FAIL] {tf}: is_win missing.\")
            continue
        tr = df[df[\"datetime\"] < CUTOFF].copy()
        te = df[df[\"datetime\"] >= CUTOFF].copy()
        if tr.empty or te.empty:
            print(f\"[WARN] {tf}: empty split; skipping.\")
            continue
        Xtr, ytr = tr[feats].astype(float), tr[\"is_win\"].astype(int)
        Xte, yte = te[feats].astype(float), te[\"is_win\"].astype(int)
        model = lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=31,
            subsample=0.9, colsample_bytree=0.9, random_state=42
        )
        model.fit(Xtr, ytr)
        model_path = os.path.join(DATA_DIR, f\"zeno_lgbm_{tf}.pkl\")
        joblib.dump(model, model_path)
        te_out = te[[\"datetime\",\"is_win\"] + feats].copy()
        te_out[\"prob_win_raw\"] = model.predict_proba(Xte)[:,1]
        out_sig = os.path.join(SIGNAL_DIR, f\"ML_signals_{tf}.csv\")
        write_csv(te_out, out_sig)
        try:
            auc = roc_auc_score(yte, te_out[\"prob_win_raw\"])
            print(f\"[{tf}] AUC={auc:.4f} | model: {model_path}\")
        except Exception as e:
            print(f\"[{tf}] AUC failed: {e}\")

# ---------- 4) CALIBRATION + THRESHOLDS ----------
def _calibrate(x, y, mode):
    if mode == \"isotonic\":
        iso = IsotonicRegression(out_of_bounds=\"clip\")
        p = iso.fit_transform(x, y)
        return p, iso
    if mode == \"platt\":
        lr = LogisticRegression(max_iter=1000)
        lr.fit(x.reshape(-1,1), y)
        p = lr.predict_proba(x.reshape(-1,1))[:,1]
        return p, lr
    return x, None

def _pick_mode(tf):
    # Use your last choices
    return \"isotonic\" if tf in (\"M15\",\"H4\") else \"platt\"

def _best_thr(prob, y, min_trades=150, grid=None):
    if grid is None:
        grid = [round(0.50 + i*0.01,2) for i in range(0,31)]  # 0.50..0.80
    best=(0.0,0.0)
    for t in grid:
        sub = (prob >= t)
        n = int(sub.sum())
        if n < min_trades: continue
        wr = float(y[sub].mean()) if n>0 else 0.0
        if wr > best[1]: best=(t,wr)
    return best

def step_calibrate_and_thresholds():
    policy={}
    for tf in TIMEFRAMES:
        path = os.path.join(SIGNAL_DIR, f\"ML_signals_{tf}.csv\")
        df = read_events_csv(path)
        df[\"is_win\"] = df[\"is_win\"].astype(int)
        x = df[\"prob_win_raw\"].values
        y = df[\"is_win\"].values
        mode = _pick_mode(tf)
        pcal, cal = _calibrate(x, y, mode)
        df[\"prob_win_cal\"] = pcal
        outc = os.path.join(SIGNAL_DIR, f\"ML_signals_{tf}_calibrated.csv\")
        write_csv(df[[\"datetime\",\"is_win\",\"prob_win_raw\",\"prob_win_cal\"]], outc)
        try:
            b_raw = brier_score_loss(y, x)
            b_cal = brier_score_loss(y, pcal)
            auc_r = roc_auc_score(y, x)
            auc_c = roc_auc_score(y, pcal)
            print(f\"[{tf}] Calibrator={mode} | Brier raw={b_raw:.4f} -> cal={b_cal:.4f} | AUC raw={auc_r} cal={auc_c}\")
        except: pass
        thr, wr = _best_thr(df[\"prob_win_cal\"].values, df[\"is_win\"].values, min_trades=150)
        if thr==0.0: thr = 0.70
        policy[tf] = {\"GLOBAL\":{\"thr\":float(thr)}}
    pol_path = os.path.join(LOGS_DIR, \"thresholds_FINAL.json\")
    with open(pol_path, \"w\") as f: json.dump(policy, f, indent=2)
    print(f\"[OK] thresholds_FINAL.json -> {pol_path}\")
    return policy

# ---------- 5) WALKFORWARD ----------
def step_walkforward(policy):
    R_POS, R_NEG = 100.0, -100.0
    for tf in TIMEFRAMES:
        path = os.path.join(SIGNAL_DIR, f\"ML_signals_{tf}_calibrated.csv\")
        df = read_events_csv(path)
        thr = float(policy[tf][\"GLOBAL\"][\"thr\"])
        take = df[df[\"prob_win_cal\"] >= thr].copy()
        if take.empty:
            print(f\"[{tf}] No trades at thr={thr}\"); continue
        take[\"reward\"] = np.where(take[\"is_win\"].astype(int)==1, R_POS, R_NEG)
        take[\"balance\"] = 10000 + take[\"reward\"].cumsum()
        outlog = os.path.join(LOGS_DIR, f\"walkforward_{tf}_FINAL_log.csv\")
        write_csv(take[[\"datetime\",\"is_win\",\"prob_win_cal\",\"reward\",\"balance\"]], outlog)
        wr = float(take[\"is_win\"].mean())
        print(f\"[{tf}] FINAL Walkforward: thr={thr} | winrate={wr:.2%} | trades={len(take)} | end_balance={take['balance'].iloc[-1]:,.2f}\")

# ---------- 6) ORDERS (sized) ----------
def session_from_dt(ts:pd.Timestamp):
    h = ts.hour
    if 7 <= h < 16: return \"London\"
    if 13 <= h < 21: return \"NY\"
    return \"Off\"

def step_orders_sized(policy, equity=10000.0, risk_pct=0.005, atr_mult=1.0, pip_value=100.0):
    stamp = datetime.utcnow().strftime(\"%Y%m%dT%H%MZ\")
    for tf in TIMEFRAMES:
        path = os.path.join(SIGNAL_DIR, f\"ML_signals_{tf}_calibrated.csv\")
        df = read_events_csv(path)
        thr = float(policy[tf][\"GLOBAL\"][\"thr\"])
        df = df[df[\"prob_win_cal\"]>=thr].copy()
        if df.empty: 
            print(f\"[{tf}] no orders.\"); 
            continue
        # entry: use close; SL/TP from ATR
        if \"close\" not in df.columns or \"atr\" not in df.columns:
            # backfill from clean file
            base = read_events_csv(os.path.join(DATA_DIR, f\"trade_events_{tf}_FULL_clean.csv\"))[ [\"datetime\",\"close\",\"atr\"] ]
            df = df.merge(base, on=\"datetime\", how=\"left\")
        df[\"session\"] = df[\"datetime\"].apply(session_from_dt)
        df[\"side\"] = np.where(df.get(\"bias_bull\", pd.Series(1,index=df.index)).fillna(1)>0, \"buy\",\"sell\")
        entry = df[\"close\"].astype(float)
        atr   = df[\"atr\"].astype(float).fillna(0.5)
        sl = np.where(df[\"side\"]==\"buy\", entry - atr*atr_mult, entry + atr*atr_mult)
        tp = np.where(df[\"side\"]==\"buy\", entry + 2*(entry - sl), entry - 2*(sl - entry))
        pos_risk = equity * risk_pct
        eff_stop = (entry - sl).abs() + 0.5
        lot = (pos_risk / (eff_stop * pip_value)).clip(lower=0.01)
        out = pd.DataFrame({
          \"datetime\": df[\"datetime\"],
          \"symbol\": \"XAUUSD\",
          \"tf\": tf,
          \"session\": df[\"session\"],
          \"side\": df[\"side\"],
          \"entry_price\": entry.round(2),
          \"sl_price\": sl.round(2),
          \"tp_price\": tp.round(2),
          \"lot\": lot.round(3),
          \"prob_win\": df[\"prob_win_raw\"].round(4),
          \"prob_win_cal\": df[\"prob_win_cal\"].round(4),
          \"comment\": \"ML\"
        })
        outf = os.path.join(SIGNAL_DIR, f\"orders_enriched_{tf}_{stamp}.csv\")
        write_csv(out, outf)
    print(\"[PANEL] orders_enriched_* written per TF\")

# ---------- 7) TICKETS + JOURNAL + ZIP ----------
def step_tickets_and_outbox():
    stamp = datetime.utcnow().strftime(\"%Y%m%dT%H%MZ\")
    all_tickets=[]
    all_journal=[]
    used={}
    for tf in TIMEFRAMES:
        glob = sorted([p for p in os.listdir(SIGNAL_DIR) if p.startswith(f\"orders_enriched_{tf}_\") and p.endswith(\".csv\")])
        if not glob: 
            print(f\"[SKIP] {tf}: no orders_enriched csv\"); 
            continue
        path = os.path.join(SIGNAL_DIR, glob[-1])
        used[tf]=path
        df = read_events_csv(path)
        # tickets minimal
        tix = df[[\"datetime\",\"symbol\",\"tf\",\"session\",\"side\",\"entry_price\",\"sl_price\",\"tp_price\",\"lot\",\"prob_win\",\"prob_win_cal\",\"comment\"]].copy()
        tix_path = os.path.join(SIGNAL_DIR, f\"tickets_{tf}_{stamp}.csv\")
        write_csv(tix, tix_path)
        # journal
        j = df.copy()
        j_path = os.path.join(SIGNAL_DIR, f\"journal_{tf}_{stamp}.csv\")
        write_csv(j, j_path)
        all_tickets.append(tix.assign(tf=tf))
        all_journal.append(j.assign(tf=tf))
        # outbox jsonl
        outbox = os.path.join(OUTBOX_DIR, f\"orders_{tf}_{stamp}.jsonl\")
        with open(outbox, \"w\", encoding=\"utf-8\") as f:
            for _,r in tix.iterrows():
                f.write(json.dumps({k:(r[k].item() if hasattr(r[k],\"item\") else r[k]) for k in tix.columns}, default=str)+\"\\n\")
        print(f\"[WRITE] {outbox} | orders={len(tix)}\")
        led = os.path.join(LOGS_DIR, f\"orders_ledger_{tf}_{stamp}.csv\")
        write_csv(tix, led)
    if all_tickets:
        all_t = pd.concat(all_tickets, ignore_index=True)
        all_j = pd.concat(all_journal, ignore_index=True)
        all_t_path = os.path.join(SIGNAL_DIR, f\"tickets_ALL_{stamp}.csv\")
        all_j_path = os.path.join(SIGNAL_DIR, f\"journal_ALL_{stamp}.csv\")
        write_csv(all_t, all_t_path)
        write_csv(all_j, all_j_path)
        zip_path = os.path.join(SIGNAL_DIR, f\"tickets_ALL_{stamp}.zip\")
        with zipfile.ZipFile(zip_path, \"w\", zipfile.ZIP_DEFLATED) as z:
            z.write(all_t_path, os.path.basename(all_t_path))
        print(f\"[ZIP] {zip_path}\")
        print(\"\\n[SUMMARY] inputs used:\")
        for k,v in used.items(): print(f\"  {k}: {v}\")
        print(\"[OK] Tickets ready; outbox JSONL packages written.\")
    else:
        print(\"[WARN] No tickets generated.\")

def main():
    print(\"[1/7] CLEAN\"); step_clean()
    print(\"[2/7] FEATURE POLICY\"); pol = step_feature_policy()
    print(\"[3/7] TRAIN+INFER\"); step_train_infer(pol)
    print(\"[4/7] CALIB+THR\"); thr = step_calibrate_and_thresholds()
    print(\"[5/7] WALKFORWARD\"); step_walkforward(thr)
    print(\"[6/7] ORDERS (SIZED)\"); step_orders_sized(thr, equity=10000.0, risk_pct=0.005, atr_mult=1.0, pip_value=100.0)
    print(\"[7/7] TICKETS+OUTBOX\"); step_tickets_and_outbox()
    print(\"[DONE] Full ML rebuild complete.\")

if __name__ == \"__main__\":
    main()
'@; Set-Content -Encoding UTF8 \"%ROOT%\modules\zeno_pipeline.py\""

if errorlevel 1 (
  echo [FATAL] Failed writing pipeline file.
  exit /b 1
)

REM ---- RUN the pipeline ----
python "%ROOT%\modules\zeno_pipeline.py"
if errorlevel 1 (
  echo [FATAL] Pipeline run failed.
  exit /b 1
)

echo.
echo [OK] ZENO ML full rebuild finished.
exit /b 0
