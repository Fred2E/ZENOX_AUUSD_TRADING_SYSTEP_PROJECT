@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM =======================================
REM ZENO ML STEP-BY-STEP DRIVER (CMD)
REM =======================================

set "ROOT=C:\Users\open\Documents\ZENO_XAUUSD"
set "DATA_DIR=%ROOT%\outputs\ml_data"
set "SIGNAL_DIR=%ROOT%\outputs\ml_signals"
set "LOGS_DIR=%ROOT%\logs"
set "OUTBOX_DIR=%ROOT%\outputs\bridge_outbox"

if not exist "%ROOT%\.venv_zeno" (
  py -3 -m venv "%ROOT%\.venv_zeno"
)
call "%ROOT%\.venv_zeno\Scripts\activate.bat"
python -m pip install --upgrade pip >nul
python -m pip install pandas numpy scikit-learn lightgbm joblib >nul

REM ---- Ensure pipeline file exists (idempotent) ----
if not exist "%ROOT%\modules\zeno_pipeline.py" (
  echo [SETUP] Writing modules\zeno_pipeline.py ...
  powershell -NoProfile -Command ^
    "$c=Get-Content -Raw '%ROOT%\modules\zeno_pipeline.py' 2>$null; if(-not $c){$c=@'
print(\"This placeholder will be replaced when you run zeno_rebuild_all.cmd first.\")
'@ | Set-Content -Encoding UTF8 '%ROOT%\modules\zeno_pipeline.py'}"
)

REM ===================================================
REM HOW TO USE:
REM   Remove 'REM ' in front of the lines you want to run.
REM   Each phase calls the same Python file with a --step flag.
REM   (We’ll call small wrappers that live inside the script.)
REM ===================================================

REM ---- Phase 1: CLEAN labeled -> *_FULL_clean.csv
REM python - <<PY
REM from modules.zeno_pipeline import step_clean; step_clean()
REM PY

REM ---- Phase 2: Feature policy (keep/drop) -> logs\feature_policy.json
REM python - <<PY
REM from modules.zeno_pipeline import step_feature_policy; step_feature_policy()
REM PY

REM ---- Phase 3: Train + infer (base) -> ML_signals_{tf}.csv
REM python - <<PY
REM from modules.zeno_pipeline import step_feature_policy, step_train_infer
REM pol = step_feature_policy()
REM step_train_infer(pol)
REM PY

REM ---- Phase 4: Calibrate + thresholds -> ML_signals_{tf}_calibrated.csv + thresholds_FINAL.json
REM python - <<PY
REM from modules.zeno_pipeline import step_calibrate_and_thresholds
REM step_calibrate_and_thresholds()
REM PY

REM ---- Phase 5: Walkforward -> logs\walkforward_{tf}_FINAL_log.csv
REM python - <<PY
REM import json, os
REM from modules.zeno_pipeline import step_walkforward, LOGS_DIR
REM thr = json.load(open(os.path.join(LOGS_DIR,'thresholds_FINAL.json')))
REM step_walkforward(thr)
REM PY

REM ---- Phase 6: Orders sized -> orders_enriched_{tf}_*.csv
REM python - <<PY
REM import json, os
REM from modules.zeno_pipeline import step_orders_sized, LOGS_DIR
REM thr = json.load(open(os.path.join(LOGS_DIR,'thresholds_FINAL.json')))
REM step_orders_sized(thr, equity=10000.0, risk_pct=0.005, atr_mult=1.0, pip_value=100.0)
REM PY

REM ---- Phase 7: Tickets + Outbox JSONL (+ ZIP)
REM python - <<PY
REM from modules.zeno_pipeline import step_tickets_and_outbox
REM step_tickets_and_outbox()
REM PY

echo.
echo [INFO] Open this file and uncomment the phase(s) you want to run.
echo        Save, then run the CMD again.
exit /b 0
