# zeno_master_pipeline.py

import os
import subprocess
import logging
from datetime import datetime

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()]
)

# --- Ordered Pipeline Steps (label, shell command) ---
STEPS = [
    ("MT5 LIVE EXPORT",           "python mt5_live_export.py"),                # 1. Export new live data from MT5/MetaTrader
    ("LIVE FEATURE ENGINEERING",  "python zeno_live_feature_pipeline.py"),     # 2. Transform raw candles into features
    ("ML PREDICTION",             "python zeno_live_predict_ml.py"),           # 3. Score features with ML model
    ("RL ACTION",                 "python zeno_live_rl_action.py"),            # 4. Run RL-based trade logic/actions
]

def run_step(name, command):
    """
    Execute a shell command as part of the pipeline.
    Logs start, success, duration, and any errors with output for audit/troubleshooting.
    """
    start = datetime.utcnow()
    logging.info(f"\n=== {name} START ===")
    try:
        # Execute step, capture all stdout/stderr for pipeline traceability
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        print(result.stdout)  # Display output to console for live review
        if result.returncode != 0:
            # Print error details to both log and console for immediate visibility
            logging.error(f"{name} FAILED.\nStdout:\n{result.stdout}\nStderr:\n{result.stderr}")
            raise RuntimeError(f"{name} failed with return code {result.returncode}")
    except Exception as e:
        # Catch *all* exceptions: nothing is allowed to silently fail in production
        logging.error(f"CRITICAL: {name} crashed with exception: {str(e)}")
        raise
    elapsed = (datetime.utcnow() - start).total_seconds()
    logging.info(f"=== {name} SUCCESS ({elapsed:.2f}s) ===")

def preflight_check():
    """
    Abort early if any required scripts are missing from the project directory.
    Avoids partial pipeline runs and hard-to-diagnose downstream errors.
    """
    for label, cmd in STEPS:
        script = cmd.split()[-1]
        if not os.path.exists(script):
            raise FileNotFoundError(f"Preflight FAIL: Required script not found: {script}")
        else:
            logging.info(f"[PRE-CHECK] Found required script: {script}")

def main():
    """
    Orchestrates the entire ZENO pipeline:
    - Checks all component scripts are present
    - Runs each step in sequence, logging progress and errors
    - Logs total elapsed time for monitoring and operational review
    """
    logging.info("=== ZENO MASTER PIPELINE INITIATED ===")
    pipeline_start = datetime.utcnow()

    # --- Pre-execution: make sure all scripts exist ---
    preflight_check()

    # --- Sequentially execute each pipeline step ---
    for name, command in STEPS:
        run_step(name, command)

    # --- Log total pipeline runtime ---
    total_time = (datetime.utcnow() - pipeline_start).total_seconds()
    logging.info(f"\n=== ZENO MASTER PIPELINE COMPLETED in {total_time:.2f} seconds ===")

if __name__ == "__main__":
    main()
