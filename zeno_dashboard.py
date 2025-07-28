import sys
import os
import streamlit as st
import pandas as pd

# === Add ZENO modules directory to path ===
sys.path.append(r"C:\Users\open\Documents\ZENO_XAUUSD\modules")

# === Import ZENO Modules ===
from structure_detector import detect_structure
from candle_patterns import detect_candle_patterns
from confluence_scanner import evaluate_confluence

# === SETTINGS ===
data_dir_live = r"C:\Users\open\Documents\ZENO_XAUUSD\data\live\XAUUSD"
data_dir_backtest = r"C:\Users\open\Documents\ZENO_XAUUSD\outputs\setups"
timeframes = ["M5", "M15", "H1", "H4", "D1"]
refresh_interval = 30
score_threshold_default = 4

# === STREAMLIT SETUP ===
st.set_page_config(page_title="ZENO Unified Dashboard", layout="wide")
st.title("🧠 ZENO Strategy Engine - Unified Dashboard")
st.caption("Live signal scanning + historical strategy review")

# === SIDEBAR ===
st.sidebar.header("⚙️ Controls")
mode = st.sidebar.radio("Mode", ["Live Signals", "Historical Review"])
score_threshold = st.sidebar.slider("Score Threshold", 1, 5, score_threshold_default)
timeframes_selected = st.sidebar.multiselect("Timeframes", timeframes, default=timeframes)
st.sidebar.markdown(f"⏱ Refreshes every `{refresh_interval}` seconds in Live Mode")

# === DASHBOARD DISPLAY ===
signal_cols = st.columns(len(timeframes_selected))

for idx, tf in enumerate(timeframes_selected):
    with signal_cols[idx]:
        st.subheader(f"{tf} - {mode}")

        if mode == "Live Signals":
            file_path = os.path.join(data_dir_live, tf, f"XAUUSD_{tf}_LIVE.csv")
        else:
            file_path = os.path.join(data_dir_backtest, f"ZENO_A+_signals_{tf}.csv")

        if not os.path.exists(file_path):
            st.warning("❌ No data found.")
            continue

        try:
            df = pd.read_csv(file_path, parse_dates=['datetime'])
            df.columns = [c.lower() for c in df.columns]
            df.set_index('datetime', inplace=True)

            if mode == "Live Signals":
                df = detect_structure(df)
                df = detect_candle_patterns(df)
                df = evaluate_confluence(df, tf)
                # Ensure correct case for filtering and display
                filter_cols = ['close', 'confluences', 'score', 'candle_pattern']
                for c in filter_cols:
                    if c not in df.columns:
                        df[c] = None
                df = df[df['score'] >= score_threshold][filter_cols].tail(10)
            else:
                filter_cols = ['close', 'confluences', 'score', 'candle_pattern']
                for c in filter_cols:
                    if c not in df.columns:
                        df[c] = None
                df = df[df['score'] >= score_threshold][filter_cols].tail(10)

            if not df.empty:
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No high-confluence signals found.")

        except Exception as e:
            st.error(f"Processing error: {e}")

# === AUTO REFRESH ONLY IN LIVE MODE ===
if mode == "Live Signals":
    import time
    time.sleep(refresh_interval)
    st.rerun()
