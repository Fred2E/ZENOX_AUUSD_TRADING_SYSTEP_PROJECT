import os
import requests
from datetime import datetime

# Telegram setup (replace with your real token/chat_id)
TELEGRAM_TOKEN = "7282917429:AAG66-KnJd7lMilPoxoi1thiZPnDFyez9aU"
CHAT_ID = "7598848875"

LOG_DIR = r"C:\Users\open\Documents\ZENO_XAUUSD\logs"
MODELS_DIR = r"C:\Users\open\Documents\ZENO_XAUUSD\models"
CRITICAL_LOGS = [
    "rl_training.log",
    "walkforward.log"
]

def send_alert(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    requests.post(url, data={"chat_id": CHAT_ID, "text": message})

def check_logs():
    errors = []
    for log_file in CRITICAL_LOGS:
        path = os.path.join(LOG_DIR, log_file)
        if not os.path.exists(path):
            errors.append(f"Missing log: {log_file}")
            continue
        with open(path, "r") as f:
            lines = f.readlines()[-50:]  # Check only last 50 lines for efficiency
            for line in lines:
                if "ERROR" in line or "Failed" in line or "Exception" in line:
                    errors.append(f"Error found in {log_file}: {line.strip()}")

    return errors

def check_models():
    for tf in ["M5", "M15", "H1", "H4", "D1"]:
        path = os.path.join(MODELS_DIR, f"rl_policy_{tf}_latest.zip")
        if not os.path.exists(path):
            return [f"Model missing: {path}"]
    return []

if __name__ == "__main__":
    errors = check_logs() + check_models()
    if errors:
        alert = f"[{datetime.now()}] CRITICAL ZENO SYSTEM ERROR:\n" + "\n".join(errors)
        send_alert(alert)
        print(alert)
        # Optionally, you could add: os.system("taskkill /IM python.exe /F")  # Windows kill all Python processes
    else:
        print(f"[{datetime.now()}] ZENO SYSTEM: All clear, no errors detected.")
