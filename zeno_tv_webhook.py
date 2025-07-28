# zeno_tv_webhook.py
"""
ZENO TradingView Webhook Listener
Usage  (PowerShell / CMD):
    python "C:\\Users\\open\\Documents\\ZENO_XAUUSD\\zeno_tv_webhook.py"

After launch, set TradingView Webhook URL to:
    http://<YOUR-IP>:8008/zeno/webhook
The payload JSON is appended to outputs/alerts/tradingview_log.txt
"""

import os, json, datetime
from fastapi import FastAPI, Request
import uvicorn

# --- CONFIG ---
BASE = r"C:\Users\open\Documents\ZENO_XAUUSD"
LOG_FILE = os.path.join(BASE, r"outputs\alerts\tradingview_log.txt")
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

app = FastAPI()

@app.post("/zeno/webhook")
async def zeno_webhook(req: Request):
    data = await req.json()
    ts = datetime.datetime.utcnow().isoformat()
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{ts}  {json.dumps(data)}\n")
    return {"status": "ok"}

if __name__ == "__main__":
    print("🚀 ZENO TradingView webhook running at http://0.0.0.0:8008/zeno/webhook")
    uvicorn.run(app, host="0.0.0.0", port=8008)