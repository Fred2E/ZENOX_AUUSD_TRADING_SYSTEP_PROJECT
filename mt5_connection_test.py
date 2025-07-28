import MetaTrader5 as mt5
import os

# ---- USER CONFIGURATION ----
LOGIN    = 76847754
PASSWORD = 'Lovetr!ck2000'
SERVER   = 'Exness-MT5Trial5'  # Double check: Should be EXACT as shown in your MT5 platform

# SCAN for terminal64.exe if you are unsure
MT5_PATHS = [
    r"C:\Program Files\MetaTrader 5\terminal64.exe",
    r"C:\Program Files\Exness Technologies Ltd\MetaTrader 5\terminal64.exe",
    r"C:\Program Files (x86)\MetaTrader 5\terminal64.exe",
]

# Find first valid MT5 terminal
MT5_PATH = None
for path in MT5_PATHS:
    if os.path.exists(path):
        MT5_PATH = path
        break

if MT5_PATH is None:
    print("[CRITICAL] MetaTrader 5 terminal64.exe not found. Edit the script to add the correct path.")
    exit(1)

print(f"[INFO] Using MT5 terminal: {MT5_PATH}")
print(f"[INFO] Trying login: {LOGIN} on server: {SERVER}")

# ---- MT5 INIT ----
if not mt5.initialize(MT5_PATH, login=LOGIN, password=PASSWORD, server=SERVER):
    print("[ERROR] initialize() failed, error code:", mt5.last_error())
    print("[HINT] Double-check server name, login, password, and terminal path.")
else:
    print("[SUCCESS] MT5 initialized OK! Shutting down...")
    mt5.shutdown()
