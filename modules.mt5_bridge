# modules/mt5_bridge.py

import MetaTrader5 as mt5
import time

# --- adjust these to your MT5 installation if needed ---
MT5_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe"

def initialize(login=None, password=None, server=None):
    """
    Initialize connection to MT5. 
    If you've already logged in manually, this will still attach to the running terminal.
    """
    # try to start terminal if not already running
    if not mt5.initialize():
        mt5.shutdown()
        if not mt5.initialize(path=MT5_PATH, login=login, password=password, server=server):
            raise RuntimeError(f"MT5 initialize() failed, error code = {mt5.last_error()}")

def place_order(symbol, direction, lot, price, stoploss, takeprofit, deviation=10):
    """
    Send a market order.
      direction: "buy" or "sell"
      lot: float
      price: float
      stoploss, takeprofit: floats
    Returns (True, ticket) or (False, error_msg)
    """
    # ensure MT5 is initialized
    if not mt5.initialize():
        initialize()

    # prepare request
    side = mt5.ORDER_TYPE_BUY if direction.lower() == "buy" else mt5.ORDER_TYPE_SELL
    price = mt5.symbol_info_tick(symbol).ask if side == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).bid

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(lot),
        "type": side,
        "price": price,
        "sl": float(stoploss),
        "tp": float(takeprofit),
        "deviation": deviation,
        "magic": 123456,               # your magic number
        "comment": "ZENO AutoTrader",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        return False, f"retcode={result.retcode}"
    return True, result.order

def shutdown():
    """Shutdown MT5 connection."""
    mt5.shutdown()
