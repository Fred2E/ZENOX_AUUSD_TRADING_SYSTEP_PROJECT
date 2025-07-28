import MetaTrader5 as mt5
import time
import logging

MT5_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe"

class MT5Bridge:
    def __init__(self, login=None, password=None, server=None):
        self.login = login
        self.password = password
        self.server = server
        self.is_initialized = False

    def initialize(self):
        """Robust MT5 initialization."""
        if not mt5.initialize():
            mt5.shutdown()
            if not mt5.initialize(path=MT5_PATH, login=self.login, password=self.password, server=self.server):
                logging.error(f"MT5 init failed, error: {mt5.last_error()}")
                raise RuntimeError(f"MT5 initialize() failed, error code = {mt5.last_error()}")
        self.is_initialized = True
        print("MT5 Initialized")

    def place_order(self, symbol, direction, lot, stoploss, takeprofit, deviation=10, magic=123456, comment="ZENO AutoTrader"):
        """Safe, robust market order with ALL checks and logging."""
        if not self.is_initialized:
            self.initialize()
        # --- Symbol check ---
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None or not symbol_info.visible:
            logging.error(f"Symbol {symbol} not available in MT5")
            return False, "Symbol not found"

        # --- Price check ---
        tick = mt5.symbol_info_tick(symbol)
        if tick is None or tick.bid is None or tick.ask is None:
            logging.error(f"Could not retrieve price tick for {symbol}")
            return False, "Tick data error"

        side = mt5.ORDER_TYPE_BUY if direction.lower() == "buy" else mt5.ORDER_TYPE_SELL
        price = tick.ask if side == mt5.ORDER_TYPE_BUY else tick.bid

        # --- Compose request ---
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(lot),
            "type": side,
            "price": price,
            "sl": float(stoploss),
            "tp": float(takeprofit),
            "deviation": deviation,
            "magic": magic,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        # --- Send order ---
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logging.error(f"Order send failed: retcode={result.retcode}, details={result}")
            return False, f"retcode={result.retcode} ({result.comment})"
        logging.info(f"Order SUCCESS: ticket={result.order}, symbol={symbol}, side={direction}, price={price}")
        return True, result.order

    def shutdown(self):
        """Cleanly shutdown."""
        mt5.shutdown()
        print("MT5 Shutdown")
