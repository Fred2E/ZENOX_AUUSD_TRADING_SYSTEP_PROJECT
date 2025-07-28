import numpy as np

class RiskManager:
    """
    Institutional-grade risk manager for XAUUSD:
    - Dynamic lot sizing based on account equity, drawdown, confluence
    - ATR-based volatility stops, 1:2 RRR, and breakeven automation
    """

    def __init__(self, pip_value=100, spread=0.5):
        """
        Args:
            pip_value (float): Dollar value per $1 move, per standard lot for XAUUSD (usually $100).
            spread (float): Typical spread in dollars for XAUUSD (e.g., 0.5).
        """
        self.pip_value = pip_value
        self.spread = spread

    def get_risk_pct(self, account_equity, account_status='baseline', drawdown_pct=0.0, primary_score=4, secondary_score=0):
        """
        Calculate risk % per trade based on account health and confluence.
        - 1.0% in profit, 0.5% baseline, 0.2% in deep drawdown.
        - Reduce risk 20% if account is down by 20% (compounded).
        - Only allow if all primary confluences met (primary_score==4).

        Returns:
            float: risk % (e.g., 0.005 for 0.5%)
        """
        # Core risk tiers
        if account_status == 'profit':
            risk_pct = 0.01
        elif account_status == 'drawdown' and drawdown_pct > 0.5:
            risk_pct = 0.002
        else:
            risk_pct = 0.005

        # Dynamic adjustment for drawdown (20% reduction per -20% equity)
        reduction_factor = int(drawdown_pct // 0.2) * 0.2
        risk_pct = risk_pct * (1 - reduction_factor)

        # Only allow risk if all 4 primary confluences present
        if primary_score < 4:
            return 0.0

        # Allow higher tier if 2+ secondary confluences present (could be enhanced further)
        # But per your plan: all primary required, secondaries bonus only
        return risk_pct

    def compute_stop_loss(self, entry_price, atr, direction='long'):
        """
        ATR-based stop loss: adapts to current market volatility.

        Returns:
            float: stop loss price
        """
        if direction == 'long':
            sl = entry_price - atr * 1  # Multiplier=1 per your plan
        else:
            sl = entry_price + atr * 1
        return round(sl, 2)

    def compute_take_profit(self, entry_price, stop_loss, direction='long'):
        """
        Always enforces 1:2 risk:reward.
        Returns:
            float: take profit price
        """
        stop_distance = abs(entry_price - stop_loss)
        if direction == 'long':
            tp = entry_price + 2 * stop_distance
        else:
            tp = entry_price - 2 * stop_distance
        return round(tp, 2)

    def position_size(self, account_equity, risk_pct, entry_price, stop_loss):
        """
        Calculate dynamic lot size so max risk = % of current account.
        Always includes spread in calculation.

        Returns:
            float: lot size (rounded to 0.01 lots)
        """
        # Dollar risk per trade
        risk_amt = account_equity * risk_pct
        stop_dist = abs(entry_price - stop_loss) + self.spread
        if stop_dist == 0:
            return 0.0  # Avoid div-by-zero
        lot_size = risk_amt / (stop_dist * self.pip_value)
        return np.round(lot_size, 2)

    def breakeven_trigger(self, entry_price, take_profit, direction='long'):
        """
        Breakeven trigger: move SL to entry after price moves 70% toward TP.

        Returns:
            float: price at which SL should be moved to entry
        """
        if direction == 'long':
            be_level = entry_price + 0.7 * (take_profit - entry_price)
        else:
            be_level = entry_price - 0.7 * (entry_price - take_profit)
        return round(be_level, 2)

    def generate_trade_plan(
        self, account_equity, account_status, drawdown_pct,
        entry_price, atr, direction, primary_score, secondary_score
    ):
        """
        Core function: returns complete trade sizing and risk plan dictionary
        based on all risk and confluence logic.

        Returns:
            dict: {risk_pct, stop_loss, take_profit, lot_size, breakeven_trigger, trade_allowed}
        """
        # 1. Compute allowed risk percent
        risk_pct = self.get_risk_pct(
            account_equity, account_status, drawdown_pct,
            primary_score, secondary_score
        )
        if risk_pct == 0.0:
            return {
                'trade_allowed': False,
                'reason': 'Not all primary confluences met or account risk logic blocked.'
            }

        # 2. Calculate stop loss (ATR), TP (1:2 RRR)
        stop_loss = self.compute_stop_loss(entry_price, atr, direction)
        take_profit = self.compute_take_profit(entry_price, stop_loss, direction)

        # 3. Calculate lot size (dynamic)
        lot_size = self.position_size(account_equity, risk_pct, entry_price, stop_loss)

        # 4. Calculate breakeven trigger level
        be_trigger = self.breakeven_trigger(entry_price, take_profit, direction)

        # 5. Return full plan
        return {
            'trade_allowed': True,
            'risk_pct': risk_pct,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'lot_size': lot_size,
            'breakeven_trigger': be_trigger
        }

# Example usage:
# risk_mgr = RiskManager(pip_value=100, spread=0.5)
# plan = risk_mgr.generate_trade_plan(
#     account_equity=10000, account_status='baseline', drawdown_pct=0.0,
#     entry_price=1950, atr=3.0, direction='long',
#     primary_score=4, secondary_score=3
# )
# print(plan)
