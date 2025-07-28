import math

def get_pip_value(symbol='XAUUSD'):
    # XAUUSD: pip = 0.01, $1 move = $100/lot, so 1 pip (0.1) = $10, 0.01 = $1 per lot
    # Broker-dependent, adjust accordingly!
    return 1.0 if symbol.upper() == 'XAUUSD' else 10.0

def cap_sl(sl_pips, min_sl=10, max_sl=50):
    """Apply institutional SL hard cap in pips."""
    return max(min(max_sl, sl_pips), min_sl)

def position_sizer(
    account_balance,
    risk_tier='neutral',   # 'profit', 'neutral', 'drawdown'
    spread=2.0,            # in pips (realistic for XAUUSD)
    sl_pips=50,            # raw SL in pips (capped later)
    commission_per_lot=7,  # round-trip cost, adjust for broker
    slippage=2.0,          # in pips
    symbol='XAUUSD',
    min_lot=0.01,
    max_lot=100.0,
    min_sl=10,
    max_sl=50,
    enforce_1R2R=True,
    verbose=False
):
    # 1. Cap the stop loss
    capped_sl_pips = cap_sl(sl_pips, min_sl=min_sl, max_sl=max_sl)
    # 2. Get risk percent for system state
    risk_perc = {'profit': 0.01, 'neutral': 0.005, 'drawdown': 0.002}
    risk_pct = risk_perc.get(risk_tier, 0.005)
    pip_value = get_pip_value(symbol)
    total_sl_pips = capped_sl_pips + max(spread, 0) + max(slippage, 0)

    # 3. Calculate USD risk per trade
    risk_usd = max(account_balance * risk_pct, 1.0)  # $1 min floor to avoid no-trade edge cases

    # 4. Lot size calculation
    denom = max(total_sl_pips * pip_value, 1e-6)
    raw_lots = risk_usd / denom
    lots = round(max(min_lot, min(raw_lots, max_lot)), 2)

    # 5. Check for negative or impossible lots
    if lots <= 0 or math.isnan(lots):
        lots = min_lot

    # 6. Calculate true USD risk at this lot size
    usd_risk_actual = lots * capped_sl_pips * pip_value
    risk_diff = usd_risk_actual - risk_usd

    # 7. 1R:2R enforcement
    tp_pips = capped_sl_pips * 2 if enforce_1R2R else capped_sl_pips
    expected_tp_usd = lots * tp_pips * pip_value
    expected_sl_usd = lots * capped_sl_pips * pip_value

    breakdown = {
        'account_balance': round(account_balance, 2),
        'risk_tier': risk_tier,
        'risk_pct': risk_pct,
        'risk_usd_target': round(risk_usd, 2),
        'sl_pips': sl_pips,
        'capped_sl_pips': capped_sl_pips,
        'spread': spread,
        'slippage': slippage,
        'total_sl_pips': total_sl_pips,
        'pip_value': pip_value,
        'lots': lots,
        'commission_per_lot': commission_per_lot,
        'tp_pips': tp_pips,
        'expected_tp_usd': round(expected_tp_usd, 2),
        'expected_sl_usd': round(expected_sl_usd, 2),
        'usd_risk_actual': round(usd_risk_actual, 2),
        'risk_diff_usd': round(risk_diff, 2),
        'WARN_UNDERRISK': lots == min_lot and usd_risk_actual < risk_usd,
        'WARN_MIN_SL_EXCEEDED': sl_pips < min_sl
    }

    if verbose:
        print(f"[PositionSizer] {breakdown}")
        if breakdown['WARN_UNDERRISK']:
            print(f"[WARNING] Actual $ risk ({usd_risk_actual:.2f}) below target ({risk_usd:.2f}) due to min_lot floor.")
        if breakdown['WARN_MIN_SL_EXCEEDED']:
            print(f"[WARNING] SL pips ({sl_pips}) was below minimum allowed ({min_sl}).")

    return breakdown

if __name__ == "__main__":
    test_cases = [
        {'account_balance': 10000, 'risk_tier': 'profit',   'sl_pips': 5,  'spread': 2.0},
        {'account_balance': 10000, 'risk_tier': 'profit',   'sl_pips': 50, 'spread': 2.0},
        {'account_balance': 10000, 'risk_tier': 'neutral',  'sl_pips': 25, 'spread': 3.0},
        {'account_balance': 5000,  'risk_tier': 'drawdown', 'sl_pips': 75, 'spread': 2.5},
    ]
    for case in test_cases:
        print("\n--- Position Sizing Test ---")
        print(position_sizer(**case, verbose=True))
