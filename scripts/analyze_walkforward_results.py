import pandas as pd
import numpy as np
import os

LOG_DIR = r"C:\Users\open\Documents\ZENO_XAUUSD\logs"
TIMEFRAMES = ["M5", "M15", "H1", "H4", "D1"]

def analyze_trade_log(trade_log):
    if trade_log.empty:
        return {
            "Num Trades": 0,
            "Total Pips": 0,
            "Mean Pips/Trade": 0,
            "Win Rate (%)": 0,
            "Sharpe Ratio": 0,
            "Max Drawdown": 0
        }
    pnl = trade_log['reward_pips']
    cum_pnl = pnl.cumsum()
    max_dd = np.min(cum_pnl - np.maximum.accumulate(cum_pnl))
    sharpe = pnl.mean() / (pnl.std() + 1e-8) * np.sqrt(len(pnl))
    return {
        "Num Trades": len(trade_log),
        "Total Pips": pnl.sum(),
        "Mean Pips/Trade": pnl.mean(),
        "Win Rate (%)": (pnl > 0).mean() * 100,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_dd
    }

def main():
    summary = []
    print("\n=== Automated Walk-Forward Backtest Analysis ===\n")
    for tf in TIMEFRAMES:
        path = os.path.join(LOG_DIR, f"trade_log_{tf}.csv")
        if not os.path.exists(path):
            print(f"{tf}: trade log missing, skipping.")
            continue
        df = pd.read_csv(path)
        stats = analyze_trade_log(df)
        stats = {k: (round(v, 2) if isinstance(v, (float, int)) else v) for k,v in stats.items()}
        stats['Timeframe'] = tf
        summary.append(stats)

        print(f"--- {tf} ---")
        for k, v in stats.items():
            if k != 'Timeframe':
                print(f"{k}: {v}")
        print()

    # Sort and highlight
    df = pd.DataFrame(summary)
    df = df[['Timeframe', 'Num Trades', 'Total Pips', 'Mean Pips/Trade', 'Win Rate (%)', 'Sharpe Ratio', 'Max Drawdown']]
    df.to_csv(os.path.join(LOG_DIR, "walkforward_analysis.csv"), index=False)
    print(df)
    print("\nAnalysis saved to walkforward_analysis.csv in logs folder.")

    # Identify best/worst timeframe by Sharpe
    if not df.empty:
        best = df.loc[df['Sharpe Ratio'].idxmax()]
        worst = df.loc[df['Sharpe Ratio'].idxmin()]
        print(f"\n**Best Sharpe:** {best['Timeframe']} ({best['Sharpe Ratio']:.2f})")
        print(f"**Worst Sharpe:** {worst['Timeframe']} ({worst['Sharpe Ratio']:.2f})")
        print("\nRecommendation: Investigate what the best timeframe is doing right. Ruthlessly review the worst for improvement or possible exclusion.")

if __name__ == "__main__":
    main()
