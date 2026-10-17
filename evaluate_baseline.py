"""
Multi-asset baseline evaluation — rule-based agent on real data.

Runs the identical backtest across the asset universe and prints a comparison
table (strategy vs buy-and-hold) plus run-health diagnostics, so any zero-trade
result can be explained rather than guessed at.

    PYTHONPATH=. python evaluate_baseline.py
"""
import logging
import warnings

import pandas as pd

from llm_trader.config import AgentConfig
from llm_trader.backtest.engine import BacktestEngine
from llm_trader.backtest.evaluator import Evaluator

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

FIXED_SIZE = 0.05        # same size for every trade, as in the live job
START, END = "2025-10-01", "2026-07-28"
ASSETS = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "NVDA": "Nvidia",
    "JPM":  "JPMorgan",
    "XOM":  "ExxonMobil",
    "GLD":  "Gold ETF",
    "SLV":  "Silver ETF",
}

rows, health = [], []

for ticker, name in ASSETS.items():
    cfg = AgentConfig(
        tickers=[ticker], start_date=START, end_date=END,
        use_mock_llm=True,      # rule-based baseline (no LLM, no API cost)
        fixed_size_pct=FIXED_SIZE,
        use_synthetic=False,    # REAL market data
        cache_dir="./data/cache_eval",
        chart_dir="./data/charts_eval",
    )
    try:
        engine = BacktestEngine(cfg)
        result = engine.run(ticker)
        ev = Evaluator()
        strat = ev.evaluate(result, label=ticker)
        price = engine.fetcher.fetch(ticker).loc[START:END, "Close"]
        bnh = ev.buy_and_hold(price, cfg.initial_capital)

        rows.append({
            "Asset": f"{name} ({ticker})",
            "Strat ret %": round(strat.total_return_pct, 2),
            "B&H ret %": round(bnh.total_return_pct, 2),
            "Excess %": round(strat.total_return_pct - bnh.total_return_pct, 2),
            "Strat maxDD %": round(strat.max_drawdown_pct, 2),
            "B&H maxDD %": round(bnh.max_drawdown_pct, 2),
            "Win %": round(strat.win_rate_pct, 1),
            "Avg trade %": round(strat.avg_trade_pct, 2),
            "Trades": strat.n_trades,
        })
        health.append({
            "Asset": ticker, "approved": len(result.signals),
            "rejected": len(result.rejected), "errors": len(result.errors),
            "warmup": result.n_warmup, "bars": len(result.equity_curve),
        })
        print(f"  done {ticker}: {strat.n_trades} trades, "
              f"{strat.total_return_pct:+.2f}% vs B&H {bnh.total_return_pct:+.2f}%")
    except Exception as exc:
        print(f"  FAILED {ticker}: {exc}")
        health.append({"Asset": ticker, "approved": "-", "rejected": "-",
                       "errors": f"RUN FAILED: {exc}", "warmup": "-", "bars": "-"})

df = pd.DataFrame(rows)
hf = pd.DataFrame(health)

print("\n" + "=" * 100)
print(f" BASELINE (rule-based agent, real data)   {START} -> {END}")
print("=" * 100)
print(df.to_string(index=False))

print("\n RUN HEALTH (verifies results are real, not silent failures)")
print(hf.to_string(index=False))

if rows:
    print("\n AGGREGATE")
    print(f"   mean strategy return : {df['Strat ret %'].mean():+.2f}%")
    print(f"   mean buy&hold return : {df['B&H ret %'].mean():+.2f}%")
    print(f"   mean excess          : {df['Excess %'].mean():+.2f}%")
    print(f"   mean strategy maxDD  : {df['Strat maxDD %'].mean():.2f}%")
    print(f"   mean buy&hold maxDD  : {df['B&H maxDD %'].mean():.2f}%")
    print(f"   assets beating B&H   : {(df['Excess %'] > 0).sum()} / {len(df)}")
    print(f"   total trades         : {df['Trades'].sum()}")

df.to_csv("baseline_results.csv", index=False)
print("\n saved -> baseline_results.csv")
