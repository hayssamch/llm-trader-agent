"""
Multi-asset baseline evaluation — rule-based agent on real data.

Runs the identical backtest across the asset universe and prints two tables:

  * equity, compared with an EXPOSURE-MATCHED buy-and-hold (the same average capital
    held in the asset), because a fixed-size, long-only strategy is invested only a
    fraction of the time and a fully invested buy-and-hold is not a like-for-like yardstick;
  * per-trade metrics that do not depend on position size, with the average trade compared
    with entering on any day and holding for the same length of time.

Run-health diagnostics are printed so a zero-trade result is explained, not guessed at.

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
        price = engine.fetcher.fetch(ticker).loc[START:END, "Close"]
        row = {"Asset": ticker, **ev.summarise(result, price, label=ticker)}
        rows.append(row)
        health.append({
            "Asset": ticker, "approved": len(result.signals),
            "rejected": len(result.rejected), "errors": len(result.errors),
            "warmup": result.n_warmup, "bars": len(result.equity_curve),
        })
        print(f"  done {ticker}: {row['Trades']} trades, {row['Return %']:+.2f}% "
              f"(exposure {row['Exposure %']:.1f}%, matched B&H {row['B&H matched %']:+.2f}%)")
    except Exception as exc:
        print(f"  FAILED {ticker}: {exc}")
        health.append({"Asset": ticker, "approved": "-", "rejected": "-",
                       "errors": f"RUN FAILED: {exc}", "warmup": "-", "bars": "-"})

df = pd.DataFrame(rows)
hf = pd.DataFrame(health)

print("\n" + "=" * 100)
print(f" BASELINE (rule-based agent, real data, {FIXED_SIZE:.0%} per trade)   {START} -> {END}")
print("=" * 100)

if rows:
    d = df.round(2)
    print("\n EQUITY, EXPOSURE-MATCHED (B&H matched holds the same average capital in the asset)")
    print(d[["Asset", "Return %", "Exposure %", "B&H matched %", "Excess vs matched",
             "B&H full %", "MaxDD %", "B&H matched DD %"]].to_string(index=False))
    print("\n PER TRADE (independent of position size; net of round-trip commission)")
    print(d[["Asset", "Trades", "Win %", "Avg trade % (net)", "Median trade %",
             "Avg hold (days)", "Any-entry same-hold %", "Trade edge (pts)"]].to_string(index=False))

print("\n RUN HEALTH (verifies results are real, not silent failures)")
print(hf.to_string(index=False))

if rows:
    print("\n AGGREGATE")
    print(f"   mean exposure          : {df['Exposure %'].mean():.1f}%")
    print(f"   mean excess vs matched : {df['Excess vs matched'].mean():+.2f} pts")
    print(f"   assets beating matched : {(df['Excess vs matched'] > 0).sum()} / {len(df)}")
    print(f"   mean trade edge        : {df['Trade edge (pts)'].mean():+.2f} pts")
    print(f"   total trades           : {int(df['Trades'].sum())}")

df.to_csv("baseline_results.csv", index=False)
print("\n saved -> baseline_results.csv")
