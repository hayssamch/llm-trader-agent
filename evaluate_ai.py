"""
Full-AI vs rule-based comparison — the core experiment.

Runs BOTH decision-makers over the same assets, window, data, tools and risk
filter. The only variable is who decides: the EMA/RSI rule, or the LLM reading
the rendered chart. Everything else is held constant, so any difference in
results is attributable to the decision-maker.

Results are reported two size-independent ways:
  * equity versus an EXPOSURE-MATCHED buy-and-hold (same average capital in the asset);
  * per-trade metrics, with the average trade compared with entering on any day and
    holding for the same length of time.

Requires an OpenRouter key in the environment (never hard-code it):

    # PowerShell
    $env:OPENAI_API_KEY = "sk-or-v1-..."
    $env:OPENAI_BASE_URL = "https://openrouter.ai/api/v1"
    python evaluate_ai.py

    # bash
    export OPENAI_API_KEY="sk-or-v1-..."
    export OPENAI_BASE_URL="https://openrouter.ai/api/v1"
    python evaluate_ai.py

Set EVAL_REPEATS=2 (or more) to repeat the AI run and measure run-to-run variance.

Cost warning: one LLM call per trading day per asset (~145 per asset for the
default window). Keep ASSETS short. openai/gpt-4o-mini is the cheap option.
"""
import logging
import os
import sys
import warnings

import pandas as pd

from llm_trader.config import AgentConfig
from llm_trader.backtest.engine import BacktestEngine
from llm_trader.backtest.evaluator import Evaluator

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

FIXED_SIZE = 0.05        # both agents trade this size, so only the decisions differ
START, END = "2025-10-01", "2026-07-28"
MODEL  = os.environ.get("EVAL_MODEL", "openai/gpt-4o-mini")
ASSETS = {"AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "Nvidia", "JPM": "JPMorgan"}
REPEATS = int(os.environ.get("EVAL_REPEATS", "1"))   # >1 repeats the AI run to measure variance

if not os.environ.get("OPENAI_API_KEY"):
    sys.exit("OPENAI_API_KEY is not set. See the docstring at the top of this file.")
os.environ.setdefault("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")


def run(ticker: str, use_mock: bool):
    cfg = AgentConfig(
        tickers=[ticker], start_date=START, end_date=END,
        use_mock_llm=use_mock, use_synthetic=False, agent_model=MODEL,
        fixed_size_pct=FIXED_SIZE,
        cache_dir="./data/cache_eval", chart_dir="./data/charts_eval",
    )
    engine = BacktestEngine(cfg)
    result = engine.run(ticker)
    price = engine.fetcher.fetch(ticker).loc[START:END, "Close"]
    return result, Evaluator().summarise(result, price, label=ticker)


rows = []
for ticker, name in ASSETS.items():
    runs = [("Rule-based", True)] + [(f"AI #{k + 1}", False) for k in range(REPEATS)]
    for label, use_mock in runs:
        print(f"  running {ticker} — {label} …", flush=True)
        try:
            result, summ = run(ticker, use_mock)
            n_err = len(result.errors)
            if n_err:
                print(f"    WARNING: {n_err} decisions FAILED — "
                      f"first: {result.errors[0]['error'][:120]}", flush=True)
            rows.append({"Asset": ticker, "Agent": label, **summ,
                         "Approved": len(result.signals),
                         "Rejected": len(result.rejected), "Errors": n_err})
            print(f"    -> {summ['Trades']} trades, {summ['Return %']:+.2f}%, "
                  f"exposure {summ['Exposure %']:.1f}%", flush=True)
        except Exception as exc:
            print(f"    FAILED: {exc}", flush=True)
            rows.append({"Asset": ticker, "Agent": label, "Errors": f"RUN FAILED: {exc}"})

df = pd.DataFrame(rows)
d = df.round(2)
print("\n" + "=" * 100)
print(f" RULE-BASED vs FULL AI   {START} -> {END}   model={MODEL}   size={FIXED_SIZE:.0%}")
print("=" * 100)
print("\n EQUITY, EXPOSURE-MATCHED (B&H matched holds the same average capital in the asset)")
print(d[["Asset", "Agent", "Return %", "Exposure %", "B&H matched %", "Excess vs matched",
         "MaxDD %"]].to_string(index=False))
print("\n PER TRADE (independent of position size; net of round-trip commission)")
print(d[["Asset", "Agent", "Trades", "Win %", "Avg trade % (net)", "Avg hold (days)",
         "Any-entry same-hold %", "Trade edge (pts)"]].to_string(index=False))
print("\n RUN HEALTH")
print(d[["Asset", "Agent", "Approved", "Rejected", "Errors"]].to_string(index=False))
print("\n Errors must be 0 for a result to be trustworthy — a run where calls")
print(" failed produces 0 trades that must NOT be read as a deliberate HOLD.")
df.to_csv("ai_comparison_results.csv", index=False)
print("\n saved -> ai_comparison_results.csv")
