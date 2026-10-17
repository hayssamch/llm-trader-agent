# LLM Trader Agent

A single **chart-reading agent** that visually analyses stock candlestick
charts and issues explainable BUY / SELL / HOLD signals. The agent reads a
rendered chart image and calls analytical **tools** (technical indicators,
support/resistance, order blocks) to back up its visual reasoning, then a
deterministic **risk filter** turns its decision into a position-sized trade.

> Master's thesis prototype — for research, simulation and backtesting only.
> Not financial advice. Stock-market prediction is uncertain and risky.

## Architecture (v2 — single tool-calling agent)

```
Market data ─▶ DataFetcher ─┬─▶ IndicatorEngine ┐
                            ├─▶ SRDetector      ├─ tools ─┐
                            └─▶ OBDetector      ┘         │
                                                          ▼
                     ChartRenderer ─▶ [chart image] ─▶ ChartReadingAgent
                                                          │ (ReAct loop:
                                                          │  see → call tool →
                                                          │  reason → decide)
                                                          ▼
                                                     RiskFilter
                                                          ▼
                                            BUY · SELL · HOLD + rationale
                                                          ▼
                                        BacktestEngine ─▶ Evaluator
```

Sentiment / news / fundamentals are **deferred to future work** (see thesis
Chapter 9); the corresponding modules are intentionally out of the pipeline.

## Install

```bash
pip install -r requirements.txt
```

## Easy start — the dashboard (recommended)

The friendliest way to use this is the web dashboard. From the folder that
CONTAINS the `llm_trader/` package:

**Mac/Linux:** `./start.sh`  •  **Windows:** double-click `start.bat`

That creates a virtual environment, installs everything (first run only), and
opens the dashboard in your browser at `http://localhost:8501`. Pick a ticker,
a date range and a run mode, then click **Run backtest**.

Prefer to launch it yourself?

```bash
pip install -r llm_trader/requirements.txt
streamlit run app.py
```

Three run modes in the sidebar:
- **Demo (offline, instant)** — no API key, no internet. Explore the UI.
- **Real data · rule-based** — real prices, no API cost.
- **Full AI agent (GPT-4o)** — real prices + the VLM agent (needs an API key).

To put it online later (e.g. for remote examiners): push to GitHub and deploy
free on [Streamlit Community Cloud](https://share.streamlit.io) — same `app.py`,
no code changes.

---

## Command-line usage (advanced)


Run from the directory that CONTAINS the `llm_trader/` folder, so the package
is importable:

```bash
# 1) FIRST RUN — offline, no API key, deterministic (verifies all wiring):
PYTHONPATH=. python -m llm_trader.run --ticker AAPL --mock --synthetic --verbose

# 2) Real market data, still no API cost (checks yfinance + chart rendering):
PYTHONPATH=. python -m llm_trader.run --ticker AAPL --mock

# 3) FULL RUN — real data + GPT-4o agent (needs OPENAI_API_KEY):
export OPENAI_API_KEY=sk-...
PYTHONPATH=. python -m llm_trader.run --ticker AAPL --start 2024-06-01 --end 2024-09-30 --verbose
```

Inspect a single decision (and the agent's tool-call trace):

```bash
PYTHONPATH=. python -m llm_trader.debug_one --ticker AAPL --date 2024-09-16 --mock --synthetic
```

Run the tests (leakage guard, indicator shift, risk gates):

```bash
PYTHONPATH=. python -m llm_trader.tests.test_pipeline
```

## Daily signals to your phone (Telegram)

A scheduled GitHub Action runs the agent once per trading day (22:00 UTC, after the
US close) over the 11 assets and sends **one** Telegram digest: new BUYs with entry,
stop-loss, target and the agent's reasoning (plus the chart it analysed), positions to
close, and a short "no action" summary on quiet days. It **recommends only; it never
places orders.**

Because the system is long-only, it tracks *paper* positions so you are told to open a
position once, told when to close it (stop, target or agent exit), and never re-told to
buy something already held. Every decision is appended to `signals_log.csv` and every
closed paper trade to `paper_trades.csv` on the `live-data` branch. That log is a genuine
**forward test**: each decision is made before its outcome exists.

**Setup** (once): create a bot with `@BotFather`, message it, find your chat id via
`https://api.telegram.org/bot<TOKEN>/getUpdates`, then add four repository secrets
(Settings -> Secrets and variables -> Actions): `TELEGRAM_BOT_TOKEN`,
`TELEGRAM_CHAT_ID`, `OPENAI_API_KEY` (an OpenRouter key) and `ALPHAVANTAGE_API_KEY`.

**Test it:** Actions -> *Daily signals* -> *Run workflow*. The defaults are a safe test
(free rule-based mode, marked "TEST RUN", nothing saved). Choose `ai` to test the LLM.
Optional repository variables: `LIVE_MODEL` changes the model (default `openai/gpt-4o-mini`),
`LIVE_SIZE_PCT` changes the position size (default `0.05`).

**Same position size for both agents.** Every trade, by the AI and by the rule alike, uses
one size (5% by default, `fixed_size_pct` in `config.py`). Position size is then a portfolio
policy and not part of either agent's judgment, so an AI-versus-rule comparison measures
the BUY/SELL/HOLD decisions only. Set it to `agent` to let each agent propose its own size.

**Late price data.** Yahoo has returned the latest daily bar a day late on some nights, and
the job only analyses the newest bar. A second run at 09:00 UTC (Mon-Sat, before the US
open) catches a bar that was missing the night before. The message says so when prices are
behind ("Price data is N trading days behind ...", which also appears on market holidays),
`runs_log.csv` records every run (data source, newest bar, expected bar, days behind), and
each log row has a `data_source` column. If nothing is new, the morning run sends no message.
Decisions for days no run could see (for example 14 Oct 2026) are simply absent from the log.

**Locally** (prints instead of sending): `python -m llm_trader.live.daily --mode rule --dry-run`

The signal on a given close is what a backtest would decide for the *next* session
(`as_of` = last bar + 1 day), so live behaviour matches the evaluated behaviour.

### Reviewing decisions in Google Sheets

The log is a plain CSV on the `live-data` branch, so a Google Sheet can show it live with
one formula (no credentials; the repo is public, so the log is public too). In a new
Sheet, rename the first tab `Log` and put this in cell `A1`:

```
=IMPORTDATA("https://raw.githubusercontent.com/hayssamch/llm-trader-agent/live-data/signals_log.csv")
```

Google refreshes it about hourly. One row per decision: the **AI** (`agent = ai`) and the
free **rule-based baseline** (`agent = rule`) are logged on the same bars, so they are judged
on identical days. Columns: `last_bar, ticker, agent, action, confidence, approved, event,
close, size_pct, stop_loss, take_profit, fwd_1d, fwd_5d, fwd_10d, fwd_20d, reject_reason,
trend, model, run_utc, rationale`.

`fwd_Nd` is the % change of the close N trading days after the decision bar, measured from
that bar's close. It is blank until N days have passed and is filled in by later runs, which
is what lets you assess a past decision. Read it as "what the market did next", not as a
backtested trade result (see `paper_trades.csv` for closed paper trades).

A second tab (`Review`) with a start date in `B1` and an end date in `B2` can then compute,
per asset and agent, how BUY recommendations fared (5-day outcome in column M, 20-day in O).
Filter by date with `FILTER` first and aggregate with `QUERY` after: IMPORTDATA delivers the
dates as plain numbers, and a `QUERY` condition such as `A >= date '2026-10-13'` then silently
matches nothing.

```
=IFERROR(QUERY(FILTER(Log!A:T,Log!A:A>=$B$1,Log!A:A<=$B$2,Log!D:D="BUY",Log!M:M<>""),"select Col2, Col3, count(Col4), avg(Col13), avg(Col15) group by Col2, Col3 order by Col2, Col3 label Col2 'Ticker', Col3 'Agent', count(Col4) 'BUYs', avg(Col13) 'Avg 5-day move %', avg(Col15) 'Avg 20-day move %'",0),"No BUY with a 5-day outcome in this period yet.")
```

(Verified against an independent calculation on the live log, using the 1-day column.)

Always compare against the benchmark: the average `fwd_5d` of **all** `agent = ai` rows in
the same period is what a random pick would have earned. A BUY average only means something
if it beats that. Expect few BUY rows per month, so treat early numbers as anecdotes.

## Run modes

| Flag           | Effect                                                        |
|----------------|--------------------------------------------------------------|
| `--mock`       | Deterministic rule-based agent, **no API calls**             |
| `--synthetic`  | Generates synthetic OHLCV, **no network**                    |
| neither        | Real yfinance data + real GPT-4o agent (needs API key)       |

The `--mock --synthetic` combination is the recommended first run: it exercises
the whole pipeline with zero external dependencies.

## Leakage prevention (two layers)

1. `DataFetcher.fetch_window(as_of=...)` returns only bars **strictly before**
   the decision date.
2. `IndicatorEngine.compute()` applies `.shift(1)` to every indicator column,
   so the value read at bar *T* reflects only information available at *T-1*.

Both are covered by tests in `tests/test_pipeline.py`.

## Module map

| Module                          | Role                                        |
|---------------------------------|---------------------------------------------|
| `config.py`                     | All parameters (`AgentConfig`)              |
| `models.py`                     | Typed data contracts                        |
| `data/fetcher.py`               | Market data + synthetic generator           |
| `indicators/engine.py`          | Technical indicators (tool)                 |
| `indicators/sr_detector.py`     | Support/resistance (tool)                   |
| `indicators/ob_detector.py`     | Order blocks (tool)                         |
| `chart/renderer.py`             | Candlestick chart → base64 PNG              |
| `agent/tools.py`                | Tool schemas + leakage-safe executor        |
| `agent/chart_agent.py`          | The chart-reading agent (ReAct loop)        |
| `decision/risk_filter.py`       | Hard risk rules                             |
| `backtest/engine.py`            | Chronological simulation                    |
| `backtest/evaluator.py`         | Metrics + baseline comparison               |
| `run.py` / `debug_one.py`       | Entry points                                |

## Switching to Claude instead of GPT-4o

Set `agent_model="claude-3-5-sonnet-20241022"` and adapt the tool-call block in
`agent/chart_agent.py` to the Anthropic Messages API (tools + `tool_use` /
`tool_result` blocks). The tool schemas in `agent/tools.py` map directly.
