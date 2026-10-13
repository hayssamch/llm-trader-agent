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
Optional repository variable `LIVE_MODEL` changes the model (default `openai/gpt-4o-mini`).

**Locally** (prints instead of sending): `python -m llm_trader.live.daily --mode rule --dry-run`

The signal on a given close is what a backtest would decide for the *next* session
(`as_of` = last bar + 1 day), so live behaviour matches the evaluated behaviour.

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
