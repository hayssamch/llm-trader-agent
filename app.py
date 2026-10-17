"""
app.py — LLM Trader Agent dashboard (Streamlit front end).

Run from the folder that CONTAINS the `llm_trader/` package:
    streamlit run app.py

Turns the CLI backtester into a clickable product: choose a ticker, a date
range and a run mode, click Run, and see the equity curve, performance metrics,
trade log and the agent's per-decision reasoning. A second tab inspects a single
day — showing the exact candlestick chart the agent looked at and how it reasoned.
"""
import os
os.environ.setdefault("MPLBACKEND", "Agg")   # headless plotting

import datetime as dt

import pandas as pd
import streamlit as st

from llm_trader.config import AgentConfig
from llm_trader.data.fetcher import DataFetcher
from llm_trader.indicators.engine import IndicatorEngine
from llm_trader.chart.renderer import ChartRenderer
from llm_trader.agent.tools import ToolExecutor
from llm_trader.agent.chart_agent import ChartReadingAgent
from llm_trader.decision.risk_filter import RiskFilter
from llm_trader.backtest.engine import BacktestEngine
from llm_trader.backtest.evaluator import Evaluator


# ── Run-mode presets ─────────────────────────────────────────────────────────
MODES = {
    "Demo (offline, instant)":   dict(use_mock_llm=True,  use_synthetic=True),
    "Real data · rule-based":    dict(use_mock_llm=True,  use_synthetic=False),
    "Full AI agent (GPT-4o)":    dict(use_mock_llm=False, use_synthetic=False),
}

# ── Selectable assets ────────────────────────────────────────────────────────
# Gold and silver use the GLD/SLV ETFs: they trade like stocks, so both the
# yfinance primary source and the Alpha Vantage cloud fallback support them
# (futures symbols like GC=F are Yahoo-only and would break on Streamlit Cloud).
ASSETS = {
    "Apple (AAPL)":        "AAPL",
    "Microsoft (MSFT)":    "MSFT",
    "Nvidia (NVDA)":       "NVDA",
    "Amazon (AMZN)":       "AMZN",
    "Alphabet (GOOGL)":    "GOOGL",
    "Meta (META)":         "META",
    "Tesla (TSLA)":        "TSLA",
    "JPMorgan (JPM)":      "JPM",
    "ExxonMobil (XOM)":    "XOM",
    "Gold ETF (GLD)":      "GLD",
    "Silver ETF (SLV)":    "SLV",
}

st.set_page_config(page_title="LLM Trader Agent", page_icon="📈", layout="wide")

# Cloud data fallback: expose ALPHAVANTAGE_API_KEY from Streamlit secrets to the
# fetcher (Yahoo rate-limits cloud IPs; Alpha Vantage takes over when set).
try:
    if "ALPHAVANTAGE_API_KEY" in st.secrets:
        os.environ.setdefault("ALPHAVANTAGE_API_KEY", st.secrets["ALPHAVANTAGE_API_KEY"])
except Exception:
    pass  # no secrets file configured (e.g. local run) — fine


def build_config(mode_key, ticker, start, end, capital, max_pos, min_conf, model,
                 same_size=True):
    m = MODES[mode_key]
    return AgentConfig(
        tickers=[ticker],
        start_date=start.strftime("%Y-%m-%d"),
        end_date=end.strftime("%Y-%m-%d"),
        initial_capital=float(capital),
        max_position_pct=max_pos / 100.0,
        fixed_size_pct=(max_pos / 100.0) if same_size else None,
        min_confidence=min_conf,
        agent_model=model,
        use_mock_llm=m["use_mock_llm"],
        use_synthetic=m["use_synthetic"],
    )


# ── Sidebar: configuration ───────────────────────────────────────────────────
st.sidebar.title("📈 LLM Trader Agent")
st.sidebar.caption("Research prototype.")

mode_key = st.sidebar.selectbox("Run mode", list(MODES.keys()), index=0,
    help="Demo needs nothing. Rule-based uses real prices, no API. "
         "Full AI calls GPT-4o and needs an API key.")

asset_label = st.sidebar.selectbox("Asset", list(ASSETS.keys()), index=0)
ticker = ASSETS[asset_label]

# Rolling default window: ends today, starts ~10 months back. The agent needs
# ~60 trading days of warm-up before its first decision, so the evaluated
# period is roughly the last 7 months. Recent dates also sit safely after LLM
# training cutoffs (no memorized-price leakage).
_end_default   = dt.date.today()
_start_default = _end_default - dt.timedelta(days=300)

col_a, col_b = st.sidebar.columns(2)
start = col_a.date_input("Start", value=_start_default)
end   = col_b.date_input("End",   value=_end_default)

st.sidebar.markdown("**Portfolio & risk**")
capital  = st.sidebar.number_input("Initial capital ($)", 1_000, 10_000_000, 100_000, step=1_000)
max_pos  = st.sidebar.slider("Position size (%)", 1, 50, 5)
min_conf = st.sidebar.slider("Min confidence to trade", 0.0, 1.0, 0.6, 0.05)
same_size = st.sidebar.checkbox(
    "Same size for every trade", value=True,
    help="On: every trade, by the AI and the rule alike, uses exactly the position size "
         "above, so comparing them reflects the decisions and not the size. "
         "Off: the AI picks its own size, up to the position size above.")

model = "gpt-4o"
if MODES[mode_key]["use_mock_llm"] is False:
    model = st.sidebar.selectbox("Model",
    ["openai/gpt-4o", "openai/gpt-4o-mini", "anthropic/claude-3.5-sonnet"], index=0)
    api_key = st.sidebar.text_input("OpenRouter API key", type="password",
        help="Paste the full key from openrouter.ai/settings/keys "
             "(starts with sk-or-v1-). Used only for this session; never stored.")
    api_key = (api_key or "").strip()
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
        if not api_key.startswith("sk-or-"):
            st.sidebar.warning("This doesn't look like an OpenRouter key "
                               "(should start with sk-or-). Double-check it.")


# ── Main area: tabs ──────────────────────────────────────────────────────────
tab_run, tab_inspect, tab_about = st.tabs(
    ["▶ Backtest", "🔍 Inspect one decision", "ℹ About"]
)

# =============================================================================
# TAB 1 — BACKTEST
# =============================================================================
with tab_run:
    st.subheader(f"Backtest — {asset_label}")

    if MODES[mode_key]["use_mock_llm"] is False and not os.environ.get("OPENAI_API_KEY"):
        st.warning("Full AI mode needs an OpenRouter API key (enter it in the sidebar).")

    run = st.button("▶ Run backtest", type="primary", use_container_width=True)

    if run:
        cfg = build_config(mode_key, ticker, start, end, capital, max_pos, min_conf, model, same_size)
        try:
            with st.spinner(f"Running the agent over {ticker}… "
                            f"({'instant' if cfg.use_synthetic else 'fetching real data'})"):
                engine = BacktestEngine(cfg)
                result = engine.run(ticker)
                ev = Evaluator()
                strat = ev.evaluate(result, label="Chart Agent")
                price = engine.fetcher.fetch(ticker).loc[cfg.start_date: cfg.end_date, "Close"]
                bnh = ev.buy_and_hold(price, cfg.initial_capital)
            st.session_state["result"] = result
            st.session_state["strat"] = strat
            st.session_state["bnh"] = bnh
            st.session_state["table"] = ev.comparison_table([strat, bnh])
        except Exception as exc:
            st.error(f"Run failed: {exc}")
            if not cfg.use_synthetic:
                st.info("Tip: try **Demo (offline, instant)** first to confirm the pipeline works.")

    if "strat" in st.session_state:
        strat  = st.session_state["strat"]
        bnh    = st.session_state["bnh"]
        result = st.session_state["result"]

        # KPI row
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total return", f"{strat.total_return_pct:.1f}%",
                  delta=f"{strat.total_return_pct - bnh.total_return_pct:.1f}% vs B&H")
        c2.metric("Sharpe", f"{strat.sharpe:.2f}")
        c3.metric("Max drawdown", f"{strat.max_drawdown_pct:.1f}%")
        c4.metric("Win rate", f"{strat.win_rate_pct:.0f}%")
        c5.metric("Trades", f"{strat.n_trades}")

        # Equity curve vs buy & hold
        st.markdown("##### Equity curve")
        eq = pd.DataFrame({
            "Chart Agent": strat.equity_curve,
            "Buy & Hold":  bnh.equity_curve,
        })
        st.line_chart(eq)

        # Metrics table + trades side by side
        left, right = st.columns([1, 1])
        with left:
            st.markdown("##### Metrics vs benchmark")
            st.dataframe(st.session_state["table"], use_container_width=True)
        with right:
            st.markdown("##### Trade log")
            if result.trades:
                tdf = pd.DataFrame(result.trades)
                tdf["pnl_%"] = (tdf["pnl_pct"] * 100).round(2)
                st.dataframe(
                    tdf[["entry_date", "exit_date", "entry_price",
                         "exit_price", "pnl_%", "reason"]],
                    use_container_width=True, height=280)
            else:
                st.info("No completed trades in this period.")

        # Agent reasoning
        st.markdown("##### Agent decisions")

        # Run health — report failures explicitly. A run where every LLM call
        # errored must never be presented as if the agent chose to do nothing.
        n_err, n_rej = len(result.errors), len(result.rejected)
        if n_err:
            st.error(
                f"**{n_err} decision(s) FAILED — the agent produced no answer on "
                f"those dates.** These are errors, not decisions; any result below "
                f"is based only on the dates that succeeded.\n\n"
                f"First error ({result.errors[0]['as_of']}): "
                f"`{result.errors[0]['error'][:300]}`")
        if n_rej:
            st.warning(f"{n_rej} decision(s) were blocked by the risk filter — "
                       f"e.g. {result.rejected[0].as_of}: "
                       f"{result.rejected[0].reject_reason}")
        if result.n_warmup:
            st.caption(f"{result.n_warmup} early day(s) skipped: the agent needs "
                       f"~60 trading days of history before its first decision.")

        if result.signals:
            st.caption(f"{len(result.signals)} approved decision(s)"
                       f"{' — showing first 25' if len(result.signals) > 25 else ''}:")
            for s in result.signals[:25]:
                with st.expander(f"{s.as_of} — {s.action.value} "
                                 f"(confidence {s.confidence:.2f})"):
                    st.write(f"**Chart reading:** {s.chart_reading or '—'}")
                    st.write(f"**Rationale:** {s.rationale or '—'}")
        elif not n_err and not n_rej:
            st.info("No decisions were produced. The window is likely too short — "
                    "it needs ~60 trading days of warm-up plus the period you want "
                    "evaluated (so ~6 months minimum).")


# =============================================================================
# TAB 2 — SINGLE DECISION INSPECTOR
# =============================================================================
with tab_inspect:
    st.subheader("Inspect a single decision")
    st.caption("See the exact chart the agent looked at and how it reasoned.")

    insp_date = st.date_input("Decision date", value=dt.date.today() - dt.timedelta(days=7),
                              key="insp_date")
    go = st.button("🔍 Analyse this date", type="primary")

    if go:
        cfg = build_config(mode_key, ticker, start, end, capital, max_pos, min_conf, model, same_size)
        as_of = insp_date.strftime("%Y-%m-%d")
        try:
            with st.spinner("Rendering chart and running the agent…"):
                fetcher = DataFetcher(cfg)
                window  = fetcher.fetch_window(ticker, as_of=as_of)
                if len(window) < 60:
                    st.error(f"Only {len(window)} bars before {as_of} — need ≥60. "
                             f"Pick a later date or widen the data range.")
                    st.stop()
                enriched = IndicatorEngine(cfg).compute(window)
                chart_path = os.path.join(cfg.chart_dir, f"{ticker}_{as_of}.png")
                ChartRenderer(cfg).render(enriched, ticker, as_of, save_path=chart_path)
                executor = ToolExecutor(window, ticker, as_of, cfg)
                decision = ChartReadingAgent(cfg).analyze(
                    ChartRenderer.encode_file(chart_path), executor, ticker, as_of)
                price = float(window["Close"].iloc[-1])
                signal = RiskFilter(cfg).filter(decision, price)

            img_col, txt_col = st.columns([3, 2])
            with img_col:
                st.image(chart_path, caption=f"What the agent saw — {ticker} as of {as_of} "
                         f"(last bar {window.index[-1].date()})", use_container_width=True)
            with txt_col:
                verdict = signal.action.value if signal.approved else f"{decision.action} → HOLD"
                st.metric("Decision", verdict, delta=f"confidence {decision.confidence:.2f}")
                st.write(f"**Trend read:** {decision.trend}")
                st.write(f"**Tools called:** {', '.join(decision.tools_used) or '—'}")
                if decision.key_levels:
                    st.write(f"**Key levels:** {', '.join(f'{x:.2f}' for x in decision.key_levels)}")
                if not signal.approved:
                    st.warning(f"Risk filter rejected: {signal.reject_reason}")
                else:
                    tp = f"{signal.take_profit:.2f}" if signal.take_profit else "—"
                    st.write(f"**Stop-loss:** {signal.stop_loss:.2f}  |  **Take-profit:** {tp}")

            st.markdown("##### Chart reading")
            st.info(decision.chart_reading or "—")
            st.markdown("##### Rationale")
            st.write(decision.rationale or "—")

        except Exception as exc:
            st.error(f"Analysis failed: {exc}")


# =============================================================================
# TAB 3 — ABOUT
# =============================================================================
with tab_about:
    st.markdown("""
### What this is
A single **chart-reading agent** that visually analyses candlestick charts and
issues explainable BUY / SELL / HOLD signals. It reads a rendered chart image
and calls analytical **tools** (indicators, support/resistance, order blocks)
to back up its visual reasoning, then a deterministic **risk filter** turns the
decision into a position-sized trade.

### Run modes
- **Demo (offline, instant)** — synthetic data + rule-based agent. No API key,
  no network. Use this to explore the interface.
- **Real data · rule-based** — real prices from yfinance, still no API cost.
- **Full AI agent (GPT-4o)** — real prices + the actual VLM agent. Needs an
  OpenAI API key (entered in the sidebar, used only for the session).

### Leakage prevention
The agent only ever sees data **strictly before** the decision date, and all
indicators are shifted by one bar. See the thesis methodology chapter.

> For research, simulation and backtesting only. Stock-market prediction is
> uncertain and risky. This is **not** financial advice.
""")
