"""
live/daily.py — daily signal runner (the "live" mode of the LLM Trader Agent).

Run once per trading day after the US close. For every asset it applies the
SAME leakage-safe pipeline used in the backtests (fetch -> indicators -> chart ->
agent -> risk filter) to the latest available bar, then sends one Telegram digest.

It recommends only; it never places orders. To make "BUY"/"SELL" meaningful for a
long-only system it tracks PAPER positions, so you are told to open a position
once, told when it should be closed (stop, target or agent exit), and never
re-told to buy something already held. Every decision is appended to a log, which
doubles as a genuine forward test (each decision is made before its outcome exists).

    python -m llm_trader.live.daily --mode rule --dry-run        # free, prints only
    python -m llm_trader.live.daily --mode ai                    # real run

Timing: a decision with bars up to day L is the decision a backtest would make
for day L+1, so as_of = L+1 (the window is strictly before as_of, as always).
"""
import os
os.environ.setdefault("MPLBACKEND", "Agg")   # headless: must be set before matplotlib loads

import argparse
import csv
import html
import json
import logging
import sys
import warnings
from dataclasses import dataclass, field, replace
from datetime import date, datetime, time, timedelta, timezone
from collections import Counter
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from llm_trader.config import AgentConfig
from llm_trader.data.fetcher import DataFetcher
from llm_trader.indicators.engine import IndicatorEngine
from llm_trader.chart.renderer import ChartRenderer
from llm_trader.agent.tools import ToolExecutor
from llm_trader.agent.chart_agent import ChartReadingAgent
from llm_trader.decision.risk_filter import RiskFilter
from llm_trader.models import Action, AgentDecision
from llm_trader.live.notifier import TelegramNotifier

logger = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")

UNIVERSE = {
    "AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "Nvidia", "AMZN": "Amazon",
    "GOOGL": "Alphabet", "META": "Meta", "TSLA": "Tesla", "JPM": "JPMorgan",
    "XOM": "ExxonMobil", "GLD": "Gold ETF", "SLV": "Silver ETF",
}
SECRET_ENV = ("TELEGRAM_BOT_TOKEN", "OPENAI_API_KEY", "ALPHAVANTAGE_API_KEY")
APP_URL = os.environ.get("APP_URL", "https://llm-trading-agent.streamlit.app")
HORIZONS = (1, 5, 10, 20)           # trading days after the decision bar
FWD_FIELDS = [f"fwd_{h}d" for h in HORIZONS]
# Most useful columns first, the long rationale last. One row per
# (last_bar, ticker, agent); the fwd_* columns fill in as prices arrive.
LOG_FIELDS = (["last_bar", "ticker", "agent", "action", "confidence", "approved",
               "event", "close", "size_pct", "stop_loss", "take_profit"] + FWD_FIELDS +
              ["reject_reason", "trend", "model", "run_utc", "rationale",
               "data_source"])          # appended last: columns A-T keep their letters in the Sheet
LOG_KEY = ("last_bar", "ticker", "agent")
TRADE_FIELDS = ["ticker", "entry_date", "entry_price", "exit_date", "exit_price",
                "pnl_pct", "reason", "model"]
# One row per run: makes data gaps and source switches visible after the fact.
RUN_FIELDS = ["run_utc", "trigger", "mode", "model", "size_policy", "analysed", "skipped",
              "failed", "newest_bar", "expected_bar", "days_behind", "sources"]


def scrub(text: str) -> str:
    """Remove any secret value from text before it is logged or sent."""
    for name in SECRET_ENV:
        val = os.environ.get(name, "")
        if val and len(val) > 6:
            text = text.replace(val, "***")
    return text


# ── state ────────────────────────────────────────────────────────────────────
@dataclass
class LiveState:
    last_bar: Dict[str, str] = field(default_factory=dict)    # newest bar already processed
    positions: Dict[str, dict] = field(default_factory=dict)  # open PAPER positions

    @classmethod
    def load(cls, path: str) -> "LiveState":
        if not os.path.exists(path):
            return cls()
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
        return cls(raw.get("last_bar", {}), raw.get("positions", {}))

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"last_bar": self.last_bar, "positions": self.positions},
                      fh, indent=2, sort_keys=True)


@dataclass
class Candidate:
    ticker:     str
    last_bar:   str            # date of the newest bar the agent saw
    close:      float
    decision:   AgentDecision
    chart_path: str
    rule_decision: Optional[AgentDecision] = None   # free baseline, logged alongside the AI
    data_source: str = ""                            # yfinance / alphavantage / cache / synthetic


# ── data / analysis ──────────────────────────────────────────────────────────
def drop_partial_bar(full: pd.DataFrame, now_utc: datetime) -> pd.DataFrame:
    """Drop today's bar if the US market has not closed yet (it would be partial)."""
    if full.empty:
        return full
    now_et = now_utc.astimezone(ET)
    pending = now_et.weekday() < 5 and now_et.time() < time(16, 30)
    if full.index[-1].date() == now_et.date() and pending:
        return full.iloc[:-1]
    return full


def expected_last_bar(now_utc: datetime) -> date:
    """Date of the newest US trading day whose close should already be published: today on a
    weekday after 16:30 ET, otherwise the previous weekday. Market holidays are not modelled,
    so a holiday shows up as 'one day behind' (the message says so)."""
    now_et = now_utc.astimezone(ET)
    d = now_et.date()
    if not (now_et.weekday() < 5 and now_et.time() >= time(16, 30)):
        d -= timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def days_behind(newest: date, expected: date) -> int:
    """Whole trading days (Mon-Fri) that the newest bar lags the expected one; 0 if current."""
    if newest >= expected:
        return 0
    return int(np.busday_count(newest, expected))


def analyse_ticker(ticker: str, cfg: AgentConfig, agent: ChartReadingAgent,
                   state: LiveState, now_utc: datetime, force: bool = False,
                   rule_agent: Optional[ChartReadingAgent] = None,
                   series_out: Optional[Dict[str, pd.Series]] = None,
                   source_out: Optional[Dict[str, str]] = None) -> Optional[Candidate]:
    """Returns a Candidate, or None when there is no new bar since the last run.

    rule_agent: when given (AI mode), the free rule-based decision for the same bar is
                computed too, so both decision-makers are judged on identical days.
    series_out: receives the asset's closing prices (used later to fill in outcomes,
                including for assets that have no new bar today).
    source_out: receives where the prices came from (yfinance / alphavantage / ...)."""
    fetcher = DataFetcher(cfg)
    raw = fetcher.fetch(ticker)
    source = fetcher.last_source          # read now: fetch_window() below re-reads the cache
    if source_out is not None:
        source_out[ticker] = source
    full = drop_partial_bar(raw, now_utc)
    if full.empty:
        raise ValueError("no price data returned")
    if series_out is not None:
        series_out[ticker] = full["Close"].astype(float)
    newest = full.index[-1]
    last_bar = newest.strftime("%Y-%m-%d")
    if (now_utc.date() - newest.date()).days > 6:
        raise ValueError(f"price data is stale (newest bar {last_bar})")
    if not force and state.last_bar.get(ticker) == last_bar:
        return None

    as_of = (newest + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    window = fetcher.fetch_window(ticker, as_of=as_of)
    if len(window) < 60:
        raise ValueError(f"only {len(window)} bars of history (need 60)")

    enriched = IndicatorEngine(cfg).compute(window)
    chart_path = os.path.join(cfg.chart_dir, f"{ticker}_{last_bar}.png")
    image_b64 = ChartRenderer(cfg).render(enriched, ticker, as_of, save_path=chart_path)
    executor = ToolExecutor(window, ticker, as_of, cfg)
    decision = agent.analyze(image_b64, executor, ticker, as_of)
    if not decision.raw:      # the parser falls back to HOLD on bad JSON — not a real decision
        raise ValueError("AI response could not be parsed (treated as a failure, not a HOLD)")
    rule_decision = None
    if rule_agent is not None:        # separate executor so tool audit trails stay separate
        rule_decision = rule_agent.analyze(
            image_b64, ToolExecutor(window, ticker, as_of, cfg), ticker, as_of)
    return Candidate(ticker, last_bar, float(window["Close"].iloc[-1]), decision,
                     chart_path, rule_decision, source)


# ── portfolio logic (pure: easy to test) ─────────────────────────────────────
def _close_position(state, c, reason, model, events, closed):
    pos = state.positions.pop(c.ticker)
    pnl = (c.close - pos["entry_price"]) / pos["entry_price"]
    events.append({"kind": "CLOSE", "ticker": c.ticker, "reason": reason,
                   "entry_price": pos["entry_price"], "entry_date": pos["entry_date"],
                   "exit_price": c.close, "pnl_pct": pnl, "last_bar": c.last_bar})
    closed.append({"ticker": c.ticker, "entry_date": pos["entry_date"],
                   "entry_price": round(pos["entry_price"], 4), "exit_date": c.last_bar,
                   "exit_price": round(c.close, 4), "pnl_pct": round(pnl * 100, 3),
                   "reason": reason, "model": model})


def process(candidates: List[Candidate], state: LiveState, risk: RiskFilter,
            run_utc: str, model: str, agent: str = "ai"):
    """Apply exits, then agent signals (most confident first). Mutates `state`.

    Returns (events, rows, closed). `rows` holds one audit row per decision. The
    primary agent's rows carry an `event`; the free rule-based baseline rows (AI mode
    only) are logged for comparison, have event "" and never touch positions."""
    events, rows, closed = [], [], []
    event_of: Dict[str, str] = {}

    for c in candidates:                                   # 1) stop-loss / target
        pos = state.positions.get(c.ticker)
        if not pos:
            continue
        if c.close <= pos["stop_loss"]:
            _close_position(state, c, "stop_loss", model, events, closed)
            event_of[c.ticker] = "CLOSE_STOP"
        elif pos.get("take_profit") and c.close >= pos["take_profit"]:
            _close_position(state, c, "take_profit", model, events, closed)
            event_of[c.ticker] = "CLOSE_TARGET"

    # 2) agent decisions — with limited open-position slots, the best ideas win
    sigs = {}
    for c in sorted(candidates, key=lambda x: -x.decision.confidence):
        d = c.decision
        sig = risk.filter(d, c.close, 0.0, len(state.positions))
        sigs[c.ticker] = sig
        pos = state.positions.get(c.ticker)
        if sig.approved and sig.action == Action.BUY:
            if pos is None:
                state.positions[c.ticker] = {
                    "entry_date": c.last_bar, "entry_price": c.close,
                    "stop_loss": sig.stop_loss, "take_profit": sig.take_profit,
                    "size_pct": sig.size_pct, "confidence": sig.confidence}
                events.append({"kind": "OPEN", "ticker": c.ticker, "close": c.close,
                               "stop": sig.stop_loss, "target": sig.take_profit,
                               "size": sig.size_pct, "conf": sig.confidence,
                               "rationale": d.rationale or d.chart_reading,
                               "last_bar": c.last_bar, "chart": c.chart_path})
                event_of[c.ticker] = "OPEN"
            else:
                event_of.setdefault(c.ticker, "HELD")      # already long
        elif sig.approved and sig.action == Action.SELL:
            if pos is not None:
                _close_position(state, c, "signal", model, events, closed)
                event_of[c.ticker] = "CLOSE_SIGNAL"
            else:
                event_of.setdefault(c.ticker, "FLAT")      # long-only: nothing to sell

    def log_row(c, d, sig, who, event, mdl):
        return {
            "run_utc": run_utc, "ticker": c.ticker, "last_bar": c.last_bar,
            "agent": who, "close": round(c.close, 4), "action": d.action,
            "confidence": round(d.confidence, 3), "approved": sig.approved,
            "reject_reason": sig.reject_reason, "event": event,
            "size_pct": round(sig.size_pct, 4), "stop_loss": round(sig.stop_loss, 4),
            "take_profit": round(sig.take_profit, 4) if sig.take_profit else "",
            "trend": d.trend, "model": mdl, "data_source": c.data_source,
            "rationale": " ".join((d.rationale or "").split())}

    for c in candidates:                                   # 3) full audit log
        rows.append(log_row(c, c.decision, sigs[c.ticker], agent,
                            event_of.get(c.ticker, "NONE"), model))
    for c in candidates:                                   # 4) baseline, for comparison only
        if c.rule_decision is not None:
            rsig = risk.filter(c.rule_decision, c.close, 0.0, 0)
            rows.append(log_row(c, c.rule_decision, rsig, "rule", "", "rule-based"))
    return events, rows, closed


def primary_rows(rows):
    """The rows behind alerts (baseline comparison rows have an empty event)."""
    return [r for r in rows if r["event"]]


# ── message formatting ───────────────────────────────────────────────────────
def _name(t: str) -> str:
    return f"{UNIVERSE.get(t, t)} ({t})" if t in UNIVERSE else t


def _day(iso: str) -> str:
    return datetime.strptime(iso, "%Y-%m-%d").strftime("%d %b")


def _short(text: str, n: int = 280) -> str:
    text = " ".join((text or "").split())
    return html.escape(text if len(text) <= n else text[:n - 1].rstrip() + "…")


def format_digest(events, rows, skipped, failures, state, closes, mode_label,
                  persist: bool, lag_note: str = "") -> str:
    esc = html.escape
    rows = primary_rows(rows)
    bars = sorted({r["last_bar"] for r in rows})
    head = (f"📊 <b>Daily signals — closes of {_day(bars[-1])}</b>" if bars
            else "📊 <b>Daily signals</b>")
    out = [("🧪 <b>TEST RUN — not saved</b>\n" if not persist else "") + head,
           f"<i>{esc(mode_label)}</i>", ""]
    if lag_note:                      # data is behind: say so before anything else
        out[2:2] = ["⚠️ <b>" + esc(lag_note) + "</b>"]

    for e in events:
        if e["kind"] == "OPEN":
            out.append(f"🟢 <b>BUY — {esc(_name(e['ticker']))}</b>\n"
                       f"confidence {e['conf']:.2f} · size {e['size']:.0%} of portfolio\n"
                       f"Entry ≈ ${e['close']:,.2f} (close {_day(e['last_bar'])}) · "
                       f"Stop-loss ${e['stop']:,.2f}"
                       + (f" · Target ${e['target']:,.2f}" if e["target"] else "")
                       + f"\n<i>{_short(e['rationale'])}</i>\n")
        else:
            why = {"stop_loss": "🛑 stop-loss hit", "take_profit": "🎯 target reached",
                   "signal": "🔴 agent says SELL"}[e["reason"]]
            out.append(f"🔔 <b>CLOSE — {esc(_name(e['ticker']))}</b> · {why}\n"
                       f"Entry ${e['entry_price']:,.2f} ({_day(e['entry_date'])}) → "
                       f"${e['exit_price']:,.2f} · <b>{e['pnl_pct']:+.1%}</b>\n")
    if not events:
        out.append("✅ <b>No action today.</b>\n")

    blocked = [r for r in rows if r["action"] != "HOLD" and not r["approved"]]
    quiet = [r["ticker"] for r in rows if r["event"] in ("NONE", "HELD", "FLAT")
             and not (r["action"] != "HOLD" and not r["approved"])]
    if quiet:
        out.append("No action: " + ", ".join(quiet))
    if blocked:
        out.append("Blocked by risk filter: " + "; ".join(
            f"{r['ticker']} {r['action']} ({esc(str(r['reject_reason']))})" for r in blocked))
    if skipped:
        out.append("No new price bar (holiday or data delay): " + ", ".join(skipped))
    if failures:
        out.append("⚠️ <b>Failed</b>: " + "; ".join(
            f"{t}: {esc(m[:120])}" for t, m in failures.items()))

    opened_now = {e["ticker"] for e in events if e["kind"] == "OPEN"}
    carried = {t: p for t, p in state.positions.items() if t not in opened_now}
    if carried:                                  # positions already open before today
        bits = []
        for t, p in carried.items():
            if t in closes:
                bits.append(f"{t} {(closes[t] - p['entry_price']) / p['entry_price']:+.1%}")
            else:
                bits.append(f"{t} (no update)")
        out.append("\n📂 Open paper positions: " + ", ".join(bits))
    out.append(f'\n<a href="{APP_URL}">Open the dashboard</a> · '
               f"research output, not financial advice")
    return "\n".join(out)


# ── persistence ──────────────────────────────────────────────────────────────
def _append_csv(path: str, fields: List[str], rows: List[dict]) -> None:
    if not rows:
        return
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerows(rows)


def forward_returns(series: pd.Series, last_bar: str) -> Dict[str, float]:
    """% change of the close h trading days after `last_bar`, relative to the close of
    `last_bar`. Only horizons that have already happened are returned."""
    days = list(series.index.strftime("%Y-%m-%d"))
    if last_bar not in days:
        return {}
    pos = days.index(last_bar)
    base = float(series.iloc[pos])
    return {f"fwd_{h}d": round((float(series.iloc[pos + h]) / base - 1) * 100, 3)
            for h in HORIZONS if pos + h < len(series)}


def _read_log(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return [{k: row.get(k, "") or "" for k in LOG_FIELDS} for row in csv.DictReader(fh)]


def update_log(path: str, new_rows: List[dict],
               series: Optional[Dict[str, pd.Series]] = None) -> None:
    """Merge today's rows into the log (a row is identified by last_bar+ticker+agent, so
    re-running a day replaces rather than duplicates) and fill in any outcome columns
    that have become known. Outcomes are what let you ASSESS a past decision."""
    merged: Dict[tuple, dict] = {tuple(r[k] for k in LOG_KEY): r for r in _read_log(path)}
    for r in new_rows:
        row = {k: r.get(k, "") for k in LOG_FIELDS}
        if isinstance(row["approved"], bool):          # TRUE/FALSE is what Sheets parses
            row["approved"] = "TRUE" if row["approved"] else "FALSE"
        merged[tuple(row[k] for k in LOG_KEY)] = row
    for row in merged.values():
        ser = (series or {}).get(row["ticker"])
        if ser is None:
            continue
        for col, val in forward_returns(ser, row["last_bar"]).items():
            if row[col] == "":
                row[col] = val
    if not merged:
        return
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=LOG_FIELDS)
        w.writeheader()
        w.writerows(merged.values())


def persist(out_dir: str, state: LiveState, rows, closed,
            series: Optional[Dict[str, pd.Series]] = None,
            run_row: Optional[dict] = None) -> None:
    os.makedirs(out_dir, exist_ok=True)
    state.save(os.path.join(out_dir, "state.json"))
    update_log(os.path.join(out_dir, "signals_log.csv"), rows, series)
    _append_csv(os.path.join(out_dir, "paper_trades.csv"), TRADE_FIELDS, closed)
    _append_csv(os.path.join(out_dir, "runs_log.csv"), RUN_FIELDS, [run_row] if run_row else [])


# ── orchestration ────────────────────────────────────────────────────────────
def run(args, notifier: TelegramNotifier, now_utc: Optional[datetime] = None) -> int:
    now_utc = now_utc or datetime.now(timezone.utc)
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()] \
        or list(UNIVERSE)
    ai = args.mode == "ai"
    if ai and not args.synthetic and not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY (your OpenRouter key) is not set for AI mode")

    size_arg = str(getattr(args, "size_pct", "0.05")).strip().lower()
    fixed_size = None if size_arg in ("agent", "none", "") else float(size_arg)

    cfg = AgentConfig(
        tickers=tickers,
        start_date=(now_utc - timedelta(days=420)).strftime("%Y-%m-%d"),
        end_date=(now_utc + timedelta(days=1)).strftime("%Y-%m-%d"),   # yfinance end is exclusive
        use_mock_llm=not ai, use_synthetic=args.synthetic, agent_model=args.model,
        fixed_size_pct=fixed_size,          # same size for AI and rule: a fair comparison
        cache_dir="./data/cache_live", chart_dir="./data/charts_live")
    state = LiveState.load(os.path.join(args.out_dir, "state.json"))
    agent, risk = ChartReadingAgent(cfg), RiskFilter(cfg)
    # In AI mode the free rule-based decision is computed on the same bars as well, so
    # both decision-makers can be compared on identical days (forward test).
    rule_agent = ChartReadingAgent(replace(cfg, use_mock_llm=True)) if ai else None
    series: Dict[str, pd.Series] = {}
    sources: Dict[str, str] = {}

    candidates, skipped, failures = [], [], {}
    for t in tickers:
        try:
            c = analyse_ticker(t, cfg, agent, state, now_utc, force=args.force,
                               rule_agent=rule_agent, series_out=series,
                               source_out=sources)
            (skipped.append(t) if c is None else candidates.append(c))
        except Exception as exc:                           # one asset must not kill the run
            failures[t] = scrub(f"{type(exc).__name__}: {exc}")[:200]
            print(f"  {t}: FAILED — {failures[t]}")

    run_utc = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    events, rows, closed = process(candidates, state, risk, run_utc,
                                   args.model if ai else "rule-based",
                                   agent="ai" if ai else "rule")
    for c in candidates:
        state.last_bar[c.ticker] = c.last_bar
    # Is the price data current? (Yahoo has returned bars a day late on some nights.)
    expected = expected_last_bar(now_utc)
    behind = {t: days_behind(ser.index[-1].date(), expected) for t, ser in series.items()}
    lagging = {t: n for t, n in behind.items() if n >= 1}
    lag_note = ""
    if lagging:
        n = max(lagging.values())
        who = (f"all {len(series)} assets" if len(lagging) == len(series)
               else ", ".join(sorted(lagging)))
        oldest = min(series[t].index[-1].date() for t in lagging)
        lag_note = (f"Price data is {n} trading day{'s' if n > 1 else ''} behind for {who} "
                    f"(data delay or a market holiday). Newest bar {oldest:%d %b}, "
                    f"expected {expected:%d %b}.")

    run_row = {
        "run_utc": run_utc, "trigger": os.environ.get("GITHUB_EVENT_NAME", "manual"),
        "mode": args.mode, "model": args.model if ai else "rule-based",
        "size_policy": "agent" if fixed_size is None else f"fixed {fixed_size:.0%}",
        "analysed": len(candidates), "skipped": len(skipped), "failed": len(failures),
        "newest_bar": max((x.index[-1].strftime("%Y-%m-%d") for x in series.values()), default=""),
        "expected_bar": expected.strftime("%Y-%m-%d"),
        "days_behind": max(behind.values(), default=0),
        "sources": ", ".join(f"{k}:{v}" for k, v in sorted(Counter(sources.values()).items()))}

    do_persist = not args.no_persist
    if do_persist:
        persist(args.out_dir, state, rows, closed, series, run_row)

    label = f"AI · {args.model}" if ai else "rule-based EMA/RSI baseline"
    text = format_digest(events, rows, skipped, failures, state,
                         {c.ticker: c.close for c in candidates}, label, do_persist,
                         lag_note=lag_note)
    if candidates or failures or lag_note:
        notifier.send(text)
    else:                      # e.g. the morning catch-up run when last night was complete
        print("nothing new to report - no message sent")
    for e in events:                                       # chart for each new BUY
        if e["kind"] == "OPEN":
            try:
                notifier.send_photo(e["chart"], f"{e['ticker']} — chart the agent analysed "
                                    f"(last candle {_day(e['last_bar'])})")
            except Exception as exc:
                print(f"  chart for {e['ticker']} not sent: {scrub(str(exc))}")

    print(f"done: {len(candidates)} analysed, {len(skipped)} skipped (no new bar), "
          f"{len(failures)} failed, {len(events)} action(s); data sources: "
          f"{run_row['sources'] or 'n/a'}; days behind: {run_row['days_behind']}")
    # Only a total failure is an error; partial failures are reported in the digest.
    return 1 if failures and not candidates and not skipped else 0


def main(argv=None) -> int:
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.WARNING)
    ap = argparse.ArgumentParser(description="Daily LLM Trader signal run")
    ap.add_argument("--mode", choices=["ai", "rule"], default=os.environ.get("LIVE_MODE", "ai"))
    ap.add_argument("--model", default=os.environ.get("LIVE_MODEL", "openai/gpt-4o-mini"))
    ap.add_argument("--tickers", default=os.environ.get("LIVE_TICKERS", ""),
                    help="comma-separated; default is the full 11-asset universe")
    ap.add_argument("--out-dir", default=os.environ.get("LIVE_DATA_DIR", "live_data"))
    ap.add_argument("--dry-run", action="store_true", help="print the message instead of sending")
    ap.add_argument("--no-persist", action="store_true", help="do not write state or logs")
    ap.add_argument("--force", action="store_true", help="ignore the 'no new bar' check")
    ap.add_argument("--synthetic", action="store_true", help="offline test data (no network)")
    ap.add_argument("--size-pct", default=os.environ.get("LIVE_SIZE_PCT", "0.05"),
                    help="position size used for EVERY trade by both agents (default 0.05); "
                         "'agent' lets each agent propose its own size")
    args = ap.parse_args(argv)

    if os.environ.get("OPENAI_API_KEY"):
        os.environ.setdefault("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    notifier = TelegramNotifier(dry_run=args.dry_run)
    try:
        return run(args, notifier)
    except Exception as exc:
        msg = scrub(f"{type(exc).__name__}: {exc}")
        try:
            notifier.send(f"❌ <b>Daily signal run FAILED</b>\n<code>{html.escape(msg[:500])}</code>")
        except Exception:
            pass
        print("FAILED: " + msg, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
