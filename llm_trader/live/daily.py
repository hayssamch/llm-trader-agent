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
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta, timezone
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

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
LOG_FIELDS = ["run_utc", "ticker", "last_bar", "close", "action", "confidence",
              "approved", "reject_reason", "event", "size_pct", "stop_loss",
              "take_profit", "trend", "model", "rationale"]
TRADE_FIELDS = ["ticker", "entry_date", "entry_price", "exit_date", "exit_price",
                "pnl_pct", "reason", "model"]


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


def analyse_ticker(ticker: str, cfg: AgentConfig, agent: ChartReadingAgent,
                   state: LiveState, now_utc: datetime,
                   force: bool = False) -> Optional[Candidate]:
    """Returns a Candidate, or None when there is no new bar since the last run."""
    fetcher = DataFetcher(cfg)
    full = drop_partial_bar(fetcher.fetch(ticker), now_utc)
    if full.empty:
        raise ValueError("no price data returned")
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
    return Candidate(ticker, last_bar, float(window["Close"].iloc[-1]), decision, chart_path)


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
            run_utc: str, model: str):
    """Apply exits, then agent signals (most confident first). Mutates `state`."""
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

    for c in candidates:                                   # 3) full audit log
        d, sig = c.decision, sigs[c.ticker]
        rows.append({
            "run_utc": run_utc, "ticker": c.ticker, "last_bar": c.last_bar,
            "close": round(c.close, 4), "action": d.action,
            "confidence": round(d.confidence, 3), "approved": sig.approved,
            "reject_reason": sig.reject_reason, "event": event_of.get(c.ticker, "NONE"),
            "size_pct": round(sig.size_pct, 4), "stop_loss": round(sig.stop_loss, 4),
            "take_profit": round(sig.take_profit, 4) if sig.take_profit else "",
            "trend": d.trend, "model": model,
            "rationale": " ".join((d.rationale or "").split())})
    return events, rows, closed


# ── message formatting ───────────────────────────────────────────────────────
def _name(t: str) -> str:
    return f"{UNIVERSE.get(t, t)} ({t})" if t in UNIVERSE else t


def _day(iso: str) -> str:
    return datetime.strptime(iso, "%Y-%m-%d").strftime("%d %b")


def _short(text: str, n: int = 280) -> str:
    text = " ".join((text or "").split())
    return html.escape(text if len(text) <= n else text[:n - 1].rstrip() + "…")


def format_digest(events, rows, skipped, failures, state, closes, mode_label,
                  persist: bool) -> str:
    esc = html.escape
    bars = sorted({r["last_bar"] for r in rows})
    head = (f"📊 <b>Daily signals — closes of {_day(bars[-1])}</b>" if bars
            else "📊 <b>Daily signals</b>")
    out = [("🧪 <b>TEST RUN — not saved</b>\n" if not persist else "") + head,
           f"<i>{esc(mode_label)}</i>", ""]

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


def persist(out_dir: str, state: LiveState, rows, closed) -> None:
    os.makedirs(out_dir, exist_ok=True)
    state.save(os.path.join(out_dir, "state.json"))
    _append_csv(os.path.join(out_dir, "signals_log.csv"), LOG_FIELDS, rows)
    _append_csv(os.path.join(out_dir, "paper_trades.csv"), TRADE_FIELDS, closed)


# ── orchestration ────────────────────────────────────────────────────────────
def run(args, notifier: TelegramNotifier, now_utc: Optional[datetime] = None) -> int:
    now_utc = now_utc or datetime.now(timezone.utc)
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()] \
        or list(UNIVERSE)
    ai = args.mode == "ai"
    if ai and not args.synthetic and not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY (your OpenRouter key) is not set for AI mode")

    cfg = AgentConfig(
        tickers=tickers,
        start_date=(now_utc - timedelta(days=420)).strftime("%Y-%m-%d"),
        end_date=(now_utc + timedelta(days=1)).strftime("%Y-%m-%d"),   # yfinance end is exclusive
        use_mock_llm=not ai, use_synthetic=args.synthetic, agent_model=args.model,
        cache_dir="./data/cache_live", chart_dir="./data/charts_live")
    state = LiveState.load(os.path.join(args.out_dir, "state.json"))
    agent, risk = ChartReadingAgent(cfg), RiskFilter(cfg)

    candidates, skipped, failures = [], [], {}
    for t in tickers:
        try:
            c = analyse_ticker(t, cfg, agent, state, now_utc, force=args.force)
            (skipped.append(t) if c is None else candidates.append(c))
        except Exception as exc:                           # one asset must not kill the run
            failures[t] = scrub(f"{type(exc).__name__}: {exc}")[:200]
            print(f"  {t}: FAILED — {failures[t]}")

    run_utc = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    events, rows, closed = process(candidates, state, risk, run_utc,
                                   args.model if ai else "rule-based")
    for c in candidates:
        state.last_bar[c.ticker] = c.last_bar
    do_persist = not args.no_persist
    if do_persist:
        persist(args.out_dir, state, rows, closed)

    label = f"AI · {args.model}" if ai else "rule-based EMA/RSI baseline"
    text = format_digest(events, rows, skipped, failures, state,
                         {c.ticker: c.close for c in candidates}, label, do_persist)
    notifier.send(text)
    for e in events:                                       # chart for each new BUY
        if e["kind"] == "OPEN":
            try:
                notifier.send_photo(e["chart"], f"{e['ticker']} — chart the agent analysed "
                                    f"(last candle {_day(e['last_bar'])})")
            except Exception as exc:
                print(f"  chart for {e['ticker']} not sent: {scrub(str(exc))}")

    print(f"done: {len(candidates)} analysed, {len(skipped)} skipped (no new bar), "
          f"{len(failures)} failed, {len(events)} action(s)")
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
