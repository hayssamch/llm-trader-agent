"""
tests/test_live.py — daily signal runner: paper-position logic, de-duplication,
message safety and secret handling. No network, no API (synthetic data).
Run:  PYTHONPATH=. python -m pytest llm_trader/tests/test_live.py -v
"""
import argparse
import csv
from dataclasses import replace
from datetime import date, datetime, timezone

import pandas as pd
import pytest

from llm_trader.config import AgentConfig
from llm_trader.decision.risk_filter import RiskFilter
from llm_trader.models import AgentDecision
from llm_trader.live import daily
from llm_trader.data.fetcher import DataFetcher
from llm_trader.live.daily import (Candidate, LiveState, LOG_FIELDS, _read_log,
                                   analyse_ticker, days_behind, drop_partial_bar,
                                   expected_last_bar, format_digest, forward_returns,
                                   process, scrub, update_log)
from llm_trader.live.notifier import NotifyError, TelegramNotifier


def decision(action="BUY", conf=0.8, ticker="AAPL", raw=None, rationale="trend up"):
    return AgentDecision(ticker=ticker, as_of="2026-10-14", action=action,
                         confidence=conf, size_pct=0.05, stop_loss_pct=0.05,
                         take_profit_pct=0.10, trend="uptrend", chart_reading="x",
                         rationale=rationale, raw={"ok": 1} if raw is None else raw)


def cand(ticker="AAPL", close=100.0, day="2026-10-13", **kw):
    return Candidate(ticker, day, close, decision(ticker=ticker, **kw), f"{ticker}.png")


def run_day(cands, state, cfg=None):
    return process(cands, state, RiskFilter(cfg or AgentConfig()), "2026-10-13T22:00:00Z", "m")


# ── paper positions ──────────────────────────────────────────────────────────
def test_buy_opens_once_then_is_not_repeated():
    st = LiveState()
    ev, rows, _ = run_day([cand(close=100)], st)
    assert [e["kind"] for e in ev] == ["OPEN"] and "AAPL" in st.positions
    assert rows[0]["event"] == "OPEN"
    ev2, rows2, _ = run_day([cand(close=101, day="2026-10-14")], st)
    assert ev2 == [] and rows2[0]["event"] == "HELD"        # no repeat BUY alert


def test_stop_loss_closes_position():
    st = LiveState()
    run_day([cand(close=100)], st)                            # stop = 95
    ev, rows, closed = run_day([cand(action="HOLD", close=94, day="2026-10-14")], st)
    assert ev[0]["kind"] == "CLOSE" and ev[0]["reason"] == "stop_loss"
    assert ev[0]["pnl_pct"] == pytest.approx(-0.06) and "AAPL" not in st.positions
    assert closed[0]["pnl_pct"] == -6.0 and rows[0]["event"] == "CLOSE_STOP"


def test_take_profit_and_sell_signal_close():
    st = LiveState()
    run_day([cand(close=100)], st)                            # target = 110
    ev, _, _ = run_day([cand(action="HOLD", close=111, day="2026-10-14")], st)
    assert ev[0]["reason"] == "take_profit"
    run_day([cand(close=100)], st)
    ev, _, _ = run_day([cand(action="SELL", close=102, day="2026-10-15")], st)
    assert ev[0]["reason"] == "signal" and "AAPL" not in st.positions


def test_sell_while_flat_is_ignored_long_only():
    ev, rows, _ = run_day([cand(action="SELL")], LiveState())
    assert ev == [] and rows[0]["event"] == "FLAT"


def test_low_confidence_buy_is_blocked_and_logged():
    st = LiveState()
    ev, rows, _ = run_day([cand(conf=0.55)], st)
    assert ev == [] and not st.positions
    assert rows[0]["approved"] is False and "confidence" in rows[0]["reject_reason"]


def test_limited_slots_go_to_most_confident_ideas():
    st = LiveState()
    cands = [cand(t, conf=c) for t, c in
             [("AAA", 0.62), ("BBB", 0.90), ("CCC", 0.70), ("DDD", 0.85), ("EEE", 0.65)]]
    run_day(cands, st)                                        # max_open_trades = 3
    assert set(st.positions) == {"BBB", "DDD", "CCC"}


# ── data handling ────────────────────────────────────────────────────────────
def _bars(last_day):
    idx = pd.date_range(end=last_day, periods=5, freq="B", tz="UTC")
    return pd.DataFrame({"Close": range(5)}, index=idx)


def test_partial_bar_dropped_only_before_close():
    df = _bars("2026-10-13")                                      # Tuesday
    during = datetime(2026, 10, 13, 18, 0, tzinfo=timezone.utc)   # 14:00 ET
    after = datetime(2026, 10, 13, 22, 0, tzinfo=timezone.utc)    # 18:00 ET
    assert len(drop_partial_bar(df, during)) == 4
    assert len(drop_partial_bar(df, after)) == 5
    sat = datetime(2026, 10, 17, 12, 0, tzinfo=timezone.utc)
    assert len(drop_partial_bar(_bars("2026-10-16"), sat)) == 5   # Friday bar kept on weekends


class FakeAgent:
    def __init__(self, raw=None):
        self.raw = raw

    def analyze(self, image_b64, executor, ticker, as_of):
        return decision(ticker=ticker, raw=self.raw)


def synth_cfg(tmp_path):
    return AgentConfig(use_synthetic=True, use_mock_llm=True,
                       start_date="2025-08-01", end_date="2026-10-14",
                       cache_dir=str(tmp_path / "c"), chart_dir=str(tmp_path / "ch"))


NOW = datetime(2026, 10, 13, 22, 0, tzinfo=timezone.utc)


def test_no_new_bar_is_skipped_unless_forced(tmp_path):
    cfg, st = synth_cfg(tmp_path), LiveState()
    first = analyse_ticker("AAPL", cfg, FakeAgent({"ok": 1}), st, NOW)
    assert first is not None
    st.last_bar["AAPL"] = first.last_bar
    assert analyse_ticker("AAPL", cfg, FakeAgent({"ok": 1}), st, NOW) is None
    assert analyse_ticker("AAPL", cfg, FakeAgent({"ok": 1}), st, NOW, force=True) is not None


def test_unparseable_ai_answer_is_a_failure_not_a_hold(tmp_path):
    with pytest.raises(ValueError, match="could not be parsed"):
        analyse_ticker("AAPL", synth_cfg(tmp_path), FakeAgent(raw={}), LiveState(), NOW)


def test_state_roundtrip(tmp_path):
    st = LiveState({"AAPL": "2026-10-13"}, {"AAPL": {"entry_price": 1.0}})
    p = str(tmp_path / "s" / "state.json")
    st.save(p)
    assert LiveState.load(p).positions == st.positions
    assert LiveState.load(str(tmp_path / "missing.json")).positions == {}


# ── messages and secrets ─────────────────────────────────────────────────────
def test_digest_escapes_html_and_marks_test_runs():
    st = LiveState()
    ev, rows, _ = run_day([cand(rationale="<script>alert(1)</script> & more")], st)
    text = format_digest(ev, rows, [], {}, st, {"AAPL": 100.0}, "rule-based", persist=False)
    assert "<script>" not in text and "&lt;script&gt;" in text
    assert "TEST RUN" in text and "BUY — Apple (AAPL)" in text


def test_digest_reports_no_action_and_failures():
    ev, rows, _ = run_day([cand(action="HOLD")], LiveState())
    text = format_digest(ev, rows, ["MSFT"], {"NVDA": "ValueError: no data"},
                         LiveState(), {}, "AI", persist=True)
    assert "No action today" in text and "MSFT" in text and "Failed" in text
    assert "TEST RUN" not in text


def test_scrub_removes_secret_values(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-or-v1-supersecretvalue")
    assert "supersecret" not in scrub("401 for key sk-or-v1-supersecretvalue")


def test_notifier_never_leaks_token(monkeypatch):
    token = "123456:SECRET-TOKEN-VALUE"

    def boom(url, **kw):
        raise ConnectionError(f"failed to reach {url}")        # url contains the token

    monkeypatch.setattr("llm_trader.live.notifier.requests.post", boom)
    monkeypatch.setattr("llm_trader.live.notifier.time.sleep", lambda s: None)
    with pytest.raises(NotifyError) as err:
        TelegramNotifier(token, "42").send("hi")
    assert token not in str(err.value) and "SECRET-TOKEN" not in str(err.value)


def test_notifier_requires_credentials_unless_dry_run(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    with pytest.raises(NotifyError):
        TelegramNotifier()
    TelegramNotifier(dry_run=True).send("ok")                  # no credentials needed


def test_long_messages_are_split():
    parts = TelegramNotifier.split("\n".join(["x" * 100] * 120))
    assert len(parts) > 1 and all(len(p) <= 4000 for p in parts)


# ── outcomes, baseline rows and the Google-Sheets-friendly log ───────────────
def _series(closes, start="2026-10-01"):
    idx = pd.bdate_range(start, periods=len(closes), tz="UTC")
    return pd.Series(closes, index=idx, dtype=float)


def _row(ticker="AAPL", day="2026-10-01", agent="ai", action="BUY", approved=True):
    return {"last_bar": day, "ticker": ticker, "agent": agent, "action": action,
            "confidence": 0.8, "approved": approved, "event": "OPEN", "close": 100.0,
            "size_pct": 0.05, "stop_loss": 95.0, "take_profit": 110.0,
            "reject_reason": "", "trend": "uptrend", "model": "m",
            "run_utc": "2026-10-01T22:00:00Z", "rationale": "because"}


def test_forward_returns_only_for_horizons_that_have_happened():
    ser = _series([100, 101, 102, 103, 104, 105, 110])         # 7 bars, decision on bar 0
    out = forward_returns(ser, "2026-10-01")
    assert out == {"fwd_1d": 1.0, "fwd_5d": 5.0}                # 10d / 20d not yet known
    assert forward_returns(ser, "2026-10-06") == {"fwd_1d": 0.971}   # bar 3: 103 -> 104
    assert forward_returns(ser, "1999-01-01") == {}


def test_outcomes_fill_in_later_and_never_change_once_known(tmp_path):
    path = str(tmp_path / "signals_log.csv")
    update_log(path, [_row()], {"AAPL": _series([100, 102])})
    first = _read_log(path)[0]                                   # CSV values read back as text
    assert float(first["fwd_1d"]) == 2.0 and first["fwd_5d"] == ""
    # a later run, with more prices (and an adjusted history) fills new horizons only
    update_log(path, [], {"AAPL": _series([100, 150, 103, 104, 105, 106, 107])})
    row = _read_log(path)[0]
    assert float(row["fwd_1d"]) == 2.0                          # already known: untouched
    assert float(row["fwd_5d"]) == 6.0


def test_rerunning_a_day_replaces_the_row_instead_of_duplicating(tmp_path):
    path = str(tmp_path / "signals_log.csv")
    update_log(path, [_row()], None)
    update_log(path, [_row(action="HOLD")], None)                # same day/ticker/agent
    update_log(path, [_row(agent="rule")], None)                 # different agent = new row
    rows = _read_log(path)
    assert len(rows) == 2 and {r["action"] for r in rows if r["agent"] == "ai"} == {"HOLD"}


def test_log_is_clean_for_google_sheets(tmp_path):
    path = str(tmp_path / "signals_log.csv")
    r = _row()
    r["rationale"] = 'has "quotes", commas,\nand a newline'
    update_log(path, [r], None)
    with open(path, newline="", encoding="utf-8") as fh:
        got = list(csv.DictReader(fh))
    assert list(got[0].keys()) == LOG_FIELDS and len(got) == 1
    assert got[0]["approved"] == "TRUE" and got[0]["last_bar"] == "2026-10-01"
    assert LOG_FIELDS.index("action") < LOG_FIELDS.index("rationale")   # key columns first


def test_old_log_without_outcome_columns_is_still_readable(tmp_path):
    path = tmp_path / "signals_log.csv"
    path.write_text("last_bar,ticker,agent,action\n2026-10-01,AAPL,ai,BUY\n", encoding="utf-8")
    row = _read_log(str(path))[0]
    assert row["action"] == "BUY" and row["fwd_5d"] == ""


def test_baseline_rows_are_logged_but_never_trigger_alerts_or_positions():
    st = LiveState()
    c = cand(action="HOLD")                                      # the AI says HOLD...
    c.rule_decision = decision(action="BUY", conf=0.9)           # ...the free rule says BUY
    ev, rows, _ = run_day([c], st)
    assert ev == [] and not st.positions                         # only the AI drives alerts
    assert [(r["agent"], r["action"]) for r in rows] == [("ai", "HOLD"), ("rule", "BUY")]
    text = format_digest(ev, rows, [], {}, st, {"AAPL": 100.0}, "AI", persist=True)
    assert "BUY" not in text and "No action today" in text


def test_end_to_end_run_saves_ai_and_baseline_rows_and_dedupes(tmp_path, monkeypatch):
    real = daily.ChartReadingAgent
    monkeypatch.setattr(daily, "ChartReadingAgent",
                        lambda cfg: FakeAgent({"ok": 1}) if not cfg.use_mock_llm else real(cfg))
    monkeypatch.chdir(tmp_path)
    args = argparse.Namespace(mode="ai", model="m", tickers="AAPL,MSFT",
                              out_dir=str(tmp_path / "live"), dry_run=True,
                              no_persist=False, force=False, synthetic=True)
    assert daily.run(args, TelegramNotifier(dry_run=True)) == 0
    rows = _read_log(str(tmp_path / "live" / "signals_log.csv"))
    assert sorted((r["ticker"], r["agent"]) for r in rows) ==         [("AAPL", "ai"), ("AAPL", "rule"), ("MSFT", "ai"), ("MSFT", "rule")]
    assert set(LiveState.load(str(tmp_path / "live" / "state.json")).positions) == {"AAPL", "MSFT"}
    # same day again: nothing new to analyse, nothing duplicated
    assert daily.run(args, TelegramNotifier(dry_run=True)) == 0
    assert len(_read_log(str(tmp_path / "live" / "signals_log.csv"))) == 4


# -- one size policy for both agents (fair comparison) -----------------------
def _size_of(cfg, proposed):
    d = replace(decision(), size_pct=proposed)
    return RiskFilter(cfg).filter(d, 100.0).size_pct


def test_fixed_size_overrides_whatever_the_agent_proposes():
    cfg = AgentConfig(fixed_size_pct=0.05, max_position_pct=0.10)
    assert _size_of(cfg, 0.10) == 0.05          # an ambitious agent is brought down
    assert _size_of(cfg, 0.02) == 0.05          # a timid one is brought up: same size for all


def test_fixed_size_is_still_capped_and_default_is_unchanged():
    assert _size_of(AgentConfig(fixed_size_pct=0.20, max_position_pct=0.10), 0.05) == 0.10
    plain = AgentConfig(max_position_pct=0.10)   # fixed_size_pct=None: agent decides, as before
    assert _size_of(plain, 0.07) == 0.07 and _size_of(plain, 0.15) == 0.10


# -- is the price data current? -----------------------------------------------
UTC = timezone.utc


def test_expected_last_bar_handles_evenings_mornings_and_weekends():
    assert expected_last_bar(datetime(2026, 10, 14, 0, 42, tzinfo=UTC)) == date(2026, 10, 13)  # Tue 20:42 ET
    assert expected_last_bar(datetime(2026, 10, 13, 15, 0, tzinfo=UTC)) == date(2026, 10, 12)  # Tue 11:00 ET, market open
    assert expected_last_bar(datetime(2026, 10, 19, 9, 0, tzinfo=UTC)) == date(2026, 10, 16)   # Mon 05:00 ET
    assert expected_last_bar(datetime(2026, 10, 17, 12, 0, tzinfo=UTC)) == date(2026, 10, 16)  # Saturday


def test_days_behind_counts_trading_days_only():
    assert days_behind(date(2026, 10, 12), date(2026, 10, 13)) == 1
    assert days_behind(date(2026, 10, 16), date(2026, 10, 19)) == 1      # Fri -> Mon is one trading day
    assert days_behind(date(2026, 10, 15), date(2026, 10, 19)) == 2
    assert days_behind(date(2026, 10, 19), date(2026, 10, 19)) == 0
    assert days_behind(date(2026, 10, 20), date(2026, 10, 19)) == 0


def test_digest_puts_the_data_lag_warning_first():
    text = format_digest([], [], [], {}, LiveState(), {}, "AI", persist=True,
                         lag_note="Price data is 1 trading day behind for all 11 assets.")
    assert "Price data is 1 trading day behind" in text
    assert text.index("behind") < text.index("No action today")


def test_fetcher_records_where_prices_came_from(tmp_path):
    f = DataFetcher(synth_cfg(tmp_path))
    f.fetch("AAPL")
    assert f.last_source == "synthetic"


class Recorder:
    """Stands in for the Telegram notifier and keeps what would have been sent."""
    def __init__(self):
        self.messages = []

    def send(self, text, silent=False):
        self.messages.append(text)

    def send_photo(self, *a, **k):
        pass


def _args(tmp_path, **over):
    base = dict(mode="ai", model="m", tickers="AAPL,MSFT", out_dir=str(tmp_path / "live"),
                dry_run=True, no_persist=False, force=False, synthetic=True, size_pct="0.05")
    base.update(over)
    return argparse.Namespace(**base)


def _fake_ai(monkeypatch):
    real = daily.ChartReadingAgent
    monkeypatch.setattr(daily, "ChartReadingAgent",
                        lambda cfg: FakeAgent({"ok": 1}) if not cfg.use_mock_llm else real(cfg))


def test_run_logs_source_and_size_and_stays_quiet_when_nothing_is_new(tmp_path, monkeypatch):
    _fake_ai(monkeypatch)
    monkeypatch.chdir(tmp_path)
    rec, args = Recorder(), _args(tmp_path)
    assert daily.run(args, rec) == 0
    rows = _read_log(str(tmp_path / "live" / "signals_log.csv"))
    assert {r["data_source"] for r in rows} == {"synthetic"}
    assert {float(r["size_pct"]) for r in rows} == {0.05}        # one size for both agents
    assert len(rec.messages) == 1
    assert daily.run(args, rec) == 0                              # same bar again
    assert len(rec.messages) == 1                                 # nothing new: no second message
    with open(tmp_path / "live" / "runs_log.csv", newline="", encoding="utf-8") as fh:
        runs = list(csv.DictReader(fh))
    assert [r["analysed"] for r in runs] == ["2", "0"] and runs[0]["sources"] == "synthetic:2"
    assert runs[1]["skipped"] == "2" and runs[0]["size_policy"] == "fixed 5%"


def test_data_lag_is_reported_even_when_nothing_is_new(tmp_path, monkeypatch):
    _fake_ai(monkeypatch)
    monkeypatch.chdir(tmp_path)
    rec, args = Recorder(), _args(tmp_path)
    daily.run(args, rec)
    monkeypatch.setattr(daily, "expected_last_bar", lambda now: date(2099, 1, 5))
    n = len(rec.messages)
    daily.run(args, rec)                                          # nothing new, but data is behind
    assert len(rec.messages) == n + 1 and "behind" in rec.messages[-1]
    with open(tmp_path / "live" / "runs_log.csv", newline="", encoding="utf-8") as fh:
        assert int(list(csv.DictReader(fh))[-1]["days_behind"]) > 0
