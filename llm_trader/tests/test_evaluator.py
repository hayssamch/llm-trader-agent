"""
tests/test_evaluator.py — exposure-matched benchmark and size-independent trade metrics.
No network, no API.
Run:  PYTHONPATH=. python -m pytest llm_trader/tests/test_evaluator.py -v
"""
import pandas as pd
import pytest

from llm_trader.backtest.engine import BacktestResult
from llm_trader.backtest.evaluator import Evaluator
from llm_trader.config import AgentConfig

DAYS = pd.bdate_range("2026-01-01", periods=20)


def result(trades, equity=None):
    ec = pd.Series(100_000.0 if equity is None else equity, index=DAYS) \
        if not isinstance(equity, pd.Series) else equity
    return BacktestResult(AgentConfig(), "TEST", ec, trades=trades)


def trade(entry, exit_, pnl, notional=10_000.0, price=100.0):
    return {"ticker": "TEST", "entry_date": DAYS[entry], "exit_date": DAYS[exit_],
            "entry_price": price, "exit_price": price * (1 + pnl),
            "shares": notional / price, "pnl_pct": pnl, "reason": "signal"}


def test_exposure_is_zero_without_trades():
    assert Evaluator().exposure(result([])) == 0.0


def test_exposure_counts_days_in_market_times_size():
    # 10% of capital held on 5 of 20 bars (exit bar is not held)
    r = result([trade(0, 5, 0.02, notional=10_000)])
    assert Evaluator().exposure(r) == pytest.approx(0.10 * 5 / 20)


def test_exposure_adds_concurrent_trades():
    r = result([trade(0, 10, 0.0, notional=5_000), trade(5, 15, 0.0, notional=5_000)])
    # 0.05 for bars 0-4 and 10-14, 0.10 for bars 5-9, 0 for bars 15-19
    assert Evaluator().exposure(r) == pytest.approx((0.05 * 5 + 0.10 * 5 + 0.05 * 5) / 20)


def test_avg_hold_days():
    r = result([trade(0, 4, 0.0), trade(5, 15, 0.0)])
    assert Evaluator().avg_hold_days(r) == pytest.approx(7.0)


def test_matched_buy_and_hold_scales_the_asset_move():
    price = pd.Series([100.0, 110.0, 90.0, 120.0], index=DAYS[:4])
    full = Evaluator().buy_and_hold(price, 100_000)
    half = Evaluator().buy_and_hold(price, 100_000, exposure=0.5)
    zero = Evaluator().buy_and_hold(price, 100_000, exposure=0.0)
    assert full.total_return_pct == pytest.approx(20.0)
    assert half.total_return_pct == pytest.approx(10.0)
    assert zero.total_return_pct == pytest.approx(0.0)
    assert half.max_drawdown_pct == pytest.approx(full.max_drawdown_pct / 2, rel=0.1)


def test_default_buy_and_hold_is_unchanged():
    price = pd.Series([100.0, 200.0], index=DAYS[:2])
    assert Evaluator().buy_and_hold(price, 1_000).total_return_pct == pytest.approx(100.0)


def test_entry_benchmark_is_the_average_forward_return():
    price = pd.Series([100.0, 110.0, 121.0, 133.1], index=DAYS[:4])   # +10% every bar
    assert Evaluator().entry_benchmark(price, 1) == pytest.approx(10.0)
    assert Evaluator().entry_benchmark(price, 2) == pytest.approx(21.0)
    assert pd.isna(Evaluator().entry_benchmark(price, 0))
    assert pd.isna(Evaluator().entry_benchmark(price, 10))


def test_summarise_nets_commission_and_reports_the_edge():
    price = pd.Series(100.0, index=DAYS)               # flat market: any-entry average is 0
    r = result([trade(0, 4, 0.03), trade(5, 9, -0.01)])
    s = Evaluator().summarise(r, price, "T")
    assert s["Trades"] == 2
    assert s["Win %"] == pytest.approx(50.0)
    assert s["Avg trade % (net)"] == pytest.approx(1.0 - 0.2)    # (3 - 1)/2 minus 2 x 0.1%
    assert s["Median trade %"] == pytest.approx(1.0)
    assert s["Any-entry same-hold %"] == pytest.approx(0.0)
    assert s["Trade edge (pts)"] == pytest.approx(0.8)
    assert s["Exposure %"] == pytest.approx(10.0 * 8 / 20)


def test_summarise_with_no_trades_is_safe():
    price = pd.Series(100.0, index=DAYS)
    s = Evaluator().summarise(result([]), price, "T")
    assert s["Trades"] == 0 and s["Exposure %"] == 0.0
    assert s["Avg trade % (net)"] == 0.0
    assert pd.isna(s["Trade edge (pts)"])
    assert s["B&H matched %"] == pytest.approx(0.0)


def test_real_data_shape_tz_aware_bars_and_string_trade_dates():
    """Yahoo bars are tz-aware (UTC); the engine records trade dates as 'YYYY-MM-DD' strings."""
    days = pd.bdate_range("2026-01-01", periods=20, tz="UTC")
    ec = pd.Series(100_000.0, index=days)
    price = pd.Series(100.0, index=days)
    tr = {"ticker": "T", "entry_date": days[0].strftime("%Y-%m-%d"),
          "exit_date": days[5].strftime("%Y-%m-%d"), "entry_price": 100.0,
          "exit_price": 103.0, "shares": 100.0, "pnl_pct": 0.03, "reason": "signal"}
    r = BacktestResult(AgentConfig(), "T", ec, trades=[tr])
    ev = Evaluator()
    assert ev.exposure(r) == pytest.approx(0.10 * 5 / 20)
    assert ev.avg_hold_days(r) == pytest.approx(5.0)
    s = ev.summarise(r, price, "T")
    assert s["Trades"] == 1 and s["Exposure %"] == pytest.approx(2.5)
