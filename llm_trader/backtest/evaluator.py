"""
backtest/evaluator.py — UC: Evaluate & compare
Computes the performance metrics reported in the thesis Experiments chapter
and builds a strategy-vs-baseline comparison table.
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from llm_trader.backtest.engine import BacktestResult

logger = logging.getLogger(__name__)


@dataclass
class Metrics:
    label:            str
    total_return_pct: float
    cagr_pct:         float
    sharpe:           float
    sortino:          float
    max_drawdown_pct: float
    win_rate_pct:     float
    profit_factor:    float
    avg_trade_pct:    float
    n_trades:         int
    equity_curve:     Optional[pd.Series] = field(default=None, repr=False)


class Evaluator:
    TRADING_DAYS = 252

    def evaluate(self, result: BacktestResult, label: str = "Strategy") -> Metrics:
        ec, trades = result.equity_curve, result.trades
        m = Metrics(
            label=label,
            total_return_pct=self._total_return(ec) * 100,
            cagr_pct=self._cagr(ec) * 100,
            sharpe=self._sharpe(ec),
            sortino=self._sortino(ec),
            max_drawdown_pct=self._max_dd(ec) * 100,
            win_rate_pct=self._win_rate(trades) * 100,
            profit_factor=self._profit_factor(trades),
            avg_trade_pct=self._avg_trade(trades) * 100,
            n_trades=len(trades),
            equity_curve=ec,
        )
        self._log(m)
        return m

    def buy_and_hold(self, price: pd.Series, capital: float,
                     label: str = "Buy-and-Hold", exposure: float = 1.0) -> Metrics:
        """Hold the asset with `exposure` of capital (1.0 = fully invested), the rest in
        cash at 0%. An exposure below 1 gives the benchmark a like-for-like capital
        deployment when the strategy is only invested part of the time."""
        ec = capital * (1.0 + exposure * (price / price.iloc[0] - 1.0))
        dummy = type("R", (), {"equity_curve": ec, "trades": []})()
        return self.evaluate(dummy, label=label)

    # ── size-independent comparison helpers ──────────────────────────────────
    def exposure(self, result: BacktestResult) -> float:
        """Average fraction of initial capital invested, over every bar of the equity
        curve (days spent flat count as zero). Entry notional is shares x entry price."""
        ec, trades = result.equity_curve, result.trades
        if len(ec) == 0 or not trades:
            return 0.0
        capital = result.config.initial_capital
        idx = pd.DatetimeIndex(ec.index)
        invested = pd.Series(0.0, index=idx)
        for t in trades:
            notional = t["shares"] * t["entry_price"] / capital
            held = (idx >= pd.Timestamp(t["entry_date"])) & (idx < pd.Timestamp(t["exit_date"]))
            invested[held] += notional
        return float(invested.mean())

    def avg_hold_days(self, result: BacktestResult) -> float:
        """Average holding period of a closed trade, in bars of the equity curve."""
        ec, trades = result.equity_curve, result.trades
        if not trades:
            return 0.0
        idx = pd.DatetimeIndex(ec.index)
        days = [int(((idx >= pd.Timestamp(t["entry_date"])) &
                     (idx < pd.Timestamp(t["exit_date"]))).sum()) for t in trades]
        return float(np.mean(days))

    def entry_benchmark(self, price: pd.Series, hold_days: float) -> float:
        """Average return of entering on ANY bar and holding `hold_days` bars (percent).
        The like-for-like yardstick for the average trade: it removes market drift and
        trade length from the comparison."""
        h = int(round(hold_days))
        if h < 1 or len(price) <= h:
            return float("nan")
        fwd = price.shift(-h) / price - 1.0
        return float(fwd.dropna().mean() * 100)

    def summarise(self, result: BacktestResult, price: pd.Series, label: str) -> Dict:
        """One row of the thesis results table: strategy, full buy-and-hold, exposure-matched
        buy-and-hold, and the per-trade metrics that do not depend on position size."""
        m = self.evaluate(result, label=label)
        capital = result.config.initial_capital
        p = price.loc[result.equity_curve.index[0]:result.equity_curve.index[-1]]
        expo = self.exposure(result)
        full = self.buy_and_hold(p, capital)
        matched = self.buy_and_hold(p, capital, exposure=expo)
        hold = self.avg_hold_days(result)
        bench = self.entry_benchmark(p, hold) if m.n_trades else float("nan")
        cost = 2 * result.config.commission_pct * 100          # round-trip commission, in %
        net_avg = m.avg_trade_pct - cost if m.n_trades else 0.0
        return {
            "Return %": m.total_return_pct,
            "B&H full %": full.total_return_pct,
            "Exposure %": expo * 100,
            "B&H matched %": matched.total_return_pct,
            "Excess vs matched": m.total_return_pct - matched.total_return_pct,
            "MaxDD %": m.max_drawdown_pct,
            "B&H matched DD %": matched.max_drawdown_pct,
            "Trades": m.n_trades,
            "Win %": m.win_rate_pct,
            "Avg trade % (net)": net_avg,
            "Median trade %": float(np.median([t["pnl_pct"] for t in result.trades]) * 100)
                              if result.trades else 0.0,
            "Avg hold (days)": hold,
            "Any-entry same-hold %": bench,
            "Trade edge (pts)": (net_avg - bench) if m.n_trades and not np.isnan(bench)
                                else float("nan"),
        }

    def comparison_table(self, metrics: List[Metrics]) -> pd.DataFrame:
        rows = {
            "Total return (%)": [m.total_return_pct for m in metrics],
            "CAGR (%)":         [m.cagr_pct for m in metrics],
            "Sharpe ratio":     [m.sharpe for m in metrics],
            "Sortino ratio":    [m.sortino for m in metrics],
            "Max drawdown (%)": [m.max_drawdown_pct for m in metrics],
            "Win rate (%)":     [m.win_rate_pct for m in metrics],
            "Profit factor":    [m.profit_factor for m in metrics],
            "Avg trade (%)":    [m.avg_trade_pct for m in metrics],
            "# Trades":         [m.n_trades for m in metrics],
        }
        return pd.DataFrame(rows, index=[m.label for m in metrics]).T.round(3)

    # ── metrics ──────────────────────────────────────────────────────────────
    def _total_return(self, ec):
        return (ec.iloc[-1] / ec.iloc[0]) - 1.0 if len(ec) > 1 else 0.0

    def _cagr(self, ec):
        if len(ec) < 2:
            return 0.0
        years = len(ec) / self.TRADING_DAYS
        return (ec.iloc[-1] / ec.iloc[0]) ** (1 / years) - 1.0 if years > 0 else 0.0

    def _rets(self, ec):
        return ec.pct_change().dropna()

    def _sharpe(self, ec):
        r = self._rets(ec)
        return float(r.mean() / r.std() * np.sqrt(self.TRADING_DAYS)) if r.std() else 0.0

    def _sortino(self, ec):
        r = self._rets(ec)
        d = r[r < 0]
        return float(r.mean() / d.std() * np.sqrt(self.TRADING_DAYS)) if len(d) and d.std() else 0.0

    def _max_dd(self, ec):
        if len(ec) < 2:
            return 0.0
        dd = (ec - ec.cummax()) / ec.cummax()
        return float(abs(dd.min()))

    def _win_rate(self, trades):
        return sum(1 for t in trades if t["pnl_pct"] > 0) / len(trades) if trades else 0.0

    def _profit_factor(self, trades):
        gp = sum(t["pnl_pct"] for t in trades if t["pnl_pct"] > 0)
        gl = abs(sum(t["pnl_pct"] for t in trades if t["pnl_pct"] < 0))
        return gp / gl if gl else (float("inf") if gp > 0 else 1.0)

    def _avg_trade(self, trades):
        return float(np.mean([t["pnl_pct"] for t in trades])) if trades else 0.0

    def _log(self, m):
        logger.info("%-16s ret=%+.1f%% CAGR=%+.1f%% Sharpe=%.2f MaxDD=%.1f%% "
                    "Win=%.1f%% Trades=%d", m.label, m.total_return_pct, m.cagr_pct,
                    m.sharpe, m.max_drawdown_pct, m.win_rate_pct, m.n_trades)
