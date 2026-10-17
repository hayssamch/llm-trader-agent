"""
decision/risk_filter.py — UC: Apply risk rules
Deterministic rule layer between the agent's decision and execution.
Enforces hard constraints the agent cannot override.
"""
import logging

from llm_trader.config import AgentConfig, DEFAULT_CONFIG
from llm_trader.models import Action, AgentDecision, TradeSignal

logger = logging.getLogger(__name__)


class RiskFilter:
    def __init__(self, config: AgentConfig = DEFAULT_CONFIG):
        self.cfg = config

    def filter(
        self,
        decision:       AgentDecision,
        current_price:  float,
        portfolio_dd:   float = 0.0,
        open_positions: int   = 0,
    ) -> TradeSignal:
        action = Action(decision.action)
        conf   = decision.confidence
        proposed = (self.cfg.fixed_size_pct if self.cfg.fixed_size_pct is not None
                    else decision.size_pct)
        size   = min(proposed, self.cfg.max_position_pct)
        sl_pct = max(decision.stop_loss_pct, 0.02)
        tp_pct = decision.take_profit_pct

        # Rule 1: confidence gate
        if action != Action.HOLD and conf < self.cfg.min_confidence:
            return self._reject(decision, current_price, sl_pct,
                                f"confidence {conf:.2f} < {self.cfg.min_confidence}")

        # Rule 2: drawdown guard
        if portfolio_dd > 0.15 and action != Action.HOLD:
            return self._reject(decision, current_price, sl_pct,
                                f"portfolio drawdown {portfolio_dd:.1%} > 15%")

        # Rule 3: max open trades
        if open_positions >= self.cfg.max_open_trades and action == Action.BUY:
            return self._reject(decision, current_price, sl_pct,
                                f"max open trades ({self.cfg.max_open_trades}) reached")

        # Approved → compute price levels
        if action == Action.BUY:
            sl = current_price * (1 - sl_pct)
            tp = current_price * (1 + tp_pct)
        elif action == Action.SELL:
            sl = current_price * (1 + sl_pct)
            tp = current_price * (1 - tp_pct)
        else:
            sl, tp = current_price * (1 - sl_pct), None

        logger.info("RISK OK  %s %s @ %.2f size=%.1f%% conf=%.2f",
                    action.value, decision.ticker, current_price, size * 100, conf)
        return TradeSignal(
            ticker=decision.ticker, as_of=decision.as_of, action=action,
            confidence=conf, size_pct=size, stop_loss=sl, take_profit=tp,
            rationale=decision.rationale, chart_reading=decision.chart_reading,
            approved=True,
        )

    def _reject(self, decision, current_price, sl_pct, reason) -> TradeSignal:
        logger.info("RISK REJECT %s on %s: %s", decision.ticker, decision.as_of, reason)
        return TradeSignal(
            ticker=decision.ticker, as_of=decision.as_of, action=Action.HOLD,
            confidence=decision.confidence, size_pct=0.0,
            stop_loss=current_price * (1 - sl_pct), take_profit=None,
            rationale=decision.rationale, chart_reading=decision.chart_reading,
            approved=False, reject_reason=reason,
        )
