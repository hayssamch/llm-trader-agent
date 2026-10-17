"""
config.py — UC: Configure analysis
Central configuration for the single chart-reading agent.

Architecture note (v2)
----------------------
The multi-agent fusion design (separate quantitative / vision / sentiment
agents merged by an orchestrator) has been replaced by ONE tool-calling
agent that reads a rendered chart and calls analytical tools on demand.
Sentiment / fundamentals parameters have been removed from the active
pipeline and are deferred to future work (see thesis Chapter 9).
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class AgentConfig:
    # ── Universe ──────────────────────────────────────────────────────────────
    tickers: List[str] = field(
        default_factory=lambda: ["AAPL", "MSFT", "NVDA", "JPM", "XOM"]
    )
    start_date: str = "2024-01-01"   # after most LLM training cutoffs
    end_date:   str = "2025-06-30"
    interval:   str = "1d"

    # ── Indicators ────────────────────────────────────────────────────────────
    ema_periods:  List[int] = field(default_factory=lambda: [20, 50, 200])
    rsi_period:   int = 14
    macd_fast:    int = 12
    macd_slow:    int = 26
    macd_signal:  int = 9
    bb_period:    int = 20
    bb_std:       float = 2.0
    atr_period:   int = 14
    adx_period:   int = 14

    # ── S/R detection ────────────────────────────────────────────────────────
    sr_swing_order: int = 10
    sr_n_clusters:  int = 6
    sr_lookback:    int = 120

    # ── Order block detection ────────────────────────────────────────────────
    ob_impulse_pct: float = 0.015
    ob_max_age:     int = 40

    # ── Chart rendering ───────────────────────────────────────────────────────
    chart_lookback: int = 60
    chart_dpi:      int = 150

    # ── Agent (merged VLM reasoning + tool calling) ──────────────────────────
    agent_model:       str = "gpt-4o"   # or "claude-3-5-sonnet-20241022"
    agent_max_tokens:  int = 1500
    agent_temperature: float = 0.1      # low for reproducibility
    agent_max_iters:   int = 5          # max tool-call rounds before forcing a decision

    # ── Risk filter ───────────────────────────────────────────────────────────
    max_position_pct: float = 0.10
    # When set, EVERY trade uses this size, for the AI and the rule alike (still capped by
    # max_position_pct). Position size then becomes a portfolio policy rather than part of
    # the agent's judgment, so an AI-vs-rule comparison isolates the BUY/SELL/HOLD decisions.
    # None (default) keeps the old behaviour: the agent proposes its own size.
    fixed_size_pct:   Optional[float] = None
    stop_loss_pct:    float = 0.05
    max_open_trades:  int = 3
    min_confidence:   float = 0.6

    # ── Backtest ──────────────────────────────────────────────────────────────
    initial_capital: float = 100_000.0
    commission_pct:  float = 0.001
    slippage_pct:    float = 0.0005

    # ── Run modes (for first-run smoke testing) ──────────────────────────────
    use_mock_llm:  bool = False   # True → deterministic rule-based agent, no API calls
    use_synthetic: bool = False   # True → generate synthetic OHLCV instead of yfinance

    # ── Paths ─────────────────────────────────────────────────────────────────
    cache_dir:   str = "./data/cache"
    chart_dir:   str = "./data/charts"
    results_dir: str = "./results"


DEFAULT_CONFIG = AgentConfig()
