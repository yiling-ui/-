"""Offline backtest + historical data subsystems.

Phase 1-3 modules:
    historical_loader  — fetch OHLCV from ccxt + cache to disk
    runner             — drive PumpPhaseFSM over cached klines

Phase B.4 modules (this PR):
    data_adapter       — read-side replacement for ccxt during backtest
    matching_engine    — ExchangeAdapter implementation: fills market
                         orders against cached klines, triggers stops
                         on bar high/low breaches, accrues PnL + fees
    slippage_model     — closed-form fill-price formula + OLS calibration
    walk_forward       — rolling train/validate windowing + per-window
                         stats (used by the Phase 4 trainer)

Together these let the live ``CCXTExecutor`` run unchanged against an
in-memory simulator: the only seam between live and backtest is the
``ExchangeAdapter`` IO Protocol (``executor.py``) — exactly the
production-parity rule the plan demands.
"""

from altcoin_agent.backtest.data_adapter import BacktestDataAdapter
from altcoin_agent.backtest.historical_loader import (
    HistoricalDataLoader,
    KlineFetcher,
)
from altcoin_agent.backtest.matching_engine import (
    MatchingEngine,
    MatchingEngineConfig,
)
from altcoin_agent.backtest.runner import (
    BacktestRunner,
    PhaseTaggedBar,
    compute_phase_inputs,
)
from altcoin_agent.backtest.slippage_model import (
    SlippageModel,
    SlippageObservation,
    SlippageParams,
    fit_params,
    load_observations,
)
from altcoin_agent.backtest.walk_forward import (
    WalkForwardConfig,
    WalkForwardReport,
    WalkForwardSplit,
    Window,
    WindowStats,
    iter_splits,
    run_walk_forward,
    stats_from_pnls,
)

__all__ = [
    "BacktestDataAdapter",
    "BacktestRunner",
    "HistoricalDataLoader",
    "KlineFetcher",
    "MatchingEngine",
    "MatchingEngineConfig",
    "PhaseTaggedBar",
    "SlippageModel",
    "SlippageObservation",
    "SlippageParams",
    "WalkForwardConfig",
    "WalkForwardReport",
    "WalkForwardSplit",
    "Window",
    "WindowStats",
    "compute_phase_inputs",
    "fit_params",
    "iter_splits",
    "load_observations",
    "run_walk_forward",
    "stats_from_pnls",
]
