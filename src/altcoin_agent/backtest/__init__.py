"""Offline backtest + historical data subsystems (QUADRANT Phase 2-3).

Two modules:

    historical_loader  — fetch OHLCV from ccxt + cache to disk
    runner             — drive PumpPhaseFSM over cached klines, emit
                         phase-tagged outputs the trainer can consume

Phase 4's full backtest engine (matching, slippage, walk-forward) ships
later. The Phase 1-3 scope here is intentionally limited to the data
plumbing + phase tagging; the trainer in Phase 4 will plug in via
``runner.run_phase_tagging()``.
"""

from altcoin_agent.backtest.historical_loader import (
    HistoricalDataLoader,
    KlineFetcher,
)
from altcoin_agent.backtest.runner import (
    BacktestRunner,
    PhaseTaggedBar,
    compute_phase_inputs,
)

__all__ = [
    "BacktestRunner",
    "HistoricalDataLoader",
    "KlineFetcher",
    "PhaseTaggedBar",
    "compute_phase_inputs",
]
