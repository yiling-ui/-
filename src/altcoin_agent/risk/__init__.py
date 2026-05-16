"""Risk + Execution package for the altcoin agent.

Public surface:

    from altcoin_agent.risk import (
        Side,
        Position,
        PositionLeg,
        AccountState,
        ExchangeAdapter,
        PositionSizer,
        DynamicLeverageConfig,
        RiskGate,
        RiskGateConfig,
        RiskDecision,
        TrailingStopFSM,
        TrailingState,
        Reconciler,
        ReconcilerReport,
        PositionWatcher,
        CloseCallback,
        CCXTExecutor,
        ExecutionError,
        RollingController,
        RollingConfig,
        RollDecision,
    )
"""

from altcoin_agent.risk.atr import ATRCalculator
from altcoin_agent.risk.ccxt_adapter import CCXTExchangeAdapter, build_ccxt_adapter
from altcoin_agent.risk.executor import (
    CCXTExecutor,
    ExchangeAdapter,
    ExecutionError,
)
from altcoin_agent.risk.gate import RiskDecision, RiskGate, RiskGateConfig
from altcoin_agent.risk.position_watcher import CloseCallback, PositionWatcher
from altcoin_agent.risk.reconciler import Reconciler, ReconcilerReport
from altcoin_agent.risk.rolling import (
    RollDecision,
    RollingConfig,
    RollingController,
)
from altcoin_agent.risk.sizing import DynamicLeverageConfig, PositionSizer
from altcoin_agent.risk.state import AccountState, Position, PositionLeg, Side
from altcoin_agent.risk.trailing import TrailingState, TrailingStopFSM

__all__ = [
    "AccountState",
    "ATRCalculator",
    "CCXTExchangeAdapter",
    "CCXTExecutor",
    "CloseCallback",
    "DynamicLeverageConfig",
    "ExchangeAdapter",
    "ExecutionError",
    "Position",
    "PositionLeg",
    "PositionSizer",
    "PositionWatcher",
    "Reconciler",
    "ReconcilerReport",
    "RiskDecision",
    "RiskGate",
    "RiskGateConfig",
    "RollDecision",
    "RollingConfig",
    "RollingController",
    "Side",
    "TrailingState",
    "TrailingStopFSM",
    "build_ccxt_adapter",
]
