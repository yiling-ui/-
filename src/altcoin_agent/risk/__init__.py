"""Risk + Execution package for the altcoin agent.

Public surface:

    from altcoin_agent.risk import (
        Side,
        Position,
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
        CCXTExecutor,
        ExecutionError,
    )
"""

from altcoin_agent.risk.executor import (
    CCXTExecutor,
    ExchangeAdapter,
    ExecutionError,
)
from altcoin_agent.risk.gate import RiskDecision, RiskGate, RiskGateConfig
from altcoin_agent.risk.reconciler import Reconciler, ReconcilerReport
from altcoin_agent.risk.sizing import DynamicLeverageConfig, PositionSizer
from altcoin_agent.risk.state import AccountState, Position, Side
from altcoin_agent.risk.trailing import TrailingState, TrailingStopFSM

__all__ = [
    "AccountState",
    "CCXTExecutor",
    "DynamicLeverageConfig",
    "ExchangeAdapter",
    "ExecutionError",
    "Position",
    "PositionSizer",
    "Reconciler",
    "ReconcilerReport",
    "RiskDecision",
    "RiskGate",
    "RiskGateConfig",
    "Side",
    "TrailingState",
    "TrailingStopFSM",
]
