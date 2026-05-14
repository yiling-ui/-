"""Risk and execution module — Task D.

Public API:
    RiskGate, GateDecision    — the single hard wall (SR-1 enforced here)
    PositionSizer             — dynamic leverage + risk parity
    TrailingStopFSM           — only tightens, never weakens, the hard stop
    Reconciler                — SR-2 startup reconciliation
    CCXTExecutor              — entry + immediate exchange stop (SR-2)
    Position, OrderIntent     — domain types

All survival rules from .kiro/steering/trading_logic.md are enforced here.
"""
from altcoin_agent.risk.executor import CCXTExecutor, ExchangeAdapter
from altcoin_agent.risk.gate import (
    GateDecision,
    GateReject,
    RiskGate,
    RiskGateConfig,
)
from altcoin_agent.risk.reconciler import (
    OrphanPosition,
    Reconciler,
    ReconcilerReport,
)
from altcoin_agent.risk.sizing import (
    DynamicLeverageConfig,
    OrderIntent,
    PositionSizer,
    SizingResult,
    compute_dynamic_leverage,
)
from altcoin_agent.risk.state import AccountState, Position
from altcoin_agent.risk.trailing import TrailingState, TrailingStopFSM

__all__ = [
    "CCXTExecutor",
    "ExchangeAdapter",
    "GateDecision",
    "GateReject",
    "RiskGate",
    "RiskGateConfig",
    "OrphanPosition",
    "Reconciler",
    "ReconcilerReport",
    "DynamicLeverageConfig",
    "OrderIntent",
    "PositionSizer",
    "SizingResult",
    "compute_dynamic_leverage",
    "AccountState",
    "Position",
    "TrailingState",
    "TrailingStopFSM",
]
