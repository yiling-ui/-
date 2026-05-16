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
        # Audit batch 2 — safety hardening
        RegimeFilter,
        RegimeFilterConfig,
        ClusterMap,
        ClusterCapConfig,
        AccountPersistor,
        DecisionAuditLog,
        KillSwitchConfig,
        KillSwitchWatcher,
    )
"""

from altcoin_agent.risk.atr import ATRCalculator
from altcoin_agent.risk.audit_log import DecisionAuditLog
from altcoin_agent.risk.ccxt_adapter import CCXTExchangeAdapter, build_ccxt_adapter
from altcoin_agent.risk.cluster import (
    ClusterCapConfig,
    ClusterMap,
    cap_breached,
)
from altcoin_agent.risk.confidence_gate import (
    ConfidenceGate,
    ConfidenceVerdict,
    ConfidenceWeights,
)
from altcoin_agent.risk.executor import (
    CCXTExecutor,
    ExchangeAdapter,
    ExecutionError,
)
from altcoin_agent.risk.gate import RiskDecision, RiskGate, RiskGateConfig
from altcoin_agent.risk.kill_switch import (
    KillSwitchConfig,
    KillSwitchWatcher,
)
from altcoin_agent.risk.persistence import AccountPersistor
from altcoin_agent.risk.position_watcher import CloseCallback, PositionWatcher
from altcoin_agent.risk.pump_phase import (
    KlineBar,
    PhaseInputs,
    PhaseThresholds,
    PhaseTransition,
    PumpPhase,
    PumpPhaseFSM,
)
from altcoin_agent.risk.quadrant_factory import (
    QuadrantRiskBundle,
    QuadrantRiskFactory,
)
from altcoin_agent.risk.reconciler import Reconciler, ReconcilerReport
from altcoin_agent.risk.regime_filter import RegimeFilter, RegimeFilterConfig
from altcoin_agent.risk.rolling import (
    RollDecision,
    RollingConfig,
    RollingController,
)
from altcoin_agent.risk.sizing import DynamicLeverageConfig, PositionSizer
from altcoin_agent.risk.sqlite_persistence import SQLiteAccountStore
from altcoin_agent.risk.state import AccountState, Position, PositionLeg, Side
from altcoin_agent.risk.symbol_profile import (
    DEFAULT_QUADRANT_PARAMS,
    Quadrant,
    QuadrantParams,
    SymbolProfile,
    SymbolProfileStore,
    quadrant_params,
)
from altcoin_agent.risk.symbol_profile import (
    classify as classify_quadrant,
)
from altcoin_agent.risk.trailing import TrailingState, TrailingStopFSM

__all__ = [
    "AccountPersistor",
    "AccountState",
    "ATRCalculator",
    "CCXTExchangeAdapter",
    "CCXTExecutor",
    "CloseCallback",
    "ClusterCapConfig",
    "ClusterMap",
    "ConfidenceGate",
    "ConfidenceVerdict",
    "ConfidenceWeights",
    "DEFAULT_QUADRANT_PARAMS",
    "DecisionAuditLog",
    "DynamicLeverageConfig",
    "ExchangeAdapter",
    "ExecutionError",
    "KillSwitchConfig",
    "KillSwitchWatcher",
    "KlineBar",
    "PhaseInputs",
    "PhaseThresholds",
    "PhaseTransition",
    "Position",
    "PositionLeg",
    "PositionSizer",
    "PositionWatcher",
    "PumpPhase",
    "PumpPhaseFSM",
    "Quadrant",
    "QuadrantParams",
    "QuadrantRiskBundle",
    "QuadrantRiskFactory",
    "Reconciler",
    "ReconcilerReport",
    "RegimeFilter",
    "RegimeFilterConfig",
    "RiskDecision",
    "RiskGate",
    "RiskGateConfig",
    "RollDecision",
    "RollingConfig",
    "RollingController",
    "Side",
    "SQLiteAccountStore",
    "SymbolProfile",
    "SymbolProfileStore",
    "TrailingState",
    "TrailingStopFSM",
    "build_ccxt_adapter",
    "cap_breached",
    "classify_quadrant",
    "quadrant_params",
]
