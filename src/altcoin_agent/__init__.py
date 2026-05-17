"""Altcoin Momentum Agent V1.0 — public API surface."""

from altcoin_agent.ai_engine import (
    AIVerdict,
    DeepSeekEngine,
    EngineError,
    LLMEngine,
    SocialPost,
)
from altcoin_agent.dashboard import DashboardState, install_dashboard
from altcoin_agent.fuser import (
    Direction,
    FusedSignal,
    FuserConfig,
    RuleIndex,
    ScoreFuser,
)
from altcoin_agent.learning_engine import (
    DynamicRule,
    HistoricalSlice,
    PostMortemReport,
    RuleStore,
    extract_candidate_features,
    fetch_historical_slice,
    run_post_mortem,
    synthesize_dump_slice,
)
from altcoin_agent.llm_provider import (
    AnthropicProvider,
    LLMProvider,
    OpenAICompatibleProvider,
    build_default_provider,
)
from altcoin_agent.notifier import (
    Notifier,
    NullNotifier,
    TelegramNotifier,
    build_default_notifier,
)
from altcoin_agent.risk import (
    AccountState,
    ATRCalculator,
    CCXTExchangeAdapter,
    CCXTExecutor,
    DynamicLeverageConfig,
    ExchangeAdapter,
    ExecutionError,
    Position,
    PositionSizer,
    Reconciler,
    RiskDecision,
    RiskGate,
    RiskGateConfig,
    Side,
    TrailingState,
    TrailingStopFSM,
    build_ccxt_adapter,
)
from altcoin_agent.screener import (
    FundingAnomalyDetector,
    FundingSnapshot,
    Kline,
    LiquidityPool,
    LiquidityPoolAnalyzer,
    OISnapshot,
    OISurgeDetector,
    Screener,
    SignalEvent,
    SignalKind,
    VolumeSpikeDetector,
    WashTradingDetector,
)
from altcoin_agent.slice_cache import SliceCache

__all__ = [
    # screener
    "Screener",
    "SignalEvent",
    "SignalKind",
    "Kline",
    "FundingSnapshot",
    "OISnapshot",
    "LiquidityPool",
    "VolumeSpikeDetector",
    "FundingAnomalyDetector",
    "OISurgeDetector",
    "LiquidityPoolAnalyzer",
    "WashTradingDetector",
    # ai_engine + providers
    "LLMEngine",
    "DeepSeekEngine",
    "AIVerdict",
    "SocialPost",
    "EngineError",
    "LLMProvider",
    "OpenAICompatibleProvider",
    "AnthropicProvider",
    "build_default_provider",
    # fuser
    "ScoreFuser",
    "FuserConfig",
    "FusedSignal",
    "Direction",
    "RuleIndex",
    # learning engine + cache
    "RuleStore",
    "DynamicRule",
    "HistoricalSlice",
    "PostMortemReport",
    "fetch_historical_slice",
    "extract_candidate_features",
    "run_post_mortem",
    "synthesize_dump_slice",
    "SliceCache",
    # risk
    "Side",
    "Position",
    "AccountState",
    "ExchangeAdapter",
    "ExecutionError",
    "PositionSizer",
    "DynamicLeverageConfig",
    "RiskGate",
    "RiskGateConfig",
    "RiskDecision",
    "TrailingStopFSM",
    "TrailingState",
    "Reconciler",
    "CCXTExecutor",
    "CCXTExchangeAdapter",
    "ATRCalculator",
    "build_ccxt_adapter",
    # notifiers
    "Notifier",
    "NullNotifier",
    "TelegramNotifier",
    "build_default_notifier",
    # dashboard
    "DashboardState",
    "install_dashboard",
]
