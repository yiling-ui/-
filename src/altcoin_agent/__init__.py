"""Altcoin Momentum Agent — public API surface."""
from altcoin_agent.ai_engine import (
    AIVerdict,
    DeepSeekEngine,
    EngineError,
    SocialPost,
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
    VolumeSpikeDetector,
)

__all__ = [
    # screener
    "Screener",
    "SignalEvent",
    "Kline",
    "FundingSnapshot",
    "OISnapshot",
    "LiquidityPool",
    "VolumeSpikeDetector",
    "FundingAnomalyDetector",
    "OISurgeDetector",
    "LiquidityPoolAnalyzer",
    # ai_engine
    "DeepSeekEngine",
    "AIVerdict",
    "SocialPost",
    "EngineError",
]
