"""Social-source aggregation package.

Public surface:

    from altcoin_agent.social import (
        BinanceSquareScraper,
        ProxyConfig,
        CookieJar,
        SquarePost,
        ScraperAuthRequired,
        ScraperGeoBlocked,
        ScraperError,
        SocialSnapshot,
        focus_on_symbol,
        # KOL historical hit-rate (Phase B.6 sister deliverable).
        HistoricalAnalyzer,
        HistoricalAnalyzerConfig,
        KOLAdjustment,
        KOLHistoryStore,
        KOLObservation,
        KOLScore,
        build_observations_from_posts,
        normalize_author,
    )
"""

from altcoin_agent.social.binance_square import (
    BinanceSquareScraper,
    CookieJar,
    ProxyConfig,
    ScraperAuthRequired,
    ScraperError,
    ScraperGeoBlocked,
    SquarePost,
)
from altcoin_agent.social.crawler import SocialSnapshot, focus_on_symbol
from altcoin_agent.social.historical_analyzer import (
    HistoricalAnalyzer,
    HistoricalAnalyzerConfig,
    KOLAdjustment,
    KOLHistoryStore,
    KOLObservation,
    KOLScore,
    build_observations_from_posts,
    normalize_author,
)

__all__ = [
    "BinanceSquareScraper",
    "CookieJar",
    "HistoricalAnalyzer",
    "HistoricalAnalyzerConfig",
    "KOLAdjustment",
    "KOLHistoryStore",
    "KOLObservation",
    "KOLScore",
    "ProxyConfig",
    "ScraperAuthRequired",
    "ScraperError",
    "ScraperGeoBlocked",
    "SocialSnapshot",
    "SquarePost",
    "build_observations_from_posts",
    "focus_on_symbol",
    "normalize_author",
]
