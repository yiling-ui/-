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

__all__ = [
    "BinanceSquareScraper",
    "CookieJar",
    "ProxyConfig",
    "ScraperAuthRequired",
    "ScraperError",
    "ScraperGeoBlocked",
    "SocialSnapshot",
    "SquarePost",
    "focus_on_symbol",
]
