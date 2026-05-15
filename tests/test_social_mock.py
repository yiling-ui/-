"""Unit tests for binance_square scraper and social aggregator.

We monkey-patch BinanceSquareScraper._do_request, the single network entry
point, so all tests run offline.
"""

from __future__ import annotations

from typing import Any

import pytest

from altcoin_agent.social import (
    BinanceSquareScraper,
    CookieJar,
    ProxyConfig,
    ScraperAuthRequired,
    ScraperGeoBlocked,
    SquarePost,
)
from altcoin_agent.social.crawler import focus_on_symbol

# --------------------------------------------------------------------- #
# CookieJar parsing
# --------------------------------------------------------------------- #


def test_cookie_jar_parses_raw_header() -> None:
    raw = "csrftoken=abc; bnc-uuid=u123; p20t=tok; theme=dark"
    jar = CookieJar.from_header_string(raw)
    assert jar.cookies["csrftoken"] == "abc"
    assert jar.cookies["bnc-uuid"] == "u123"
    assert jar.cookies["p20t"] == "tok"
    assert jar.csrftoken == "abc"
    assert jar.has_session is True


def test_cookie_jar_no_session() -> None:
    jar = CookieJar()
    assert jar.has_session is False


# --------------------------------------------------------------------- #
# ProxyConfig
# --------------------------------------------------------------------- #


def test_proxy_config_picks_from_pool() -> None:
    p = ProxyConfig(pool=["http://a", "http://b"], rotate=False)
    assert p.pick() == "http://a"


def test_proxy_config_socks_detection() -> None:
    p = ProxyConfig(url="socks5://x:1080")
    assert p.is_socks() is True
    p2 = ProxyConfig(url="http://x:8080")
    assert p2.is_socks() is False


# --------------------------------------------------------------------- #
# Scraper: monkey-patched transport
# --------------------------------------------------------------------- #


def _patch_transport(monkeypatch: pytest.MonkeyPatch, scraper: BinanceSquareScraper,
                      response_chain: list) -> list[tuple[str, str]]:
    """Patch _do_request to walk through `response_chain` (dict|Exception)."""
    seen: list[tuple[str, str]] = []
    chain = list(response_chain)

    async def fake(method: str, path: str, *, json: Any = None) -> dict[str, Any]:
        seen.append((method, path))
        if not chain:
            return {}
        nxt = chain.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt

    monkeypatch.setattr(scraper, "_do_request", fake)
    return seen


@pytest.mark.asyncio
async def test_scraper_falls_back_to_announcements_when_no_cookie(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scraper = BinanceSquareScraper()  # no cookies -> no auth attempts
    cms_payload = {"data": {"articles": [
        {"id": "x1", "title": "Binance Will Add RAVE",
         "body": "details about $RAVE", "releaseDate": 100},
    ]}}
    seen = _patch_transport(monkeypatch, scraper, [cms_payload])
    posts = await scraper.fetch_by_keyword("RAVE")
    assert len(posts) == 1
    assert posts[0].source == "binance_announcements"
    assert "RAVE" in posts[0].text
    # Only the public CMS endpoint was hit (no auth attempts).
    assert all("/cms/article/list/query" in path for _, path in seen)


@pytest.mark.asyncio
async def test_scraper_geo_block_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    scraper = BinanceSquareScraper()
    _patch_transport(monkeypatch, scraper, [ScraperGeoBlocked("HTTP 451")])
    with pytest.raises(ScraperGeoBlocked):
        await scraper.fetch_by_keyword("RAVE")


@pytest.mark.asyncio
async def test_scraper_auth_endpoints_then_fallback_on_session_expired(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With cookies, scraper tries auth endpoints; on 401 it degrades to CMS."""
    cookies = CookieJar(cookies={"p20t": "tok", "bnc-uuid": "u",
                                  "csrftoken": "c"}, csrftoken="c")
    scraper = BinanceSquareScraper(cookies=cookies)
    # All three auth endpoints reject; CMS works.
    chain = [
        ScraperAuthRequired("ep1"),
        ScraperAuthRequired("ep2"),
        ScraperAuthRequired("ep3"),
        {"data": {"articles": [{"id": "a1", "title": "RAVE news",
                                  "body": "...", "releaseDate": 1}]}},
    ]
    _patch_transport(monkeypatch, scraper, chain)
    posts = await scraper.fetch_by_keyword("RAVE")
    assert len(posts) == 1
    assert posts[0].source == "binance_announcements"


# --------------------------------------------------------------------- #
# Aggregator (focus_on_symbol)
# --------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_focus_returns_snapshot_with_primary_status(monkeypatch: pytest.MonkeyPatch) -> None:
    """When no cookies & primary returns nothing useful, snapshot still
    comes back with auxiliary attempted (best-effort)."""
    # Force the BinanceSquareScraper used inside focus_on_symbol to return [].
    async def fake_fetch(self, keyword: str, page_size: int = 20) -> list[SquarePost]:
        return []
    monkeypatch.setattr(BinanceSquareScraper, "fetch_by_keyword", fake_fetch)

    # Stub the auxiliary fetchers to avoid network.
    import altcoin_agent.social.crawler as crawler_mod

    async def fake_okx(client, symbol):       # noqa: ANN001
        return {"inst_id": f"{symbol}-MOCK"}
    async def fake_dex(client, symbol):       # noqa: ANN001
        return {"query": symbol, "pair": None}
    async def fake_cg(client, symbol):        # noqa: ANN001
        return {"symbol": symbol, "trending_rank": None}

    monkeypatch.setattr(crawler_mod, "_fetch_okx", fake_okx)
    monkeypatch.setattr(crawler_mod, "_fetch_dexscreener", fake_dex)
    monkeypatch.setattr(crawler_mod, "_fetch_coingecko_trending", fake_cg)

    snap = await focus_on_symbol("RAVEUSDT")
    assert snap.symbol == "RAVEUSDT"
    assert snap.primary_status == "degraded:no_results"
    assert snap.okx is not None
    assert snap.dexscreener is not None
    assert snap.coingecko is not None
