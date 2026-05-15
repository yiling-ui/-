"""
Mock tests for the Binance Square scraper.

Strategy:
    - Patch BinanceSquareScraper._do_request (the single network seam)
      so no actual HTTP traffic occurs.
    - Verify the multi-endpoint waterfall, proxy/cookie injection,
      response parsing, and graceful failure modes.
"""

from __future__ import annotations

import pytest

from altcoin_agent.social.binance_square import (
    _ENDPOINTS,
    BinanceSquareScraper,
    CookieJar,
    ProxyConfig,
    ScraperAuthRequired,
    ScraperGeoBlocked,
    SquarePost,
    _post_from_cms_article,
    _post_from_square,
)

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _patch_responses(
    monkeypatch: pytest.MonkeyPatch,
    scraper: BinanceSquareScraper,
    responses: dict[str, tuple[int, str]],
) -> list[dict]:
    """Replace _do_request with a fake that returns canned responses by
    endpoint name. Returns the list of recorded requests for assertion."""
    recorded: list[dict] = []

    async def fake(*, method, url, headers, json_body, params, proxy_url):
        ep_name = "unknown"
        for ep in _ENDPOINTS:
            if ep.path in url:
                ep_name = ep.name
                break
        recorded.append({
            "endpoint": ep_name,
            "method": method,
            "url": url,
            "headers": headers,
            "json_body": json_body,
            "params": params,
            "proxy_url": proxy_url,
        })
        if ep_name not in responses:
            return 404, '{"error":"not configured in test"}'
        status, body = responses[ep_name]
        return status, body

    # _ensure_session has to be a no-op for tests because aiohttp is unused.
    async def fake_ensure(self):
        return None

    async def fake_close(self):
        return None

    monkeypatch.setattr(scraper, "_do_request", fake)
    monkeypatch.setattr(BinanceSquareScraper, "_ensure_session", fake_ensure)
    monkeypatch.setattr(BinanceSquareScraper, "close", fake_close)
    return recorded


# --------------------------------------------------------------------------- #
# CookieJar parsing
# --------------------------------------------------------------------------- #


def test_cookie_jar_parses_browser_cookie_header() -> None:
    raw = "csrftoken=abc123; bnc-uuid=uuid-456; p20t=session-789; junk=  "
    jar = CookieJar.from_header_string(raw)
    assert jar.cookies["csrftoken"] == "abc123"
    assert jar.cookies["bnc-uuid"] == "uuid-456"
    assert jar.cookies["p20t"] == "session-789"
    # And it round-trips back to a valid header
    h = jar.header()
    assert "csrftoken=abc123" in h
    assert "bnc-uuid=uuid-456" in h


def test_cookie_jar_handles_empty_and_malformed() -> None:
    assert CookieJar.from_header_string("").cookies == {}
    jar = CookieJar.from_header_string("good=1; ; bad ; alsogood=2")
    assert jar.cookies == {"good": "1", "alsogood": "2"}


# --------------------------------------------------------------------------- #
# ProxyConfig
# --------------------------------------------------------------------------- #


def test_proxy_config_pick_returns_url_when_set() -> None:
    p = ProxyConfig(url="http://user:pass@proxy.example:8080")
    assert p.pick() == "http://user:pass@proxy.example:8080"


def test_proxy_config_pick_returns_none_when_empty() -> None:
    assert ProxyConfig().pick() is None


def test_proxy_config_pick_rotates_pool() -> None:
    p = ProxyConfig(pool=["http://a", "http://b", "http://c"], rotate=True)
    seen = {p.pick() for _ in range(50)}
    # With 3 entries and 50 picks the chance of not seeing all 3 is vanishing
    assert seen == {"http://a", "http://b", "http://c"}


def test_proxy_config_pool_first_when_no_rotate() -> None:
    p = ProxyConfig(pool=["http://a", "http://b"], rotate=False)
    for _ in range(10):
        assert p.pick() == "http://a"


# --------------------------------------------------------------------------- #
# Endpoint waterfall — auth required
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_scraper_skips_auth_endpoints_without_cookies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No cookies → only public CMS endpoint is even attempted."""
    cms_body = (
        '{"code":"000000","data":{"catalogs":[{"articles":['
        '{"id":1001,"code":"a-b-c","title":"Binance lists $RAVE","body":"..."}'
        ']}]}}'
    )
    scraper = BinanceSquareScraper(cookies=None, proxy=None)
    rec = _patch_responses(monkeypatch, scraper, {
        "cms_announcements": (200, cms_body),
    })

    posts = await scraper.fetch_by_keyword("RAVE", limit=10)
    assert len(posts) == 1
    assert posts[0].author == "binance_official"
    assert "RAVE" in posts[0].title

    # Only the public endpoint should have been hit
    endpoints_called = [r["endpoint"] for r in rec]
    assert endpoints_called == ["cms_announcements"]


@pytest.mark.asyncio
async def test_scraper_raises_auth_required_when_no_cookies_and_cms_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scraper = BinanceSquareScraper(cookies=None)
    _patch_responses(monkeypatch, scraper, {
        # CMS returns no articles -> no usable posts
        "cms_announcements": (200, '{"code":"000000","data":{"catalogs":[]}}'),
    })
    with pytest.raises(ScraperAuthRequired):
        await scraper.fetch_by_keyword("RAVE", limit=10)


# --------------------------------------------------------------------------- #
# Endpoint waterfall — auth provided
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_scraper_uses_first_auth_endpoint_when_cookies_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = (
        '{"code":"000000","data":{"feeds":['
        '{"id":42,"user":{"nickname":"@whale","followerCount":120000},'
        '"content":"$RAVE pumping","createTime":1700000000000,'
        '"likeCount":99,"commentCount":12}'
        ']}}'
    )
    cookies = CookieJar(cookies={"csrftoken": "x", "bnc-uuid": "y", "p20t": "z"})
    scraper = BinanceSquareScraper(cookies=cookies)
    rec = _patch_responses(monkeypatch, scraper, {
        "square_feed_by_tag": (200, body),
    })

    posts = await scraper.fetch_by_keyword("RAVE", limit=10)
    assert len(posts) == 1
    p = posts[0]
    assert p.author == "@whale"
    assert p.follower_count == 120000
    assert "RAVE" in p.text
    assert p.likes == 99

    # First auth endpoint succeeds; nothing else attempted
    assert [r["endpoint"] for r in rec] == ["square_feed_by_tag"]
    # Cookie + csrftoken should have been included in headers
    assert rec[0]["headers"].get("Cookie", "").startswith("csrftoken=x")
    assert rec[0]["headers"].get("csrftoken") == "x"
    assert rec[0]["headers"].get("bnc-uuid") == "y"


@pytest.mark.asyncio
async def test_scraper_falls_through_to_next_endpoint_on_4xx(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If endpoint #1 returns a hard 4xx, scraper tries #2, #3, ..."""
    feed_list_body = (
        '{"code":"000000","data":{"feeds":['
        '{"id":7,"user":{"nickname":"alpha","followers":500},'
        '"content":"good signal","createTime":1700000001000}'
        ']}}'
    )
    cookies = CookieJar(cookies={"csrftoken": "x"})
    scraper = BinanceSquareScraper(cookies=cookies)
    rec = _patch_responses(monkeypatch, scraper, {
        "square_feed_by_tag": (404, "no such tag"),
        "square_feed_list": (200, feed_list_body),
    })
    posts = await scraper.fetch_by_keyword("RAVE", limit=10)
    assert len(posts) == 1
    assert posts[0].author == "alpha"
    # First endpoint failed, second succeeded; later ones not tried
    assert [r["endpoint"] for r in rec] == ["square_feed_by_tag", "square_feed_list"]


# --------------------------------------------------------------------------- #
# Geo / auth surface as typed exceptions
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_scraper_raises_geo_blocked_on_451(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cookies = CookieJar(cookies={"csrftoken": "x"})
    scraper = BinanceSquareScraper(cookies=cookies)
    _patch_responses(monkeypatch, scraper, {
        "square_feed_by_tag": (451, "Unavailable for legal reasons"),
    })
    with pytest.raises(ScraperGeoBlocked):
        await scraper.fetch_by_keyword("RAVE", limit=5)


@pytest.mark.asyncio
async def test_scraper_logs_auth_required_on_401_and_falls_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """401 on a single endpoint should not abort the whole waterfall:
    the scraper logs and tries the next endpoint."""
    cms_body = (
        '{"code":"000000","data":{"catalogs":[{"articles":['
        '{"id":1,"code":"abc","title":"announcement","body":""}'
        ']}]}}'
    )
    cookies = CookieJar(cookies={"csrftoken": "x"})
    scraper = BinanceSquareScraper(cookies=cookies)
    _patch_responses(monkeypatch, scraper, {
        "square_feed_by_tag": (401, '{"code":"100001005"}'),
        "square_feed_list":   (401, '{"code":"100001005"}'),
        "square_post_list":   (401, '{"code":"100001005"}'),
        "cms_announcements":  (200, cms_body),
    })
    posts = await scraper.fetch_by_keyword("RAVE", limit=5)
    assert len(posts) == 1
    assert posts[0].author == "binance_official"


# --------------------------------------------------------------------------- #
# Proxy is wired through to _do_request
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_scraper_passes_proxy_url_per_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cookies = CookieJar(cookies={"csrftoken": "x"})
    proxy = ProxyConfig(url="http://user:pass@vpn.example:8080")
    scraper = BinanceSquareScraper(cookies=cookies, proxy=proxy)
    rec = _patch_responses(monkeypatch, scraper, {
        "square_feed_by_tag": (200, '{"code":"000000","data":{"feeds":[]}}'),
        "square_feed_list":   (200, '{"code":"000000","data":{"feeds":[]}}'),
        "square_post_list":   (200, '{"code":"000000","data":{"posts":[]}}'),
        "cms_announcements":  (200, '{"code":"000000","data":{"catalogs":[]}}'),
    })
    # Empty results across all endpoints -> ScraperError with no posts;
    # we just want to assert the proxy was passed.
    try:
        await scraper.fetch_by_keyword("RAVE", limit=5)
    except Exception:
        pass  # we don't care about the outcome here

    assert rec, "at least one request should have been attempted"
    for r in rec:
        assert r["proxy_url"] == "http://user:pass@vpn.example:8080"


@pytest.mark.asyncio
async def test_scraper_pool_rotation_distributes_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cookies = CookieJar(cookies={"csrftoken": "x"})
    pool = ["http://p1", "http://p2", "http://p3"]
    proxy = ProxyConfig(pool=pool, rotate=True)
    scraper = BinanceSquareScraper(cookies=cookies, proxy=proxy)
    rec = _patch_responses(monkeypatch, scraper, {
        "square_feed_by_tag": (404, ""),
        "square_feed_list":   (404, ""),
        "square_post_list":   (404, ""),
        "cms_announcements":  (404, ""),
    })
    try:
        await scraper.fetch_by_keyword("X", limit=5)
    except Exception:
        pass

    used = {r["proxy_url"] for r in rec}
    # With 4 endpoints we should see proxies from the pool (random rotation,
    # may not see all 3, but every used one must be from the pool).
    assert used.issubset(set(pool))
    assert used  # at least one was used


# --------------------------------------------------------------------------- #
# Response parsers (pure functions)
# --------------------------------------------------------------------------- #


def test_parse_cms_article() -> None:
    raw = {
        "id": 274074,
        "code": "875f9acd",
        "title": "Binance Will Add ABC",
        "body": "details here",
        "releaseDate": 1700000000000,
    }
    p = _post_from_cms_article(raw, source_endpoint="cms_announcements")
    assert isinstance(p, SquarePost)
    assert p.author == "binance_official"
    assert "Binance Will Add ABC" in p.title
    assert p.url.endswith("875f9acd")
    assert p.ts_ms == 1700000000000


def test_parse_square_post_handles_field_aliases() -> None:
    """Different Square endpoints use slightly different field names.
    The parser must be liberal."""
    raw = {
        "id": "post-1",
        "user": {"nickname": "@bigwhale", "followerCount": 99999},
        "content": "hello",
        "createTime": 1700000123000,
    }
    p = _post_from_square(raw, source_endpoint="square_feed_by_tag")
    assert p.post_id == "post-1"
    assert p.author == "@bigwhale"
    assert p.follower_count == 99999
    assert p.text == "hello"
    assert p.ts_ms == 1700000123000


def test_parse_square_post_with_alternate_field_names() -> None:
    raw = {
        "postId": "alt-2",
        "author": {"name": "kol", "followers": 500},
        "description": "shilly text",
        "publishTime": 1700000999000,
        "likes": 7, "comments": 1, "shares": 0,
    }
    p = _post_from_square(raw, source_endpoint="square_post_list")
    assert p.post_id == "alt-2"
    assert p.author == "kol"
    assert p.follower_count == 500
    assert p.likes == 7


# --------------------------------------------------------------------------- #
# Endpoint registry sanity
# --------------------------------------------------------------------------- #


def test_endpoints_priority_auth_first_then_public() -> None:
    """The order matters: we must always try authenticated Square endpoints
    before falling back to the public CMS announcements."""
    auth_idxs = [i for i, e in enumerate(_ENDPOINTS) if e.requires_auth]
    public_idxs = [i for i, e in enumerate(_ENDPOINTS) if not e.requires_auth]
    assert auth_idxs and public_idxs
    assert max(auth_idxs) < min(public_idxs)
    # And there must be at least one CMS endpoint as a fallback.
    assert any(e.name == "cms_announcements" for e in _ENDPOINTS)
