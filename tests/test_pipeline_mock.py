"""Mock tests for pipeline.py — the bridge between Screener, social,
DeepSeek and the post-mortem learning loop.

These tests run fully offline:
  * focus_on_symbol is replaced by an injected ``social_fetcher`` so we
    don't need real cookies / network.
  * DeepSeekEngine.judge is monkey-patched so we don't need real API keys.
  * run_post_mortem (used by DelayedPostMortemScheduler) is patched at the
    pipeline import site to avoid hitting the OKX REST API.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from altcoin_agent import pipeline as pipeline_mod
from altcoin_agent.ai_engine import AIVerdict, DeepSeekEngine
from altcoin_agent.fuser import FuserConfig, ScoreFuser
from altcoin_agent.learning_engine import RuleStore
from altcoin_agent.pipeline import (
    DEFAULT_CONSULT_KINDS,
    CandidateGate,
    DelayedPostMortemScheduler,
    LLMConsultor,
    RecentSignalsCache,
    cookie_jar_from_env,
    proxy_config_from_env,
)
from altcoin_agent.screener import (
    FundingSnapshot,
    OISnapshot,
    SignalEvent,
    SignalKind,
)
from altcoin_agent.social.crawler import SocialSnapshot

# --------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------- #


def _ev(kind: SignalKind, ts: int = 1_000_000, **payload) -> SignalEvent:
    return SignalEvent(
        kind=kind, symbol="RAVEUSDT", ts=ts, exchange="binance", payload=payload,
    )


def _make_fuser(tmp_json: Path) -> ScoreFuser:
    return ScoreFuser(config=FuserConfig(dynamic_rules_path=tmp_json))


def _ok_snapshot(symbol: str = "RAVEUSDT") -> SocialSnapshot:
    return SocialSnapshot(
        symbol=symbol,
        fetched_at_ts_ms=int(time.time() * 1000),
        primary_status="ok",
    )


def _degraded_snapshot(symbol: str = "RAVEUSDT") -> SocialSnapshot:
    snap = SocialSnapshot(
        symbol=symbol,
        fetched_at_ts_ms=int(time.time() * 1000),
        primary_status="degraded:no_results",
    )
    snap.okx = {"funding_rate": -0.001, "open_interest": 1.2e6}
    snap.dexscreener = {"query": "RAVE", "pair": None}
    snap.coingecko = {"trending_rank": 3}
    return snap


# --------------------------------------------------------------------- #
# env helpers
# --------------------------------------------------------------------- #


def test_cookie_jar_from_env_returns_none_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BINANCE_SQUARE_COOKIE", raising=False)
    assert cookie_jar_from_env() is None


def test_cookie_jar_from_env_parses_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BINANCE_SQUARE_COOKIE", "csrftoken=abc; bnc-uuid=u; p20t=t")
    jar = cookie_jar_from_env()
    assert jar is not None
    assert jar.has_session is True
    assert jar.csrftoken == "abc"


def test_proxy_config_from_env_none_when_pool_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PROXY_POOL", raising=False)
    assert proxy_config_from_env() is None


def test_proxy_config_from_env_parses_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROXY_POOL", "http://a:1, http://b:2")
    monkeypatch.setenv("PROXY_ROTATE", "false")
    cfg = proxy_config_from_env()
    assert cfg is not None
    assert cfg.pool == ["http://a:1", "http://b:2"]
    assert cfg.rotate is False


# --------------------------------------------------------------------- #
# CandidateGate
# --------------------------------------------------------------------- #


def test_candidate_gate_allows_in_kind_allowlist() -> None:
    gate = CandidateGate(cooldown_sec=300)
    ev = _ev(SignalKind.OI_SILENT_BUILD, ts=1_000_000,
             oi_delta_pct=0.22, from_price=1.0, to_price=1.005)
    assert gate.should_consult(ev) is True


def test_candidate_gate_rejects_out_of_kind() -> None:
    gate = CandidateGate(cooldown_sec=300)
    ev = _ev(SignalKind.LIQUIDITY_POOL_FORMED, ts=1_000_000,
             side="sell_side", level=1.0, touch_count=2)
    assert ev.kind not in DEFAULT_CONSULT_KINDS
    assert gate.should_consult(ev) is False


def test_candidate_gate_cooldown_blocks_repeated_consults() -> None:
    gate = CandidateGate(cooldown_sec=300)
    base_ts = 1_000_000
    e1 = _ev(SignalKind.VOLUME_SPIKE, ts=base_ts, side="buy", zscore=5.0)
    assert gate.should_consult(e1) is True
    gate.mark_consulted("RAVEUSDT", base_ts)

    # 100 seconds later — still inside cooldown.
    e2 = _ev(SignalKind.VOLUME_SPIKE, ts=base_ts + 100_000, side="buy", zscore=5.0)
    assert gate.should_consult(e2) is False

    # 301 seconds later — cooldown elapsed.
    e3 = _ev(SignalKind.VOLUME_SPIKE, ts=base_ts + 301_000, side="buy", zscore=5.0)
    assert gate.should_consult(e3) is True


def test_candidate_gate_per_symbol_independent() -> None:
    gate = CandidateGate(cooldown_sec=300)
    base_ts = 1_000_000
    e1 = _ev(SignalKind.VOLUME_SPIKE, ts=base_ts, side="buy", zscore=5.0)
    e1.symbol = "RAVEUSDT"      # type: ignore[misc]
    gate.mark_consulted("RAVEUSDT", base_ts)
    e2 = SignalEvent(
        kind=SignalKind.VOLUME_SPIKE, symbol="DOGEUSDT", ts=base_ts + 1000,
        exchange="binance", payload={"side": "buy", "zscore": 5.0},
    )
    assert gate.should_consult(e2) is True


# --------------------------------------------------------------------- #
# RecentSignalsCache
# --------------------------------------------------------------------- #


def test_recent_cache_builds_smc_context_from_freshness_window() -> None:
    cache = RecentSignalsCache(window_sec=90)
    base = 1_000_000
    cache.add_signal(_ev(SignalKind.LIQUIDITY_SWEEP, ts=base,
                          side="sell_side", level=1.0, wick_to_body=2.5,
                          bar_close=0.99))
    cache.add_signal(_ev(SignalKind.VOLUME_SPIKE, ts=base + 10_000,
                          side="buy", zscore=5.5))
    cache.add_signal(_ev(SignalKind.OI_SILENT_BUILD, ts=base + 20_000,
                          oi_delta_pct=0.22, from_price=1.0, to_price=1.005))
    # this one is OUTSIDE the 90s window
    cache.add_signal(_ev(SignalKind.LIQUIDITY_SWEEP, ts=base - 200_000,
                          side="buy_side", level=0.95, wick_to_body=2.0))
    cache.add_funding(FundingSnapshot(ts=base, symbol="RAVEUSDT", rate=-0.0015))
    cache.add_oi(OISnapshot(ts=base, symbol="RAVEUSDT",
                             open_interest=1.5e6, price=1.005))

    smc = cache.build_smc_context("RAVEUSDT", now_ts=base + 25_000)
    # Only the in-window sweep is included (the older one was filtered out).
    assert len(smc.liquidity_sweeps) == 1
    assert smc.liquidity_sweeps[0]["side"] == "sell_side"
    assert smc.volume_spike is not None
    assert smc.volume_spike["zscore"] == 5.5
    assert smc.oi_event is not None
    assert smc.oi_event["kind"] == "oi_silent_build"

    assert cache.latest_funding("RAVEUSDT") is not None
    assert cache.latest_funding("RAVEUSDT").rate == pytest.approx(-0.0015)
    assert cache.latest_oi("RAVEUSDT") is not None
    assert cache.latest_oi("RAVEUSDT").open_interest == pytest.approx(1.5e6)


def test_recent_cache_latest_funding_zscore_picks_most_recent() -> None:
    cache = RecentSignalsCache(window_sec=300)
    base = 1_000_000
    cache.add_signal(_ev(SignalKind.FUNDING_DEVIATION, ts=base,
                          short_avg=0.0001, baseline_mean=0.00005,
                          zscore=-3.5))
    cache.add_signal(_ev(SignalKind.FUNDING_DEVIATION, ts=base + 5_000,
                          short_avg=0.0002, baseline_mean=0.00005,
                          zscore=2.1))
    assert cache.latest_funding_zscore("RAVEUSDT") == pytest.approx(2.1)


def test_recent_cache_returns_empty_smc_for_unseen_symbol() -> None:
    cache = RecentSignalsCache()
    smc = cache.build_smc_context("NOPEUSDT", now_ts=1_000_000)
    assert smc.liquidity_sweeps == []
    assert smc.volume_spike is None
    assert smc.oi_event is None


# --------------------------------------------------------------------- #
# LLMConsultor
# --------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_consultor_calls_engine_and_feeds_fuser(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fuser = _make_fuser(tmp_path / "rules.json")
    cache = RecentSignalsCache(window_sec=90)
    base = 1_000_000
    cache.add_signal(_ev(SignalKind.OI_SILENT_BUILD, ts=base,
                          oi_delta_pct=0.22, from_price=1.0, to_price=1.005))
    cache.add_signal(_ev(SignalKind.VOLUME_SPIKE, ts=base, side="buy", zscore=6.0))
    cache.add_funding(FundingSnapshot(ts=base, symbol="RAVEUSDT", rate=-0.0025))

    captured: dict = {}

    async def fake_judge(self, **kwargs) -> AIVerdict:    # noqa: ANN001
        captured.update(kwargs)
        return AIVerdict(
            intent="pump", confidence_score=88,
            reason="mock", kol_intent="frontrun_call",
        )

    monkeypatch.setattr(DeepSeekEngine, "judge", fake_judge)

    async def fake_social(symbol: str) -> SocialSnapshot:
        return _ok_snapshot(symbol)

    engine = DeepSeekEngine(api_key="sk-test")
    consultor = LLMConsultor(
        engine=engine, fuser=fuser, cache=cache,
        social_fetcher=fake_social,
    )

    ev = _ev(SignalKind.VOLUME_SPIKE, ts=base, side="buy", zscore=6.0)
    verdict = await consultor.consult(ev)
    assert verdict is not None
    assert verdict.intent == "pump"
    assert verdict.confidence_score == 88

    # Engine received funding rate from the cache and an SMC context with
    # both VOLUME_SPIKE and OI_SILENT_BUILD events surfaced.
    assert captured["symbol"] == "RAVEUSDT"
    assert captured["exchange"] == "binance"
    assert captured["funding_rate"] == pytest.approx(-0.0025)
    smc = captured["smc"]
    assert smc.volume_spike is not None
    assert smc.oi_event is not None

    # Primary OK -> aux source highlights MUST NOT leak in (architect rule).
    extra = captured["extra"]
    assert extra["primary_status"] == "ok"
    assert "aux_okx" not in extra
    assert "aux_dexscreener" not in extra

    # And the fuser actually saw the verdict (the consultor wired it in).
    fused = fuser.evaluate("RAVEUSDT", "binance", base)
    assert fused.llm_verdict is not None
    assert fused.llm_verdict.intent == "pump"


@pytest.mark.asyncio
async def test_consultor_surfaces_aux_only_when_primary_degraded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fuser = _make_fuser(tmp_path / "rules.json")
    cache = RecentSignalsCache()

    captured: dict = {}

    async def fake_judge(self, **kwargs) -> AIVerdict:    # noqa: ANN001
        captured.update(kwargs)
        return AIVerdict(intent="neutral", confidence_score=20,
                          reason="mock", kol_intent="neutral")

    monkeypatch.setattr(DeepSeekEngine, "judge", fake_judge)

    async def fake_social(symbol: str) -> SocialSnapshot:
        return _degraded_snapshot(symbol)

    consultor = LLMConsultor(
        engine=DeepSeekEngine(api_key="sk-test"),
        fuser=fuser, cache=cache, social_fetcher=fake_social,
    )

    ev = _ev(SignalKind.VOLUME_SPIKE, side="buy", zscore=5.0)
    await consultor.consult(ev)

    extra = captured["extra"]
    assert extra["primary_status"] == "degraded:no_results"
    # Auxiliary sources surface ONLY when primary is degraded.
    assert "aux_okx" in extra
    assert "aux_dexscreener" in extra
    assert "aux_coingecko" in extra


@pytest.mark.asyncio
async def test_consultor_returns_none_when_engine_has_no_api_key(
    tmp_path: Path,
) -> None:
    """If DEEPSEEK_API_KEY isn't set, engine.judge raises EngineError; the
    consultor must swallow it and return None so the trading bus continues
    in rule-only mode."""
    fuser = _make_fuser(tmp_path / "rules.json")
    cache = RecentSignalsCache()
    engine = DeepSeekEngine(api_key=None)

    async def fake_social(symbol: str) -> SocialSnapshot:
        return _ok_snapshot(symbol)

    consultor = LLMConsultor(
        engine=engine, fuser=fuser, cache=cache, social_fetcher=fake_social,
    )

    ev = _ev(SignalKind.VOLUME_SPIKE, side="buy", zscore=5.0)
    verdict = await consultor.consult(ev)
    assert verdict is None


@pytest.mark.asyncio
async def test_consultor_swallows_social_fetch_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A crashing social fetcher must NOT abort the consult — the LLM
    is still called with a degraded primary_status."""
    fuser = _make_fuser(tmp_path / "rules.json")
    cache = RecentSignalsCache()

    captured: dict = {}

    async def fake_judge(self, **kwargs) -> AIVerdict:    # noqa: ANN001
        captured.update(kwargs)
        return AIVerdict(intent="neutral", confidence_score=10,
                          reason="degraded social", kol_intent="neutral")

    monkeypatch.setattr(DeepSeekEngine, "judge", fake_judge)

    async def boom_social(symbol: str) -> SocialSnapshot:
        raise RuntimeError("network exploded")

    consultor = LLMConsultor(
        engine=DeepSeekEngine(api_key="sk-test"),
        fuser=fuser, cache=cache, social_fetcher=boom_social,
    )

    ev = _ev(SignalKind.VOLUME_SPIKE, side="buy", zscore=5.0)
    verdict = await consultor.consult(ev)
    assert verdict is not None
    assert captured["extra"]["primary_status"].startswith("degraded:fetch_error")


@pytest.mark.asyncio
async def test_consultor_swallows_engine_unexpected_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fuser = _make_fuser(tmp_path / "rules.json")
    cache = RecentSignalsCache()

    async def boom_judge(self, **kwargs):    # noqa: ANN001
        raise ValueError("totally unexpected")

    monkeypatch.setattr(DeepSeekEngine, "judge", boom_judge)

    async def fake_social(symbol: str) -> SocialSnapshot:
        return _ok_snapshot(symbol)

    consultor = LLMConsultor(
        engine=DeepSeekEngine(api_key="sk-test"),
        fuser=fuser, cache=cache, social_fetcher=fake_social,
    )

    ev = _ev(SignalKind.VOLUME_SPIKE, side="buy", zscore=5.0)
    verdict = await consultor.consult(ev)
    assert verdict is None    # graceful degrade


# --------------------------------------------------------------------- #
# DelayedPostMortemScheduler
# --------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_post_mortem_scheduler_runs_after_delay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = RuleStore(json_path=tmp_path / "rules.json",
                       md_path=tmp_path / "rules.md")

    seen: list[tuple[str, int]] = []

    async def fake_run_post_mortem(**kwargs):
        seen.append((kwargs["symbol"], kwargs["target_ts_ms"]))
        # Build a minimal report-shaped object that pipeline.py will log.
        from altcoin_agent.learning_engine import (
            EventResult,
            PostMortemReport,
        )
        return PostMortemReport(
            symbol=kwargs["symbol"],
            target_ts_ms=kwargs["target_ts_ms"],
            result=EventResult(direction="pump", magnitude_pct=0.05,
                                minutes_to_extremum=10,
                                realized_at_ts_ms=kwargs["target_ts_ms"]),
            candidates=[],
            picks=[],
        )

    monkeypatch.setattr(pipeline_mod, "run_post_mortem", fake_run_post_mortem)

    sched = DelayedPostMortemScheduler(store=store, engine=None, delay_sec=0)
    task = sched.schedule(symbol="RAVEUSDT", target_ts_ms=12345)
    await asyncio.wait_for(task, timeout=2.0)

    assert seen == [("RAVEUSDT", 12345)]
    # Task should have been removed from the tracking set on completion.
    assert task not in sched._tasks


@pytest.mark.asyncio
async def test_post_mortem_scheduler_shutdown_cancels_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = RuleStore(json_path=tmp_path / "rules.json",
                       md_path=tmp_path / "rules.md")

    # If shutdown didn't cancel the pending sleep, we'd hang for 60s.
    async def never_called(**kwargs):
        pytest.fail("post-mortem should not run after shutdown cancels it")

    monkeypatch.setattr(pipeline_mod, "run_post_mortem", never_called)

    sched = DelayedPostMortemScheduler(store=store, engine=None, delay_sec=60)
    sched.schedule(symbol="RAVEUSDT", target_ts_ms=99)
    sched.schedule(symbol="DOGEUSDT", target_ts_ms=100)
    await asyncio.wait_for(sched.shutdown(), timeout=2.0)
    assert sched._tasks == set()


@pytest.mark.asyncio
async def test_post_mortem_scheduler_swallows_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = RuleStore(json_path=tmp_path / "rules.json",
                       md_path=tmp_path / "rules.md")

    async def boom(**kwargs):
        raise RuntimeError("OKX 500")

    monkeypatch.setattr(pipeline_mod, "run_post_mortem", boom)

    sched = DelayedPostMortemScheduler(store=store, engine=None, delay_sec=0)
    task = sched.schedule(symbol="RAVEUSDT", target_ts_ms=1)
    # Must NOT raise — pipeline swallows post-mortem errors.
    await asyncio.wait_for(task, timeout=2.0)




@pytest.mark.asyncio
async def test_post_mortem_scheduler_forwards_entry_and_direction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bug #2 fix: when ``schedule(entry_ts_ms=..., expected_direction=...)``
    is called, the scheduler must forward both kwargs into
    ``run_post_mortem`` so the entry-aware slicing + direction-aware
    result kick in."""
    store = RuleStore(json_path=tmp_path / "rules.json",
                       md_path=tmp_path / "rules.md")

    captured: dict = {}

    async def fake_run_post_mortem(**kwargs):
        captured.update(kwargs)
        from altcoin_agent.learning_engine import (
            EventResult,
            PostMortemReport,
        )
        return PostMortemReport(
            symbol=kwargs["symbol"],
            target_ts_ms=kwargs["target_ts_ms"],
            result=EventResult(direction="pump", magnitude_pct=-0.05,
                                minutes_to_extremum=10,
                                realized_at_ts_ms=kwargs["target_ts_ms"]),
            candidates=[], picks=[],
        )

    monkeypatch.setattr(pipeline_mod, "run_post_mortem", fake_run_post_mortem)

    sched = DelayedPostMortemScheduler(store=store, engine=None, delay_sec=0)
    task = sched.schedule(
        symbol="RAVEUSDT",
        target_ts_ms=12_345 + 3_600_000,
        entry_ts_ms=12_345,
        expected_direction="pump",
    )
    await asyncio.wait_for(task, timeout=2.0)
    assert captured["symbol"] == "RAVEUSDT"
    assert captured["entry_ts_ms"] == 12_345
    assert captured["expected_direction"] == "pump"
    # target_ts_ms is the moment we *evaluate* (entry + delay).
    assert captured["target_ts_ms"] == 12_345 + 3_600_000


@pytest.mark.asyncio
async def test_post_mortem_scheduler_omits_entry_kwargs_for_legacy_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Backward-compat: legacy callers without entry_ts_ms must NOT
    receive those kwargs (so the legacy code path inside
    ``run_post_mortem`` remains untouched)."""
    store = RuleStore(json_path=tmp_path / "rules.json",
                       md_path=tmp_path / "rules.md")
    captured: dict = {}

    async def fake_run_post_mortem(**kwargs):
        captured.update(kwargs)
        from altcoin_agent.learning_engine import (
            EventResult,
            PostMortemReport,
        )
        return PostMortemReport(
            symbol=kwargs["symbol"],
            target_ts_ms=kwargs["target_ts_ms"],
            result=EventResult("pump", 0.0, 0, kwargs["target_ts_ms"]),
            candidates=[], picks=[],
        )

    monkeypatch.setattr(pipeline_mod, "run_post_mortem", fake_run_post_mortem)
    sched = DelayedPostMortemScheduler(store=store, engine=None, delay_sec=0)
    task = sched.schedule(symbol="RAVEUSDT", target_ts_ms=12_345)
    await asyncio.wait_for(task, timeout=2.0)
    assert "entry_ts_ms" not in captured
    assert "expected_direction" not in captured



# --------------------------------------------------------------------- #
# DelayedPostMortemScheduler.record() — close-event-driven learning loop.
#
# Audit-fix Req #4 invariant: rule updates are anchored to the REAL trade
# outcome (entry_price, fill_price, realized_pnl_usdt), not to a fixed-time
# market-slice synthesis fired off the open handler. These tests pin that
# contract.
# --------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_post_mortem_record_uses_realized_result_no_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``record()`` must call ``run_post_mortem`` with the EventResult we
    built from the real trade — i.e. the synthetic OKX-slice path is
    skipped — and must do so IMMEDIATELY, not after sleeping ``delay_sec``.
    """
    store = RuleStore(json_path=tmp_path / "rules.json",
                       md_path=tmp_path / "rules.md")

    captured: dict = {}

    async def fake_run_post_mortem(**kwargs):
        captured.update(kwargs)
        from altcoin_agent.learning_engine import (
            EventResult,
            PostMortemReport,
        )
        rr = kwargs.get("realized_result")
        return PostMortemReport(
            symbol=kwargs["symbol"],
            target_ts_ms=kwargs["target_ts_ms"],
            result=rr or EventResult(
                direction="pump", magnitude_pct=0.0,
                minutes_to_extremum=0,
                realized_at_ts_ms=kwargs["target_ts_ms"],
            ),
            candidates=[],
            picks=[],
        )

    monkeypatch.setattr(pipeline_mod, "run_post_mortem", fake_run_post_mortem)

    # delay_sec=3600 to prove ``record()`` does NOT honour the legacy
    # delay path — if it did, the test would hang for an hour.
    sched = DelayedPostMortemScheduler(store=store, engine=None, delay_sec=3600)

    entry_ts = 1_700_000_000_000
    close_ts = entry_ts + 5 * 60 * 1000  # closed 5 min after open
    task = sched.record(
        symbol="RAVEUSDT",
        entry_ts_ms=entry_ts,
        close_ts_ms=close_ts,
        side="long",
        entry_price=1.0000,
        fill_price=0.9700,           # stopped out -3%
        realized_pnl_usdt=-30.0,
        realized_r=-1.0,
        close_reason="exchange_close_detected",
    )
    await asyncio.wait_for(task, timeout=2.0)

    # The post-mortem ran with our synthetic EventResult, not a network slice.
    assert captured["symbol"] == "RAVEUSDT"
    assert captured["target_ts_ms"] == close_ts
    assert captured["entry_ts_ms"] == entry_ts
    assert captured["expected_direction"] == "pump"   # LONG -> pump thesis
    rr = captured["realized_result"]
    assert rr.direction == "pump"
    # Stopped-out long: magnitude is the negative fractional move from entry.
    assert rr.magnitude_pct == pytest.approx(-0.03, abs=1e-9)
    assert rr.realized_at_ts_ms == close_ts


@pytest.mark.asyncio
async def test_post_mortem_record_short_thesis_negative_magnitude(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A SHORT that gets stopped out (price went UP) must record a
    ``dump`` direction with a NEGATIVE magnitude — proving the realised
    direction is the trader's intended thesis, not the market's actual
    move, so the rule store learns from misses on the side we bet."""
    store = RuleStore(json_path=tmp_path / "rules.json",
                       md_path=tmp_path / "rules.md")

    captured: dict = {}

    async def fake_run_post_mortem(**kwargs):
        captured.update(kwargs)
        from altcoin_agent.learning_engine import (
            EventResult,
            PostMortemReport,
        )
        rr = kwargs.get("realized_result")
        return PostMortemReport(
            symbol=kwargs["symbol"],
            target_ts_ms=kwargs["target_ts_ms"],
            result=rr or EventResult(
                direction="dump", magnitude_pct=0.0,
                minutes_to_extremum=0,
                realized_at_ts_ms=kwargs["target_ts_ms"],
            ),
            candidates=[],
            picks=[],
        )

    monkeypatch.setattr(pipeline_mod, "run_post_mortem", fake_run_post_mortem)

    sched = DelayedPostMortemScheduler(store=store, engine=None, delay_sec=0)
    entry_ts = 1_700_000_000_000
    close_ts = entry_ts + 12 * 60 * 1000
    task = sched.record(
        symbol="DOGEUSDT",
        entry_ts_ms=entry_ts,
        close_ts_ms=close_ts,
        side="short",
        entry_price=0.1000,
        fill_price=0.1050,           # short stopped out, +5% adverse
        realized_pnl_usdt=-50.0,
        realized_r=-1.0,
        close_reason="trailing_stop_fill",
    )
    await asyncio.wait_for(task, timeout=2.0)

    assert captured["expected_direction"] == "dump"
    rr = captured["realized_result"]
    assert rr.direction == "dump"
    assert rr.magnitude_pct == pytest.approx(-0.05, abs=1e-9)


@pytest.mark.asyncio
async def test_post_mortem_record_winner_positive_magnitude(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A LONG that wins (price went UP) records ``pump`` with POSITIVE
    magnitude — feeding hits into the rule store on the bullish features
    that fired this winning entry."""
    store = RuleStore(json_path=tmp_path / "rules.json",
                       md_path=tmp_path / "rules.md")

    captured: dict = {}

    async def fake_run_post_mortem(**kwargs):
        captured.update(kwargs)
        from altcoin_agent.learning_engine import (
            EventResult,
            PostMortemReport,
        )
        rr = kwargs.get("realized_result")
        return PostMortemReport(
            symbol=kwargs["symbol"],
            target_ts_ms=kwargs["target_ts_ms"],
            result=rr or EventResult(
                direction="pump", magnitude_pct=0.08,
                minutes_to_extremum=42,
                realized_at_ts_ms=kwargs["target_ts_ms"],
            ),
            candidates=[],
            picks=[],
        )

    monkeypatch.setattr(pipeline_mod, "run_post_mortem", fake_run_post_mortem)

    sched = DelayedPostMortemScheduler(store=store, engine=None, delay_sec=0)
    entry_ts = 1_700_000_000_000
    close_ts = entry_ts + 42 * 60 * 1000
    await asyncio.wait_for(
        sched.record(
            symbol="WIFUSDT",
            entry_ts_ms=entry_ts,
            close_ts_ms=close_ts,
            side="long",
            entry_price=1.0000,
            fill_price=1.0800,          # +8% trail-take
            realized_pnl_usdt=80.0,
            realized_r=2.7,
            close_reason="exchange_close_detected",
        ),
        timeout=2.0,
    )

    rr = captured["realized_result"]
    assert rr.direction == "pump"
    assert rr.magnitude_pct == pytest.approx(0.08, abs=1e-9)
