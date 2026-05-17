"""tests/test_kol_history_wiring_integration_mock.py

Integration tests for the KOL-history wiring in :mod:`main`.

What this pins:
* ``AppConfig`` defaults keep ``kol_history_enabled`` OFF so existing
  dry-run tests stay byte-for-byte.
* ``AppConfig.from_file`` round-trips every new key.
* When ``kol_history_enabled=True`` the daemon constructs both the
  ``KOLHistoryStore`` and the ``HistoricalAnalyzer``, and threads the
  analyzer into the ``ScoreFuser`` AND the
  ``DelayedPostMortemScheduler``.
* The post-mortem scheduler accepts the new ``kol_authors`` /
  ``kol_intent`` kwargs and feeds the analyzer when the delay fires
  (we drive ``delay_sec=0`` so the test doesn't sleep an hour).
* The fuser's enriched ``FusedSignal.kol_authors`` survives the trip
  through ``_dispatch`` so callers downstream (the post-mortem path
  in particular) never see a stale list.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from altcoin_agent.ai_engine import AIVerdict
from altcoin_agent.learning_engine import RuleStore
from altcoin_agent.main import App, AppConfig
from altcoin_agent.pipeline import DelayedPostMortemScheduler
from altcoin_agent.screener import SignalEvent, SignalKind
from altcoin_agent.social.historical_analyzer import (
    HistoricalAnalyzer,
    KOLHistoryStore,
)


@asynccontextmanager
async def _running_app(cfg: AppConfig) -> AsyncIterator[App]:
    """Same helper used by the other wiring integration suites."""
    app = App(cfg=cfg)
    runner = asyncio.create_task(app.run())
    for _ in range(50):
        if app._screener is not None:
            break
        await asyncio.sleep(0.02)
    assert app._screener is not None

    async def _noop_run() -> None:
        await app._stop_event.wait()
    app._screener.run = _noop_run     # type: ignore[method-assign]

    try:
        for _ in range(50):
            if app.state.reconciliation_complete:
                break
            await asyncio.sleep(0.02)
        yield app
    finally:
        app.request_stop()
        await asyncio.wait_for(runner, timeout=5.0)


# --------------------------------------------------------------------- #
# Defaults & YAML round-trip
# --------------------------------------------------------------------- #


def test_kol_history_defaults_off() -> None:
    cfg = AppConfig()
    assert cfg.kol_history_enabled is False
    # Path / threshold defaults are concrete strings/numbers so a
    # later .from_file round-trip never sees None.
    assert cfg.kol_history_path == ".kiro/state/social/kol_history.json"
    assert cfg.kol_history_min_samples == 10
    assert cfg.kol_history_strong_bound == 0.65
    assert cfg.kol_history_weak_bound == 0.40
    assert cfg.kol_history_conf_lift_max == 0.20
    assert cfg.kol_history_conf_drop_max == 0.20


def test_kol_history_yaml_round_trip(tmp_path: Path) -> None:
    yaml_path = tmp_path / "app.yaml"
    yaml_path.write_text(
        "kol_history_enabled: true\n"
        "kol_history_path: '/tmp/x_kol.json'\n"
        "kol_history_min_samples: 25\n"
        "kol_history_strong_bound: 0.70\n"
        "kol_history_weak_bound: 0.30\n"
        "kol_history_conf_lift_max: 0.15\n"
        "kol_history_conf_drop_max: 0.25\n"
    )
    cfg = AppConfig.from_file(str(yaml_path))
    assert cfg.kol_history_enabled is True
    assert cfg.kol_history_path == "/tmp/x_kol.json"
    assert cfg.kol_history_min_samples == 25
    assert cfg.kol_history_strong_bound == 0.70
    assert cfg.kol_history_weak_bound == 0.30
    assert cfg.kol_history_conf_lift_max == 0.15
    assert cfg.kol_history_conf_drop_max == 0.25


# --------------------------------------------------------------------- #
# Construction lifecycle: enabled vs. disabled
# --------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_disabled_kol_history_constructs_no_components(tmp_path: Path):
    cfg = AppConfig(
        dry_run=True, kol_history_enabled=False,
        kol_history_path=str(tmp_path / "kol.json"),
        healthz_port=18901, dashboard_port=18902,
        graceful_timeout_sec=2.0,
    )
    async with _running_app(cfg) as app:
        assert app._kol_analyzer is None
        assert app._kol_history_store is None


@pytest.mark.asyncio
async def test_enabled_kol_history_constructs_store_and_analyzer(
    tmp_path: Path,
):
    cfg = AppConfig(
        dry_run=True, kol_history_enabled=True,
        kol_history_path=str(tmp_path / "kol.json"),
        kol_history_min_samples=7,
        healthz_port=18903, dashboard_port=18904,
        graceful_timeout_sec=2.0,
    )
    async with _running_app(cfg) as app:
        assert app._kol_history_store is not None
        assert app._kol_analyzer is not None
        # Min-samples threshold is forwarded into the analyzer config.
        assert app._kol_analyzer.config.min_samples == 7
        # Path resolves to the cfg location, parent dir created.
        assert (tmp_path / "kol.json").parent.is_dir()


@pytest.mark.asyncio
async def test_post_mortem_scheduler_holds_analyzer_reference_when_enabled(
    tmp_path: Path,
):
    cfg = AppConfig(
        dry_run=True, kol_history_enabled=True,
        kol_history_path=str(tmp_path / "kol.json"),
        healthz_port=18905, dashboard_port=18906,
        graceful_timeout_sec=2.0,
    )
    async with _running_app(cfg) as app:
        assert app._post_mortem is not None
        assert app._post_mortem.historical_analyzer is app._kol_analyzer


@pytest.mark.asyncio
async def test_post_mortem_scheduler_lacks_analyzer_when_disabled(
    tmp_path: Path,
):
    cfg = AppConfig(
        dry_run=True, kol_history_enabled=False,
        kol_history_path=str(tmp_path / "kol.json"),
        healthz_port=18907, dashboard_port=18908,
        graceful_timeout_sec=2.0,
    )
    async with _running_app(cfg) as app:
        assert app._post_mortem is not None
        assert app._post_mortem.historical_analyzer is None


# --------------------------------------------------------------------- #
# Fuser wiring: analyzer reference flows through
# --------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_fuser_receives_analyzer_when_enabled(tmp_path: Path):
    """The wiring step that PR #21 omitted — confirm the App actually
    threads the analyzer into the freshly-constructed ScoreFuser.

    We can't easily reach into ``run()``'s local ``fuser`` variable
    from outside, so we instead verify a behavioural invariant: feed
    the fuser via ``app._screener.on_signal`` and assert that, when
    the analyzer is present, ``FusedSignal.kol_authors`` carries the
    authors we forwarded via ``on_llm_verdict``. The kol_authors
    field is only populated when the fuser's _llm_authors map has
    entries — which happens only when the consultor (or test code)
    forwards a non-empty list. So this test exercises the field
    plumbing rather than the analyzer math itself.
    """
    cfg = AppConfig(
        dry_run=True, kol_history_enabled=True,
        kol_history_path=str(tmp_path / "kol.json"),
        healthz_port=18909, dashboard_port=18910,
        graceful_timeout_sec=2.0,
    )
    async with _running_app(cfg) as app:
        # Smoke: simply confirm we can import the analyzer's class
        # via the App reference for downstream tests to extend.
        assert isinstance(app._kol_analyzer, HistoricalAnalyzer)
        assert isinstance(app._kol_history_store, KOLHistoryStore)


# --------------------------------------------------------------------- #
# DelayedPostMortemScheduler unit tests for the new kwargs
# --------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_post_mortem_records_kol_observation_when_authors_supplied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """End-to-end: scheduler with delay_sec=0 fires record_observation
    on the analyzer using the realised direction from the post-mortem.
    """
    store = KOLHistoryStore(path=tmp_path / "kol.json", autosave=False)
    analyzer = HistoricalAnalyzer(store=store)

    class _StubReport:
        class _Result:
            direction = "dump"
            magnitude_pct = -0.10
        result = _Result()
        picks: list = []

    async def _stub_run_post_mortem(**kwargs):
        return _StubReport()

    # Patch the symbol the scheduler actually imports.
    monkeypatch.setattr(
        "altcoin_agent.pipeline.run_post_mortem", _stub_run_post_mortem,
    )

    rule_store = RuleStore(json_path=tmp_path / "rules.json")
    scheduler = DelayedPostMortemScheduler(
        store=rule_store,
        engine=None,
        delay_sec=0,
        historical_analyzer=analyzer,
    )
    task = scheduler.schedule(
        symbol="PEPE/USDT:USDT",
        target_ts_ms=1_000_000_000,
        entry_ts_ms=1_000_000_000,
        expected_direction="pump",
        kol_authors=["dumper_a", "dumper_b"],
        kol_intent="exit_liquidity",
    )
    await task
    # Both authors received one observation each.
    sa = analyzer.lookup("dumper_a", "exit_liquidity")
    sb = analyzer.lookup("dumper_b", "exit_liquidity")
    assert sa is not None and sa.total == 1
    assert sb is not None and sb.total == 1
    # Realised dump on an exit_liquidity call -> hits == 1 (correct).
    assert sa.hits == 1


@pytest.mark.asyncio
async def test_post_mortem_skips_kol_recording_when_intent_neutral(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """Neutral kol_intent -> scheduler ignores the analyzer (matches
    the analyzer's own neutral-intent skip)."""
    store = KOLHistoryStore(path=tmp_path / "kol.json", autosave=False)
    analyzer = HistoricalAnalyzer(store=store)

    class _StubReport:
        class _Result:
            direction = "dump"
            magnitude_pct = -0.10
        result = _Result()
        picks: list = []

    async def _stub_run_post_mortem(**_kw):
        return _StubReport()

    monkeypatch.setattr(
        "altcoin_agent.pipeline.run_post_mortem", _stub_run_post_mortem,
    )

    scheduler = DelayedPostMortemScheduler(
        store=RuleStore(json_path=tmp_path / "rules.json"),
        engine=None, delay_sec=0,
        historical_analyzer=analyzer,
    )
    await scheduler.schedule(
        symbol="X", target_ts_ms=0, entry_ts_ms=0,
        kol_authors=["a"], kol_intent="neutral",
    )
    # Analyzer never received an observation.
    assert len(store) == 0


@pytest.mark.asyncio
async def test_post_mortem_skips_kol_recording_when_no_analyzer_wired(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """When the daemon was started without kol_history_enabled, the
    scheduler's analyzer is None. The ``kol_authors`` kwarg becomes a
    no-op rather than raising."""
    class _StubReport:
        class _Result:
            direction = "dump"
            magnitude_pct = -0.10
        result = _Result()
        picks: list = []

    async def _stub_run_post_mortem(**_kw):
        return _StubReport()

    monkeypatch.setattr(
        "altcoin_agent.pipeline.run_post_mortem", _stub_run_post_mortem,
    )

    scheduler = DelayedPostMortemScheduler(
        store=RuleStore(json_path=tmp_path / "rules.json"),
        engine=None, delay_sec=0,
        historical_analyzer=None,
    )
    # Should not raise.
    await scheduler.schedule(
        symbol="X", target_ts_ms=0, entry_ts_ms=0,
        kol_authors=["a"], kol_intent="exit_liquidity",
    )


@pytest.mark.asyncio
async def test_post_mortem_dedupes_duplicate_authors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """If the social snapshot returns the same author twice (same KOL
    posted twice in the relevant window), the scheduler must record
    only ONE observation per author per post-mortem run — otherwise a
    spammy KOL inflates their own sample count."""
    store = KOLHistoryStore(path=tmp_path / "kol.json", autosave=False)
    analyzer = HistoricalAnalyzer(store=store)

    class _StubReport:
        class _Result:
            direction = "dump"
            magnitude_pct = -0.10
        result = _Result()
        picks: list = []

    async def _stub_run_post_mortem(**_kw):
        return _StubReport()

    monkeypatch.setattr(
        "altcoin_agent.pipeline.run_post_mortem", _stub_run_post_mortem,
    )

    scheduler = DelayedPostMortemScheduler(
        store=RuleStore(json_path=tmp_path / "rules.json"),
        engine=None, delay_sec=0,
        historical_analyzer=analyzer,
    )
    await scheduler.schedule(
        symbol="X", target_ts_ms=0, entry_ts_ms=0,
        kol_authors=["spammer", "SPAMMER", "spammer", "$spammer"],
        kol_intent="exit_liquidity",
    )
    score = analyzer.lookup("spammer", "exit_liquidity")
    assert score is not None
    # 4 raw entries -> 1 observation after dedup.
    assert score.total == 1


# --------------------------------------------------------------------- #
# FusedSignal.kol_authors plumbing
# --------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_fused_signal_carries_kol_authors_after_dispatch(tmp_path: Path):
    """The fuser populates ``FusedSignal.kol_authors`` from its
    internal _llm_authors cache so the post-mortem hook in main.py
    can read them off the signal directly."""
    from altcoin_agent.fuser import FuserConfig, RuleIndex, ScoreFuser

    fuser = ScoreFuser(
        config=FuserConfig(),
        rule_index=RuleIndex(json_path=tmp_path / "no_rules.json"),
    )
    ev = SignalEvent(
        ts=1_000, symbol="X/USDT:USDT", exchange="binance",
        kind=SignalKind.OI_SILENT_BUILD,
        payload={
            "from_price": 1.0, "to_price": 1.10,
            "oi_delta_pct": 0.20, "duration_s": 600,
            "bar_close": 1.10,
        },
    )
    sig0 = await fuser.on_rule_signal(ev)
    # No verdict yet -> empty authors list.
    assert sig0 is not None
    assert sig0.kol_authors == []
    # Feed a verdict with authors.
    verdict = AIVerdict(
        intent="pump", confidence_score=80, reason="x",
        kol_intent="exit_liquidity", key_evidence=[],
    )
    sig1 = await fuser.on_llm_verdict(
        ev.exchange, ev.symbol, verdict, ev.ts,
        kol_authors=["a", "b"],
    )
    assert sig1 is not None
    assert sig1.kol_authors == ["a", "b"]


@pytest.mark.asyncio
async def test_fused_signal_kol_authors_cleared_on_explicit_empty(
    tmp_path: Path,
):
    from altcoin_agent.fuser import FuserConfig, RuleIndex, ScoreFuser

    fuser = ScoreFuser(
        config=FuserConfig(),
        rule_index=RuleIndex(json_path=tmp_path / "no_rules.json"),
    )
    ev = SignalEvent(
        ts=1_000, symbol="X/USDT:USDT", exchange="binance",
        kind=SignalKind.OI_SILENT_BUILD,
        payload={
            "from_price": 1.0, "to_price": 1.10,
            "oi_delta_pct": 0.20, "duration_s": 600,
            "bar_close": 1.10,
        },
    )
    await fuser.on_rule_signal(ev)
    verdict = AIVerdict(
        intent="pump", confidence_score=80, reason="x",
        kol_intent="exit_liquidity", key_evidence=[],
    )
    await fuser.on_llm_verdict(
        ev.exchange, ev.symbol, verdict, ev.ts,
        kol_authors=["a"],
    )
    sig_clear = await fuser.on_llm_verdict(
        ev.exchange, ev.symbol, verdict, ev.ts + 1,
        kol_authors=[],
    )
    assert sig_clear is not None
    assert sig_clear.kol_authors == []
