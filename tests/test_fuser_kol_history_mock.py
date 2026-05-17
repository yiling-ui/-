"""Integration tests for ``ScoreFuser`` × ``HistoricalAnalyzer``.

These tests assert that the analyzer hook in ``fuser.evaluate`` actually
flips the kol_exit branch decision when supplied with a calibrated
history, and that the V1.0 path (analyzer=None, no kol_authors) is
byte-for-byte unchanged.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from altcoin_agent.ai_engine import AIVerdict
from altcoin_agent.fuser import (
    FuserConfig,
    RuleIndex,
    ScoreFuser,
)
from altcoin_agent.screener import SignalEvent, SignalKind
from altcoin_agent.social.historical_analyzer import (
    HistoricalAnalyzer,
    HistoricalAnalyzerConfig,
    KOLHistoryStore,
)

# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #


def _make_signal_event(
    *, symbol="PEPE/USDT:USDT", exchange="binance",
    kind=SignalKind.OI_SILENT_BUILD, side_long=True, ts=1_700_000_000_000,
) -> SignalEvent:
    """Build a SignalEvent with a payload that resolves to LONG/SHORT.

    OI_SILENT_BUILD direction comes from from_price/to_price relationship:
    to_price > from_price -> LONG; to_price < from_price -> SHORT.
    """
    if side_long:
        payload = {
            "from_price": 1.00, "to_price": 1.10,
            "oi_delta_pct": 0.20, "duration_s": 600,
            "bar_close": 1.10,
        }
    else:
        payload = {
            "from_price": 1.10, "to_price": 1.00,
            "oi_delta_pct": 0.20, "duration_s": 600,
            "bar_close": 1.00,
        }
    return SignalEvent(
        ts=ts, symbol=symbol, exchange=exchange, kind=kind,
        payload=payload,
    )


def _make_verdict(
    *, intent="pump", confidence_score=55, kol_intent="exit_liquidity",
) -> AIVerdict:
    return AIVerdict(
        intent=intent,  # type: ignore[arg-type]
        confidence_score=confidence_score,
        reason="rule trigger fired",
        kol_intent=kol_intent,  # type: ignore[arg-type]
        key_evidence=[],
    )


def _make_fuser(
    *, tmp_path: Path, analyzer: HistoricalAnalyzer | None = None,
    cfg: FuserConfig | None = None,
) -> ScoreFuser:
    # Disable rule-index persistence by pointing it at a non-existent
    # path under tmp_path so the hot-reload throttle never fires.
    return ScoreFuser(
        config=cfg or FuserConfig(),
        rule_index=RuleIndex(json_path=tmp_path / "no_rules.json"),
        historical_analyzer=analyzer,
    )


def _seed_dumper(
    analyzer: HistoricalAnalyzer, *, author: str, hits: int, total: int,
) -> None:
    """Seed an exit_liquidity track record for ``author``."""
    for i in range(hits):
        analyzer.record_observation(
            author=author, symbol="X", intent="exit_liquidity",
            ts_ms=1_000 + i, realised_direction="dump",
            magnitude_pct=-0.05,
        )
    for i in range(total - hits):
        analyzer.record_observation(
            author=author, symbol="X", intent="exit_liquidity",
            ts_ms=2_000 + i, realised_direction="pump",
            magnitude_pct=0.03,
        )


# --------------------------------------------------------------------- #
# Backward compatibility: V1.0 path is unchanged
# --------------------------------------------------------------------- #


def test_evaluate_unchanged_when_analyzer_is_none(tmp_path: Path):
    fuser = _make_fuser(tmp_path=tmp_path, analyzer=None)
    ev = _make_signal_event(side_long=True)
    fuser._rules.setdefault(
        fuser._key(ev.exchange, ev.symbol), __import__("collections").deque(),
    ).append(ev)
    verdict = _make_verdict(confidence_score=55, kol_intent="exit_liquidity")
    fuser._llm[fuser._key(ev.exchange, ev.symbol)] = verdict
    fuser._llm_ts[fuser._key(ev.exchange, ev.symbol)] = ev.ts
    sig = fuser.evaluate(ev.symbol, ev.exchange, ev.ts)
    # Confidence 0.55 < 0.70 hard-veto threshold -> SOFT CAP path applies
    # but does NOT block. Final score capped at kol_exit_soft_cap_score=70.
    assert not sig.blocked
    assert sig.final_score <= fuser.cfg.kol_exit_soft_cap_score
    # No "kol_history" annotation in notes when analyzer was off.
    assert all("kol_history" not in n for n in sig.notes)


@pytest.mark.asyncio
async def test_legacy_on_llm_verdict_signature_still_works(tmp_path: Path):
    """A caller that omits ``kol_authors`` (the V1.0 signature) must
    still produce a usable FusedSignal — proves backward-compat at the
    API surface."""
    fuser = _make_fuser(tmp_path=tmp_path, analyzer=None)
    ev = _make_signal_event(side_long=True)
    await fuser.on_rule_signal(ev)
    verdict = _make_verdict(confidence_score=80, kol_intent="neutral")
    sig = await fuser.on_llm_verdict(
        ev.exchange, ev.symbol, verdict, ev.ts,
    )
    assert sig is not None
    # Legacy callers shouldn't surface kol_history notes.
    assert all("kol_history" not in n for n in sig.notes)


# --------------------------------------------------------------------- #
# History promotes a borderline verdict into the hard-veto branch
# --------------------------------------------------------------------- #


def test_known_dumper_promotes_below_threshold_to_hard_veto(tmp_path: Path):
    """Confidence 0.55 + known-bad caller -> analyzer lifts to >= 0.70 ->
    fuser hits the hard-veto branch instead of the soft cap."""
    store = KOLHistoryStore(path=tmp_path / "kol.json", autosave=False)
    analyzer = HistoricalAnalyzer(
        store=store,
        config=HistoricalAnalyzerConfig(
            min_samples=10, strong_bound=0.65, conf_lift_max=0.20,
        ),
    )
    # 20-of-20 perfect dumper (Laplace ≈ 21/22 = 0.955 -> full lift).
    _seed_dumper(analyzer, author="dumper_x", hits=20, total=20)
    fuser = _make_fuser(tmp_path=tmp_path, analyzer=analyzer)
    ev = _make_signal_event(side_long=True)
    # Inject the rule signal AND the verdict-with-authors.
    import asyncio
    asyncio.run(fuser.on_rule_signal(ev))
    verdict = _make_verdict(confidence_score=55, kol_intent="exit_liquidity")
    asyncio.run(fuser.on_llm_verdict(
        ev.exchange, ev.symbol, verdict, ev.ts,
        kol_authors=["dumper_x"],
    ))
    sig = fuser.evaluate(ev.symbol, ev.exchange, ev.ts)
    assert sig.blocked is True
    assert sig.block_reason == "kol_exit_liquidity_hard_veto"
    assert any("kol_history adjusted confidence" in n for n in sig.notes)


def test_unreliable_caller_demotes_above_threshold_to_soft_cap(tmp_path: Path):
    """Confidence 0.72 + known-unreliable caller -> analyzer drops to
    < 0.70 -> fuser falls into the soft-cap path."""
    store = KOLHistoryStore(path=tmp_path / "kol.json", autosave=False)
    analyzer = HistoricalAnalyzer(
        store=store,
        config=HistoricalAnalyzerConfig(
            min_samples=10, weak_bound=0.40, conf_drop_max=0.20,
        ),
    )
    # 0/20 hits ≈ 0.045 Laplace -> well below weak_bound.
    _seed_dumper(analyzer, author="permabear", hits=0, total=20)
    fuser = _make_fuser(tmp_path=tmp_path, analyzer=analyzer)
    ev = _make_signal_event(side_long=True)
    import asyncio
    asyncio.run(fuser.on_rule_signal(ev))
    verdict = _make_verdict(confidence_score=72, kol_intent="exit_liquidity")
    asyncio.run(fuser.on_llm_verdict(
        ev.exchange, ev.symbol, verdict, ev.ts,
        kol_authors=["permabear"],
    ))
    sig = fuser.evaluate(ev.symbol, ev.exchange, ev.ts)
    assert sig.blocked is False
    # Soft cap pinned the final score at <= 70.
    assert sig.final_score <= fuser.cfg.kol_exit_soft_cap_score
    assert any("kol_history adjusted confidence" in n for n in sig.notes)


# --------------------------------------------------------------------- #
# Edge cases: insufficient samples, neutral intent, no authors
# --------------------------------------------------------------------- #


def test_below_min_samples_keeps_original_decision(tmp_path: Path):
    """A KOL with only 3 observations doesn't move the needle —
    confidence stays at 0.55, soft cap (not hard veto) applies."""
    store = KOLHistoryStore(path=tmp_path / "kol.json", autosave=False)
    analyzer = HistoricalAnalyzer(
        store=store,
        config=HistoricalAnalyzerConfig(min_samples=10),
    )
    _seed_dumper(analyzer, author="newbie", hits=3, total=3)
    fuser = _make_fuser(tmp_path=tmp_path, analyzer=analyzer)
    ev = _make_signal_event(side_long=True)
    import asyncio
    asyncio.run(fuser.on_rule_signal(ev))
    verdict = _make_verdict(confidence_score=55, kol_intent="exit_liquidity")
    asyncio.run(fuser.on_llm_verdict(
        ev.exchange, ev.symbol, verdict, ev.ts,
        kol_authors=["newbie"],
    ))
    sig = fuser.evaluate(ev.symbol, ev.exchange, ev.ts)
    assert sig.blocked is False
    # Note: the analyzer ran but had no effect; we surface that.
    assert any("kol_history evaluated (no change)" in n for n in sig.notes)


def test_no_authors_supplied_keeps_original_decision(tmp_path: Path):
    """When LLMConsultor sees a degraded social fetch, it forwards an
    empty author list. The analyzer hook MUST be a no-op rather than
    raising or silently using stale authors."""
    store = KOLHistoryStore(path=tmp_path / "kol.json", autosave=False)
    analyzer = HistoricalAnalyzer(store=store)
    fuser = _make_fuser(tmp_path=tmp_path, analyzer=analyzer)
    ev = _make_signal_event(side_long=True)
    import asyncio
    asyncio.run(fuser.on_rule_signal(ev))
    verdict = _make_verdict(confidence_score=55, kol_intent="exit_liquidity")
    asyncio.run(fuser.on_llm_verdict(
        ev.exchange, ev.symbol, verdict, ev.ts,
        # An empty list -> cache cleared.
        kol_authors=[],
    ))
    sig = fuser.evaluate(ev.symbol, ev.exchange, ev.ts)
    assert sig.blocked is False
    assert all("kol_history" not in n for n in sig.notes)


def test_neutral_kol_intent_skips_analyzer(tmp_path: Path):
    """neutral kol_intent -> analyzer hook should not run at all
    (no notes, no adjustment)."""
    store = KOLHistoryStore(path=tmp_path / "kol.json", autosave=False)
    analyzer = HistoricalAnalyzer(store=store)
    fuser = _make_fuser(tmp_path=tmp_path, analyzer=analyzer)
    ev = _make_signal_event(side_long=True)
    import asyncio
    asyncio.run(fuser.on_rule_signal(ev))
    verdict = _make_verdict(confidence_score=80, kol_intent="neutral")
    asyncio.run(fuser.on_llm_verdict(
        ev.exchange, ev.symbol, verdict, ev.ts,
        kol_authors=["any_author"],
    ))
    sig = fuser.evaluate(ev.symbol, ev.exchange, ev.ts)
    # neutral intent -> no kol-related notes / no veto.
    assert sig.blocked is False
    assert all("kol_history" not in n for n in sig.notes)


# --------------------------------------------------------------------- #
# Resilience: a buggy analyzer must NEVER block the trading hot path
# --------------------------------------------------------------------- #


def test_buggy_analyzer_falls_back_to_unadjusted_confidence(tmp_path: Path):
    """If the analyzer raises, the fuser must catch + log + use the
    LLM's original confidence (the V1.0 path) instead of crashing the
    hot path."""

    class BoomAnalyzer:
        def adjust_kol_confidence(self, **_kw):  # noqa: ANN001
            raise RuntimeError("analyzer is broken")

    fuser = _make_fuser(
        tmp_path=tmp_path,
        analyzer=BoomAnalyzer(),  # type: ignore[arg-type]
    )
    ev = _make_signal_event(side_long=True)
    import asyncio
    asyncio.run(fuser.on_rule_signal(ev))
    verdict = _make_verdict(confidence_score=80, kol_intent="exit_liquidity")
    asyncio.run(fuser.on_llm_verdict(
        ev.exchange, ev.symbol, verdict, ev.ts,
        kol_authors=["whoever"],
    ))
    # Should not raise.
    sig = fuser.evaluate(ev.symbol, ev.exchange, ev.ts)
    # Falls back to verdict.confidence=0.80 -> hard veto fires (>= 0.70).
    assert sig.blocked is True
    assert sig.block_reason == "kol_exit_liquidity_hard_veto"


# --------------------------------------------------------------------- #
# Author-cache lifecycle: stale entries cleared on explicit empty
# --------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_author_cache_cleared_on_explicit_empty_list(tmp_path: Path):
    fuser = _make_fuser(tmp_path=tmp_path)
    ev = _make_signal_event()
    key = fuser._key(ev.exchange, ev.symbol)
    verdict = _make_verdict()
    # First feed: authors present.
    await fuser.on_llm_verdict(
        ev.exchange, ev.symbol, verdict, ev.ts,
        kol_authors=["a", "b"],
    )
    assert fuser._llm_authors[key] == ["a", "b"]
    # Second feed with explicit empty list -> cache cleared.
    await fuser.on_llm_verdict(
        ev.exchange, ev.symbol, verdict, ev.ts + 1,
        kol_authors=[],
    )
    assert key not in fuser._llm_authors
    # Third feed with None (legacy callers) -> cache untouched (stays
    # cleared in this case because we just removed it, but if there
    # had been entries, they'd persist).
    await fuser.on_llm_verdict(
        ev.exchange, ev.symbol, verdict, ev.ts + 2,
    )
    assert key not in fuser._llm_authors
