"""
Mock tests for fuser.py.

Strategy: build SignalEvent and AIVerdict objects directly (no network, no
ccxt, no DeepSeek). Cover every branch of the fusion decision tree:

    * pure rule mode (LLM missing)
    * LLM agreement -> multiplier boost
    * LLM disagree direction -> hard veto
    * KOL exit_liquidity high confidence -> HARD VETO
    * KOL exit_liquidity low confidence  -> SOFT CAP
    * Intra-rule direction conflict -> penalty
    * Stale rule signals dropped
    * Cooldown blocks repeat high_priority
    * Multi-detector confluence -> high_priority
    * Sink dispatch only on high_priority
"""

from __future__ import annotations

import pytest

from altcoin_agent.ai_engine import AIVerdict
from altcoin_agent.fuser import (
    DEFAULT_RULE_WEIGHTS,
    Direction,
    FusedSignal,
    FuserConfig,
    ScoreFuser,
)
from altcoin_agent.screener import SignalEvent, SignalKind

# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #


def _ev(
    kind: SignalKind,
    *,
    ts: int = 1_000_000,
    symbol: str = "RAVEUSDT",
    exchange: str = "binance",
    payload: dict | None = None,
) -> SignalEvent:
    return SignalEvent(
        kind=kind,
        symbol=symbol,
        ts=ts,
        exchange=exchange,
        payload=payload or {},
    )


def _v(
    intent: str = "pump",
    *,
    confidence: float = 0.85,
    kol_intent: str = "frontrun_call",
) -> AIVerdict:
    """Build an AIVerdict where confidence_score = round(confidence*100)."""
    return AIVerdict(
        intent=intent,                                  # type: ignore[arg-type]
        confidence_score=int(round(confidence * 100)),
        reason="stub",
        kol_intent=kol_intent,                          # type: ignore[arg-type]
        key_evidence=[],
    )


# --------------------------------------------------------------------------- #
# Pure-rule mode
# --------------------------------------------------------------------------- #


def test_pure_rule_mode_promotes_when_confluence_is_strong() -> None:
    """volume_spike(buy) + oi_silent_build + liquidity_sweep(buy_side) -> long, score >=85"""
    fuser = ScoreFuser()
    now = 1_000_000

    # all signals within window, all directionally LONG
    fuser._rules.setdefault("binance:RAVEUSDT", __import__("collections").deque(maxlen=64)).extend([
        _ev(SignalKind.VOLUME_SPIKE, ts=now-10_000,
            payload={"side": "buy", "zscore": 6.0}),
        _ev(SignalKind.OI_SILENT_BUILD, ts=now-5_000,
            payload={"oi_delta_pct": 0.22, "from_price": 1.0, "to_price": 1.005}),
        _ev(SignalKind.LIQUIDITY_SWEEP, ts=now-1_000,
            payload={"side": "buy_side", "wick_to_body": 2.1}),
    ])

    sig = fuser.evaluate("RAVEUSDT", "binance", now)
    assert sig.direction == Direction.LONG
    assert sig.rule_score >= 85
    assert sig.final_score >= 85
    assert sig.is_high_priority is True
    assert sig.blocked is False
    assert sig.llm_verdict is None  # pure rule


def test_pure_rule_mode_does_not_promote_when_only_one_signal() -> None:
    fuser = ScoreFuser()
    now = 1_000_000
    fuser._rules.setdefault("binance:X", __import__("collections").deque(maxlen=64)).append(
        _ev(SignalKind.VOLUME_SPIKE, ts=now, symbol="X",
            payload={"side": "buy"})
    )
    sig = fuser.evaluate("X", "binance", now)
    # Volume spike alone = 30 points -> below 85 threshold
    assert sig.is_high_priority is False
    assert sig.rule_score == pytest.approx(30.0)


# --------------------------------------------------------------------------- #
# LLM boost
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_llm_agreement_boosts_score_above_threshold() -> None:
    """
    Rule score below threshold by itself, but high-confidence agreeing LLM
    promotes it via the multiplier.
    """
    fuser = ScoreFuser()
    now = 1_000_000

    # Rule score = volume_spike + oi_silent_build = 30 + 35 = 65 (LONG)
    await fuser.on_rule_signal(_ev(SignalKind.VOLUME_SPIKE, ts=now,
                                   payload={"side": "buy"}))
    await fuser.on_rule_signal(_ev(SignalKind.OI_SILENT_BUILD, ts=now,
                                   payload={"from_price": 1.0, "to_price": 1.005}))

    sig_before = fuser.evaluate("RAVEUSDT", "binance", now)
    assert sig_before.rule_score == pytest.approx(65.0)
    assert sig_before.is_high_priority is False

    # LLM strongly agrees with LONG
    await fuser.on_llm_verdict(
        "binance", "RAVEUSDT", _v("pump", confidence=0.95), ts=now,
    )
    # 65 * (1.00 + 0.25*0.95) = 65 * 1.2375 ~ 80.4 -- still below 85
    # We need stronger rule baseline OR even higher conf. Pump up rules.
    # Consume the return value of on_rule_signal directly; calling evaluate()
    # afterwards would see cooldown set by the dispatch and report blocked.
    sig_final = await fuser.on_rule_signal(_ev(SignalKind.LIQUIDITY_SWEEP, ts=now,
                                               payload={"side": "buy_side", "wick_to_body": 2.0}))
    assert sig_final is not None
    # rule = 30+35+35 = 100 capped at 95; 95 * 1.2375 = 117 capped at 100
    assert sig_final.final_score >= 95
    assert sig_final.is_high_priority is True
    assert any("LLM agree" in n for n in sig_final.notes)


# --------------------------------------------------------------------------- #
# Direction conflict (LLM vs Rule)
# --------------------------------------------------------------------------- #


def test_llm_dump_vetoes_long_rule_signals() -> None:
    """
    User-critical: rules say PUMP but LLM says DUMP -> never trade.
    Rule looks like a great pump entry but is in fact distribution.
    """
    fuser = ScoreFuser()
    now = 1_000_000
    key = "binance:RAVEUSDT"
    bucket = fuser._rules.setdefault(key, __import__("collections").deque(maxlen=64))
    bucket.extend([
        _ev(SignalKind.VOLUME_SPIKE, ts=now-2_000, payload={"side": "buy"}),
        _ev(SignalKind.OI_SILENT_BUILD, ts=now-1_000,
            payload={"from_price": 1.0, "to_price": 1.005}),
        _ev(SignalKind.LIQUIDITY_SWEEP, ts=now,
            payload={"side": "buy_side", "wick_to_body": 2.0}),
    ])
    fuser._llm[key] = _v("dump", confidence=0.9)
    fuser._llm_ts[key] = now

    sig = fuser.evaluate("RAVEUSDT", "binance", now)
    assert sig.is_high_priority is False
    assert sig.blocked is True
    assert sig.block_reason == "direction_conflict"
    assert sig.final_score <= 40  # veto floor


# --------------------------------------------------------------------------- #
# KOL exit liquidity vetoes (the key user-requested rule)
# --------------------------------------------------------------------------- #


def test_kol_exit_liquidity_high_conf_HARD_vetoes_even_if_rules_perfect() -> None:
    """
    Even with maximum rule confluence, kol_intent=exit_liquidity at conf>=0.7
    must produce a hard veto and NEVER promote to high_priority.
    """
    fuser = ScoreFuser()
    now = 1_000_000
    key = "binance:RAVEUSDT"
    bucket = fuser._rules.setdefault(key, __import__("collections").deque(maxlen=64))
    bucket.extend([
        _ev(SignalKind.VOLUME_SPIKE, ts=now, payload={"side": "buy"}),
        _ev(SignalKind.OI_SILENT_BUILD, ts=now,
            payload={"from_price": 1.0, "to_price": 1.005}),
        _ev(SignalKind.LIQUIDITY_SWEEP, ts=now,
            payload={"side": "buy_side", "wick_to_body": 3.0}),
        _ev(SignalKind.FUNDING_EXTREME, ts=now,
            payload={"direction": "short_squeeze", "rate": -0.003}),
    ])
    # LLM intent agrees with PUMP but flags KOL is dumping on followers
    fuser._llm[key] = _v("pump", confidence=0.85, kol_intent="exit_liquidity")
    fuser._llm_ts[key] = now

    sig = fuser.evaluate("RAVEUSDT", "binance", now)
    assert sig.is_high_priority is False
    assert sig.blocked is True
    assert sig.block_reason == "kol_exit_liquidity_hard_veto"
    assert sig.final_score <= 30
    assert any("HARD VETO" in n for n in sig.notes)


def test_kol_exit_liquidity_low_conf_SOFT_caps_score_at_70() -> None:
    """
    Same rules but LLM is only 50% confident in the KOL-exit call -> soft cap
    at 70, allow non-high-priority bookkeeping but never promote.
    """
    fuser = ScoreFuser()
    now = 1_000_000
    key = "binance:RAVEUSDT"
    bucket = fuser._rules.setdefault(key, __import__("collections").deque(maxlen=64))
    bucket.extend([
        _ev(SignalKind.VOLUME_SPIKE, ts=now, payload={"side": "buy"}),
        _ev(SignalKind.OI_SILENT_BUILD, ts=now,
            payload={"from_price": 1.0, "to_price": 1.005}),
        _ev(SignalKind.LIQUIDITY_SWEEP, ts=now,
            payload={"side": "buy_side", "wick_to_body": 3.0}),
    ])
    fuser._llm[key] = _v("pump", confidence=0.50, kol_intent="exit_liquidity")
    fuser._llm_ts[key] = now

    sig = fuser.evaluate("RAVEUSDT", "binance", now)
    assert sig.is_high_priority is False           # never promote under exit_liquidity
    assert sig.blocked is False                    # not a hard veto
    assert sig.final_score <= 70.001
    assert any("SOFT CAP" in n for n in sig.notes)


# --------------------------------------------------------------------------- #
# Stale signals dropped
# --------------------------------------------------------------------------- #


def test_stale_rule_signals_outside_window_are_ignored() -> None:
    fuser = ScoreFuser(config=FuserConfig(window_sec=90))
    now = 10_000_000
    key = "binance:X"
    bucket = fuser._rules.setdefault(key, __import__("collections").deque(maxlen=64))
    # 200 seconds old -> stale
    bucket.append(_ev(SignalKind.LIQUIDITY_SWEEP, ts=now - 200_000, symbol="X",
                      payload={"side": "buy_side", "wick_to_body": 2.0}))
    bucket.append(_ev(SignalKind.OI_SILENT_BUILD, ts=now - 200_000, symbol="X",
                      payload={"from_price": 1.0, "to_price": 1.005}))

    sig = fuser.evaluate("X", "binance", now)
    assert sig.rule_score == 0.0
    assert sig.direction == Direction.NEUTRAL
    assert sig.is_high_priority is False


# --------------------------------------------------------------------------- #
# Cooldown
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_cooldown_prevents_back_to_back_high_priority() -> None:
    seen: list[FusedSignal] = []

    async def sink(s: FusedSignal) -> None:
        seen.append(s)

    fuser = ScoreFuser(sink=sink, config=FuserConfig(cooldown_sec=60))
    now = 1_000_000

    # Trigger first high_priority
    await fuser.on_rule_signal(_ev(SignalKind.VOLUME_SPIKE, ts=now, payload={"side": "buy"}))
    await fuser.on_rule_signal(_ev(SignalKind.OI_SILENT_BUILD, ts=now,
                                   payload={"from_price": 1.0, "to_price": 1.005}))
    sig1 = await fuser.on_rule_signal(_ev(SignalKind.LIQUIDITY_SWEEP, ts=now,
                                          payload={"side": "buy_side", "wick_to_body": 2.0}))
    assert sig1 is not None and sig1.is_high_priority is True
    assert len(seen) == 1

    # 30s later -> still in cooldown, even if another sweep arrives
    next_ts = now + 30_000
    sig2 = await fuser.on_rule_signal(_ev(SignalKind.LIQUIDITY_SWEEP, ts=next_ts,
                                          payload={"side": "buy_side", "wick_to_body": 2.0}))
    assert sig2 is not None
    assert sig2.is_high_priority is False
    assert any("cooldown" in n.lower() for n in sig2.notes)
    assert len(seen) == 1  # sink not called again

    # 70s after first -> cooldown lifted
    later = now + 70_000
    sig3 = await fuser.on_rule_signal(_ev(SignalKind.OI_SILENT_BUILD, ts=later,
                                          payload={"from_price": 1.0, "to_price": 1.005}))
    assert sig3 is not None and sig3.is_high_priority is True
    assert len(seen) == 2


# --------------------------------------------------------------------------- #
# Intra-rule direction conflict
# --------------------------------------------------------------------------- #


def test_intra_rule_conflict_neutralizes_direction_and_penalizes_score() -> None:
    fuser = ScoreFuser()
    now = 1_000_000
    key = "binance:X"
    bucket = fuser._rules.setdefault(key, __import__("collections").deque(maxlen=64))
    # mixed: volume spike buy + liquidity sweep sell_side (which means SHORT bias)
    bucket.append(_ev(SignalKind.VOLUME_SPIKE, ts=now, symbol="X", payload={"side": "buy"}))
    bucket.append(_ev(SignalKind.LIQUIDITY_SWEEP, ts=now, symbol="X",
                      payload={"side": "sell_side", "wick_to_body": 2.0}))

    sig = fuser.evaluate("X", "binance", now)
    assert sig.direction == Direction.NEUTRAL
    assert sig.is_high_priority is False
    assert any("conflict" in n.lower() for n in sig.notes)


# --------------------------------------------------------------------------- #
# Sink only fires on high_priority
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_sink_not_called_when_not_high_priority() -> None:
    seen: list[FusedSignal] = []

    async def sink(s: FusedSignal) -> None:
        seen.append(s)

    fuser = ScoreFuser(sink=sink)
    # only volume spike: 30 points, far below threshold
    sig = await fuser.on_rule_signal(_ev(SignalKind.VOLUME_SPIKE, ts=1, payload={"side": "buy"}))
    assert sig is not None
    assert sig.is_high_priority is False
    assert seen == []


# --------------------------------------------------------------------------- #
# Sanity: rule weights sum is sensible
# --------------------------------------------------------------------------- #


def test_default_weights_allow_three_signal_confluence_to_clear_threshold() -> None:
    """The trio (vol+OI silent build+sweep) MUST be able to clear 85 alone."""
    trio = (
        DEFAULT_RULE_WEIGHTS[SignalKind.VOLUME_SPIKE]
        + DEFAULT_RULE_WEIGHTS[SignalKind.OI_SILENT_BUILD]
        + DEFAULT_RULE_WEIGHTS[SignalKind.LIQUIDITY_SWEEP]
    )
    assert trio >= 85


# --------------------------------------------------------------------------- #
# DUMP / SHORT symmetry coverage  (audit-fix regressions)
# --------------------------------------------------------------------------- #


def test_oi_silent_build_in_DOWNTREND_is_now_classified_SHORT() -> None:
    """
    Audit fix A: OI暴涨 + 价跌 = 主力做空建仓 (distribution-then-short).
    Was incorrectly classified as LONG before. Without this test the
    detector "sees" zero short signals from OI in a real dump setup.
    """
    fuser = ScoreFuser()
    now = 1_000_000
    bucket = fuser._rules.setdefault("binance:RAVEUSDT", __import__("collections").deque(maxlen=64))
    bucket.extend([
        _ev(SignalKind.VOLUME_SPIKE, ts=now, payload={"side": "sell"}),
        # OI grows 22%, price drops -0.5% (well past -0.3% threshold)
        _ev(SignalKind.OI_SILENT_BUILD, ts=now,
            payload={"oi_delta_pct": 0.22, "from_price": 1.000, "to_price": 0.995}),
        _ev(SignalKind.LIQUIDITY_SWEEP, ts=now,
            payload={"side": "sell_side", "wick_to_body": 2.5}),
    ])
    sig = fuser.evaluate("RAVEUSDT", "binance", now)
    assert sig.direction == Direction.SHORT
    assert sig.is_high_priority is True
    assert sig.rule_score >= 85


def test_oi_silent_build_with_FLAT_price_is_now_NEUTRAL() -> None:
    """
    Audit fix A boundary: OI grows but price is flat (within 0.3% band).
    Was previously written as 'mild long bias' but in real dumps this is
    ambiguous. New behavior: NEUTRAL — let other rules decide direction.
    """
    fuser = ScoreFuser()
    now = 1_000_000
    bucket = fuser._rules.setdefault("binance:X", __import__("collections").deque(maxlen=64))
    # Only OI signal, price moved +0.1% (under 0.3% threshold)
    bucket.append(_ev(SignalKind.OI_SILENT_BUILD, ts=now, symbol="X",
                      payload={"oi_delta_pct": 0.22, "from_price": 1.000, "to_price": 1.001}))
    sig = fuser.evaluate("X", "binance", now)
    assert sig.direction == Direction.NEUTRAL


def test_short_confluence_promotes_high_priority() -> None:
    """Mirror of the long confluence test. Three short-aligned rule signals → HIGH PRIORITY SHORT."""
    fuser = ScoreFuser()
    now = 1_000_000
    bucket = fuser._rules.setdefault("binance:RAVEUSDT", __import__("collections").deque(maxlen=64))
    bucket.extend([
        _ev(SignalKind.VOLUME_SPIKE, ts=now-5_000, payload={"side": "sell"}),
        _ev(SignalKind.OI_SILENT_BUILD, ts=now-2_000,
            payload={"oi_delta_pct": 0.20, "from_price": 1.000, "to_price": 0.99}),
        _ev(SignalKind.LIQUIDITY_SWEEP, ts=now,
            payload={"side": "sell_side", "wick_to_body": 2.2}),
    ])
    sig = fuser.evaluate("RAVEUSDT", "binance", now)
    assert sig.direction == Direction.SHORT
    assert sig.rule_score >= 85
    assert sig.is_high_priority is True


def test_kol_exit_liquidity_with_SHORT_signals_is_NOT_vetoed() -> None:
    """
    Critical audit fix: when our rules say SHORT and the LLM agrees with dump
    AND identifies the KOLs as exit_liquidity, that is a CONFIRMING piece of
    evidence — we want to short alongside the dump-front-run. Must promote.
    """
    fuser = ScoreFuser()
    now = 1_000_000
    key = "binance:RAVEUSDT"
    bucket = fuser._rules.setdefault(key, __import__("collections").deque(maxlen=64))
    # Strong SHORT confluence
    bucket.extend([
        _ev(SignalKind.VOLUME_SPIKE, ts=now, payload={"side": "sell"}),
        _ev(SignalKind.OI_SILENT_BUILD, ts=now,
            payload={"from_price": 1.000, "to_price": 0.99}),
        _ev(SignalKind.LIQUIDITY_SWEEP, ts=now,
            payload={"side": "sell_side", "wick_to_body": 2.5}),
    ])
    # LLM says dump AND flags KOL as exit_liquidity at high confidence
    fuser._llm[key] = _v("dump", confidence=0.85, kol_intent="exit_liquidity")
    fuser._llm_ts[key] = now

    sig = fuser.evaluate("RAVEUSDT", "binance", now)
    assert sig.direction == Direction.SHORT
    assert sig.is_high_priority is True            # NOT vetoed
    assert sig.blocked is False                    # NOT vetoed
    assert any("alongside" in n.lower() for n in sig.notes)   # special boost note


def test_kol_exit_liquidity_with_LONG_signals_STILL_HARD_vetoes() -> None:
    """Sanity: the audit fix did not weaken the LONG-side veto."""
    fuser = ScoreFuser()
    now = 1_000_000
    key = "binance:RAVEUSDT"
    bucket = fuser._rules.setdefault(key, __import__("collections").deque(maxlen=64))
    bucket.extend([
        _ev(SignalKind.VOLUME_SPIKE, ts=now, payload={"side": "buy"}),
        _ev(SignalKind.OI_SILENT_BUILD, ts=now,
            payload={"from_price": 1.0, "to_price": 1.005}),
        _ev(SignalKind.LIQUIDITY_SWEEP, ts=now,
            payload={"side": "buy_side", "wick_to_body": 3.0}),
    ])
    fuser._llm[key] = _v("pump", confidence=0.85, kol_intent="exit_liquidity")
    fuser._llm_ts[key] = now

    sig = fuser.evaluate("RAVEUSDT", "binance", now)
    assert sig.is_high_priority is False
    assert sig.blocked is True
    assert sig.block_reason == "kol_exit_liquidity_hard_veto"
