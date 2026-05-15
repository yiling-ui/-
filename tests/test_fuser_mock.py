"""Mock tests for ScoreFuser, including hot-loaded learned rules."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from altcoin_agent.ai_engine import AIVerdict
from altcoin_agent.fuser import (
    Direction,
    FusedSignal,
    FuserConfig,
    RuleIndex,
    ScoreFuser,
)
from altcoin_agent.screener import SignalEvent, SignalKind

# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #


def _ev(kind: SignalKind, ts: int, **payload) -> SignalEvent:
    payload.setdefault("symbol", "RAVEUSDT")
    sym = payload.pop("symbol")
    return SignalEvent(
        kind=kind, symbol=sym, ts=ts, exchange="binance", payload=payload,
    )


def _verdict(intent: str, score: int, *, kol_intent: str = "neutral") -> AIVerdict:
    return AIVerdict(
        intent=intent,            # type: ignore[arg-type]
        confidence_score=score,
        reason="mock verdict",
        kol_intent=kol_intent,    # type: ignore[arg-type]
        key_evidence=[],
    )


def _make_fuser(tmp_json: Path | None = None) -> ScoreFuser:
    cfg = FuserConfig(
        dynamic_rules_path=tmp_json or Path("/nonexistent/dynamic_rules.json"),
    )
    return ScoreFuser(config=cfg)


def _make_fuser_with_sink(tmp_json: Path | None = None) -> tuple[ScoreFuser, list[FusedSignal]]:
    cfg = FuserConfig(
        dynamic_rules_path=tmp_json or Path("/nonexistent/dynamic_rules.json"),
    )
    emitted: list[FusedSignal] = []

    async def sink(s: FusedSignal) -> None:
        emitted.append(s)

    return ScoreFuser(sink=sink, config=cfg), emitted


# --------------------------------------------------------------------- #
# Direction inference
# --------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_long_confluence_promotes_high_priority() -> None:
    f, emitted = _make_fuser_with_sink()
    now = 1_000_000
    await f.on_rule_signal(_ev(SignalKind.FUNDING_EXTREME, now,
                                rate=-0.0025, direction="short_squeeze"))
    await f.on_rule_signal(_ev(SignalKind.OI_SILENT_BUILD, now,
                                oi_delta_pct=0.22, from_price=1.0, to_price=1.005,
                                price_move_pct=0.005))
    await f.on_rule_signal(_ev(SignalKind.VOLUME_SPIKE, now,
                                side="buy", zscore=5.5))
    await f.on_rule_signal(_ev(SignalKind.LIQUIDITY_SWEEP, now,
                                side="buy_side", wick_to_body=2.4, bar_close=1.01))
    # The sink only sees the FIRST high-priority emission per cooldown window.
    assert emitted, "expected at least one high-priority emission"
    sig = emitted[0]
    assert sig.direction == Direction.LONG
    assert sig.is_high_priority


@pytest.mark.asyncio
async def test_oi_with_price_dropping_is_short_distribution() -> None:
    """Architect SR rule: OI up + price down = SHORT distribution, not LONG."""
    f = _make_fuser()
    now = 1_000_000
    sig = await f.on_rule_signal(_ev(SignalKind.OI_SILENT_BUILD, now,
                                      oi_delta_pct=0.22, from_price=2.0, to_price=1.99))
    sig = await f.on_rule_signal(_ev(SignalKind.VOLUME_SPIKE, now,
                                      side="sell", zscore=6.0))
    assert sig is not None
    assert sig.direction == Direction.SHORT


# --------------------------------------------------------------------- #
# LLM agreement boost & veto
# --------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_llm_disagreement_vetoes_signal() -> None:
    f = _make_fuser()
    now = 1_000_000
    await f.on_rule_signal(_ev(SignalKind.FUNDING_EXTREME, now,
                                rate=-0.0025, direction="short_squeeze"))
    await f.on_rule_signal(_ev(SignalKind.OI_SILENT_BUILD, now,
                                oi_delta_pct=0.22, from_price=1.0, to_price=1.005))
    sig = await f.on_llm_verdict("binance", "RAVEUSDT", _verdict("dump", 80), now)
    assert sig is not None
    assert sig.blocked
    assert sig.block_reason == "direction_conflict"


@pytest.mark.asyncio
async def test_kol_exit_high_conf_hard_vetoes_long_only() -> None:
    f = _make_fuser()
    now = 1_000_000
    await f.on_rule_signal(_ev(SignalKind.FUNDING_EXTREME, now,
                                rate=-0.0025, direction="short_squeeze"))
    await f.on_rule_signal(_ev(SignalKind.OI_SILENT_BUILD, now,
                                oi_delta_pct=0.22, from_price=1.0, to_price=1.005))
    await f.on_rule_signal(_ev(SignalKind.VOLUME_SPIKE, now, side="buy", zscore=6.0))
    sig = await f.on_llm_verdict(
        "binance", "RAVEUSDT",
        _verdict("pump", 90, kol_intent="exit_liquidity"), now,
    )
    assert sig is not None
    assert sig.blocked
    assert sig.block_reason == "kol_exit_liquidity_hard_veto"


@pytest.mark.asyncio
async def test_kol_exit_alongside_short_is_NOT_vetoed() -> None:
    """Shorting INTO KOL distribution is the right play."""
    f = _make_fuser()
    now = 1_000_000
    await f.on_rule_signal(_ev(SignalKind.FUNDING_EXTREME, now,
                                rate=0.002, direction="long_fragile"))
    await f.on_rule_signal(_ev(SignalKind.OI_SILENT_BUILD, now,
                                oi_delta_pct=0.22, from_price=2.0, to_price=1.99))
    await f.on_rule_signal(_ev(SignalKind.VOLUME_SPIKE, now, side="sell", zscore=6.0))
    sig = await f.on_llm_verdict(
        "binance", "RAVEUSDT",
        _verdict("dump", 90, kol_intent="exit_liquidity"), now,
    )
    assert sig is not None
    assert not sig.blocked
    assert sig.direction == Direction.SHORT


# --------------------------------------------------------------------- #
# Wash trading hard veto on LONG (SR-3)
# --------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_wash_trading_hard_vetoes_long() -> None:
    f = _make_fuser()
    now = 1_000_000
    await f.on_rule_signal(_ev(SignalKind.VOLUME_SPIKE, now, side="buy", zscore=6.0))
    await f.on_rule_signal(_ev(SignalKind.OI_SILENT_BUILD, now,
                                oi_delta_pct=0.22, from_price=1.0, to_price=1.005))
    sig = await f.on_rule_signal(
        _ev(SignalKind.WASH_TRADING_DETECTED, now,
            patterns=["ghost_volume"], volume_to_count_z_ratio=3.5),
    )
    assert sig is not None
    assert sig.blocked
    assert sig.block_reason == "wash_trading_detected"
    assert sig.direction == Direction.NEUTRAL


@pytest.mark.asyncio
async def test_wash_trading_does_not_veto_short() -> None:
    """Fake-pump bars often precede real dumps — shorts are not vetoed."""
    f = _make_fuser()
    now = 1_000_000
    await f.on_rule_signal(_ev(SignalKind.FUNDING_EXTREME, now,
                                rate=0.002, direction="long_fragile"))
    await f.on_rule_signal(_ev(SignalKind.OI_SILENT_BUILD, now,
                                oi_delta_pct=0.22, from_price=2.0, to_price=1.99))
    await f.on_rule_signal(_ev(SignalKind.VOLUME_SPIKE, now, side="sell", zscore=6.0))
    sig = await f.on_rule_signal(
        _ev(SignalKind.WASH_TRADING_DETECTED, now, patterns=["ghost_volume"]),
    )
    assert sig is not None
    assert not sig.blocked


# --------------------------------------------------------------------- #
# Hot-loaded learned rules (RuleIndex)
# --------------------------------------------------------------------- #


def _write_rules(path: Path, rules: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"version": 1, "rules": rules}))


def test_rule_index_hot_reload_is_throttled(tmp_path: Path) -> None:
    p = tmp_path / "rules.json"
    _write_rules(p, [{"feature_name": "f", "bucket": "b", "side": "pump",
                       "hits": 5, "total": 10}])
    idx = RuleIndex(json_path=p, min_check_interval_sec=0.05)

    idx.maybe_reload()
    assert len(idx) == 1

    # Within the throttle window, edits to the file are NOT picked up.
    _write_rules(p, [
        {"feature_name": "f", "bucket": "b", "side": "pump", "hits": 5, "total": 10},
        {"feature_name": "g", "bucket": "x", "side": "dump", "hits": 1, "total": 1},
    ])
    idx.maybe_reload()
    assert len(idx) == 1

    # After the throttle window, reload picks up changes.
    time.sleep(0.06)
    # Bump mtime to be safe (some FS resolutions are coarse).
    import os
    now = time.time()
    os.utime(p, (now, now))
    idx.maybe_reload()
    assert len(idx) == 2


def test_rule_index_io_error_keeps_previous_index(tmp_path: Path) -> None:
    p = tmp_path / "rules.json"
    _write_rules(p, [{"feature_name": "f", "bucket": "b", "side": "pump",
                       "hits": 5, "total": 10}])
    idx = RuleIndex(json_path=p, min_check_interval_sec=0.0)
    idx.maybe_reload()
    assert len(idx) == 1

    # Corrupt the file.
    p.write_text("{not valid json")
    import os
    now = time.time()
    os.utime(p, (now, now))
    idx.maybe_reload()
    # Previous good index is kept.
    assert len(idx) == 1


@pytest.mark.asyncio
async def test_learned_reward_is_capped_at_1_30x(tmp_path: Path) -> None:
    """Even a hit_rate=99% rule cannot push the reward beyond 1.30x of base."""
    p = tmp_path / "rules.json"
    _write_rules(p, [
        # 99/100 hit rate, big total -> shrinkage is near 1, raw lift near +0.96
        # but cap clamps the lift to +0.20, total reward capped at 1.30x.
        {"feature_name": "volume_zscore_last1h", "bucket": "pos_extreme",
         "side": "pump", "hits": 99, "total": 100},
    ])
    cfg = FuserConfig(dynamic_rules_path=p)
    f = ScoreFuser(config=cfg)
    f.rule_index.min_check_interval_sec = 0.0  # reload on every evaluate

    now = 1_000_000
    # Single rule signal — its base score is 30 (volume_spike weight). With a
    # 1.30x cap the final cannot exceed 39.
    await f.on_rule_signal(_ev(SignalKind.VOLUME_SPIKE, now,
                                side="buy", zscore=10.0))
    sig = f.evaluate("RAVEUSDT", "binance", now)
    assert sig.final_score <= 30.0 * 1.30 + 0.01


@pytest.mark.asyncio
async def test_learned_penalty_drags_score_down(tmp_path: Path) -> None:
    p = tmp_path / "rules.json"
    _write_rules(p, [
        # 5/100 hit rate -> raw lift ~ -0.90, shrunk + capped to -0.30.
        {"feature_name": "volume_zscore_last1h", "bucket": "pos_extreme",
         "side": "pump", "hits": 5, "total": 100},
    ])
    cfg = FuserConfig(dynamic_rules_path=p)
    f = ScoreFuser(config=cfg)
    f.rule_index.min_check_interval_sec = 0.0

    now = 1_000_000
    await f.on_rule_signal(_ev(SignalKind.VOLUME_SPIKE, now,
                                side="buy", zscore=10.0))
    await f.on_rule_signal(_ev(SignalKind.OI_SILENT_BUILD, now,
                                oi_delta_pct=0.22, from_price=1.0, to_price=1.005))
    sig = f.evaluate("RAVEUSDT", "binance", now)
    base = 30.0 + 35.0  # vol_spike + oi_silent_build
    # Penalty cap is -30%, so floor on (final / base) is 0.70
    assert sig.final_score < base, "penalty should drag final below base"
    assert sig.final_score >= base * 0.70 - 0.01


# --------------------------------------------------------------------- #
# Window + cooldown
# --------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_stale_events_outside_window_are_dropped() -> None:
    cfg = FuserConfig(window_sec=10)
    f = ScoreFuser(config=cfg)
    # Old event well outside the 10s window.
    old = _ev(SignalKind.VOLUME_SPIKE, ts=0, side="buy", zscore=5.0)
    f._rules.setdefault("binance:RAVEUSDT", __import__("collections").deque(maxlen=64)).append(old)
    sig: FusedSignal = f.evaluate("RAVEUSDT", "binance", now_ts=60_000)
    assert sig.rule_score == 0.0
    assert sig.direction == Direction.NEUTRAL
