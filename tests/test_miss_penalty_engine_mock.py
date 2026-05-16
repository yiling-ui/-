"""Unit tests for risk/miss_penalty_engine.py (Phase A.1)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from altcoin_agent.risk.miss_penalty_engine import (
    MissedOpportunity,
    MissPenaltyConfig,
    MissPenaltyEngine,
    bucket_reject_reason,
    compute_mfe_mae,
    fixed_klines_fetcher,
    iter_decisions_jsonl,
    iter_decisions_with_rotations,
)

# --------------------------------------------------------------------- #
# bucket_reject_reason
# --------------------------------------------------------------------- #


def test_bucket_reject_reason_strips_trailing_data() -> None:
    assert bucket_reject_reason(
        "slippage_too_high:0.0341>0.0212@lev=10.00",
    ) == "slippage_too_high"
    assert bucket_reject_reason("anti_chase:0.034") == "anti_chase"


def test_bucket_reject_reason_keeps_prefix_for_simple_reasons() -> None:
    assert bucket_reject_reason("ok") == "ok"
    assert bucket_reject_reason("reconciliation_pending") == "reconciliation_pending"


def test_bucket_reject_reason_handles_empty_or_leading_colon() -> None:
    # The collapse only happens for separators AFTER position 0; ":foo"
    # is still treated as a single bucket because the prefix is empty.
    assert bucket_reject_reason("") == "unknown"
    assert bucket_reject_reason(":foo") == ":foo"


def test_bucket_reject_reason_takes_first_separator_only() -> None:
    # "signal_blocked:foo:bar" should bucket on "signal_blocked" so the
    # scorer aggregates same-rule rejections together regardless of the
    # specific blocked sub-reason.
    assert bucket_reject_reason("signal_blocked:foo:bar") == "signal_blocked"


# --------------------------------------------------------------------- #
# compute_mfe_mae
# --------------------------------------------------------------------- #


def _bar(ts_ms: int, o: float, h: float, l: float, c: float, v: float = 1.0):  # noqa: E741
    return (ts_ms, o, h, l, c, v)


def test_compute_mfe_mae_long_basic_case() -> None:
    # Reference 100; 24h bar window goes up to 250 (high) with a -10%
    # drawdown (low 90) -> MFE=+1.5, MAE=-0.10.
    bars = [
        _bar(0, 100, 110, 90, 105),
        _bar(60_000, 105, 250, 100, 240),
    ]
    mfe, mae, n = compute_mfe_mae(bars, reference_price=100.0, direction="long")
    assert n == 2
    assert mfe == pytest.approx(1.5)
    assert mae == pytest.approx(-0.10)


def test_compute_mfe_mae_short_basic_case() -> None:
    # SHORT thesis on a falling chart: ref=100, low=20, high bounce 110.
    # MFE = (100-20)/100 = 0.8 (favorable, downward).
    # MAE = (100-110)/100 = -0.10 (against the short thesis).
    bars = [
        _bar(0, 100, 110, 20, 30),
    ]
    mfe, mae, n = compute_mfe_mae(bars, reference_price=100.0, direction="short")
    assert n == 1
    assert mfe == pytest.approx(0.8)
    assert mae == pytest.approx(-0.10)


def test_compute_mfe_mae_handles_empty_input() -> None:
    assert compute_mfe_mae([], 100.0, "long") == (0.0, 0.0, 0)


def test_compute_mfe_mae_handles_zero_reference() -> None:
    assert compute_mfe_mae(
        [_bar(0, 1, 2, 0.5, 1.5)], 0.0, "long",
    ) == (0.0, 0.0, 0)


def test_compute_mfe_mae_unknown_direction_treated_as_long() -> None:
    bars = [_bar(0, 100, 200, 80, 150)]
    mfe, mae, _ = compute_mfe_mae(bars, 100.0, "weird")
    assert mfe == pytest.approx(1.0)
    assert mae == pytest.approx(-0.20)


# --------------------------------------------------------------------- #
# iter_decisions_jsonl
# --------------------------------------------------------------------- #


def test_iter_decisions_jsonl_skips_blank_and_garbage_lines(tmp_path: Path) -> None:
    p = tmp_path / "decisions.jsonl"
    p.write_text(
        json.dumps({"ts": 1.0, "trace_id": "a"}) + "\n"
        "\n"
        "not-json\n"
        + json.dumps({"ts": 2.0, "trace_id": "b"}) + "\n"
    )
    rows = list(iter_decisions_jsonl(p))
    assert [r["trace_id"] for r in rows] == ["a", "b"]


def test_iter_decisions_jsonl_filters_by_since_ts(tmp_path: Path) -> None:
    p = tmp_path / "d.jsonl"
    p.write_text(
        "\n".join(
            json.dumps({"ts": ts, "trace_id": str(ts)})
            for ts in (10.0, 20.0, 30.0)
        )
    )
    rows = list(iter_decisions_jsonl(p, since_ts=20.0))
    assert [r["trace_id"] for r in rows] == ["20.0", "30.0"]


def test_iter_decisions_jsonl_returns_empty_for_missing_file(tmp_path: Path) -> None:
    assert list(iter_decisions_jsonl(tmp_path / "nope.jsonl")) == []


def test_iter_decisions_with_rotations_reads_oldest_first(tmp_path: Path) -> None:
    base = tmp_path / "decisions.jsonl"
    # Largest N == oldest, so should appear first.
    (tmp_path / "decisions.jsonl.2").write_text(
        json.dumps({"ts": 1.0, "trace_id": "older"}) + "\n",
    )
    (tmp_path / "decisions.jsonl.1").write_text(
        json.dumps({"ts": 2.0, "trace_id": "old"}) + "\n",
    )
    base.write_text(json.dumps({"ts": 3.0, "trace_id": "active"}) + "\n")
    ids = [r["trace_id"] for r in iter_decisions_with_rotations(base)]
    assert ids == ["older", "old", "active"]


# --------------------------------------------------------------------- #
# Engine — full integration with fake fetcher
# --------------------------------------------------------------------- #


def _write_decision(
    path: Path, *, trace_id: str, ts: float, approved: bool,
    symbol: str = "PEPE/USDT:USDT", direction: str = "long",
    current_price: float = 100.0, reason: str = "anti_chase:0.04",
) -> None:
    rec = {
        "ts": ts,
        "trace_id": trace_id,
        "symbol": symbol,
        "direction": direction,
        "current_price": current_price,
        "approved": approved,
        "reason": reason,
        "final_score": 78.0,
        "signal_kind": "volume_spike",
        "rule_score": 60.0,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def _gen_kline_window(
    *, start_ms: int, peak_pct: float, dip_pct: float,
    bars: int = 1440, base: float = 100.0,
):
    """Synthetic 24h 1m window: linear ramp to peak, then half retrace.

    - First half: walks linearly from base to base*(1+peak_pct).
    - Second half: walks linearly down to base*(1+peak_pct)*(1+dip_pct/2).
    Each bar's high/low straddle the close by ±0.1% so MFE/MAE pickups
    aren't an artefact of close-only sampling.
    """
    out = []
    half = bars // 2
    peak = base * (1 + peak_pct)
    bottom_after_peak = peak * (1 + dip_pct)
    for i in range(bars):
        ts = start_ms + i * 60_000
        if i <= half:
            close = base + (peak - base) * (i / half)
        else:
            close = peak + (bottom_after_peak - peak) * ((i - half) / (bars - half))
        high = close * 1.001
        low = close * 0.999
        # Inject the actual extremes precisely at the peak bar so MFE
        # is exactly the configured peak and the dip bar bottoms at
        # the configured dip.
        if i == half:
            high = peak
        if i == bars - 1:
            low = bottom_after_peak
        out.append((ts, close, high, low, close, 1.0))
    return out


@pytest.mark.asyncio
async def test_engine_labels_pump_when_long_reject_moons(tmp_path: Path) -> None:
    log_path = tmp_path / "decisions.jsonl"
    rejected_at_s = 1_000_000.0
    rejected_at_ms = int(rejected_at_s * 1000)

    _write_decision(
        log_path, trace_id="t-pump", ts=rejected_at_s, approved=False,
        direction="long", current_price=100.0,
        reason="anti_chase:0.04",
    )

    # Symbol mooned +200% with only -5% drawdown -> classic missed pump.
    bars = _gen_kline_window(
        start_ms=rejected_at_ms, peak_pct=2.0, dip_pct=-0.05, bars=1440,
    )
    fetcher = fixed_klines_fetcher({"PEPE/USDT:USDT": bars})

    cfg = MissPenaltyConfig(
        state_dir=str(tmp_path / "state"),
        require_window_closed=True,
    )
    engine = MissPenaltyEngine(
        decisions_log_path=log_path, kline_fetcher=fetcher, config=cfg,
        clock=lambda: rejected_at_s + 25 * 3600,  # 25h after reject
    )
    new = await engine.run_audit(lookback_hours=48)
    assert len(new) == 1
    opp = new[0]
    assert opp.is_missed_pump is True
    assert opp.realized_max_favorable_pct >= 1.0
    assert abs(opp.realized_max_adverse_pct) <= 0.30
    assert opp.miss_severity > 0.5
    assert opp.would_have_pnl_pct > 0
    assert opp.rejected_reason_bucket == "anti_chase"


@pytest.mark.asyncio
async def test_engine_does_not_label_when_drawdown_too_deep(tmp_path: Path) -> None:
    log_path = tmp_path / "decisions.jsonl"
    rejected_at_s = 1_000_000.0
    _write_decision(
        log_path, trace_id="t-dd", ts=rejected_at_s, approved=False,
        direction="long", current_price=100.0, reason="anti_chase:0.04",
    )

    # Big +200% pump but it first crashed -50% -> we'd have been
    # stopped out, so this is NOT a missed pump.
    bars = _gen_kline_window(
        start_ms=int(rejected_at_s * 1000),
        peak_pct=2.0, dip_pct=-0.05, bars=1440,
    )
    # Inject a deep dip in the first 10 bars.
    bars[5] = (bars[5][0], 100, 101, 49, 50, 1.0)

    fetcher = fixed_klines_fetcher({"PEPE/USDT:USDT": bars})
    cfg = MissPenaltyConfig(state_dir=str(tmp_path / "state"))
    engine = MissPenaltyEngine(
        decisions_log_path=log_path, kline_fetcher=fetcher, config=cfg,
        clock=lambda: rejected_at_s + 25 * 3600,
    )
    new = await engine.run_audit()
    assert len(new) == 1
    assert new[0].is_missed_pump is False
    assert abs(new[0].realized_max_adverse_pct) > 0.30


@pytest.mark.asyncio
async def test_engine_short_miss_recognised(tmp_path: Path) -> None:
    log_path = tmp_path / "decisions.jsonl"
    rejected_at_s = 2_000_000.0
    _write_decision(
        log_path, trace_id="t-short", ts=rejected_at_s, approved=False,
        direction="short", current_price=100.0, reason="anti_chase:0.04",
    )
    # Symbol crashed -60% from 100 with only a 10% bounce -> missed dump.
    # peak_pct < 0 means downside; we generate manually.
    bars = []
    start_ms = int(rejected_at_s * 1000)
    for i in range(1440):
        ts = start_ms + i * 60_000
        # Linear walk from 100 -> 40 (-60%).
        close = 100 - (60 * i / 1440)
        high = close * 1.005   # mild upside wicks
        low = close * 0.999
        bars.append((ts, close, high, low, close, 1.0))
    # Bar 5 has a brief 8% bounce (within tolerance 20%).
    bars[5] = (bars[5][0], 100, 108, 95, 96, 1.0)

    fetcher = fixed_klines_fetcher({"PEPE/USDT:USDT": bars})
    cfg = MissPenaltyConfig(state_dir=str(tmp_path / "state"))
    engine = MissPenaltyEngine(
        decisions_log_path=log_path, kline_fetcher=fetcher, config=cfg,
        clock=lambda: rejected_at_s + 25 * 3600,
    )
    new = await engine.run_audit()
    assert len(new) == 1
    assert new[0].is_missed_pump is True
    assert new[0].direction == "short"


@pytest.mark.asyncio
async def test_engine_skips_recent_rejections_window_not_closed(tmp_path: Path) -> None:
    log_path = tmp_path / "decisions.jsonl"
    rejected_at_s = 3_000_000.0
    _write_decision(
        log_path, trace_id="t-fresh", ts=rejected_at_s, approved=False,
    )
    fetcher = fixed_klines_fetcher({})
    cfg = MissPenaltyConfig(state_dir=str(tmp_path / "state"))
    engine = MissPenaltyEngine(
        decisions_log_path=log_path, kline_fetcher=fetcher, config=cfg,
        # Clock only 30 minutes after the reject -> window not closed.
        clock=lambda: rejected_at_s + 30 * 60,
    )
    new = await engine.run_audit()
    assert new == []


@pytest.mark.asyncio
async def test_engine_skips_approved_decisions(tmp_path: Path) -> None:
    log_path = tmp_path / "decisions.jsonl"
    rejected_at_s = 1_000_000.0
    _write_decision(
        log_path, trace_id="approved-trade", ts=rejected_at_s,
        approved=True, reason="ok",
    )
    fetcher = fixed_klines_fetcher({"PEPE/USDT:USDT": []})
    cfg = MissPenaltyConfig(state_dir=str(tmp_path / "state"))
    engine = MissPenaltyEngine(
        decisions_log_path=log_path, kline_fetcher=fetcher, config=cfg,
        clock=lambda: rejected_at_s + 25 * 3600,
    )
    assert await engine.run_audit() == []


@pytest.mark.asyncio
async def test_engine_run_is_idempotent(tmp_path: Path) -> None:
    log_path = tmp_path / "decisions.jsonl"
    rejected_at_s = 1_000_000.0
    _write_decision(
        log_path, trace_id="t-dup", ts=rejected_at_s, approved=False,
    )
    bars = _gen_kline_window(
        start_ms=int(rejected_at_s * 1000),
        peak_pct=2.0, dip_pct=-0.05, bars=1440,
    )
    fetcher = fixed_klines_fetcher({"PEPE/USDT:USDT": bars})
    cfg = MissPenaltyConfig(state_dir=str(tmp_path / "state"))

    engine_a = MissPenaltyEngine(
        decisions_log_path=log_path, kline_fetcher=fetcher, config=cfg,
        clock=lambda: rejected_at_s + 25 * 3600,
    )
    first = await engine_a.run_audit()
    assert len(first) == 1

    # Fresh engine simulates a daily-cron re-invocation in a new
    # process. It must NOT re-record the same trace_id.
    engine_b = MissPenaltyEngine(
        decisions_log_path=log_path, kline_fetcher=fetcher, config=cfg,
        clock=lambda: rejected_at_s + 25 * 3600,
    )
    second = await engine_b.run_audit()
    assert second == []


@pytest.mark.asyncio
async def test_engine_marks_insufficient_data_when_few_bars(tmp_path: Path) -> None:
    log_path = tmp_path / "decisions.jsonl"
    rejected_at_s = 1_000_000.0
    _write_decision(
        log_path, trace_id="t-thin", ts=rejected_at_s, approved=False,
    )
    bars = _gen_kline_window(
        start_ms=int(rejected_at_s * 1000),
        peak_pct=2.0, dip_pct=-0.05, bars=10,  # very thin
    )
    fetcher = fixed_klines_fetcher({"PEPE/USDT:USDT": bars})
    cfg = MissPenaltyConfig(
        state_dir=str(tmp_path / "state"),
        min_bars_for_label=60,
    )
    engine = MissPenaltyEngine(
        decisions_log_path=log_path, kline_fetcher=fetcher, config=cfg,
        clock=lambda: rejected_at_s + 25 * 3600,
    )
    new = await engine.run_audit()
    assert len(new) == 1
    assert new[0].insufficient_data is True
    assert new[0].is_missed_pump is False


@pytest.mark.asyncio
async def test_engine_persists_to_disk_and_loads_back(tmp_path: Path) -> None:
    log_path = tmp_path / "decisions.jsonl"
    rejected_at_s = 1_000_000.0
    _write_decision(
        log_path, trace_id="persist-1", ts=rejected_at_s, approved=False,
    )
    bars = _gen_kline_window(
        start_ms=int(rejected_at_s * 1000),
        peak_pct=2.0, dip_pct=-0.05, bars=1440,
    )
    state_dir = tmp_path / "state"
    fetcher = fixed_klines_fetcher({"PEPE/USDT:USDT": bars})
    cfg = MissPenaltyConfig(state_dir=str(state_dir))
    engine = MissPenaltyEngine(
        decisions_log_path=log_path, kline_fetcher=fetcher, config=cfg,
        clock=lambda: rejected_at_s + 25 * 3600,
    )
    await engine.run_audit()

    out_path = state_dir / "missed_opportunities.jsonl"
    assert out_path.exists()
    on_disk = json.loads(out_path.read_text().splitlines()[0])
    assert on_disk["trace_id"] == "persist-1"

    loaded = engine.load_recent_missed()
    assert len(loaded) == 1
    assert isinstance(loaded[0], MissedOpportunity)
    assert loaded[0].is_missed_pump is True


@pytest.mark.asyncio
async def test_engine_handles_malformed_audit_rows_gracefully(tmp_path: Path) -> None:
    log_path = tmp_path / "decisions.jsonl"
    # Mix of: legit reject, missing trace_id, missing direction,
    # bad current_price, garbage line.
    rejected_at_s = 1_000_000.0
    _write_decision(
        log_path, trace_id="ok", ts=rejected_at_s, approved=False,
    )
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": rejected_at_s, "approved": False}) + "\n")
        f.write(json.dumps({
            "ts": rejected_at_s, "trace_id": "x", "approved": False,
            "symbol": "X", "direction": "long",  # missing current_price
        }) + "\n")
        f.write("not-json\n")
    bars = _gen_kline_window(
        start_ms=int(rejected_at_s * 1000),
        peak_pct=2.0, dip_pct=-0.05, bars=1440,
    )
    fetcher = fixed_klines_fetcher({"PEPE/USDT:USDT": bars})
    cfg = MissPenaltyConfig(state_dir=str(tmp_path / "state"))
    engine = MissPenaltyEngine(
        decisions_log_path=log_path, kline_fetcher=fetcher, config=cfg,
        clock=lambda: rejected_at_s + 25 * 3600,
    )
    new = await engine.run_audit()
    # Only the well-formed "ok" row survives all filters.
    assert len(new) == 1
    assert new[0].trace_id == "ok"
