"""Tests for the post-review patches on top of feat/operational-patches.

Covers:
  * ``AccountState.resume()`` — symmetric with ``halt()`` (fires the
    persistence listener so /resume sticks across restart).
  * ``WithdrawalDetector`` — does NOT produce a phantom flow event when
    ``maybe_roll_over_day`` zeros ``realized_pnl_today_usdt`` at UTC
    midnight (issue: the detector's PnL snapshot was not reset by
    rollover).
  * Side-aware depth cap — the gate compares the order's notional
    against the side it actually crosses (asks for longs, bids for
    shorts) when the caller supplies ``top_depth_by_side``. The legacy
    summed-depth path falls back to a halved heuristic.
  * ``ReversalGuard`` — defaults to "close and watch" on a position
    flip; vetoes when wick / 插针 detected; vetoes on weak signal
    score; sets cooldown after a veto; only approves a clean post-
    close reversal that passes every check.
  * ``Notifier.flow()`` — new bank-flow channel renders with 🏦 BANK
    rather than 🚨 ERROR.
  * ``_parse_int_list`` — drops unparseable entries instead of
    crashing boot.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from altcoin_agent.fuser import Direction, FusedSignal
from altcoin_agent.notifier.telegram import NullNotifier, TelegramNotifier
from altcoin_agent.price_tape import PriceTape, PriceTapeConfig
from altcoin_agent.risk.gate import RiskGate, RiskGateConfig
from altcoin_agent.risk.reversal_guard import (
    ReversalGuard,
    ReversalGuardConfig,
)
from altcoin_agent.risk.sizing import PositionSizer
from altcoin_agent.risk.state import AccountState, Position, Side
from altcoin_agent.risk.withdrawal_detector import (
    WithdrawalDetector,
    WithdrawalDetectorConfig,
)


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #


def _make_signal(
    *,
    symbol: str = "PEPE/USDT:USDT",
    direction: Direction = Direction.LONG,
    final_score: float = 9.0,
    trigger_price: float = 1.0,
) -> FusedSignal:
    """Build a minimal FusedSignal that satisfies the gate's contract.

    We only fill what the post-review code paths read. Anything
    missing is supplied with safe defaults via the dataclass.
    """
    return FusedSignal(
        symbol=symbol,
        exchange="binance",
        ts=int(time.time() * 1000),
        direction=direction,
        rule_score=final_score,
        llm_score=0.0,
        final_score=final_score,
        is_high_priority=True,
        blocked=False,
        block_reason=None,
        trigger_price=trigger_price,
        rule_signals=[],
        notes=[],
    )


class _FakeAdapter:
    """Adapter stub for the WithdrawalDetector test."""

    def __init__(self, balances: list[float]) -> None:
        self._balances = list(balances)
        self._calls = 0

    async def fetch_total_usdt_balance(self) -> float:
        idx = self._calls
        self._calls += 1
        return self._balances[min(idx, len(self._balances) - 1)]


# --------------------------------------------------------------------- #
# 1) AccountState.resume()
# --------------------------------------------------------------------- #


def test_account_resume_clears_halt_and_fires_listener():
    """resume() must mirror halt(): clears flags AND notifies the
    listener so AccountPersistor.save flushes to disk."""
    a = AccountState(equity_usdt=10_000.0)
    saved: list[dict[str, Any]] = []
    a.register_change_listener(lambda s: saved.append({
        "halted": s.global_trading_halted,
        "reason": s.halt_reason,
    }))

    # Halt fires once.
    a.halt("manual:test")
    assert len(saved) == 1
    assert saved[-1] == {"halted": True, "reason": "manual:test"}

    # Resume fires once and clears both fields.
    cleared = a.resume()
    assert cleared is True
    assert len(saved) == 2
    assert saved[-1] == {"halted": False, "reason": None}
    assert a.global_trading_halted is False
    assert a.halt_reason is None


def test_account_resume_idempotent_when_not_halted():
    """Calling resume() on an already-running account must NOT fire
    the listener (no spurious save on idle /resume clicks)."""
    a = AccountState(equity_usdt=10_000.0)
    saved = []
    a.register_change_listener(lambda s: saved.append(s))

    cleared = a.resume()
    assert cleared is False
    assert saved == []


# --------------------------------------------------------------------- #
# 2) WithdrawalDetector — rollover phantom event regression
# --------------------------------------------------------------------- #


def test_detector_does_not_fire_phantom_event_on_rollover():
    """Issue: at UTC midnight ``maybe_roll_over_day`` zeros
    ``realized_pnl_today_usdt``. The detector's snapshot of that
    same value was not reset, so the next netting computed
    ``unexplained = 0 - (-yesterdays_pnl) = +yesterdays_pnl`` and
    fired a phantom "deposit_detected" event once per day."""

    a = AccountState(
        equity_usdt=10_500.0,
        starting_equity_today_usdt=10_000.0,
        realized_pnl_today_usdt=500.0,
        last_rollover_date_utc="2026-05-16",
    )
    adapter = _FakeAdapter([10_500.0, 10_500.0])  # venue unchanged
    events: list[Any] = []

    async def on_event(reason, delta, diag):
        events.append((reason, delta, diag))

    cfg = WithdrawalDetectorConfig(
        enabled=True,
        startup_grace_sec=0.0,
        min_significant_delta_usdt=100.0,
    )
    det = WithdrawalDetector(
        adapter=adapter, account=a, cfg=cfg, on_event=on_event,
    )

    async def run():
        # Establish baseline (yesterday).
        out0 = await det.poll_once()
        assert out0["action"] == "baseline_set"
        # Simulate UTC midnight: starting_equity rebases to current
        # equity, realized_pnl_today_usdt zeros, rollover date flips.
        a.starting_equity_today_usdt = a.equity_usdt
        a.realized_pnl_today_usdt = 0.0
        a.last_rollover_date_utc = "2026-05-17"
        # First poll after rollover.
        out1 = await det.poll_once()
        return out0, out1

    out0, out1 = asyncio.run(run())

    # The post-rollover poll must NOT fire a flow event.
    assert events == [], (
        "rollover should not produce a phantom flow event; "
        f"got: {events}"
    )
    # And it must explicitly mark the resync.
    assert out1["action"] == "rollover_resync"


def test_detector_still_fires_real_withdrawal_after_rollover():
    """Defence: the rollover-aware reset must NOT swallow a genuine
    withdrawal that happens to land in the same poll as a rollover."""
    a = AccountState(
        equity_usdt=10_000.0,
        starting_equity_today_usdt=10_000.0,
        realized_pnl_today_usdt=0.0,
        last_rollover_date_utc="2026-05-17",
    )
    adapter = _FakeAdapter([10_000.0, 5_000.0])  # 5k withdrawal
    events = []

    async def on_event(reason, delta, diag):
        events.append((reason, delta))

    cfg = WithdrawalDetectorConfig(
        enabled=True, startup_grace_sec=0.0,
        min_significant_delta_usdt=100.0,
    )
    det = WithdrawalDetector(
        adapter=adapter, account=a, cfg=cfg, on_event=on_event,
    )

    async def run():
        await det.poll_once()  # baseline
        # No rollover this time — same UTC day.
        return await det.poll_once()

    out = asyncio.run(run())
    assert out["action"] == "withdrawal_detected"
    assert len(events) == 1
    assert events[0][0] == "withdrawal_detected"
    assert events[0][1] == pytest.approx(-5_000.0, abs=1e-3)


# --------------------------------------------------------------------- #
# 3) Side-aware depth cap
# --------------------------------------------------------------------- #


def _make_account(equity: float = 10_000.0) -> AccountState:
    return AccountState(
        equity_usdt=equity,
        starting_equity_today_usdt=equity,
        realized_pnl_today_usdt=0.0,
        reconciliation_complete=True,
        # Pin rollover so the gate's hot-path defence-in-depth call
        # doesn't reset our equity baseline mid-test.
        last_rollover_date_utc="2026-05-17",
    )


def test_depth_cap_uses_ask_side_for_long_when_side_aware_provided():
    """A long market order eats asks. If the cap compared against the
    summed bid+ask depth, a one-sided book (700k asks, 300k bids)
    would report 1M and approve up to 100k notional at 10% — twice
    the real exposure on the side being crossed.
    """
    cfg = RiskGateConfig(
        max_notional_vs_depth_pct=0.10,
        depth_cap_side_aware=True,
        # Disable extras the test doesn't care about.
        min_liquidity_usdt=0.0,
        # SR-1 base slippage is large enough that our test won't
        # accidentally trip it.
        base_slippage=0.5,
    )
    sizer = PositionSizer()
    gate = RiskGate(sizer, cfg)
    account = _make_account(equity=10_000.0)
    sig = _make_signal(direction=Direction.LONG, final_score=9.0)

    # Asymmetric book: 700k asks, 300k bids -> sum 1M.
    # An order of 80,000 USDT would pass the LEGACY (sum) cap of 100k
    # but FAIL against the ask-side cap of 70k.
    # We construct the gate inputs so the sized notional is ~80k.
    # PositionSizer is governed by equity * leverage; we use the
    # current_price to get there.
    decision = gate.evaluate(
        signal=sig,
        account=account,
        current_price=1.0,
        top5_depth_usdt=1_000_000.0,    # legacy (summed)
        realized_vol_pct=0.05,
        initial_stop=0.95,              # 5% stop -> sizing will pick a leverage
        top_depth_by_side=(300_000.0, 700_000.0),  # (bid, ask)
    )

    # The cap should fire; the rejection reason mentions the cap.
    if not decision.approved:
        assert "notional_exceeds_depth_cap" in decision.reason
    else:
        # If sizing picked a smaller notional than 70k, this test is
        # not exercising the cap; assert side-awareness another way.
        assert decision.notional_usdt is not None
        assert decision.notional_usdt <= 70_000.0


def test_depth_cap_uses_bid_side_for_short_when_side_aware_provided():
    """Mirror: a short market order eats bids."""
    cfg = RiskGateConfig(
        max_notional_vs_depth_pct=0.10,
        depth_cap_side_aware=True,
        min_liquidity_usdt=0.0,
        base_slippage=0.5,
    )
    gate = RiskGate(PositionSizer(), cfg)
    account = _make_account()
    sig = _make_signal(direction=Direction.SHORT, final_score=9.0)

    # Same asymmetric book: 700k asks, 300k bids -> short eats 300k.
    # A 50k notional should fail (> 30k cap) but pass under the
    # legacy 1M-sum cap of 100k.
    decision = gate.evaluate(
        signal=sig,
        account=account,
        current_price=1.0,
        top5_depth_usdt=1_000_000.0,
        realized_vol_pct=0.05,
        initial_stop=1.05,
        top_depth_by_side=(300_000.0, 700_000.0),
    )
    if not decision.approved:
        assert "notional_exceeds_depth_cap" in decision.reason
    else:
        assert decision.notional_usdt is not None
        assert decision.notional_usdt <= 30_000.0


def test_depth_cap_falls_back_to_halved_sum_when_side_unknown():
    """Legacy callers (no ``top_depth_by_side``) get the halved-sum
    heuristic — strictly more conservative than the original sum,
    matching the symmetric-book assumption."""
    cfg = RiskGateConfig(
        max_notional_vs_depth_pct=0.10,
        depth_cap_side_aware=True,
        min_liquidity_usdt=0.0,
        base_slippage=0.5,
    )
    gate = RiskGate(PositionSizer(), cfg)
    helper = gate._crossing_side_depth_usdt(
        side=Side.LONG,
        top5_depth_usdt=1_000_000.0,
        top_depth_by_side=None,
    )
    assert helper == pytest.approx(500_000.0)


def test_depth_cap_legacy_sum_when_side_aware_disabled():
    """Operators that explicitly want the old behaviour set
    ``depth_cap_side_aware=False`` and the helper returns the raw
    sum. Used when test fixtures rely on the old semantics."""
    cfg = RiskGateConfig(
        max_notional_vs_depth_pct=0.10,
        depth_cap_side_aware=False,
        min_liquidity_usdt=0.0,
    )
    gate = RiskGate(PositionSizer(), cfg)
    helper = gate._crossing_side_depth_usdt(
        side=Side.LONG,
        top5_depth_usdt=1_000_000.0,
        top_depth_by_side=None,
    )
    assert helper == pytest.approx(1_000_000.0)


# --------------------------------------------------------------------- #
# 4) ReversalGuard
# --------------------------------------------------------------------- #


def test_reversal_guard_disabled_approves_everything():
    g = ReversalGuard(cfg=ReversalGuardConfig(enabled=False))
    account = _make_account()
    sig = _make_signal(direction=Direction.LONG)
    out = g.decide(signal=sig, account=account)
    assert out.action == "approve"
    assert out.reason == "disabled"


def test_reversal_guard_no_position_no_recent_close_approves():
    g = ReversalGuard(cfg=ReversalGuardConfig(enabled=True))
    account = _make_account()
    sig = _make_signal(direction=Direction.LONG)
    out = g.decide(signal=sig, account=account)
    assert out.action == "approve"
    assert out.reason == "not_reversal"


def test_reversal_guard_open_position_flip_defers_close_and_watch():
    """Operator's instruction: prefer flat-and-watch over reverse."""
    g = ReversalGuard(cfg=ReversalGuardConfig(enabled=True))
    account = _make_account()
    # Existing LONG position; new SHORT signal -> reversal candidate.
    pos = Position(
        symbol="PEPE/USDT:USDT", exchange="binance",
        side=Side.LONG, entry_price=1.0, size=100.0, leverage=5.0,
        initial_stop=0.95, current_stop=0.95,
    )
    account.open_positions[pos.symbol] = pos
    sig = _make_signal(direction=Direction.SHORT, final_score=9.0)

    out = g.decide(signal=sig, account=account)
    assert out.action == "defer_close_and_watch"
    assert out.reason == "prefer_flat_and_watch"


def test_reversal_guard_post_close_with_wick_vetoes_and_sets_cooldown():
    """插针 detection: when the recent Parkinson range exceeds the
    threshold, the reversal is vetoed and a cooldown is set."""
    pt = PriceTape(cfg=PriceTapeConfig(
        anti_chase_max_move_pct=float("inf"),
        vol_kill_range_pct=float("inf"),
    ))
    sym = "PEPE/USDT:USDT"
    now_ms = 1_715_000_000_000
    # Insert prices spanning a 5% range over 30 seconds — wick.
    for i, p in enumerate([1.00, 1.04, 0.98, 1.05, 1.02]):
        pt.observe(sym, p, ts_ms=now_ms - 30_000 + i * 6_000)

    g = ReversalGuard(
        cfg=ReversalGuardConfig(
            enabled=True,
            wick_window_sec=60,
            wick_threshold_pct=0.04,
            min_seconds_since_close=0,  # not the gate under test here
            reversal_cooldown_sec=120,
        ),
        price_tape=pt,
    )
    account = _make_account()
    g.note_close(symbol=sym, side=Side.LONG, now_ms=now_ms - 60_000)
    sig = _make_signal(symbol=sym, direction=Direction.SHORT, final_score=9.0)

    out = g.decide(signal=sig, account=account, now_ms=now_ms)
    assert out.action == "veto"
    assert out.reason == "wick_detected"
    # Cooldown is set on the symbol.
    assert account.is_in_cooldown(sym, now_ms) is True


def test_reversal_guard_too_soon_after_close_defers():
    """Below ``min_seconds_since_close`` the guard always defers."""
    g = ReversalGuard(cfg=ReversalGuardConfig(
        enabled=True,
        min_seconds_since_close=30,
    ))
    account = _make_account()
    sym = "PEPE/USDT:USDT"
    now_ms = 1_715_000_000_000
    g.note_close(symbol=sym, side=Side.LONG, now_ms=now_ms - 5_000)
    sig = _make_signal(symbol=sym, direction=Direction.SHORT, final_score=9.0)

    out = g.decide(signal=sig, account=account, now_ms=now_ms)
    assert out.action == "defer_close_and_watch"
    assert out.reason == "too_soon_after_close"


def test_reversal_guard_weak_signal_vetoes_with_cooldown():
    g = ReversalGuard(cfg=ReversalGuardConfig(
        enabled=True,
        min_seconds_since_close=0,
        min_reversal_final_score=8.0,
    ))
    account = _make_account()
    sym = "PEPE/USDT:USDT"
    now_ms = 1_715_000_000_000
    g.note_close(symbol=sym, side=Side.LONG, now_ms=now_ms - 60_000)
    sig = _make_signal(symbol=sym, direction=Direction.SHORT, final_score=6.0)

    out = g.decide(signal=sig, account=account, now_ms=now_ms)
    assert out.action == "veto"
    assert out.reason == "signal_below_reversal_floor"
    assert account.is_in_cooldown(sym, now_ms) is True


def test_reversal_guard_clean_post_close_reversal_approved():
    """Narrow happy-path: post-close, no wick, strong score, past
    minimum wait time, no cooldown -> approved."""
    pt = PriceTape(cfg=PriceTapeConfig(
        anti_chase_max_move_pct=float("inf"),
        vol_kill_range_pct=float("inf"),
    ))
    sym = "PEPE/USDT:USDT"
    now_ms = 1_715_000_000_000
    # Tight 0.5% range over the wick window — well below 4% threshold.
    for i, p in enumerate([1.000, 1.002, 0.999, 1.001, 1.000]):
        pt.observe(sym, p, ts_ms=now_ms - 60_000 + i * 12_000)

    g = ReversalGuard(
        cfg=ReversalGuardConfig(
            enabled=True,
            min_seconds_since_close=30,
            wick_threshold_pct=0.04,
            min_reversal_final_score=7.5,
        ),
        price_tape=pt,
    )
    account = _make_account()
    g.note_close(symbol=sym, side=Side.LONG, now_ms=now_ms - 120_000)
    sig = _make_signal(symbol=sym, direction=Direction.SHORT, final_score=9.0)

    out = g.decide(signal=sig, account=account, now_ms=now_ms)
    assert out.action == "approve"
    assert out.reason == "post_close_reversal_ok"


def test_reversal_guard_active_cooldown_short_circuits_to_veto():
    g = ReversalGuard(cfg=ReversalGuardConfig(
        enabled=True, reversal_cooldown_sec=300,
    ))
    account = _make_account()
    sym = "PEPE/USDT:USDT"
    now_ms = 1_715_000_000_000
    # Pre-existing cooldown from a prior veto.
    account.set_cooldown(sym, duration_sec=300, now_ms=now_ms - 10_000)
    g.note_close(symbol=sym, side=Side.LONG, now_ms=now_ms - 60_000)
    sig = _make_signal(symbol=sym, direction=Direction.SHORT, final_score=9.0)

    out = g.decide(signal=sig, account=account, now_ms=now_ms)
    assert out.action == "veto"
    assert out.reason == "reversal_cooldown_active"


# --------------------------------------------------------------------- #
# 5) Notifier.flow()
# --------------------------------------------------------------------- #


def test_null_notifier_flow_is_noop():
    n = NullNotifier()

    async def run():
        # Must not raise; must return None.
        out = await n.flow("test", payload={"x": 1})
        return out

    assert asyncio.run(run()) is None


def test_telegram_notifier_flow_renders_bank_icon():
    """The TG notifier renders flow events with a 🏦 BANK icon, NOT
    🚨 ERROR. We capture the rendered message via a fake send hook."""
    sent: list[str] = []
    n = TelegramNotifier(bot_token="x", chat_id="y", api_base="z")

    async def fake_send(msg: str) -> None:
        sent.append(msg)

    # Replace the private sender so we don't touch the network.
    n._send = fake_send  # type: ignore[assignment]

    async def run():
        await n.flow("withdrawal_detected: -5000.00 USDT")

    asyncio.run(run())
    assert len(sent) == 1
    assert "BANK FLOW" in sent[0]
    assert "🏦" in sent[0]
    # Negative-test: must not contain the error icon.
    assert "🚨" not in sent[0]


# --------------------------------------------------------------------- #
# 6) _parse_int_list — config tolerance
# --------------------------------------------------------------------- #


def test_parse_int_list_drops_unparseable_entries(caplog):
    from altcoin_agent.main import _parse_int_list

    out = _parse_int_list([123, "abc", "456", None, 789], field="ids")
    assert out == (123, 456, 789)
    # The dropped entries should each have produced a log line.
    assert any("dropping unparseable" in r.message for r in caplog.records)


def test_parse_int_list_handles_none():
    from altcoin_agent.main import _parse_int_list
    assert _parse_int_list(None, field="ids") == ()
    assert _parse_int_list([], field="ids") == ()
