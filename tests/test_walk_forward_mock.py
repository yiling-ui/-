"""tests/test_walk_forward_mock.py — Phase B.4 walk-forward windowing."""

from __future__ import annotations

import pytest

from altcoin_agent.backtest.walk_forward import (
    DAY_MS,
    MONTH_MS,
    WalkForwardConfig,
    WalkForwardSplit,
    WindowStats,
    iter_splits,
    run_walk_forward,
    stats_from_pnls,
)

# ----------------------- iter_splits ----------------------- #


def test_iter_splits_emits_consecutive_non_overlapping_when_step_eq_train():
    cfg = WalkForwardConfig(
        train_window_ms=MONTH_MS, validate_window_ms=MONTH_MS, step_ms=MONTH_MS,
    )
    splits = list(iter_splits(start_ms=0, end_ms=4 * MONTH_MS, cfg=cfg))
    # First split: train [0, 1mo), validate [1mo, 2mo) — fits.
    # Second:        train [1mo, 2mo), validate [2mo, 3mo) — fits.
    # Third would need validate to end at 4mo — fits.
    # Fourth would need validate to end at 5mo — out of range.
    assert len(splits) == 3
    assert splits[0].train.start_ms == 0
    assert splits[0].validate.end_ms == 2 * MONTH_MS
    assert splits[2].validate.end_ms == 4 * MONTH_MS


def test_iter_splits_supports_overlapping_step():
    cfg = WalkForwardConfig(
        train_window_ms=2 * MONTH_MS,
        validate_window_ms=MONTH_MS,
        step_ms=MONTH_MS,
    )
    splits = list(iter_splits(start_ms=0, end_ms=4 * MONTH_MS, cfg=cfg))
    # Splits: idx 0 -> train [0, 2mo), validate [2mo, 3mo)
    #         idx 1 -> train [1mo, 3mo), validate [3mo, 4mo)
    assert len(splits) == 2
    assert splits[1].train.start_ms == MONTH_MS
    assert splits[1].validate.end_ms == 4 * MONTH_MS


def test_iter_splits_no_emit_if_window_doesnt_fit():
    cfg = WalkForwardConfig(
        train_window_ms=2 * MONTH_MS, validate_window_ms=2 * MONTH_MS,
        step_ms=MONTH_MS,
    )
    splits = list(iter_splits(start_ms=0, end_ms=2 * MONTH_MS, cfg=cfg))
    assert splits == []


def test_iter_splits_rejects_invalid_durations():
    bad = WalkForwardConfig(train_window_ms=0, validate_window_ms=1, step_ms=1)
    with pytest.raises(ValueError):
        list(iter_splits(start_ms=0, end_ms=10, cfg=bad))
    bad2 = WalkForwardConfig(train_window_ms=1, validate_window_ms=1, step_ms=0)
    with pytest.raises(ValueError):
        list(iter_splits(start_ms=0, end_ms=10, cfg=bad2))


def test_iter_splits_empty_range_yields_nothing():
    cfg = WalkForwardConfig()
    assert list(iter_splits(start_ms=10, end_ms=10, cfg=cfg)) == []
    assert list(iter_splits(start_ms=20, end_ms=10, cfg=cfg)) == []


# ----------------------- stats_from_pnls ----------------------- #


def test_stats_from_pnls_empty():
    s = stats_from_pnls([])
    assert s.samples == 0
    assert s.win_rate == 0.0


def test_stats_from_pnls_basic_counts():
    s = stats_from_pnls([0.10, -0.05, 0.20, -0.02])
    assert s.samples == 4
    assert s.wins == 2
    assert s.losses == 2
    assert s.win_rate == pytest.approx(0.5)
    assert s.avg_pnl_pct == pytest.approx((0.10 - 0.05 + 0.20 - 0.02) / 4)
    assert s.sharpe > 0  # mean > 0 here


def test_stats_zero_volatility_yields_zero_sharpe():
    s = stats_from_pnls([0.05, 0.05, 0.05])
    assert s.sharpe == 0.0


def test_stats_max_drawdown_from_compounded_curve():
    # +50% then -50% then -10%: peak after first trade, then 30% dd.
    pnls = [0.5, -0.5, -0.10]
    s = stats_from_pnls(pnls, starting_equity=1.0)
    # Equity: 1.0 -> 1.5 -> 0.75 -> 0.675
    # Peak = 1.5, trough = 0.675, dd = (1.5 - 0.675) / 1.5 = 0.55
    assert s.max_drawdown_pct == pytest.approx(0.55, abs=1e-6)


# ----------------------- run_walk_forward ----------------------- #


def test_run_walk_forward_with_pnl_list_callback():
    cfg = WalkForwardConfig(
        train_window_ms=DAY_MS, validate_window_ms=DAY_MS, step_ms=DAY_MS,
    )

    def fake_run(split: WalkForwardSplit) -> list[float]:
        # Each split produces 5 winning trades.
        return [0.05] * 5

    report = run_walk_forward(
        start_ms=0, end_ms=4 * DAY_MS, cfg=cfg, run_window=fake_run,
    )
    assert len(report.window_stats) == 3
    assert all(ws.wins == 5 for ws in report.window_stats)
    agg = report.aggregate()
    assert agg["windows_evaluated"] == 3
    assert agg["consecutive_winning_windows"] == 3
    assert agg["win_rate_overall"] == pytest.approx(1.0)


def test_run_walk_forward_with_window_stats_callback():
    cfg = WalkForwardConfig(
        train_window_ms=DAY_MS, validate_window_ms=DAY_MS, step_ms=DAY_MS,
    )

    def fake_run(split: WalkForwardSplit) -> WindowStats:
        return WindowStats(
            index=99, samples=10, wins=8, losses=2,
            avg_pnl_pct=0.05, sharpe=2.0, max_drawdown_pct=0.10,
        )

    report = run_walk_forward(
        start_ms=0, end_ms=2 * DAY_MS, cfg=cfg, run_window=fake_run,
    )
    assert len(report.window_stats) == 1
    # The driver overrides ``index`` with the split's index.
    assert report.window_stats[0].index == 0


def test_run_walk_forward_recovers_from_callback_exception():
    cfg = WalkForwardConfig(
        train_window_ms=DAY_MS, validate_window_ms=DAY_MS, step_ms=DAY_MS,
    )
    calls = {"n": 0}

    def fake_run(split: WalkForwardSplit) -> list[float]:
        calls["n"] += 1
        if split.index == 1:
            raise RuntimeError("boom")
        return [0.01] * 3

    report = run_walk_forward(
        start_ms=0, end_ms=3 * DAY_MS, cfg=cfg, run_window=fake_run,
    )
    # 2 splits emitted (validate must fit), 1 of which threw.
    assert calls["n"] == 2
    assert len(report.window_stats) == 1


def test_consecutive_winning_windows_resets_on_failure():
    cfg = WalkForwardConfig(
        train_window_ms=DAY_MS, validate_window_ms=DAY_MS, step_ms=DAY_MS,
    )
    sequence = iter([
        [0.1] * 10,            # win_rate 1.0
        [0.1] * 10,            # win_rate 1.0
        [-0.1] * 10,           # win_rate 0.0  -> resets streak
        [0.1] * 10,            # win_rate 1.0  -> streak length 1
    ])

    def fake_run(split: WalkForwardSplit) -> list[float]:
        return next(sequence)

    report = run_walk_forward(
        start_ms=0, end_ms=5 * DAY_MS, cfg=cfg, run_window=fake_run,
    )
    agg = report.aggregate()
    assert agg["consecutive_winning_windows"] == 1


def test_unsupported_callback_return_skipped():
    cfg = WalkForwardConfig(
        train_window_ms=DAY_MS, validate_window_ms=DAY_MS, step_ms=DAY_MS,
    )

    def fake_run(split: WalkForwardSplit):
        return "garbage"

    report = run_walk_forward(
        start_ms=0, end_ms=2 * DAY_MS, cfg=cfg, run_window=fake_run,
    )
    assert report.splits  # iter_splits emitted at least one
    assert report.window_stats == []
