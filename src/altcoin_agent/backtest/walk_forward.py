"""walk_forward.py — Rolling train/validate windowing (Phase B.4 / plan B.5.4).

Splits a calendar range ``[start_ms, end_ms)`` into successive
``(train_window, validate_window)`` pairs sliding by ``step``. The
trainer in Phase 4 will plug into this to:

    1. Train a candidate ruleset on each train window.
    2. Score the candidate on the immediately-following validate window.
    3. Aggregate per-window stats into a single walk-forward report.

This module ships only the windowing logic + the per-window stats
shape. The actual rule mining is deliberately separate (it lives in
``training/trainer.py``) so the windowing here is reusable for any
future analysis (e.g. replaying decisions.jsonl with a new sizer).

Pure-data, no I/O. Tests can drive it with a fake ``run_window``
callback to verify the windowing is anchored correctly without
needing a full backtest engine.
"""

from __future__ import annotations

import logging
import statistics
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# Convenience: 24h in ms, used by default windows.
DAY_MS = 24 * 3600 * 1000
WEEK_MS = 7 * DAY_MS
MONTH_MS = 30 * DAY_MS  # calendar-agnostic; trainer can override


# --------------------------------------------------------------------- #
# Window shape
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class Window:
    """Half-open ``[start_ms, end_ms)`` interval."""

    start_ms: int
    end_ms: int

    @property
    def duration_ms(self) -> int:
        return max(0, self.end_ms - self.start_ms)


@dataclass(frozen=True)
class WalkForwardSplit:
    """One (train, validate) pair the trainer iterates over."""

    train: Window
    validate: Window
    index: int  # 0-based

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "train": {"start_ms": self.train.start_ms, "end_ms": self.train.end_ms},
            "validate": {
                "start_ms": self.validate.start_ms,
                "end_ms": self.validate.end_ms,
            },
        }


@dataclass(frozen=True)
class WalkForwardConfig:
    """Operator-tunable knobs.

    The plan suggests:
      * train = 30d, validate = 30d, step = 30d  → non-overlapping
      * 3 consecutive validate windows with win_rate ≥ 0.80 → promote

    We expose the three durations + step independently so a tighter
    validation cadence (weekly) is possible without code changes.
    """

    train_window_ms: int = MONTH_MS
    validate_window_ms: int = MONTH_MS
    step_ms: int = MONTH_MS
    # Minimum number of bars required in train window to even attempt
    # a fit. The runner skips splits below this; useful at the very
    # start of a symbol's history.
    min_train_bars: int = 0


def iter_splits(
    *,
    start_ms: int,
    end_ms: int,
    cfg: WalkForwardConfig,
) -> Iterator[WalkForwardSplit]:
    """Yield successive ``(train, validate)`` splits anchored at
    ``start_ms`` and stepping by ``cfg.step_ms``.

    A split is yielded only if its *validate* window is fully contained
    in ``[start_ms, end_ms)``. The train window may extend earlier than
    ``start_ms`` if the caller wants to retain pre-history; we don't,
    so the first train window starts exactly at ``start_ms``.
    """
    if start_ms >= end_ms:
        return
    if cfg.train_window_ms <= 0 or cfg.validate_window_ms <= 0:
        raise ValueError("walk_forward: train/validate windows must be positive")
    if cfg.step_ms <= 0:
        raise ValueError("walk_forward: step must be positive")

    cursor = start_ms
    idx = 0
    while True:
        train = Window(cursor, cursor + cfg.train_window_ms)
        validate = Window(train.end_ms, train.end_ms + cfg.validate_window_ms)
        if validate.end_ms > end_ms:
            return
        yield WalkForwardSplit(train=train, validate=validate, index=idx)
        idx += 1
        cursor += cfg.step_ms


# --------------------------------------------------------------------- #
# Per-window stats
# --------------------------------------------------------------------- #


@dataclass
class WindowStats:
    """Aggregated metrics for one validate window's PnL / trades."""

    index: int
    samples: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl_usdt: float = 0.0
    avg_pnl_pct: float = 0.0
    sharpe: float = 0.0
    max_drawdown_pct: float = 0.0

    @property
    def win_rate(self) -> float:
        total = self.wins + self.losses
        return self.wins / total if total else 0.0

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["win_rate"] = self.win_rate
        return d


def stats_from_pnls(
    pnls_pct: list[float],
    *,
    index: int = 0,
    starting_equity: float = 1.0,
) -> WindowStats:
    """Compute ``WindowStats`` from a sequence of *fractional* trade PnLs.

    ``pnls_pct[i] = 0.05`` means the i-th trade made +5% on its capital.
    Sharpe is the simple mean / stddev ratio (no annualisation, no
    risk-free rate) — matches what the trainer needs for the 80% gate.
    Max drawdown is computed on a synthetic equity curve compounding
    the trades in order.
    """
    if not pnls_pct:
        return WindowStats(index=index)
    wins = sum(1 for p in pnls_pct if p > 0)
    losses = sum(1 for p in pnls_pct if p < 0)
    avg = statistics.fmean(pnls_pct)
    if len(pnls_pct) >= 2:
        sd = statistics.pstdev(pnls_pct)
        sharpe = avg / sd if sd > 0 else 0.0
    else:
        sharpe = 0.0
    # Max drawdown on the compounded equity curve.
    eq = starting_equity
    peak = eq
    max_dd = 0.0
    for p in pnls_pct:
        eq *= (1.0 + p)
        if eq > peak:
            peak = eq
        if peak > 0:
            dd = (peak - eq) / peak
            if dd > max_dd:
                max_dd = dd
    total_pnl = (eq - starting_equity)
    return WindowStats(
        index=index,
        samples=len(pnls_pct),
        wins=wins,
        losses=losses,
        total_pnl_usdt=float(total_pnl),
        avg_pnl_pct=float(avg),
        sharpe=float(sharpe),
        max_drawdown_pct=float(max_dd),
    )


# --------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------- #


# A run_window callback takes the split and returns either a
# ``WindowStats`` directly or a list of fractional PnLs (we accept both
# so the trainer can plug in incrementally).
RunWindowFn = Callable[
    [WalkForwardSplit],
    "WindowStats | list[float]",
]


@dataclass
class WalkForwardReport:
    splits: list[WalkForwardSplit] = field(default_factory=list)
    window_stats: list[WindowStats] = field(default_factory=list)

    def aggregate(self) -> dict[str, Any]:
        """Roll all per-window stats into one summary dict."""
        all_pnls: list[float] = []
        for ws in self.window_stats:
            # Reconstruct a flat pnl list weighted by samples for an
            # overall sharpe — coarse but matches what the trainer wants.
            if ws.samples > 0:
                # We don't have the raw trades; approximate by treating
                # the avg as a single trade. The trainer can pass in
                # WindowStats already containing accurate sharpe.
                all_pnls.extend([ws.avg_pnl_pct] * ws.samples)
        agg = stats_from_pnls(all_pnls, index=-1)
        consecutive_winning = self._consecutive_winning_windows(threshold=0.80)
        return {
            "splits": len(self.splits),
            "windows_evaluated": len(self.window_stats),
            "samples": agg.samples,
            "win_rate_overall": agg.win_rate,
            "avg_pnl_pct": agg.avg_pnl_pct,
            "sharpe_overall": agg.sharpe,
            "max_drawdown_pct": agg.max_drawdown_pct,
            "consecutive_winning_windows": consecutive_winning,
            "windows": [w.as_dict() for w in self.window_stats],
        }

    def _consecutive_winning_windows(self, threshold: float = 0.80) -> int:
        """Longest tail-streak of windows with ``win_rate >= threshold``.

        Used by the trainer's promotion gate: 3 consecutive validate
        windows passing 0.80 → eligible to promote.
        """
        run = 0
        for ws in self.window_stats:
            if ws.win_rate >= threshold:
                run += 1
            else:
                run = 0
        return run


def run_walk_forward(
    *,
    start_ms: int,
    end_ms: int,
    cfg: WalkForwardConfig,
    run_window: RunWindowFn,
) -> WalkForwardReport:
    """Iterate splits, invoke ``run_window`` on each, collect a report.

    ``run_window`` returns either a ``WindowStats`` (full control) or a
    list of fractional PnLs (we'll aggregate). Either way the report
    contains one ``WindowStats`` per evaluated split.
    """
    report = WalkForwardReport()
    for split in iter_splits(start_ms=start_ms, end_ms=end_ms, cfg=cfg):
        report.splits.append(split)
        try:
            res = run_window(split)
        except Exception:
            logger.exception(
                "walk_forward: run_window raised on split %d", split.index
            )
            continue
        if isinstance(res, WindowStats):
            ws = res
            ws.index = split.index
        elif isinstance(res, list):
            ws = stats_from_pnls(res, index=split.index)
        else:
            logger.warning(
                "walk_forward: ignoring unsupported run_window result %r",
                type(res),
            )
            continue
        report.window_stats.append(ws)
    return report


__all__ = [
    "DAY_MS",
    "MONTH_MS",
    "RunWindowFn",
    "WEEK_MS",
    "WalkForwardConfig",
    "WalkForwardReport",
    "WalkForwardSplit",
    "Window",
    "WindowStats",
    "iter_splits",
    "run_walk_forward",
    "stats_from_pnls",
]
