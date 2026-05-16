"""tests/test_phase_4_e2e_mock.py — Phase 4 end-to-end integration.

Drives the full Phase B.4 + Phase 4 stack:

    HistoricalDataLoader (with stub fetcher)
        → BacktestDataAdapter
            → MatchingEngine (ExchangeAdapter Protocol)
                → fills + realized PnL
                    → TradeObservation
                        → WalkforwardTrainer
                            → RulesPromoter

The point of this test is **not** to assert specific PnL numbers; it
is to verify that the seams hold: the backtest path generates the
same shape of TradeObservation the trainer expects, and the trainer
emits LearnedRules with sensible aggregates that make it into the
RulesPromoter pools.

This is also the deferred B.4.4 ("实盘/回测一致性验证") deliverable —
we use the live ExchangeAdapter Protocol to drive a backtest, and the
test would catch any drift in that contract.
"""

from __future__ import annotations

import pytest

from altcoin_agent.backtest.data_adapter import BacktestDataAdapter
from altcoin_agent.backtest.historical_loader import (
    TIMEFRAME_MS,
    HistoricalDataLoader,
)
from altcoin_agent.backtest.matching_engine import (
    MatchingEngine,
    MatchingEngineConfig,
)
from altcoin_agent.backtest.slippage_model import SlippageModel, SlippageParams
from altcoin_agent.risk.executor import ExchangeAdapter
from altcoin_agent.risk.state import Side
from altcoin_agent.training.rule_miner import MinerConfig, TradeObservation
from altcoin_agent.training.rules_promoter import PromotionConfig
from altcoin_agent.training.walkforward_trainer import (
    WalkforwardTrainer,
    WalkforwardTrainerConfig,
    list_observation_provider,
)

# --------------------------------------------------------------------- #
# Fixture builders
# --------------------------------------------------------------------- #


class _StubFetcher:
    def __init__(self, bars):
        self.bars = sorted(bars, key=lambda b: b[0])

    def fetch_ohlcv(self, symbol, timeframe, since, limit):
        return [list(b) for b in self.bars if b[0] >= since][:limit]


def _winning_round_trip_bars(start_ts: int) -> list[list[float]]:
    """Three bars: enter at 100, hold, exit at 110 → +10% gross."""
    step = TIMEFRAME_MS["1m"]
    return [
        [start_ts,            100.0, 100.0, 100.0, 100.0, 100.0],
        [start_ts + step,     100.0, 100.0, 100.0, 100.0, 100.0],
        [start_ts + 2 * step, 110.0, 110.0, 110.0, 110.0, 100.0],
    ]


def _losing_round_trip_bars(start_ts: int) -> list[list[float]]:
    """Three bars: enter at 100, exit at 95 → -5% gross."""
    step = TIMEFRAME_MS["1m"]
    return [
        [start_ts,            100.0, 100.0, 100.0, 100.0, 100.0],
        [start_ts + step,     100.0, 100.0, 100.0, 100.0, 100.0],
        [start_ts + 2 * step,  95.0,  95.0,  95.0,  95.0, 100.0],
    ]


def _build_engine(tmp_path, bars) -> tuple[MatchingEngine, BacktestDataAdapter]:
    loader = HistoricalDataLoader(
        fetcher=_StubFetcher(bars),
        cache_root=str(tmp_path),
        inter_call_sleep_sec=0.0,
        sleep_fn=lambda *_: None,
    )
    loader.download(
        symbol="X", timeframe="1m",
        start_ms=int(bars[0][0]),
        end_ms=int(bars[-1][0]) + TIMEFRAME_MS["1m"],
    )
    adapter = BacktestDataAdapter(loader=loader, timeframe="1m")
    adapter.preload(
        symbols=["X"],
        start_ms=int(bars[0][0]),
        end_ms=int(bars[-1][0]) + TIMEFRAME_MS["1m"],
    )
    engine = MatchingEngine(
        data=adapter,
        slippage=SlippageModel(SlippageParams(
            base_spread=0.0, impact_coeff=0.0, vol_premium_coeff=0.0,
            taker_fee=0.0,
        )),
        cfg=MatchingEngineConfig(
            default_top_depth_usdt=100_000.0,
            default_realized_vol=0.0,
        ),
        starting_balance_usdt=10_000.0,
    )
    return engine, adapter


async def _drive_round_trip(engine: MatchingEngine, adapter: BacktestDataAdapter) -> float:
    """Open at first bar's close, close at last bar's close. Return PnL %."""
    it = adapter.iter_bars("X")
    list(it)  # exhaust so cursor sits on last bar
    # Re-iterate manually to control fills bar-by-bar.
    adapter._cursor_ts_by_symbol.clear()  # noqa: SLF001 — test drive
    it = adapter.iter_bars("X")
    next(it)
    entry_price = engine.data.current_bar("X").close
    await engine.market_order("X", Side.LONG, size=1.0)
    # Skip middle bar.
    next(it)
    next(it)
    exit_price = engine.data.current_bar("X").close
    await engine.market_order("X", Side.SHORT, size=1.0, reduce_only=True)
    return (exit_price - entry_price) / entry_price


# --------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_matching_engine_pnl_matches_phase_4_expectation(tmp_path):
    """Smoke: a +10% bar pair through the matching engine returns
    realised PnL 10% of notional, which is what the trainer would
    receive as a TradeObservation."""
    engine, adapter = _build_engine(tmp_path, _winning_round_trip_bars(0))
    pct = await _drive_round_trip(engine, adapter)
    assert pct == pytest.approx(0.10)
    assert engine.realized_pnl_usdt == pytest.approx(10.0)


@pytest.mark.asyncio
async def test_round_trip_fills_translate_to_trade_observation(tmp_path):
    """Verify the live → trainer seam: a closed round trip's PnL
    becomes a single TradeObservation the trainer can mine."""
    engine, adapter = _build_engine(tmp_path, _winning_round_trip_bars(0))
    pct = await _drive_round_trip(engine, adapter)
    assert engine.fills[-1]["reduce_only"] is True
    obs = TradeObservation(
        ts_ms=engine.fills[-1]["ts_ms"],
        symbol="X",
        pnl_pct=pct,
        features={
            "quadrant": "A", "phase": "ramp",
            "score_bucket": "score>=80", "rejected_reason": "none",
        },
    )
    assert obs.pnl_pct == pytest.approx(0.10)


@pytest.mark.asyncio
async def test_full_pipeline_promotes_consistent_winner(tmp_path):
    """End-to-end: drive 4 walk-forward windows where every trade is
    a winner. The "quadrant=A" rule should hit 4 qualifying validate
    windows and (with sufficient samples) get promoted into the
    production pool."""
    # Build a fixed observation set: 12 winners spread across 5 days,
    # all on quadrant A. Phase 4's RulesPromoter caps the production
    # gate at samples >= 30 (default), so we craft the windows to clear
    # that cap when pooled.
    obs: list[TradeObservation] = []
    for day in range(5):
        for i in range(15):
            obs.append(TradeObservation(
                ts_ms=day * (24 * 3600 * 1000) + i * 1000,
                symbol="X",
                pnl_pct=0.05,
                features={
                    "quadrant": "A", "phase": "ramp",
                    "score_bucket": "score>=80", "rejected_reason": "none",
                },
            ))

    cfg = WalkforwardTrainerConfig(
        train_window_ms=24 * 3600 * 1000,
        validate_window_ms=24 * 3600 * 1000,
        step_ms=24 * 3600 * 1000,
        miner=MinerConfig(
            min_samples_per_bucket=10,
            feature_combos=(("quadrant",),),
        ),
        promotion=PromotionConfig(
            min_samples=30,
            min_win_rate=0.80,
            min_sharpe=0.0,
            min_validation_months=3,
        ),
        state_dir=str(tmp_path),
    )
    trainer = WalkforwardTrainer(cfg=cfg)
    provider = list_observation_provider(obs)

    report = trainer.run(
        start_ms=0,
        end_ms=5 * 24 * 3600 * 1000,
        provider=provider,
        now_ts=1_700_000_000,
    )
    assert report.splits == 4
    assert report.rules_promoted_now == 1
    assert report.rules_production_total == 1


@pytest.mark.asyncio
async def test_full_pipeline_loser_stays_in_candidates(tmp_path):
    """A consistent loser must NOT clear the 80% promotion gate."""
    obs = [
        TradeObservation(
            ts_ms=day * (24 * 3600 * 1000) + i * 1000,
            symbol="X",
            pnl_pct=-0.05,
            features={
                "quadrant": "A", "phase": "ramp",
                "score_bucket": "score>=80", "rejected_reason": "none",
            },
        )
        for day in range(5) for i in range(15)
    ]
    cfg = WalkforwardTrainerConfig(
        train_window_ms=24 * 3600 * 1000,
        validate_window_ms=24 * 3600 * 1000,
        step_ms=24 * 3600 * 1000,
        miner=MinerConfig(
            min_samples_per_bucket=10,
            feature_combos=(("quadrant",),),
        ),
        promotion=PromotionConfig(min_samples=30, min_win_rate=0.80,
                                  min_sharpe=0.0, min_validation_months=3),
        state_dir=str(tmp_path),
    )
    trainer = WalkforwardTrainer(cfg=cfg)
    report = trainer.run(
        start_ms=0,
        end_ms=5 * 24 * 3600 * 1000,
        provider=list_observation_provider(obs),
        now_ts=1_700_000_000,
    )
    assert report.rules_promoted_now == 0
    assert report.rules_production_total == 0
    candidate_ids = {r.rule_id for r in trainer.promoter.candidate_rules()}
    assert "quadrant=A" in candidate_ids


def test_phase_4_pipeline_is_token_free():
    """Audit guarantee: the trainer + miner make zero LLM calls in v1.
    This test asserts the report's tokens_used field stays at 0 even
    on a substantial mining run."""
    obs = [
        TradeObservation(
            ts_ms=i * 60_000, symbol="X", pnl_pct=0.05,
            features={
                "quadrant": "A", "phase": "ramp",
                "score_bucket": "score>=80", "rejected_reason": "none",
            },
        )
        for i in range(500)
    ]
    cfg = WalkforwardTrainerConfig(
        train_window_ms=60 * 60_000,
        validate_window_ms=60 * 60_000,
        step_ms=60 * 60_000,
    )
    trainer = WalkforwardTrainer(cfg=cfg)
    report = trainer.run(
        start_ms=0, end_ms=500 * 60_000,
        provider=list_observation_provider(obs),
    )
    assert report.tokens_used == 0


@pytest.mark.asyncio
async def test_matching_engine_fills_remain_protocol_compliant(tmp_path):
    """B.4.4: re-affirm the live ExchangeAdapter Protocol contract is
    satisfied by the simulator after Phase 4 plumbing lands."""
    engine, _ = _build_engine(tmp_path, _winning_round_trip_bars(0))
    assert isinstance(engine, ExchangeAdapter)
