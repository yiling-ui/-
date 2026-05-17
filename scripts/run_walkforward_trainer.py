"""run_walkforward_trainer.py — operator/cron entrypoint for Phase 4 training.

Drives ``WalkforwardTrainer.run`` over a window of historical klines,
mines rules, applies the 80% promotion gate, and writes
``production_rules.json`` + ``candidate_rules.json`` under the
configured state dir. The live daemon's
``ProductionRulesLoader`` picks the new file up on its next mtime
check, no restart required.

Two observation sources are supported:

  --observations PATH        a JSONL file where each line is one
                             ``{ts_ms, symbol, pnl_pct, features:{...}}``
                             record. This is the format
                             :class:`MatchingEngine` will dump after a
                             backtest run.

  --replay-only              skip mining; just re-read the already-
                             written observations in ``state_dir`` and
                             re-promote against the configured
                             ``PromotionConfig``. Useful for tuning the
                             gate without re-mining.

Usage example:

    # First time: run a 90-day historical backtest, dump observations.
    python scripts/backtest_30d.py ...           # produces obs.jsonl

    # Then mine + promote:
    python scripts/run_walkforward_trainer.py \\
        --observations obs.jsonl \\
        --start 2024-09-01T00:00:00Z \\
        --end   2025-02-01T00:00:00Z \\
        --state-dir .kiro/state/training

    # Subsequently, re-tune promotion gates against the same obs:
    python scripts/run_walkforward_trainer.py --replay-only \\
        --observations obs.jsonl --state-dir .kiro/state/training \\
        --min-win-rate 0.85
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from altcoin_agent.training.rule_miner import MinerConfig, TradeObservation
from altcoin_agent.training.rules_promoter import PromotionConfig, RulesPromoter
from altcoin_agent.training.walkforward_trainer import (
    WalkforwardTrainer,
    WalkforwardTrainerConfig,
    list_observation_provider,
)

logger = logging.getLogger("walkforward_trainer")


# --------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------- #


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="run_walkforward_trainer",
        description=(
            "Mine rules from trade observations and promote the ones "
            "that clear the 80% production gate."
        ),
    )
    p.add_argument("--observations", required=True,
                   help="path to JSONL file with TradeObservation records")
    span = p.add_mutually_exclusive_group()
    span.add_argument("--days", type=int,
                      help="span ending at now, in days")
    span.add_argument("--start", type=str,
                      help="ISO-8601 UTC start of training window")
    p.add_argument("--end", type=str, default="",
                   help="ISO-8601 UTC end (default: now)")

    p.add_argument("--state-dir", required=True,
                   help="output dir for production_rules.json + candidate_rules.json")

    # Walk-forward window sizes (default: 1 month / 1 month / 1 month).
    p.add_argument("--train-days", type=int, default=30)
    p.add_argument("--validate-days", type=int, default=30)
    p.add_argument("--step-days", type=int, default=30)

    # Miner knobs.
    p.add_argument("--min-samples-per-bucket", type=int, default=5)
    # Promotion gate.
    p.add_argument("--min-samples", type=int, default=30,
                   help="rule must have at least this many samples to promote")
    p.add_argument("--min-win-rate", type=float, default=0.80)
    p.add_argument("--min-sharpe", type=float, default=1.5)
    p.add_argument("--min-validation-months", type=int, default=3)

    p.add_argument("--replay-only", action="store_true",
                   help="re-promote existing observations against the "
                        "current gate without re-mining the windows")

    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def _iso_to_ms(iso: str) -> int:
    from datetime import datetime, timezone
    txt = iso.strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(txt)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _load_observations(path: str) -> list[TradeObservation]:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"observations file not found: {path}")
    out: list[TradeObservation] = []
    with p.open(encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("skipping line %d (%s): %s", lineno, exc, line[:80])
                continue
            try:
                ob = TradeObservation(
                    ts_ms=int(d["ts_ms"]),
                    symbol=str(d["symbol"]),
                    pnl_pct=float(d["pnl_pct"]),
                    features={
                        str(k): str(v)
                        for k, v in (d.get("features") or {}).items()
                    },
                )
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning("skipping line %d (%s): %s", lineno, exc, line[:80])
                continue
            out.append(ob)
    return out


# --------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------- #


def run(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    observations = _load_observations(args.observations)
    if not observations:
        logger.error("no observations loaded from %s", args.observations)
        return 2
    logger.info("loaded %d observations from %s",
                len(observations), args.observations)

    end_ms = (
        _iso_to_ms(args.end) if args.end else int(time.time() * 1000)
    )
    if args.days:
        start_ms = end_ms - int(args.days) * 86_400_000
    elif args.start:
        start_ms = _iso_to_ms(args.start)
    else:
        # Infer from observations if no explicit window.
        start_ms = min(o.ts_ms for o in observations)
        end_ms = max(o.ts_ms for o in observations) + 1
    if start_ms >= end_ms:
        logger.error("inverted window: start=%s >= end=%s", start_ms, end_ms)
        return 2

    state_dir = args.state_dir
    Path(state_dir).mkdir(parents=True, exist_ok=True)

    cfg = WalkforwardTrainerConfig(
        train_window_ms=args.train_days * 86_400_000,
        validate_window_ms=args.validate_days * 86_400_000,
        step_ms=args.step_days * 86_400_000,
        promotion=PromotionConfig(
            min_samples=args.min_samples,
            min_win_rate=args.min_win_rate,
            min_sharpe=args.min_sharpe,
            min_validation_months=args.min_validation_months,
        ),
        miner=MinerConfig(
            min_samples_per_bucket=args.min_samples_per_bucket,
        ),
        state_dir=state_dir,
    )

    if args.replay_only:
        # Just feed the already-built observations through the
        # promoter as a single "window" (skip mining + walk-forward).
        from altcoin_agent.training.rule_miner import mine_rules
        rules = mine_rules(observations, cfg.miner)
        promoter = RulesPromoter(cfg=cfg.promotion, state_dir=state_dir)
        promoted, demoted = promoter.promote_all(rules, now_ts=int(time.time()))
        logger.info(
            "replay-only: mined=%d promoted_now=%d demoted_now=%d "
            "production_total=%d candidate_total=%d",
            len(rules), len(promoted), len(demoted),
            len(promoter.production_rules()),
            len(promoter.candidate_rules()),
        )
        return 0

    trainer = WalkforwardTrainer(cfg=cfg)
    provider = list_observation_provider(observations)
    report = trainer.run(
        start_ms=start_ms, end_ms=end_ms, provider=provider,
    )

    logger.info(
        "training complete: splits=%d windows_observed=%d "
        "rules_pooled=%d promoted_now=%d demoted_now=%d "
        "production_total=%d candidate_total=%d",
        report.splits,
        sum(1 for w in report.windows
            if w.train_observations or w.validate_observations),
        report.rules_after_pooling,
        report.rules_promoted_now,
        report.rules_demoted_now,
        report.rules_production_total,
        report.rules_candidate_total,
    )
    summary_path = Path(state_dir) / "last_training_run.json"
    summary_path.write_text(
        json.dumps(report.as_dict(), indent=2, sort_keys=True)
    )
    logger.info("wrote %s", summary_path)
    return 0


if __name__ == "__main__":
    sys.exit(run())
