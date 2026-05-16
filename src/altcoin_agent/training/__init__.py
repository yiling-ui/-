"""Self-learning trainer + rules promoter (QUADRANT Phase 4).

Modules:

    rules_promoter           — 80% gate for production-rule promotion
    rule_miner               — turn (features, pnl) observations into
                               LearnedRule candidates per feature bucket
    walkforward_trainer      — drive walk-forward training: split into
                               (train, validate) windows, mine rules on
                               the train window, regrade on validate,
                               accumulate cross-window, promote
    trainer                  — minimal daily-cycle wrapper (Phase 1-3
                               carry-over; kept for backward compat)
"""

from altcoin_agent.training.rule_miner import (
    MinerConfig,
    RuleAccumulator,
    TradeObservation,
    bucketize_pct,
    bucketize_score,
    mine_rules,
)
from altcoin_agent.training.rules_promoter import (
    LearnedRule,
    PromotionConfig,
    RulesPromoter,
)
from altcoin_agent.training.trainer import (
    Trainer,
    TrainerConfig,
    TrainingReport,
)
from altcoin_agent.training.walkforward_trainer import (
    ObservationProvider,
    WalkforwardTrainer,
    WalkforwardTrainerConfig,
    WalkforwardTrainerReport,
    WalkforwardWindowReport,
    list_observation_provider,
)

__all__ = [
    "LearnedRule",
    "MinerConfig",
    "ObservationProvider",
    "PromotionConfig",
    "RuleAccumulator",
    "RulesPromoter",
    "TradeObservation",
    "Trainer",
    "TrainerConfig",
    "TrainingReport",
    "WalkforwardTrainer",
    "WalkforwardTrainerConfig",
    "WalkforwardTrainerReport",
    "WalkforwardWindowReport",
    "bucketize_pct",
    "bucketize_score",
    "list_observation_provider",
    "mine_rules",
]
