"""Self-learning trainer + rules promoter (QUADRANT Phase 4).

Phase 1-3 ships skeleton scaffolding:

    Trainer        — walk-forward training driver (TODO Phase 4)
    RulesPromoter  — 80% gate for production-rule promotion

Both are intentionally minimal: dataclasses + the rule promotion logic,
which is small + pure and worth landing now so the trainer can plug in
without touching the schema. The actual walk-forward loop lands in
Phase 4 when historical data is wired and PnL math is settled.
"""

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

__all__ = [
    "LearnedRule",
    "PromotionConfig",
    "RulesPromoter",
    "Trainer",
    "TrainerConfig",
    "TrainingReport",
]
