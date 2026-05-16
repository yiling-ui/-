"""LLM auxiliary subsystems: budget enforcement + verdict cache.

The actual provider plumbing (DeepSeek/OpenAI/Anthropic) lives in the
top-level ``llm_provider.py`` and is intentionally kept separate so the
budget + cache modules can be unit-tested without any HTTP backend.

The Phase B.5 ``LLMPreRater`` worker is intentionally NOT re-exported
from this package because it depends on ``ai_engine`` which in turn
depends on this package — re-exporting it would create a circular
import. Callers that need the worker import it directly:

    from altcoin_agent.llm.pre_rater import LLMPreRater, PreRateRequest
"""

from altcoin_agent.llm.cache import LLMCache, LLMCacheEntry
from altcoin_agent.llm.token_budget import (
    BudgetMode,
    TokenBudgetManager,
    TokenBudgetState,
)

__all__ = [
    "BudgetMode",
    "LLMCache",
    "LLMCacheEntry",
    "TokenBudgetManager",
    "TokenBudgetState",
]
