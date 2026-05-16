"""LLM auxiliary subsystems: budget enforcement + verdict cache.

The actual provider plumbing (DeepSeek/OpenAI/Anthropic) lives in the
top-level ``llm_provider.py`` and is intentionally kept separate so the
budget + cache modules can be unit-tested without any HTTP backend.
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
