"""
Context window management utilities.

Smart management of LLM context windows for memory injection.
"""

from .manager import (
    ContextBudget,
    ContextManager,
    PackingResult,
    TokenCounter,
)

__all__ = [
    "ContextManager",
    "TokenCounter",
    "ContextBudget",
    "PackingResult",
]
