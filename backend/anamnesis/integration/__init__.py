"""
LLM Integration layer for memory injection.

Handles:
- Formatting memories for LLM context
- Managing context budget
- Prioritizing what memories to include
- Different injection styles (system prompt, user context, etc.)
"""

from .context_manager import (
    PRESETS,
    ContextAllocation,
    ContextBudget,
    ContextManager,
    PriorityStrategy,
    get_preset,
)
from .formatter import FormatStyle, FormattedMemory, MemoryFormatter
from .injector import (
    InjectionMode,
    InjectionResult,
    MemoryInjector,
    create_injector_for_model,
    quick_inject,
)

__all__ = [
    # Formatter
    "MemoryFormatter",
    "FormatStyle",
    "FormattedMemory",
    # Context Manager
    "ContextManager",
    "ContextBudget",
    "ContextAllocation",
    "PriorityStrategy",
    "PRESETS",
    "get_preset",
    # Injector
    "MemoryInjector",
    "InjectionMode",
    "InjectionResult",
    "create_injector_for_model",
    "quick_inject",
]
