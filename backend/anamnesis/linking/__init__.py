"""
Memory linking module.

Provides explicit linking between related memories.
"""

from .manager import (
    LinkManager,
    LinkStats,
    LinkType,
    MemoryLink,
)

__all__ = [
    "LinkManager",
    "LinkType",
    "MemoryLink",
    "LinkStats",
]
