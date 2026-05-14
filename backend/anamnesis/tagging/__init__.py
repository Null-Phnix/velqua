"""
Memory tagging module.

Provides tag management, auto-tagging, and tag-based retrieval.
"""

from .tagger import (
    AutoTagger,
    TagManager,
    TagStats,
)

__all__ = [
    "TagManager",
    "AutoTagger",
    "TagStats",
]
