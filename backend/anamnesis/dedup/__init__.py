"""
Duplicate memory detection module.

Provides detection and deduplication of similar memories.
"""

from .detector import (
    DuplicateCandidate,
    DuplicateDetector,
    DuplicateStats,
)
from .similarity import TFIDFSimilarity, quick_similarity, tokenize

__all__ = [
    "DuplicateDetector",
    "DuplicateCandidate",
    "DuplicateStats",
    "TFIDFSimilarity",
    "quick_similarity",
    "tokenize",
]
