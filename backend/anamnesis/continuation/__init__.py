"""
Conversation continuation detection.

Detects when conversations continue previous topics
and links related episodes together.
"""

from .detector import (
    ContinuationDetector,
    ContinuationLink,
    ContinuationResult,
)

__all__ = [
    "ContinuationDetector",
    "ContinuationResult",
    "ContinuationLink",
]
