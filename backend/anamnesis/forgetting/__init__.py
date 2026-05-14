"""
Forgetting mechanisms for memory management.

Implements:
- Importance decay over time
- Memory strength calculation
- Garbage collection for weak memories
- Compression (details fade, gist remains)
"""

from .compressor import MemoryCompressor
from .decay import DEFAULT_DECAY, AdaptiveDecay, DecayFunction
from .manager import ForgettingManager

__all__ = [
    "DecayFunction",
    "AdaptiveDecay",
    "DEFAULT_DECAY",
    "ForgettingManager",
    "MemoryCompressor",
]
