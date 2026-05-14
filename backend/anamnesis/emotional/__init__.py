"""
Emotional memory and context tracking.

Provides emotional intelligence for the memory system:
- Sentiment analysis
- Emotional pattern tracking
- Mood-aware recall
"""

from .analyzer import (
    EmotionalIntensity,
    EmotionCategory,
    SentimentAnalyzer,
    SentimentResult,
)
from .recall import (
    EmotionalRecall,
    MoodAwareResult,
)
from .tracker import (
    EmotionalPattern,
    EmotionalState,
    EmotionalTracker,
    MoodBaseline,
)

__all__ = [
    # Analyzer
    "SentimentAnalyzer",
    "SentimentResult",
    "EmotionCategory",
    "EmotionalIntensity",
    # Tracker
    "EmotionalTracker",
    "EmotionalState",
    "EmotionalPattern",
    "MoodBaseline",
    # Recall
    "EmotionalRecall",
    "MoodAwareResult",
]
