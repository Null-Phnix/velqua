"""
Confidence scoring for extracted facts.

Combines multiple signals to calculate final confidence:
- Context (fiction vs reality)
- Fantasy keywords
- Contradictions
- Confirmations (fact appears multiple times)
"""

from typing import Literal


def calculate_confidence(
    base_confidence: float,
    context: Literal["fiction", "reality", "uncertain"],
    has_fantasy_keywords: bool = False,
    has_contradictions: bool = False,
    confirmation_count: int = 0,
    is_meta_fact: bool = False
) -> float:
    """
    Calculate final confidence score for a fact.

    Args:
        base_confidence: Initial confidence from extractor (0.7-0.9)
        context: Conversation context (fiction/reality/uncertain)
        has_fantasy_keywords: Fact contains fantasy terms
        has_contradictions: Fact contradicts other facts
        confirmation_count: Number of times fact was confirmed
        is_meta_fact: Fact is about user's creative work (not fictional content)

    Returns:
        Final confidence score (0.0 to 1.0)

    Scoring strategy:
    - Fiction context → severely penalize (× 0.3)
    - Fantasy keywords → moderate penalty (× 0.6)
    - Meta-facts → boost (× 1.2)
    - Contradictions → penalize (× 0.7)
    - Confirmations → small boost (+0.05 per confirmation)
    """
    confidence = base_confidence

    # Context multipliers
    if context == "fiction":
        if is_meta_fact:
            # Meta-fact in fiction context is actually good
            # e.g., "Working on novel called The Talker"
            confidence *= 1.1
        else:
            # Regular fact in fiction context → probably about characters
            confidence *= 0.3
    elif context == "uncertain":
        # Slightly reduce confidence for uncertain context
        confidence *= 0.9

    # Fantasy keyword penalty
    if has_fantasy_keywords and not is_meta_fact:
        confidence *= 0.6

    # Contradiction penalty
    if has_contradictions:
        confidence *= 0.7

    # Confirmation bonus
    if confirmation_count > 0:
        confidence += (confirmation_count * 0.05)

    # Clamp to valid range
    confidence = max(0.0, min(1.0, confidence))

    return confidence


def classify_quality(confidence: float) -> str:
    """
    Classify fact quality based on confidence score.

    Returns:
        "high" (>0.7), "medium" (0.4-0.7), "low" (<0.4)
    """
    if confidence >= 0.7:
        return "high"
    elif confidence >= 0.4:
        return "medium"
    else:
        return "low"


def should_store(confidence: float, min_threshold: float = 0.4) -> bool:
    """
    Determine if fact should be stored based on confidence.

    Default threshold: 0.4 (medium quality minimum)
    """
    return confidence >= min_threshold
