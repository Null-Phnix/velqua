"""
Memory importance scoring model.

Calculates importance scores based on multiple factors:
- Base importance
- Access frequency
- Recency of access
- Emotional intensity
- Confirmation count (for facts)
- Time decay
"""

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..models import EmotionalValence, Episode, Fact


@dataclass
class ScoringConfig:
    """Configuration for importance scoring."""

    # Weight factors (should sum to ~1.0 for normalized scoring)
    base_weight: float = 0.3        # Weight of original importance
    access_weight: float = 0.25     # Weight of access frequency
    recency_weight: float = 0.2     # Weight of recent access
    emotion_weight: float = 0.15    # Weight of emotional intensity
    confirm_weight: float = 0.1     # Weight of confirmation count (facts only)

    # Decay parameters
    recency_half_life_days: float = 14.0  # Half-life for recency decay
    access_log_base: float = 2.0          # Log base for access count scaling

    # Bounds
    min_importance: float = 0.0
    max_importance: float = 1.0

    # Emotional intensity mapping
    emotion_intensity: Dict[EmotionalValence, float] = field(default_factory=lambda: {
        EmotionalValence.VERY_POSITIVE: 1.0,
        EmotionalValence.VERY_NEGATIVE: 1.0,
        EmotionalValence.POSITIVE: 0.7,
        EmotionalValence.NEGATIVE: 0.7,
        EmotionalValence.NEUTRAL: 0.3,
    })


@dataclass
class ScoringResult:
    """Result of importance scoring."""
    final_score: float
    components: Dict[str, float]
    explanation: str


class ImportanceScorer:
    """
    Calculates memory importance scores.

    Uses a weighted combination of factors to determine
    how important/memorable a piece of information is.
    """

    def __init__(self, config: Optional[ScoringConfig] = None):
        self.config = config or ScoringConfig()

    def score_episode(
        self,
        episode: Episode,
        access_count: int = 0,
        last_accessed: Optional[datetime] = None,
    ) -> ScoringResult:
        """
        Calculate importance score for an episode.

        Args:
            episode: The episode to score
            access_count: Number of times accessed
            last_accessed: When last accessed (or episode started_at if never)

        Returns:
            ScoringResult with final score and breakdown
        """
        components = {}

        # Base importance
        base = episode.importance
        components['base'] = base * self.config.base_weight

        # Access frequency (logarithmic scaling)
        access_score = self._calculate_access_score(access_count)
        components['access'] = access_score * self.config.access_weight

        # Recency
        ref_time = last_accessed or episode.started_at or datetime.now()
        recency_score = self._calculate_recency_score(ref_time)
        components['recency'] = recency_score * self.config.recency_weight

        # Emotional intensity
        emotion_score = self._calculate_emotion_score(episode.overall_valence)
        components['emotion'] = emotion_score * self.config.emotion_weight

        # For episodes, confirmation doesn't apply - use neutral value
        components['confirm'] = 0.5 * self.config.confirm_weight

        # Sum components
        raw_score = sum(components.values())

        # Normalize to [min, max]
        final_score = self._normalize(raw_score)

        explanation = self._generate_explanation(components, final_score)

        return ScoringResult(
            final_score=final_score,
            components=components,
            explanation=explanation,
        )

    def score_fact(
        self,
        fact: Fact,
        access_count: int = 0,
        last_accessed: Optional[datetime] = None,
    ) -> ScoringResult:
        """
        Calculate importance score for a fact.

        Args:
            fact: The fact to score
            access_count: Number of times accessed
            last_accessed: When last accessed

        Returns:
            ScoringResult with final score and breakdown
        """
        components = {}

        # Base importance (weighted by confidence)
        base = fact.importance * fact.confidence
        components['base'] = base * self.config.base_weight

        # Access frequency
        access_score = self._calculate_access_score(access_count)
        components['access'] = access_score * self.config.access_weight

        # Recency
        ref_time = last_accessed or fact.last_confirmed or fact.first_learned or datetime.now()
        recency_score = self._calculate_recency_score(ref_time)
        components['recency'] = recency_score * self.config.recency_weight

        # Facts don't have emotional valence - use neutral
        components['emotion'] = 0.5 * self.config.emotion_weight

        # Confirmation count (logarithmic)
        confirm_score = min(1.0, math.log1p(fact.confirmation_count) / 3.0)
        components['confirm'] = confirm_score * self.config.confirm_weight

        # Sum components
        raw_score = sum(components.values())

        # Normalize
        final_score = self._normalize(raw_score)

        explanation = self._generate_explanation(components, final_score)

        return ScoringResult(
            final_score=final_score,
            components=components,
            explanation=explanation,
        )

    def _calculate_access_score(self, access_count: int) -> float:
        """Calculate score from access count (logarithmic scaling)."""
        if access_count <= 0:
            return 0.0

        # Log scale: 1 access = ~0.5, 10 accesses = ~0.9
        score = math.log(access_count + 1, self.config.access_log_base) / 5.0
        return min(1.0, score)

    def _calculate_recency_score(self, last_time: datetime) -> float:
        """Calculate score based on recency (exponential decay)."""
        now = datetime.now()
        age_days = (now - last_time).total_seconds() / (24 * 3600)

        # Exponential decay with half-life
        half_life = self.config.recency_half_life_days
        decay = 0.5 ** (age_days / half_life)

        return decay

    def _calculate_emotion_score(self, valence: EmotionalValence) -> float:
        """Calculate score from emotional valence (intensity)."""
        return self.config.emotion_intensity.get(valence, 0.3)

    def _normalize(self, score: float) -> float:
        """Normalize score to configured bounds."""
        return max(
            self.config.min_importance,
            min(self.config.max_importance, score)
        )

    def _generate_explanation(
        self,
        components: Dict[str, float],
        final_score: float,
    ) -> str:
        """Generate human-readable explanation of the score."""
        parts = []

        if components.get('base', 0) > 0.15:
            parts.append("high base importance")
        if components.get('access', 0) > 0.1:
            parts.append("frequently accessed")
        if components.get('recency', 0) > 0.15:
            parts.append("recently accessed")
        if components.get('emotion', 0) > 0.1:
            parts.append("emotionally significant")
        if components.get('confirm', 0) > 0.05:
            parts.append("repeatedly confirmed")

        if not parts:
            parts.append("low activity")

        return f"Score {final_score:.2f}: {', '.join(parts)}"

    def recalculate_importance(
        self,
        episode: Episode,
        access_count: int = 0,
        last_accessed: Optional[datetime] = None,
    ) -> float:
        """
        Convenience method to get just the final score for an episode.
        """
        result = self.score_episode(episode, access_count, last_accessed)
        return result.final_score

    def batch_score_episodes(
        self,
        episodes: List[Episode],
        access_data: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> List[ScoringResult]:
        """
        Score multiple episodes at once.

        Args:
            episodes: List of episodes to score
            access_data: Optional dict mapping episode_id to {access_count, last_accessed}

        Returns:
            List of ScoringResults in same order
        """
        access_data = access_data or {}
        results = []

        for ep in episodes:
            data = access_data.get(ep.id, {})
            result = self.score_episode(
                ep,
                access_count=data.get('access_count', 0),
                last_accessed=data.get('last_accessed'),
            )
            results.append(result)

        return results


def calculate_importance(
    memory,
    access_count: int = 0,
    last_accessed: Optional[datetime] = None,
    config: Optional[ScoringConfig] = None,
) -> float:
    """
    Convenience function to calculate importance score.

    Args:
        memory: Episode or Fact object
        access_count: Number of accesses
        last_accessed: Last access time
        config: Optional scoring configuration

    Returns:
        Importance score (0.0 to 1.0)
    """
    scorer = ImportanceScorer(config)

    if isinstance(memory, Episode):
        return scorer.score_episode(memory, access_count, last_accessed).final_score
    elif isinstance(memory, Fact):
        return scorer.score_fact(memory, access_count, last_accessed).final_score
    else:
        return 0.5  # Default for unknown types
