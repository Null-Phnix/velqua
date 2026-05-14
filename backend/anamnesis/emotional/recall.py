"""
Mood-aware memory recall.

Retrieves memories in a way that's sensitive to emotional context.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..models import EmotionalValence, Episode, Fact
from ..stores.episodic import EpisodicStore
from ..stores.semantic import SemanticStore
from .analyzer import SentimentAnalyzer
from .tracker import EmotionalTracker


@dataclass
class MoodAwareResult:
    """Result of mood-aware recall."""
    episodes: List[Episode] = field(default_factory=list)
    facts: List[Fact] = field(default_factory=list)
    current_mood: Optional[EmotionalValence] = None
    mood_adjusted: bool = False
    adjustment_reason: Optional[str] = None
    comfort_memories: List[Episode] = field(default_factory=list)  # Positive memories when mood is low

    @property
    def total_count(self) -> int:
        """Total memories returned."""
        return len(self.episodes) + len(self.facts)


class EmotionalRecall:
    """
    Mood-aware memory recall system.

    Adjusts memory retrieval based on emotional context:
    - When user is down, surface positive memories
    - When discussing emotional topics, prioritize emotionally relevant memories
    - Track emotional associations with memories
    """

    def __init__(
        self,
        episodic_store: EpisodicStore,
        semantic_store: SemanticStore,
        tracker: Optional[EmotionalTracker] = None,
        analyzer: Optional[SentimentAnalyzer] = None,
    ):
        """
        Initialize emotional recall.

        Args:
            episodic_store: Episodic memory store
            semantic_store: Semantic memory store
            tracker: Emotional tracker (created if not provided)
            analyzer: Sentiment analyzer (created if not provided)
        """
        self.episodic_store = episodic_store
        self.semantic_store = semantic_store
        self.analyzer = analyzer or SentimentAnalyzer()
        self.tracker = tracker or EmotionalTracker(
            episodic_store=episodic_store,
            analyzer=self.analyzer,
        )

    def recall(
        self,
        query: str,
        max_episodes: int = 10,
        max_facts: int = 10,
        mood_aware: bool = True,
        comfort_memories_count: int = 3,
    ) -> MoodAwareResult:
        """
        Recall memories with mood awareness.

        Args:
            query: Search query
            max_episodes: Max episodes to return
            max_facts: Max facts to return
            mood_aware: Whether to adjust for mood
            comfort_memories_count: How many positive memories when mood is low

        Returns:
            MoodAwareResult with memories and mood info
        """
        # Analyze current query
        query_sentiment = self.analyzer.analyze(query)

        # Get baseline
        self.tracker.get_baseline()

        # Search for memories
        episodes = self.episodic_store.search(query, limit=max_episodes)
        facts = self.semantic_store.search(query, limit=max_facts)

        result = MoodAwareResult(
            episodes=episodes,
            facts=facts,
            current_mood=query_sentiment.valence,
        )

        # Apply mood-aware adjustments
        if mood_aware:
            # Check if mood is low
            if (query_sentiment.is_negative or
                self.tracker.is_mood_low(query)):

                # Add comfort memories
                comfort = self._get_comfort_memories(comfort_memories_count)
                result.comfort_memories = comfort
                result.mood_adjusted = True
                result.adjustment_reason = "Added positive memories to help lift mood"

            # If query is emotionally charged, boost emotionally relevant memories
            if query_sentiment.intensity.value >= 3:
                # Re-rank episodes by emotional relevance
                result.episodes = self._rank_by_emotional_relevance(
                    episodes,
                    target_valence=query_sentiment.valence,
                )
                result.mood_adjusted = True
                if not result.adjustment_reason:
                    result.adjustment_reason = "Ranked by emotional relevance"

        return result

    def recall_by_emotion(
        self,
        emotion: EmotionalValence,
        limit: int = 10,
    ) -> List[Episode]:
        """
        Recall memories by emotional valence.

        Args:
            emotion: Target emotional valence
            limit: Max episodes to return

        Returns:
            Episodes matching the emotional valence
        """
        all_episodes = self.episodic_store.list_all(limit=limit * 3)

        matching = [
            ep for ep in all_episodes
            if ep.overall_valence == emotion
        ]

        return matching[:limit]

    def get_positive_memories(
        self,
        limit: int = 10,
        topic: Optional[str] = None,
    ) -> List[Episode]:
        """
        Get positive memories.

        Args:
            limit: Max episodes to return
            topic: Optional topic filter

        Returns:
            Positive episodes
        """
        if topic:
            candidates = self.episodic_store.search(topic, limit=limit * 2)
        else:
            candidates = self.episodic_store.list_all(limit=limit * 2)

        positive = [
            ep for ep in candidates
            if ep.overall_valence == EmotionalValence.POSITIVE
        ]

        # If not enough tagged positive, analyze summaries
        if len(positive) < limit:
            for ep in candidates:
                if ep not in positive and ep.summary:
                    sentiment = self.analyzer.analyze(ep.summary)
                    if sentiment.is_positive:
                        positive.append(ep)

        return positive[:limit]

    def get_negative_memories(
        self,
        limit: int = 10,
        topic: Optional[str] = None,
    ) -> List[Episode]:
        """
        Get negative memories.

        Args:
            limit: Max episodes to return
            topic: Optional topic filter

        Returns:
            Negative episodes
        """
        if topic:
            candidates = self.episodic_store.search(topic, limit=limit * 2)
        else:
            candidates = self.episodic_store.list_all(limit=limit * 2)

        negative = [
            ep for ep in candidates
            if ep.overall_valence == EmotionalValence.NEGATIVE
        ]

        return negative[:limit]

    def suggest_for_mood(
        self,
        current_text: str,
        limit: int = 5,
    ) -> List[Episode]:
        """
        Suggest memories appropriate for current mood.

        If user seems down, suggest uplifting memories.
        If user seems happy, suggest related positive memories.

        Args:
            current_text: Current message/context
            limit: Max suggestions

        Returns:
            Suggested episodes
        """
        sentiment = self.analyzer.analyze(current_text)

        if sentiment.is_negative:
            # User is down - suggest positive memories
            return self.get_positive_memories(limit=limit)

        elif sentiment.is_positive:
            # User is happy - find related positive memories
            return self._find_related_positive(current_text, limit)

        else:
            # Neutral - return mixed
            return self.episodic_store.get_recent(days=30, limit=limit)

    def analyze_emotional_history(
        self,
        days_back: int = 30,
    ) -> Dict[str, Any]:
        """
        Analyze emotional history of memories.

        Args:
            days_back: Days to analyze

        Returns:
            Analysis with patterns and statistics
        """
        episodes = self.episodic_store.get_recent(days=days_back, limit=100)

        # Count valences
        valence_counts = {v: 0 for v in EmotionalValence}
        sentiment_scores = []

        for ep in episodes:
            if ep.overall_valence:
                valence_counts[ep.overall_valence] += 1

            if ep.summary:
                sentiment = self.analyzer.analyze(ep.summary)
                sentiment_scores.append(sentiment.sentiment_score)

        # Calculate stats
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0

        # Most common valence
        dominant = max(valence_counts, key=valence_counts.get)

        # Positivity ratio
        total = sum(valence_counts.values())
        positive_ratio = valence_counts[EmotionalValence.POSITIVE] / total if total > 0 else 0.5

        return {
            "episode_count": len(episodes),
            "days_analyzed": days_back,
            "valence_distribution": {k.value: v for k, v in valence_counts.items()},
            "dominant_valence": dominant.value,
            "average_sentiment": avg_sentiment,
            "positive_ratio": positive_ratio,
            "mood_assessment": self._assess_mood(avg_sentiment, positive_ratio),
        }

    def _get_comfort_memories(self, count: int) -> List[Episode]:
        """Get positive comfort memories."""
        positive = self.get_positive_memories(limit=count * 2)

        # Sort by importance
        positive.sort(key=lambda ep: ep.importance, reverse=True)

        return positive[:count]

    def _rank_by_emotional_relevance(
        self,
        episodes: List[Episode],
        target_valence: EmotionalValence,
    ) -> List[Episode]:
        """Rank episodes by emotional relevance to target."""
        def relevance(ep: Episode) -> float:
            score = 0.0

            # Valence match
            if ep.overall_valence == target_valence:
                score += 0.5

            # Importance boost
            score += ep.importance * 0.3

            # Analyze summary if available
            if ep.summary:
                sentiment = self.analyzer.analyze(ep.summary)
                if sentiment.valence == target_valence:
                    score += 0.2

            return score

        return sorted(episodes, key=relevance, reverse=True)

    def _find_related_positive(
        self,
        text: str,
        limit: int,
    ) -> List[Episode]:
        """Find positive memories related to the given text."""
        # Search by text
        related = self.episodic_store.search(text, limit=limit * 2)

        # Filter to positive
        positive_related = [
            ep for ep in related
            if ep.overall_valence == EmotionalValence.POSITIVE
        ]

        # Add general positive if not enough
        if len(positive_related) < limit:
            general_positive = self.get_positive_memories(limit=limit - len(positive_related))
            positive_related.extend([
                ep for ep in general_positive
                if ep not in positive_related
            ])

        return positive_related[:limit]

    def _assess_mood(
        self,
        avg_sentiment: float,
        positive_ratio: float,
    ) -> str:
        """Generate mood assessment string."""
        if avg_sentiment > 0.3 and positive_ratio > 0.6:
            return "Generally positive - most interactions have been upbeat"
        elif avg_sentiment < -0.3 or positive_ratio < 0.3:
            return "Below average - recent interactions show lower mood"
        elif abs(avg_sentiment) < 0.1 and 0.4 < positive_ratio < 0.6:
            return "Neutral and balanced"
        else:
            return "Mixed - emotions vary across interactions"
