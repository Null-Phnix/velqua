"""
Emotional state and pattern tracking.

Tracks emotional states over time to build a picture of
the user's emotional patterns and baseline.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from ..models import EmotionalValence, Episode
from ..stores.episodic import EpisodicStore
from .analyzer import EmotionalIntensity, EmotionCategory, SentimentAnalyzer


@dataclass
class EmotionalState:
    """Current emotional state snapshot."""
    timestamp: datetime
    valence: EmotionalValence
    primary_emotion: EmotionCategory
    intensity: EmotionalIntensity
    sentiment_score: float
    context: Optional[str] = None  # What triggered this state
    source_episode_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "valence": self.valence.value,
            "primary_emotion": self.primary_emotion.value,
            "intensity": self.intensity.value,
            "sentiment_score": self.sentiment_score,
            "context": self.context,
            "source_episode_id": self.source_episode_id,
        }


@dataclass
class EmotionalPattern:
    """Pattern of emotions over time."""
    emotion: EmotionCategory
    frequency: int  # How often it appears
    average_intensity: float
    typical_triggers: List[str] = field(default_factory=list)
    time_of_day_distribution: Dict[str, int] = field(default_factory=dict)

    @property
    def is_common(self) -> bool:
        """Check if this is a frequently occurring emotion."""
        return self.frequency >= 5


@dataclass
class MoodBaseline:
    """User's typical emotional baseline."""
    dominant_valence: EmotionalValence
    dominant_emotion: EmotionCategory
    average_sentiment: float
    variability: float  # How much mood varies
    positive_ratio: float  # Percentage of positive interactions
    calculated_at: datetime
    sample_size: int

    def is_below_baseline(self, current_sentiment: float) -> bool:
        """Check if current sentiment is below normal."""
        return current_sentiment < self.average_sentiment - self.variability

    def is_above_baseline(self, current_sentiment: float) -> bool:
        """Check if current sentiment is above normal."""
        return current_sentiment > self.average_sentiment + self.variability


class EmotionalTracker:
    """
    Tracks emotional states and patterns over time.

    Builds a model of the user's emotional baseline and
    detects when current mood deviates from normal.
    """

    def __init__(
        self,
        episodic_store: Optional[EpisodicStore] = None,
        analyzer: Optional[SentimentAnalyzer] = None,
        history_limit: int = 100,
    ):
        """
        Initialize tracker.

        Args:
            episodic_store: Optional episodic store for history
            analyzer: Sentiment analyzer (created if not provided)
            history_limit: Max states to keep in memory
        """
        self.episodic_store = episodic_store
        self.analyzer = analyzer or SentimentAnalyzer()
        self.history_limit = history_limit

        # In-memory state tracking
        self._states: List[EmotionalState] = []
        self._baseline: Optional[MoodBaseline] = None

    def record_state(
        self,
        text: str,
        context: Optional[str] = None,
        episode_id: Optional[str] = None,
    ) -> EmotionalState:
        """
        Record an emotional state from text.

        Args:
            text: Text to analyze
            context: Optional context description
            episode_id: Optional source episode ID

        Returns:
            Recorded emotional state
        """
        result = self.analyzer.analyze(text)

        state = EmotionalState(
            timestamp=datetime.now(),
            valence=result.valence,
            primary_emotion=result.primary_emotion,
            intensity=result.intensity,
            sentiment_score=result.sentiment_score,
            context=context,
            source_episode_id=episode_id,
        )

        self._states.append(state)

        # Trim if over limit
        if len(self._states) > self.history_limit:
            self._states = self._states[-self.history_limit:]

        return state

    def record_from_episode(self, episode: Episode) -> EmotionalState:
        """
        Record emotional state from an episode.

        Args:
            episode: Episode to analyze

        Returns:
            Recorded emotional state
        """
        # Get text to analyze
        text_parts = []
        for msg in episode.messages[:10]:
            if isinstance(msg, dict):
                text_parts.append(msg.get("content", ""))
            elif hasattr(msg, "content"):
                text_parts.append(msg.content)

        text = " ".join(text_parts)

        return self.record_state(
            text=text,
            context=episode.topic,
            episode_id=episode.id,
        )

    def get_current_state(self) -> Optional[EmotionalState]:
        """Get most recent emotional state."""
        return self._states[-1] if self._states else None

    def get_recent_states(
        self,
        hours: int = 24,
        limit: int = 20,
    ) -> List[EmotionalState]:
        """
        Get recent emotional states.

        Args:
            hours: How far back to look
            limit: Maximum states to return

        Returns:
            List of recent states
        """
        cutoff = datetime.now() - timedelta(hours=hours)
        recent = [s for s in self._states if s.timestamp >= cutoff]
        return recent[-limit:]

    def calculate_baseline(
        self,
        days_back: int = 30,
    ) -> MoodBaseline:
        """
        Calculate emotional baseline from history.

        Args:
            days_back: Days of history to consider

        Returns:
            Calculated baseline
        """
        # Get states from memory
        cutoff = datetime.now() - timedelta(days=days_back)
        relevant_states = [s for s in self._states if s.timestamp >= cutoff]

        # Also analyze episodes if store available
        if self.episodic_store:
            episodes = self.episodic_store.get_recent(days=days_back, limit=100)
            for ep in episodes:
                if ep.overall_valence:
                    relevant_states.append(EmotionalState(
                        timestamp=ep.started_at or datetime.now(),
                        valence=ep.overall_valence,
                        primary_emotion=EmotionCategory.NEUTRAL,
                        intensity=EmotionalIntensity.MODERATE,
                        sentiment_score=self._valence_to_score(ep.overall_valence),
                    ))

        if not relevant_states:
            return MoodBaseline(
                dominant_valence=EmotionalValence.NEUTRAL,
                dominant_emotion=EmotionCategory.NEUTRAL,
                average_sentiment=0.0,
                variability=0.3,
                positive_ratio=0.5,
                calculated_at=datetime.now(),
                sample_size=0,
            )

        # Calculate statistics
        sentiment_scores = [s.sentiment_score for s in relevant_states]
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)

        # Variability (standard deviation approximation)
        variance = sum((s - avg_sentiment) ** 2 for s in sentiment_scores) / len(sentiment_scores)
        variability = variance ** 0.5

        # Valence distribution
        valence_counts = defaultdict(int)
        emotion_counts = defaultdict(int)
        for s in relevant_states:
            valence_counts[s.valence] += 1
            emotion_counts[s.primary_emotion] += 1

        # Dominant valence and emotion
        dominant_valence = max(valence_counts, key=valence_counts.get)
        dominant_emotion = max(emotion_counts, key=emotion_counts.get)

        # Positive ratio
        positive_count = valence_counts.get(EmotionalValence.POSITIVE, 0)
        positive_ratio = positive_count / len(relevant_states)

        self._baseline = MoodBaseline(
            dominant_valence=dominant_valence,
            dominant_emotion=dominant_emotion,
            average_sentiment=avg_sentiment,
            variability=max(variability, 0.1),  # Minimum variability
            positive_ratio=positive_ratio,
            calculated_at=datetime.now(),
            sample_size=len(relevant_states),
        )

        return self._baseline

    def get_baseline(self) -> MoodBaseline:
        """Get or calculate baseline."""
        if self._baseline is None:
            return self.calculate_baseline()
        return self._baseline

    def is_mood_low(self, current_text: Optional[str] = None) -> bool:
        """
        Check if current mood is below baseline.

        Args:
            current_text: Optional text to analyze (uses current state if None)

        Returns:
            True if mood is notably lower than baseline
        """
        baseline = self.get_baseline()

        if current_text:
            result = self.analyzer.analyze(current_text)
            current = result.sentiment_score
        elif self._states:
            current = self._states[-1].sentiment_score
        else:
            return False

        return baseline.is_below_baseline(current)

    def is_mood_elevated(self, current_text: Optional[str] = None) -> bool:
        """
        Check if current mood is above baseline.

        Args:
            current_text: Optional text to analyze

        Returns:
            True if mood is notably higher than baseline
        """
        baseline = self.get_baseline()

        if current_text:
            result = self.analyzer.analyze(current_text)
            current = result.sentiment_score
        elif self._states:
            current = self._states[-1].sentiment_score
        else:
            return False

        return baseline.is_above_baseline(current)

    def get_emotional_patterns(self) -> List[EmotionalPattern]:
        """
        Analyze emotional patterns from history.

        Returns:
            List of detected patterns
        """
        # Count emotions
        emotion_data: Dict[EmotionCategory, Dict] = defaultdict(
            lambda: {"count": 0, "intensities": [], "contexts": [], "hours": []}
        )

        for state in self._states:
            data = emotion_data[state.primary_emotion]
            data["count"] += 1
            data["intensities"].append(state.intensity.value)
            if state.context:
                data["contexts"].append(state.context)
            data["hours"].append(state.timestamp.hour)

        # Build patterns
        patterns = []
        for emotion, data in emotion_data.items():
            if data["count"] > 0:
                # Time of day distribution
                hour_counts = defaultdict(int)
                for h in data["hours"]:
                    if h < 6:
                        hour_counts["night"] += 1
                    elif h < 12:
                        hour_counts["morning"] += 1
                    elif h < 18:
                        hour_counts["afternoon"] += 1
                    else:
                        hour_counts["evening"] += 1

                # Most common triggers
                trigger_counts = defaultdict(int)
                for ctx in data["contexts"]:
                    for word in ctx.lower().split():
                        if len(word) > 4:
                            trigger_counts[word] += 1

                top_triggers = sorted(
                    trigger_counts.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]

                patterns.append(EmotionalPattern(
                    emotion=emotion,
                    frequency=data["count"],
                    average_intensity=sum(data["intensities"]) / len(data["intensities"]),
                    typical_triggers=[t[0] for t in top_triggers],
                    time_of_day_distribution=dict(hour_counts),
                ))

        # Sort by frequency
        patterns.sort(key=lambda p: p.frequency, reverse=True)
        return patterns

    def get_mood_trend(
        self,
        hours: int = 24,
    ) -> str:
        """
        Get trend of mood over time.

        Args:
            hours: Hours to analyze

        Returns:
            Trend description: 'improving', 'declining', 'stable', 'fluctuating'
        """
        recent = self.get_recent_states(hours=hours)
        if len(recent) < 2:
            return "stable"

        scores = [s.sentiment_score for s in recent]

        # Simple trend detection
        first_half = sum(scores[:len(scores)//2]) / (len(scores)//2)
        second_half = sum(scores[len(scores)//2:]) / (len(scores) - len(scores)//2)

        diff = second_half - first_half

        # Calculate variance
        avg = sum(scores) / len(scores)
        variance = sum((s - avg) ** 2 for s in scores) / len(scores)

        if variance > 0.25:
            return "fluctuating"
        elif diff > 0.2:
            return "improving"
        elif diff < -0.2:
            return "declining"
        else:
            return "stable"

    def _valence_to_score(self, valence: EmotionalValence) -> float:
        """Convert valence to sentiment score."""
        mapping = {
            EmotionalValence.POSITIVE: 0.5,
            EmotionalValence.NEGATIVE: -0.5,
            EmotionalValence.NEUTRAL: 0.0,
            EmotionalValence.NEUTRAL: 0.0,
        }
        return mapping.get(valence, 0.0)
