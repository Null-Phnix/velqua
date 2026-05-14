"""
Memory quality scoring system.

Evaluates memory quality based on:
- Completeness: Has all expected fields
- Richness: Amount of detail/content
- Reliability: Confidence and confirmation
- Activity: Access patterns
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Tuple

from ..models import Episode, Fact, FactType


class QualityLevel(Enum):
    """Quality level categories."""
    EXCELLENT = "excellent"  # 0.8-1.0
    GOOD = "good"           # 0.6-0.8
    FAIR = "fair"           # 0.4-0.6
    POOR = "poor"           # 0.2-0.4
    LOW = "low"             # 0.0-0.2


@dataclass
class QualityReport:
    """Detailed quality assessment for a memory."""
    memory_id: str
    memory_type: str  # "episode" or "fact"
    overall_score: float  # 0.0 to 1.0
    quality_level: QualityLevel

    # Individual dimension scores
    completeness_score: float
    richness_score: float
    reliability_score: float
    activity_score: float

    # Details
    missing_fields: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityStats:
    """Statistics about quality across the memory system."""
    total_memories: int
    avg_quality: float
    quality_distribution: Dict[str, int]  # QualityLevel -> count
    lowest_quality: List[Tuple[str, float]]  # (id, score)
    highest_quality: List[Tuple[str, float]]
    common_issues: Dict[str, int]


class QualityScorer:
    """
    Scores memory quality.

    Provides quality assessments for individual memories
    and aggregate statistics across the system.
    """

    def __init__(
        self,
        completeness_weight: float = 0.3,
        richness_weight: float = 0.25,
        reliability_weight: float = 0.25,
        activity_weight: float = 0.2,
    ):
        """
        Initialize scorer with dimension weights.

        Args:
            completeness_weight: Weight for completeness (0-1)
            richness_weight: Weight for richness (0-1)
            reliability_weight: Weight for reliability (0-1)
            activity_weight: Weight for activity (0-1)
        """
        self.completeness_weight = completeness_weight
        self.richness_weight = richness_weight
        self.reliability_weight = reliability_weight
        self.activity_weight = activity_weight

    def score_episode(self, episode: Episode) -> QualityReport:
        """
        Score an episode's quality.

        Args:
            episode: Episode to score

        Returns:
            QualityReport with detailed assessment
        """
        missing_fields = []
        suggestions = []
        metrics = {}

        # Completeness: Has expected fields
        completeness = self._score_episode_completeness(episode, missing_fields)
        metrics["completeness_breakdown"] = self._episode_completeness_details(episode)

        # Richness: Amount of content
        richness = self._score_episode_richness(episode, metrics)

        # Reliability: Importance and emotional clarity
        reliability = self._score_episode_reliability(episode, metrics)

        # Activity: Access patterns
        activity = self._score_episode_activity(episode, metrics)

        # Calculate overall score
        overall = (
            completeness * self.completeness_weight +
            richness * self.richness_weight +
            reliability * self.reliability_weight +
            activity * self.activity_weight
        )

        # Generate suggestions
        self._generate_episode_suggestions(
            episode, completeness, richness, reliability, activity, suggestions
        )

        return QualityReport(
            memory_id=episode.id,
            memory_type="episode",
            overall_score=overall,
            quality_level=self._score_to_level(overall),
            completeness_score=completeness,
            richness_score=richness,
            reliability_score=reliability,
            activity_score=activity,
            missing_fields=missing_fields,
            suggestions=suggestions,
            metrics=metrics,
        )

    def score_fact(self, fact: Fact) -> QualityReport:
        """
        Score a fact's quality.

        Args:
            fact: Fact to score

        Returns:
            QualityReport with detailed assessment
        """
        missing_fields = []
        suggestions = []
        metrics = {}

        # Completeness: Has expected fields
        completeness = self._score_fact_completeness(fact, missing_fields)

        # Richness: Content quality
        richness = self._score_fact_richness(fact, metrics)

        # Reliability: Confidence and confirmations
        reliability = self._score_fact_reliability(fact, metrics)

        # Activity: Access patterns
        activity = self._score_fact_activity(fact, metrics)

        # Calculate overall score
        overall = (
            completeness * self.completeness_weight +
            richness * self.richness_weight +
            reliability * self.reliability_weight +
            activity * self.activity_weight
        )

        # Generate suggestions
        self._generate_fact_suggestions(
            fact, completeness, richness, reliability, activity, suggestions
        )

        return QualityReport(
            memory_id=fact.id,
            memory_type="fact",
            overall_score=overall,
            quality_level=self._score_to_level(overall),
            completeness_score=completeness,
            richness_score=richness,
            reliability_score=reliability,
            activity_score=activity,
            missing_fields=missing_fields,
            suggestions=suggestions,
            metrics=metrics,
        )

    def score_batch_episodes(
        self,
        episodes: List[Episode],
    ) -> List[QualityReport]:
        """Score multiple episodes."""
        return [self.score_episode(ep) for ep in episodes]

    def score_batch_facts(
        self,
        facts: List[Fact],
    ) -> List[QualityReport]:
        """Score multiple facts."""
        return [self.score_fact(f) for f in facts]

    def get_stats(
        self,
        episodes: List[Episode],
        facts: List[Fact],
    ) -> QualityStats:
        """
        Get quality statistics across all memories.

        Args:
            episodes: All episodes
            facts: All facts

        Returns:
            QualityStats with aggregate data
        """
        all_reports = []
        all_reports.extend(self.score_batch_episodes(episodes))
        all_reports.extend(self.score_batch_facts(facts))

        if not all_reports:
            return QualityStats(
                total_memories=0,
                avg_quality=0.0,
                quality_distribution={level.value: 0 for level in QualityLevel},
                lowest_quality=[],
                highest_quality=[],
                common_issues={},
            )

        # Calculate distribution
        distribution = {level.value: 0 for level in QualityLevel}
        for report in all_reports:
            distribution[report.quality_level.value] += 1

        # Find lowest/highest
        sorted_reports = sorted(all_reports, key=lambda r: r.overall_score)
        lowest = [(r.memory_id, r.overall_score) for r in sorted_reports[:5]]
        highest = [(r.memory_id, r.overall_score) for r in sorted_reports[-5:][::-1]]

        # Count common issues
        issues = {}
        for report in all_reports:
            for field_name in report.missing_fields:
                issues[f"missing_{field_name}"] = issues.get(f"missing_{field_name}", 0) + 1
            for suggestion in report.suggestions:
                # Extract issue type from suggestion
                if "topic" in suggestion.lower():
                    issues["no_topic"] = issues.get("no_topic", 0) + 1
                elif "summary" in suggestion.lower():
                    issues["no_summary"] = issues.get("no_summary", 0) + 1
                elif "message" in suggestion.lower():
                    issues["few_messages"] = issues.get("few_messages", 0) + 1
                elif "confidence" in suggestion.lower():
                    issues["low_confidence"] = issues.get("low_confidence", 0) + 1

        return QualityStats(
            total_memories=len(all_reports),
            avg_quality=sum(r.overall_score for r in all_reports) / len(all_reports),
            quality_distribution=distribution,
            lowest_quality=lowest,
            highest_quality=highest,
            common_issues=issues,
        )

    def _score_episode_completeness(
        self,
        episode: Episode,
        missing_fields: List[str],
    ) -> float:
        """Score episode completeness (0-1)."""
        scores = []

        # Has topic
        if episode.topic:
            scores.append(1.0)
        else:
            scores.append(0.0)
            missing_fields.append("topic")

        # Has summary
        if episode.summary:
            scores.append(1.0)
        else:
            scores.append(0.0)
            missing_fields.append("summary")

        # Has messages
        if episode.messages and len(episode.messages) > 0:
            scores.append(1.0)
        else:
            scores.append(0.0)
            missing_fields.append("messages")

        # Has timestamps
        if episode.started_at:
            scores.append(1.0)
        else:
            scores.append(0.5)  # Partial credit
            missing_fields.append("started_at")

        # Has tags
        if episode.tags:
            scores.append(1.0)
        else:
            scores.append(0.3)  # Tags are optional

        return sum(scores) / len(scores) if scores else 0.0

    def _episode_completeness_details(self, episode: Episode) -> Dict[str, bool]:
        """Get completeness details for episode."""
        return {
            "has_topic": bool(episode.topic),
            "has_summary": bool(episode.summary),
            "has_messages": bool(episode.messages),
            "has_started_at": episode.started_at is not None,
            "has_ended_at": episode.ended_at is not None,
            "has_tags": bool(episode.tags),
            "has_source_id": bool(episode.source_id),
        }

    def _score_episode_richness(
        self,
        episode: Episode,
        metrics: Dict[str, Any],
    ) -> float:
        """Score episode richness/detail (0-1)."""
        scores = []

        # Message count (more messages = richer)
        msg_count = len(episode.messages)
        metrics["message_count"] = msg_count
        if msg_count >= 10:
            scores.append(1.0)
        elif msg_count >= 5:
            scores.append(0.7)
        elif msg_count >= 2:
            scores.append(0.4)
        elif msg_count >= 1:
            scores.append(0.2)
        else:
            scores.append(0.0)

        # Summary length (longer = more detailed)
        summary_len = len(episode.summary) if episode.summary else 0
        metrics["summary_length"] = summary_len
        if summary_len >= 200:
            scores.append(1.0)
        elif summary_len >= 100:
            scores.append(0.7)
        elif summary_len >= 50:
            scores.append(0.5)
        elif summary_len > 0:
            scores.append(0.3)
        else:
            scores.append(0.0)

        # Message content length (total chars)
        total_content = sum(
            len(m.get("content", "")) for m in episode.messages
        )
        metrics["total_content_length"] = total_content
        if total_content >= 2000:
            scores.append(1.0)
        elif total_content >= 1000:
            scores.append(0.7)
        elif total_content >= 500:
            scores.append(0.5)
        else:
            scores.append(0.3)

        # Tag count
        tag_count = len(episode.tags) if episode.tags else 0
        metrics["tag_count"] = tag_count
        if tag_count >= 3:
            scores.append(1.0)
        elif tag_count >= 1:
            scores.append(0.5)
        else:
            scores.append(0.2)

        return sum(scores) / len(scores) if scores else 0.0

    def _score_episode_reliability(
        self,
        episode: Episode,
        metrics: Dict[str, Any],
    ) -> float:
        """Score episode reliability (0-1)."""
        scores = []

        # Importance (higher = more reliable/important)
        importance = episode.importance
        metrics["importance"] = importance
        scores.append(importance)

        # Has emotional valence
        if episode.overall_valence is not None:
            scores.append(0.8)
        else:
            scores.append(0.4)

        # Has source ID (verifiable origin)
        if episode.source_id:
            scores.append(1.0)
        else:
            scores.append(0.5)

        # Has metadata
        if episode.metadata:
            scores.append(0.8)
        else:
            scores.append(0.5)

        return sum(scores) / len(scores) if scores else 0.0

    def _score_episode_activity(
        self,
        episode: Episode,
        metrics: Dict[str, Any],
    ) -> float:
        """Score episode activity/engagement (0-1)."""
        scores = []

        # Access count
        access_count = episode.access_count
        metrics["access_count"] = access_count
        if access_count >= 10:
            scores.append(1.0)
        elif access_count >= 5:
            scores.append(0.7)
        elif access_count >= 1:
            scores.append(0.4)
        else:
            scores.append(0.2)

        # Recency of access
        last_accessed = episode.last_accessed
        if last_accessed:
            try:
                last_dt = last_accessed
                days_ago = (datetime.now() - last_dt).days
                metrics["days_since_access"] = days_ago
                if days_ago <= 7:
                    scores.append(1.0)
                elif days_ago <= 30:
                    scores.append(0.7)
                elif days_ago <= 90:
                    scores.append(0.4)
                else:
                    scores.append(0.2)
            except (ValueError, TypeError):
                scores.append(0.3)
        else:
            scores.append(0.3)

        # Recency of creation
        if episode.started_at:
            days_old = (datetime.now() - episode.started_at).days
            metrics["days_old"] = days_old
            if days_old <= 7:
                scores.append(0.9)
            elif days_old <= 30:
                scores.append(0.8)
            elif days_old <= 90:
                scores.append(0.6)
            else:
                scores.append(0.4)
        else:
            scores.append(0.3)

        return sum(scores) / len(scores) if scores else 0.0

    def _generate_episode_suggestions(
        self,
        episode: Episode,
        completeness: float,
        richness: float,
        reliability: float,
        activity: float,
        suggestions: List[str],
    ):
        """Generate improvement suggestions for episode."""
        if not episode.topic:
            suggestions.append("Add a topic to describe the conversation theme")

        if not episode.summary:
            suggestions.append("Add a summary to capture key points")

        if len(episode.messages) < 3:
            suggestions.append("Episode has few messages - may be incomplete")

        if not episode.tags:
            suggestions.append("Add tags to improve discoverability")

        if activity < 0.3:
            suggestions.append("Low activity - consider reviewing or archiving")

    def _score_fact_completeness(
        self,
        fact: Fact,
        missing_fields: List[str],
    ) -> float:
        """Score fact completeness (0-1)."""
        scores = []

        # Has content
        if fact.content and len(fact.content) > 10:
            scores.append(1.0)
        elif fact.content:
            scores.append(0.5)
        else:
            scores.append(0.0)
            missing_fields.append("content")

        # Has fact type
        if fact.fact_type and fact.fact_type not in (FactType.GENERAL, "unknown"):
            scores.append(1.0)
        elif fact.fact_type:
            scores.append(0.5)
        else:
            scores.append(0.2)
            missing_fields.append("fact_type")

        # Has source episodes
        if fact.source_episodes:
            scores.append(1.0)
        else:
            scores.append(0.3)

        # Has timestamps
        if fact.first_learned:
            scores.append(1.0)
        else:
            scores.append(0.5)

        # Has tags
        if fact.tags:
            scores.append(1.0)
        else:
            scores.append(0.3)

        return sum(scores) / len(scores) if scores else 0.0

    def _score_fact_richness(
        self,
        fact: Fact,
        metrics: Dict[str, Any],
    ) -> float:
        """Score fact richness (0-1)."""
        scores = []

        # Content length
        content_len = len(fact.content) if fact.content else 0
        metrics["content_length"] = content_len
        if content_len >= 100:
            scores.append(1.0)
        elif content_len >= 50:
            scores.append(0.7)
        elif content_len >= 20:
            scores.append(0.5)
        else:
            scores.append(0.3)

        # Number of source episodes
        source_count = len(fact.source_episodes)
        metrics["source_count"] = source_count
        if source_count >= 3:
            scores.append(1.0)
        elif source_count >= 1:
            scores.append(0.6)
        else:
            scores.append(0.3)

        # Has metadata
        if fact.metadata:
            scores.append(0.8)
        else:
            scores.append(0.4)

        return sum(scores) / len(scores) if scores else 0.0

    def _score_fact_reliability(
        self,
        fact: Fact,
        metrics: Dict[str, Any],
    ) -> float:
        """Score fact reliability (0-1)."""
        scores = []

        # Confidence
        confidence = fact.confidence
        metrics["confidence"] = confidence
        scores.append(confidence)

        # Confirmation count
        confirmations = fact.confirmation_count
        metrics["confirmation_count"] = confirmations
        if confirmations >= 5:
            scores.append(1.0)
        elif confirmations >= 3:
            scores.append(0.8)
        elif confirmations >= 2:
            scores.append(0.6)
        else:
            scores.append(0.4)

        # Not superseded
        if not fact.is_superseded:
            scores.append(1.0)
        else:
            scores.append(0.0)

        # Importance
        importance = fact.importance
        metrics["importance"] = importance
        scores.append(importance)

        return sum(scores) / len(scores) if scores else 0.0

    def _score_fact_activity(
        self,
        fact: Fact,
        metrics: Dict[str, Any],
    ) -> float:
        """Score fact activity (0-1)."""
        scores = []

        # Access count
        access_count = fact.metadata.get("access_count", 0)
        metrics["access_count"] = access_count
        if access_count >= 10:
            scores.append(1.0)
        elif access_count >= 5:
            scores.append(0.7)
        elif access_count >= 1:
            scores.append(0.4)
        else:
            scores.append(0.2)

        # Last confirmed recency
        if fact.last_confirmed:
            days_since = (datetime.now() - fact.last_confirmed).days
            metrics["days_since_confirmed"] = days_since
            if days_since <= 7:
                scores.append(1.0)
            elif days_since <= 30:
                scores.append(0.7)
            elif days_since <= 90:
                scores.append(0.5)
            else:
                scores.append(0.3)
        else:
            scores.append(0.3)

        return sum(scores) / len(scores) if scores else 0.0

    def _generate_fact_suggestions(
        self,
        fact: Fact,
        completeness: float,
        richness: float,
        reliability: float,
        activity: float,
        suggestions: List[str],
    ):
        """Generate improvement suggestions for fact."""
        if fact.confidence < 0.5:
            suggestions.append("Low confidence - needs verification")

        if fact.confirmation_count < 2:
            suggestions.append("Fact has few confirmations - may be unreliable")

        if fact.is_superseded:
            suggestions.append("Fact is superseded - consider removing")

        if not fact.source_episodes:
            suggestions.append("No source episodes - origin unknown")

        if not fact.tags:
            suggestions.append("Add tags to improve discoverability")

    def _score_to_level(self, score: float) -> QualityLevel:
        """Convert numeric score to quality level."""
        if score >= 0.8:
            return QualityLevel.EXCELLENT
        elif score >= 0.6:
            return QualityLevel.GOOD
        elif score >= 0.4:
            return QualityLevel.FAIR
        elif score >= 0.2:
            return QualityLevel.POOR
        else:
            return QualityLevel.LOW
