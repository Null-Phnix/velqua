"""
Memory analytics and reporting.

Analyzes memory patterns, trends, and health.
"""

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..forgetting.manager import ForgettingManager
from ..models import EmotionalValence, Episode, Fact
from ..stores.episodic import EpisodicStore
from ..stores.semantic import SemanticStore
from ..stores.sqlite_backend import SQLiteBackend


@dataclass
class TopicStats:
    """Statistics about topics."""
    topic: str
    count: int
    first_seen: Optional[datetime]
    last_seen: Optional[datetime]
    avg_importance: float
    keywords: List[str]


@dataclass
class EmotionStats:
    """Statistics about emotional patterns."""
    valence: EmotionalValence
    count: int
    percentage: float
    trend: str  # "increasing", "decreasing", "stable"


@dataclass
class TemporalStats:
    """Statistics about temporal patterns."""
    period: str  # "daily", "weekly", "monthly"
    episode_counts: Dict[str, int]  # period -> count
    fact_counts: Dict[str, int]
    peak_period: str
    activity_trend: str


@dataclass
class AnalyticsReport:
    """Complete analytics report."""
    generated_at: datetime
    total_episodes: int
    total_facts: int
    memory_span_days: int

    # Health
    healthy_memories: int
    aging_memories: int
    at_risk_memories: int
    forgotten_memories: int

    # Topics
    top_topics: List[TopicStats]
    topic_diversity: float  # 0-1, higher = more diverse

    # Emotions
    emotion_distribution: List[EmotionStats]
    emotional_balance: float  # -1 to 1, positive = more positive emotions

    # Temporal
    temporal_stats: TemporalStats

    # Activity
    most_accessed: List[Dict[str, Any]]
    most_important: List[Dict[str, Any]]

    # Quality
    avg_episode_importance: float
    avg_fact_confidence: float
    facts_by_type: Dict[str, int]


class MemoryAnalyzer:
    """
    Analyzes memory system for insights and patterns.
    """

    def __init__(
        self,
        episodic_store: EpisodicStore,
        semantic_store: SemanticStore,
        backend: Optional[SQLiteBackend] = None,
    ):
        """
        Initialize analyzer.

        Args:
            episodic_store: Episodic memory store
            semantic_store: Semantic fact store
            backend: SQLite backend for additional queries
        """
        self.episodic = episodic_store
        self.semantic = semantic_store
        self.backend = backend

    def generate_report(self) -> AnalyticsReport:
        """Generate comprehensive analytics report."""
        # Get all data
        episodes = self.episodic.list_all(limit=10000)
        facts = self.semantic.list_all(limit=10000)

        # Calculate memory span
        if episodes:
            dates = [ep.started_at for ep in episodes if ep.started_at]
            if dates:
                span = (max(dates) - min(dates)).days
            else:
                span = 0
        else:
            span = 0

        # Health analysis
        health = self._analyze_health(episodes)

        # Topic analysis
        topic_stats = self._analyze_topics(episodes)
        topic_diversity = self._calculate_topic_diversity(episodes)

        # Emotion analysis
        emotion_stats = self._analyze_emotions(episodes)
        emotional_balance = self._calculate_emotional_balance(episodes)

        # Temporal analysis
        temporal_stats = self._analyze_temporal(episodes, facts)

        # Activity analysis
        most_accessed = self._get_most_accessed(episodes)
        most_important = self._get_most_important(episodes)

        # Quality metrics
        avg_importance = (
            sum(ep.importance for ep in episodes) / len(episodes)
            if episodes else 0
        )
        avg_confidence = (
            sum(f.confidence for f in facts) / len(facts)
            if facts else 0
        )
        facts_by_type = Counter(f.fact_type for f in facts)

        return AnalyticsReport(
            generated_at=datetime.now(),
            total_episodes=len(episodes),
            total_facts=len(facts),
            memory_span_days=span,
            healthy_memories=health["healthy"],
            aging_memories=health["aging"],
            at_risk_memories=health["at_risk"],
            forgotten_memories=health["forgotten"],
            top_topics=topic_stats[:10],
            topic_diversity=topic_diversity,
            emotion_distribution=emotion_stats,
            emotional_balance=emotional_balance,
            temporal_stats=temporal_stats,
            most_accessed=most_accessed,
            most_important=most_important,
            avg_episode_importance=avg_importance,
            avg_fact_confidence=avg_confidence,
            facts_by_type=dict(facts_by_type),
        )

    def _analyze_health(self, episodes: List[Episode]) -> Dict[str, int]:
        """Analyze memory health distribution."""
        manager = ForgettingManager(episodic_store=self.episodic)

        health = {"healthy": 0, "aging": 0, "at_risk": 0, "forgotten": 0}

        for ep in episodes:
            mem_health = manager.get_memory_health(ep)
            if mem_health.current_strength >= 0.7:
                health["healthy"] += 1
            elif mem_health.current_strength >= 0.2:
                health["aging"] += 1
            elif mem_health.current_strength >= 0.05:
                health["at_risk"] += 1
            else:
                health["forgotten"] += 1

        return health

    def _analyze_topics(self, episodes: List[Episode]) -> List[TopicStats]:
        """Analyze topic distribution."""
        topic_data = defaultdict(lambda: {
            "count": 0,
            "first": None,
            "last": None,
            "importance_sum": 0,
            "keywords": Counter(),
        })

        for ep in episodes:
            topic = ep.topic or "Untitled"

            data = topic_data[topic]
            data["count"] += 1
            data["importance_sum"] += ep.importance

            if ep.started_at:
                if data["first"] is None or ep.started_at < data["first"]:
                    data["first"] = ep.started_at
                if data["last"] is None or ep.started_at > data["last"]:
                    data["last"] = ep.started_at

            # Extract keywords from topic and summary
            text = f"{topic} {ep.summary or ''}".lower()
            words = re.findall(r'\b[a-z]{4,}\b', text)
            data["keywords"].update(words)

        # Convert to TopicStats
        stats = []
        for topic, data in topic_data.items():
            avg_imp = data["importance_sum"] / data["count"] if data["count"] > 0 else 0
            keywords = [w for w, _ in data["keywords"].most_common(5)]

            stats.append(TopicStats(
                topic=topic,
                count=data["count"],
                first_seen=data["first"],
                last_seen=data["last"],
                avg_importance=avg_imp,
                keywords=keywords,
            ))

        # Sort by count
        stats.sort(key=lambda x: x.count, reverse=True)
        return stats

    def _calculate_topic_diversity(self, episodes: List[Episode]) -> float:
        """Calculate topic diversity (0-1, higher = more diverse)."""
        if not episodes:
            return 0

        topics = [ep.topic or "Untitled" for ep in episodes]
        unique_topics = len(set(topics))
        total = len(topics)

        # Diversity = unique / total, but cap at reasonable value
        return min(1.0, unique_topics / max(total / 5, 1))

    def _analyze_emotions(self, episodes: List[Episode]) -> List[EmotionStats]:
        """Analyze emotional distribution."""
        counts = Counter(ep.overall_valence for ep in episodes)
        total = len(episodes) if episodes else 1

        stats = []
        for valence in EmotionalValence:
            count = counts.get(valence, 0)
            pct = (count / total) * 100 if total > 0 else 0

            # Analyze trend (simplified - compare first vs second half)
            trend = self._calculate_emotion_trend(episodes, valence)

            stats.append(EmotionStats(
                valence=valence,
                count=count,
                percentage=pct,
                trend=trend,
            ))

        return stats

    def _calculate_emotion_trend(
        self,
        episodes: List[Episode],
        valence: EmotionalValence,
    ) -> str:
        """Calculate trend for a specific emotion."""
        if len(episodes) < 4:
            return "stable"

        # Sort by date
        sorted_eps = sorted(
            [ep for ep in episodes if ep.started_at],
            key=lambda ep: ep.started_at,
        )

        if len(sorted_eps) < 4:
            return "stable"

        # Split in half
        mid = len(sorted_eps) // 2
        first_half = sorted_eps[:mid]
        second_half = sorted_eps[mid:]

        # Count in each half
        first_count = sum(1 for ep in first_half if ep.overall_valence == valence)
        second_count = sum(1 for ep in second_half if ep.overall_valence == valence)

        # Compare
        if second_count > first_count * 1.2:
            return "increasing"
        elif second_count < first_count * 0.8:
            return "decreasing"
        return "stable"

    def _calculate_emotional_balance(self, episodes: List[Episode]) -> float:
        """Calculate emotional balance (-1 to 1, positive = more positive)."""
        if not episodes:
            return 0

        total_valence = sum(
            ep.overall_valence.value if hasattr(ep.overall_valence, 'value')
            else ep.overall_valence
            for ep in episodes
        )
        return total_valence / (len(episodes) * 2)  # Normalize to -1 to 1

    def _analyze_temporal(
        self,
        episodes: List[Episode],
        facts: List[Fact],
    ) -> TemporalStats:
        """Analyze temporal patterns."""
        # Group by week
        episode_by_week = Counter()
        fact_by_week = Counter()

        for ep in episodes:
            if ep.started_at:
                week = ep.started_at.strftime("%Y-W%W")
                episode_by_week[week] += 1

        for fact in facts:
            if fact.first_learned:
                week = fact.first_learned.strftime("%Y-W%W")
                fact_by_week[week] += 1

        # Find peak
        peak = episode_by_week.most_common(1)[0][0] if episode_by_week else "N/A"

        # Calculate trend
        if len(episode_by_week) >= 2:
            weeks = sorted(episode_by_week.keys())
            first_count = sum(episode_by_week[w] for w in weeks[:len(weeks)//2])
            last_count = sum(episode_by_week[w] for w in weeks[len(weeks)//2:])
            if last_count > first_count * 1.2:
                trend = "increasing"
            elif last_count < first_count * 0.8:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "insufficient data"

        return TemporalStats(
            period="weekly",
            episode_counts=dict(episode_by_week),
            fact_counts=dict(fact_by_week),
            peak_period=peak,
            activity_trend=trend,
        )

    def _get_most_accessed(self, episodes: List[Episode]) -> List[Dict[str, Any]]:
        """Get most accessed memories."""
        accessed = []
        for ep in episodes:
            if ep.access_count > 0:
                accessed.append({
                    "id": ep.id,
                    "topic": ep.topic or "Untitled",
                    "access_count": ep.access_count,
                    "importance": ep.importance,
                })

        accessed.sort(key=lambda x: x["access_count"], reverse=True)
        return accessed[:10]

    def _get_most_important(self, episodes: List[Episode]) -> List[Dict[str, Any]]:
        """Get most important memories."""
        important = []
        for ep in episodes:
            important.append({
                "id": ep.id,
                "topic": ep.topic or "Untitled",
                "importance": ep.importance,
                "valence": ep.overall_valence.name if hasattr(ep.overall_valence, 'name') else str(ep.overall_valence),
            })

        important.sort(key=lambda x: x["importance"], reverse=True)
        return important[:10]

    def get_quick_stats(self) -> Dict[str, Any]:
        """Get quick summary statistics."""
        episodes = self.episodic.list_all(limit=10000)
        facts = self.semantic.list_all(limit=10000)

        return {
            "total_episodes": len(episodes),
            "total_facts": len(facts),
            "avg_importance": sum(ep.importance for ep in episodes) / len(episodes) if episodes else 0,
            "avg_confidence": sum(f.confidence for f in facts) / len(facts) if facts else 0,
            "fact_types": dict(Counter(f.fact_type for f in facts)),
            "emotions": dict(Counter(ep.overall_valence.name for ep in episodes)),
        }
