"""
Temporal-aware memory recall.

Combines time expressions with memory search for
context-aware retrieval.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from ..models import Episode, Fact
from ..stores.episodic import EpisodicStore
from ..stores.semantic import SemanticStore
from .parser import TemporalExpression, TemporalParser, TemporalRange


@dataclass
class TemporalQuery:
    """A parsed query with temporal context."""
    original_query: str
    core_query: str  # Query with temporal removed
    temporal: Optional[TemporalExpression] = None
    has_time_reference: bool = False
    reference_topic: Optional[str] = None

    @property
    def time_range(self) -> Optional[TemporalRange]:
        """Get time range from temporal expression."""
        return self.temporal.range if self.temporal else None


@dataclass
class TemporalResult:
    """Result of a temporal query."""
    query: TemporalQuery
    episodes: List[Episode] = field(default_factory=list)
    facts: List[Fact] = field(default_factory=list)
    reference_episodes: List[Episode] = field(default_factory=list)  # Episodes from reference topic
    time_filtered: bool = False
    total_matches: int = 0

    @property
    def has_results(self) -> bool:
        """Check if any results were found."""
        return bool(self.episodes or self.facts)


class TemporalRecall:
    """
    Time-aware memory recall system.

    Combines temporal expression parsing with memory search
    to answer queries like:
    - "What did we discuss last week?"
    - "What Python topics have we covered recently?"
    - "Remember when we talked about machine learning?"
    """

    def __init__(
        self,
        episodic_store: EpisodicStore,
        semantic_store: SemanticStore,
        parser: Optional[TemporalParser] = None,
    ):
        """
        Initialize temporal recall.

        Args:
            episodic_store: Episodic memory store
            semantic_store: Semantic memory store
            parser: Temporal parser (created if not provided)
        """
        self.episodic_store = episodic_store
        self.semantic_store = semantic_store
        self.parser = parser or TemporalParser()

    def parse_query(self, query: str) -> TemporalQuery:
        """
        Parse a query for temporal context.

        Args:
            query: Natural language query

        Returns:
            TemporalQuery with parsed components
        """
        temporal = self.parser.parse(query)
        core_query = self.parser.remove_temporal(query)

        return TemporalQuery(
            original_query=query,
            core_query=core_query,
            temporal=temporal,
            has_time_reference=temporal is not None,
            reference_topic=temporal.reference_topic if temporal else None,
        )

    def recall(
        self,
        query: str,
        max_episodes: int = 10,
        max_facts: int = 20,
        include_related: bool = True,
    ) -> TemporalResult:
        """
        Recall memories with temporal awareness.

        Args:
            query: Natural language query with optional time context
            max_episodes: Maximum episodes to return
            max_facts: Maximum facts to return
            include_related: Include memories from reference topics

        Returns:
            TemporalResult with filtered memories
        """
        parsed = self.parse_query(query)

        # If there's a reference topic, search for it first
        reference_episodes = []
        if parsed.reference_topic and include_related:
            reference_episodes = self.episodic_store.search(
                parsed.reference_topic,
                limit=5,
            )

        # Search for core query
        if parsed.core_query:
            episodes = self.episodic_store.search(
                parsed.core_query,
                limit=max_episodes * 2,  # Get extra for filtering
            )
            facts = self.semantic_store.search(
                parsed.core_query,
                limit=max_facts,
            )
        else:
            # No core query - get recent memories
            episodes = self.episodic_store.get_recent(days=30, limit=max_episodes * 2)
            facts = []

        # Apply temporal filter if we have a time range
        time_filtered = False
        if parsed.time_range and parsed.time_range.is_bounded:
            episodes = self._filter_episodes_by_time(
                episodes,
                parsed.time_range,
            )
            time_filtered = True

        # Limit results
        episodes = episodes[:max_episodes]

        return TemporalResult(
            query=parsed,
            episodes=episodes,
            facts=facts,
            reference_episodes=reference_episodes,
            time_filtered=time_filtered,
            total_matches=len(episodes) + len(facts),
        )

    def recall_around(
        self,
        anchor_episode: Episode,
        window_days: int = 7,
        limit: int = 10,
    ) -> List[Episode]:
        """
        Recall episodes around a specific episode in time.

        Args:
            anchor_episode: Episode to anchor search around
            window_days: Days before/after to include
            limit: Maximum episodes to return

        Returns:
            List of episodes around the anchor
        """
        if not anchor_episode.started_at:
            return []

        # Create time window
        window = TemporalRange(
            start=anchor_episode.started_at - timedelta(days=window_days),
            end=anchor_episode.started_at + timedelta(days=window_days),
        )

        # Get all episodes in window
        all_episodes = self.episodic_store.list_all(limit=limit * 3)
        filtered = self._filter_episodes_by_time(all_episodes, window)

        # Remove the anchor itself
        filtered = [ep for ep in filtered if ep.id != anchor_episode.id]

        # Sort by proximity to anchor
        def proximity(ep):
            if not ep.started_at:
                return float('inf')
            return abs((ep.started_at - anchor_episode.started_at).total_seconds())

        filtered.sort(key=proximity)

        return filtered[:limit]

    def recall_sequence(
        self,
        topic: str,
        limit: int = 10,
    ) -> List[Episode]:
        """
        Recall episodes about a topic in chronological order.

        Useful for understanding the progression of a discussion
        over multiple conversations.

        Args:
            topic: Topic to search for
            limit: Maximum episodes to return

        Returns:
            Episodes sorted chronologically
        """
        episodes = self.episodic_store.search(topic, limit=limit)

        # Sort by start time
        def sort_key(ep):
            return ep.started_at or datetime.min

        episodes.sort(key=sort_key)

        return episodes

    def recall_before(
        self,
        reference_query: str,
        search_query: Optional[str] = None,
        limit: int = 10,
    ) -> List[Episode]:
        """
        Recall episodes that occurred before a reference topic.

        Args:
            reference_query: Query to find the reference point
            search_query: Optional query to filter results
            limit: Maximum episodes to return

        Returns:
            Episodes that occurred before reference
        """
        # Find reference episode(s)
        reference_eps = self.episodic_store.search(reference_query, limit=3)
        if not reference_eps:
            return []

        # Get the earliest reference
        reference_eps = [ep for ep in reference_eps if ep.started_at]
        if not reference_eps:
            return []

        reference_time = min(ep.started_at for ep in reference_eps)

        # Create time range ending at reference
        time_range = TemporalRange(end=reference_time)

        # Get episodes
        if search_query:
            candidates = self.episodic_store.search(search_query, limit=limit * 2)
        else:
            candidates = self.episodic_store.list_all(limit=limit * 2)

        # Filter by time
        filtered = self._filter_episodes_by_time(candidates, time_range)

        # Sort by recency (most recent first)
        filtered.sort(
            key=lambda ep: ep.started_at or datetime.min,
            reverse=True,
        )

        return filtered[:limit]

    def recall_after(
        self,
        reference_query: str,
        search_query: Optional[str] = None,
        limit: int = 10,
    ) -> List[Episode]:
        """
        Recall episodes that occurred after a reference topic.

        Args:
            reference_query: Query to find the reference point
            search_query: Optional query to filter results
            limit: Maximum episodes to return

        Returns:
            Episodes that occurred after reference
        """
        # Find reference episode(s)
        reference_eps = self.episodic_store.search(reference_query, limit=3)
        if not reference_eps:
            return []

        # Get the latest reference
        reference_eps = [ep for ep in reference_eps if ep.started_at]
        if not reference_eps:
            return []

        reference_time = max(ep.started_at for ep in reference_eps)

        # Create time range starting at reference
        time_range = TemporalRange(start=reference_time)

        # Get episodes
        if search_query:
            candidates = self.episodic_store.search(search_query, limit=limit * 2)
        else:
            candidates = self.episodic_store.list_all(limit=limit * 2)

        # Filter by time
        filtered = self._filter_episodes_by_time(candidates, time_range)

        # Sort by recency (oldest first)
        filtered.sort(
            key=lambda ep: ep.started_at or datetime.max,
        )

        return filtered[:limit]

    def recall_between(
        self,
        start_query: str,
        end_query: str,
        search_query: Optional[str] = None,
        limit: int = 10,
    ) -> List[Episode]:
        """
        Recall episodes between two reference points.

        Args:
            start_query: Query to find start point
            end_query: Query to find end point
            search_query: Optional query to filter results
            limit: Maximum episodes to return

        Returns:
            Episodes between the reference points
        """
        # Find start reference
        start_eps = self.episodic_store.search(start_query, limit=3)
        start_eps = [ep for ep in start_eps if ep.started_at]
        start_time = min((ep.started_at for ep in start_eps), default=datetime.min)

        # Find end reference
        end_eps = self.episodic_store.search(end_query, limit=3)
        end_eps = [ep for ep in end_eps if ep.started_at]
        end_time = max((ep.started_at for ep in end_eps), default=datetime.max)

        # Create time range
        time_range = TemporalRange(start=start_time, end=end_time)

        # Get episodes
        if search_query:
            candidates = self.episodic_store.search(search_query, limit=limit * 2)
        else:
            candidates = self.episodic_store.list_all(limit=limit * 2)

        # Filter by time
        filtered = self._filter_episodes_by_time(candidates, time_range)

        # Sort chronologically
        filtered.sort(key=lambda ep: ep.started_at or datetime.min)

        return filtered[:limit]

    def get_timeline(
        self,
        topic: Optional[str] = None,
        days_back: int = 30,
        group_by: str = "day",
    ) -> Dict[str, List[Episode]]:
        """
        Get a timeline view of episodes.

        Args:
            topic: Optional topic filter
            days_back: How many days back to include
            group_by: Grouping ('day', 'week', 'month')

        Returns:
            Dict mapping time periods to episodes
        """
        # Get episodes
        if topic:
            episodes = self.episodic_store.search(topic, limit=100)
        else:
            episodes = self.episodic_store.get_recent(days=days_back, limit=100)

        # Filter by time
        cutoff = datetime.now() - timedelta(days=days_back)
        episodes = [ep for ep in episodes if ep.started_at and ep.started_at >= cutoff]

        # Group by time period
        timeline: Dict[str, List[Episode]] = {}

        for ep in episodes:
            if not ep.started_at:
                continue

            if group_by == "day":
                key = ep.started_at.strftime("%Y-%m-%d")
            elif group_by == "week":
                # Get Monday of the week
                monday = ep.started_at - timedelta(days=ep.started_at.weekday())
                key = f"Week of {monday.strftime('%Y-%m-%d')}"
            elif group_by == "month":
                key = ep.started_at.strftime("%Y-%m")
            else:
                key = ep.started_at.strftime("%Y-%m-%d")

            if key not in timeline:
                timeline[key] = []
            timeline[key].append(ep)

        return timeline

    def _filter_episodes_by_time(
        self,
        episodes: List[Episode],
        time_range: TemporalRange,
    ) -> List[Episode]:
        """Filter episodes by time range."""
        return [
            ep for ep in episodes
            if ep.started_at and time_range.contains(ep.started_at)
        ]
