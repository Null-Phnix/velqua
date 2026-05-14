"""
Episodic Memory Store.

Stores timestamped experiences and events - "what happened."
"""

import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from ..models import EmotionalValence, Episode
from .base import MemoryStore
from .sqlite_backend import SQLiteBackend


class EpisodicStore(MemoryStore):
    """
    Store for episodic memories.

    Episodic memories are specific experiences with:
    - Temporal context (when it happened)
    - Emotional valence (how it felt)
    - Narrative structure (what happened)
    """

    def __init__(self, backend: SQLiteBackend):
        self.backend = backend

    def save(self, episode: Episode) -> str:
        """Save an episode to storage."""
        episode_dict = {
            "id": episode.id,
            "summary": episode.summary,
            "messages": [
                {"role": m["role"], "content": m["content"]}
                for m in episode.messages
            ] if episode.messages else [],
            "topic": episode.topic,
            "started_at": episode.started_at.isoformat() if episode.started_at else None,
            "ended_at": episode.ended_at.isoformat() if episode.ended_at else None,
            "overall_valence": episode.overall_valence.value if isinstance(episode.overall_valence, EmotionalValence) else episode.overall_valence,
            "importance": episode.importance,
            "source_id": episode.source_id,
            "metadata": episode.metadata,
            "tags": episode.tags if hasattr(episode, 'tags') else [],
        }
        return self.backend.save_episode(episode_dict)

    def get(self, episode_id: str) -> Optional[Episode]:
        """Retrieve an episode by ID."""
        data = self.backend.get_episode(episode_id)
        if not data:
            return None
        return self._dict_to_episode(data)

    def _dict_to_episode(self, data: Dict[str, Any]) -> Episode:
        """Convert dict to Episode object."""
        last_accessed = data.get("last_accessed")
        if last_accessed and isinstance(last_accessed, str):
            last_accessed = datetime.fromisoformat(last_accessed)

        return Episode(
            id=data["id"],
            summary=data.get("summary", ""),
            messages=data.get("messages", []),
            topic=data.get("topic"),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            ended_at=datetime.fromisoformat(data["ended_at"]) if data.get("ended_at") else None,
            overall_valence=EmotionalValence(data.get("overall_valence", 0)),
            importance=data.get("importance", 0.5),
            source_id=data.get("source_id"),
            metadata=data.get("metadata", {}),
            tags=data.get("tags", []),
            access_count=data.get("access_count", 0),
            last_accessed=last_accessed,
        )

    def delete(self, episode_id: str, hard: bool = False) -> bool:
        """
        Delete an episode.

        Args:
            episode_id: Episode ID to delete
            hard: If True, permanently remove from database
        """
        if hard:
            return self.backend.delete_episode(episode_id)

        # Soft delete - just mark as forgotten
        episode = self.get(episode_id)
        if episode:
            episode.importance = 0.0
            self.save(episode)
            return True
        return False

    def search(
        self,
        query: str,
        limit: int = 10,
        track_access: bool = True,
        **filters
    ) -> List[Episode]:
        """Search episodes by text query."""
        results = self.backend.search_episodes(query, limit)
        episodes = [self._dict_to_episode(r) for r in results]
        if track_access and episodes:
            self.touch_batch([ep.id for ep in episodes], reinforce=True)
        return episodes

    def list_all(self, limit: int = 100, offset: int = 0) -> List[Episode]:
        """List all episodes."""
        results = self.backend.list_episodes(limit, offset)
        return [self._dict_to_episode(r) for r in results]

    def count(self) -> int:
        """Count total episodes."""
        return self.backend.get_episode_count()

    def clear(self) -> int:
        """Clear all episodes."""
        count = self.count()
        # This would need implementation in backend
        return count

    # Episodic-specific methods

    def get_by_timerange(
        self,
        start: datetime,
        end: datetime,
        limit: int = 50
    ) -> List[Episode]:
        """Get episodes within a time range."""
        results = self.backend.get_episodes_by_timerange(
            start.isoformat(), end.isoformat(), limit
        )
        return [self._dict_to_episode(r) for r in results]

    def get_recent(self, days: int = 7, limit: int = 20) -> List[Episode]:
        """Get recent episodes."""
        cutoff = datetime.now() - timedelta(days=days)
        results = self.backend.get_episodes_by_timerange(
            cutoff.isoformat(), datetime.now().isoformat(), limit
        )
        return [self._dict_to_episode(r) for r in results]

    def get_by_topic(self, topic: str, limit: int = 20) -> List[Episode]:
        """Get episodes related to a topic."""
        return self.search(topic, limit)

    def get_emotional(
        self,
        valence: EmotionalValence,
        limit: int = 20
    ) -> List[Episode]:
        """Get episodes with specific emotional valence."""
        results = self.backend.get_episodes_by_valence(valence.value, limit)
        return [self._dict_to_episode(r) for r in results]

    def get_important(self, threshold: float = 0.7, limit: int = 20) -> List[Episode]:
        """Get high-importance episodes."""
        results = self.backend.get_episodes_by_importance(threshold, limit)
        return [self._dict_to_episode(r) for r in results]

    def create_from_conversation(
        self,
        conversation_id: str,
        summary: str,
        messages: List[Dict[str, str]],
        topic: Optional[str] = None,
        started_at: Optional[datetime] = None,
        ended_at: Optional[datetime] = None,
        valence: EmotionalValence = EmotionalValence.NEUTRAL,
        importance: float = 0.5,
    ) -> Episode:
        """Create and save an episode from a conversation."""
        episode = Episode(
            id=str(uuid.uuid4()),
            summary=summary,
            messages=messages,
            topic=topic,
            started_at=started_at or datetime.now(),
            ended_at=ended_at,
            overall_valence=valence,
            importance=importance,
            source_id=conversation_id,
        )
        self.save(episode)
        return episode

    def touch(self, episode_id: str, reinforce: bool = True) -> bool:
        """
        Mark an episode as accessed, updating access count and timestamp.

        This reinforces the memory, making it less likely to be forgotten.

        Args:
            episode_id: ID of the episode to touch
            reinforce: Whether to boost importance (default True)

        Returns:
            True if episode was found and touched
        """
        return self.backend.record_episode_access(
            episode_id,
            reinforce_importance=reinforce,
        )

    def touch_batch(self, episode_ids: List[str], reinforce: bool = True) -> int:
        """
        Mark multiple episodes as accessed in a single SQL statement.

        Args:
            episode_ids: List of episode IDs to touch
            reinforce: Whether to boost importance

        Returns:
            Count of episodes successfully touched
        """
        boost = 0.02 if reinforce else 0.0
        return self.backend.touch_episodes_batch(episode_ids, boost)

    def get_most_accessed(self, limit: int = 10) -> List[Episode]:
        """
        Get episodes sorted by access count.

        Args:
            limit: Maximum number of episodes to return

        Returns:
            List of most frequently accessed episodes
        """
        results = self.backend.get_episodes_most_accessed(limit)
        return [self._dict_to_episode(r) for r in results]

    def get_recently_accessed(self, days: int = 7, limit: int = 20) -> List[Episode]:
        """
        Get episodes that were recently accessed.

        Args:
            days: How many days back to look
            limit: Maximum number of episodes to return

        Returns:
            List of recently accessed episodes
        """
        cutoff = datetime.now() - timedelta(days=days)
        results = self.backend.get_episodes_recently_accessed(
            cutoff.isoformat(), limit
        )
        return [self._dict_to_episode(r) for r in results]
