"""
Semantic Memory Store.

Stores facts, preferences, and patterns - context-independent knowledge.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..models import Fact, FactType
from .base import MemoryStore
from .sqlite_backend import SQLiteBackend


class SemanticStore(MemoryStore):
    """
    Store for semantic memories (facts).

    Semantic memories are:
    - Context-independent (not tied to specific events)
    - Facts, preferences, patterns
    - Can be confirmed or contradicted over time
    """

    def __init__(self, backend: SQLiteBackend, embedder=None):
        self.backend = backend
        # Lazy import to avoid circular dependency (dedup -> stores -> dedup)
        from ..dedup.smart_detector import SmartDuplicateDetector
        self._dedup = SmartDuplicateDetector(embedder=embedder)

    def set_embedder(self, embedder):
        """Update the embedder used for smart duplicate detection."""
        self._dedup.embedder = embedder

    def save(self, fact: Fact) -> str:
        """Save a fact to storage."""
        fact_dict = {
            "id": fact.id,
            "content": fact.content,
            "fact_type": fact.fact_type,
            "confidence": fact.confidence,
            "source_episodes": fact.source_episodes,
            "first_learned": fact.first_learned.isoformat() if fact.first_learned else None,
            "last_confirmed": fact.last_confirmed.isoformat() if fact.last_confirmed else None,
            "confirmation_count": fact.confirmation_count,
            "is_superseded": fact.is_superseded,
            "importance": fact.importance,
            "metadata": fact.metadata,
            "tags": fact.tags if hasattr(fact, 'tags') else [],
        }
        return self.backend.save_fact(fact_dict)

    def get(self, fact_id: str) -> Optional[Fact]:
        """Retrieve a fact by ID."""
        data = self.backend.get_fact(fact_id)
        if not data:
            return None
        return self._dict_to_fact(data)

    def _dict_to_fact(self, data: Dict[str, Any]) -> Fact:
        """Convert dict to Fact object."""
        return Fact(
            id=data["id"],
            content=data["content"],
            fact_type=data.get("fact_type", FactType.GENERAL),
            confidence=data.get("confidence", 0.8),
            source_episodes=data.get("source_episodes", []),
            first_learned=datetime.fromisoformat(data["first_learned"]) if data.get("first_learned") else datetime.now(),
            last_confirmed=datetime.fromisoformat(data["last_confirmed"]) if data.get("last_confirmed") else datetime.now(),
            confirmation_count=data.get("confirmation_count", 1),
            is_superseded=data.get("is_superseded", False),
            importance=data.get("importance", 0.5),
            metadata=data.get("metadata", {}),
            tags=data.get("tags", []),
        )

    def delete(self, fact_id: str, hard: bool = False) -> bool:
        """
        Delete a fact.

        Args:
            fact_id: Fact ID to delete
            hard: If True, permanently remove from database
        """
        if hard:
            return self.backend.delete_fact(fact_id)

        # Soft delete - mark as superseded
        fact = self.get(fact_id)
        if fact:
            fact.is_superseded = True
            fact.confidence = 0.0
            self.save(fact)
            return True
        return False

    def search(
        self,
        query: str,
        limit: int = 10,
        track_access: bool = True,
        **filters
    ) -> List[Fact]:
        """Search facts by text query."""
        results = self.backend.search_facts(query, limit)
        facts = [self._dict_to_fact(r) for r in results]
        # Filter out superseded unless explicitly requested
        if not filters.get("include_superseded"):
            facts = [f for f in facts if not f.is_superseded]
        if track_access and facts:
            self.touch_batch([f.id for f in facts], reinforce=True)
        return facts

    def list_all(self, limit: int = 100, offset: int = 0) -> List[Fact]:
        """List all active facts."""
        results = self.backend.list_facts(limit, offset)
        return [self._dict_to_fact(r) for r in results if not r.get("is_superseded")]

    def count(self) -> int:
        """Count total active facts."""
        return self.backend.get_fact_count()

    def clear(self) -> int:
        """Clear all facts."""
        count = self.count()
        return count

    # Semantic-specific methods

    def get_by_type(self, fact_type: str, limit: int = 50) -> List[Fact]:
        """Get facts of a specific type."""
        results = self.backend.list_facts(limit, fact_type=fact_type)
        return [self._dict_to_fact(r) for r in results if not r.get("is_superseded")]

    def get_preferences(self, limit: int = 50) -> List[Fact]:
        """Get user preferences."""
        return self.get_by_type(FactType.PREFERENCE, limit)

    def get_personal_facts(self, limit: int = 50) -> List[Fact]:
        """Get personal facts about the user."""
        return self.get_by_type(FactType.PERSONAL, limit)

    def get_high_confidence(self, threshold: float = 0.9, limit: int = 50) -> List[Fact]:
        """Get high-confidence facts."""
        results = self.backend.get_facts_by_confidence(threshold, limit)
        return [self._dict_to_fact(r) for r in results]

    def find_similar(self, content: str, limit: int = 5) -> List[Fact]:
        """Find facts similar to given content (for deduplication)."""
        # Dedup lookups should not count as user access
        return self.search(content, limit, track_access=False)

    def add_fact(
        self,
        content: str,
        fact_type: FactType = FactType.GENERAL,
        source_episode_id: Optional[str] = None,
        confidence: float = 0.8,
        importance: float = 0.5,
        metadata: Optional[Dict] = None,
    ) -> Fact:
        """Add a new fact, checking for duplicates."""
        # Check for existing similar facts
        similar = self.find_similar(content, limit=3)
        for existing in similar:
            # If very similar content exists, confirm it instead
            if self._is_duplicate(content, existing.content):
                existing.confirm()
                if source_episode_id:
                    existing.source_episodes.append(source_episode_id)
                self.save(existing)
                return existing

        # Create new fact
        fact = Fact(
            id=str(uuid.uuid4()),
            content=content,
            fact_type=fact_type,
            confidence=confidence,
            source_episodes=[source_episode_id] if source_episode_id else [],
            importance=importance,
            metadata=metadata or {},
        )
        self.save(fact)
        return fact

    def _is_duplicate(self, new_content: str, existing_content: str) -> bool:
        """Check if two facts are duplicates using TF-IDF cosine similarity."""
        from ..dedup.similarity import quick_similarity
        return quick_similarity(new_content, existing_content) > 0.75

    def contradict(self, fact_id: str, contradicting_fact_id: str):
        """Mark that one fact contradicts another."""
        fact = self.get(fact_id)
        contradicting = self.get(contradicting_fact_id)

        if fact and contradicting:
            fact.contradicted_by.append(contradicting_fact_id)
            # Lower confidence of contradicted fact
            fact.confidence = max(0.1, fact.confidence - 0.2)
            self.save(fact)

    def supersede(self, old_fact_id: str, new_fact_id: str):
        """Mark an old fact as superseded by a new one."""
        old_fact = self.get(old_fact_id)
        if old_fact:
            old_fact.is_superseded = True
            old_fact.metadata["superseded_by"] = new_fact_id
            self.save(old_fact)

    def touch(self, fact_id: str, reinforce: bool = True) -> bool:
        """
        Mark a fact as accessed, updating access count and timestamp.

        This reinforces the memory, making it less likely to be forgotten.

        Args:
            fact_id: ID of the fact to touch
            reinforce: Whether to boost importance (default True)

        Returns:
            True if fact was found and touched
        """
        return self.backend.record_fact_access(
            fact_id,
            reinforce_importance=reinforce,
        )

    def touch_batch(self, fact_ids: List[str], reinforce: bool = True) -> int:
        """
        Mark multiple facts as accessed in a single SQL statement.

        Args:
            fact_ids: List of fact IDs to touch
            reinforce: Whether to boost importance

        Returns:
            Count of facts successfully touched
        """
        boost = 0.02 if reinforce else 0.0
        return self.backend.touch_facts_batch(fact_ids, boost)

    def get_most_accessed(self, limit: int = 10) -> List[Fact]:
        """
        Get facts sorted by access count.

        Args:
            limit: Maximum number of facts to return

        Returns:
            List of most frequently accessed facts
        """
        results = self.backend.get_facts_most_accessed(limit)
        return [self._dict_to_fact(r) for r in results]
