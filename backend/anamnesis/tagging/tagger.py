"""
Memory tagging system.

Provides tag management, auto-tagging based on content analysis,
and tag-based memory retrieval.
"""

import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ..models import Episode, Fact, FactType
from ..stores.episodic import EpisodicStore
from ..stores.semantic import SemanticStore


@dataclass
class TagStats:
    """Statistics about tags in the memory system."""
    total_tags: int
    unique_tags: int
    most_common: List[Tuple[str, int]]
    episodes_tagged: int
    facts_tagged: int
    avg_tags_per_episode: float
    avg_tags_per_fact: float


class TagManager:
    """
    Manages tags for memories.

    Provides CRUD operations for tags and tag-based retrieval.
    """

    def __init__(
        self,
        episodic_store: EpisodicStore,
        semantic_store: SemanticStore,
    ):
        """
        Initialize tag manager.

        Args:
            episodic_store: Episodic memory store
            semantic_store: Semantic fact store
        """
        self.episodic_store = episodic_store
        self.semantic_store = semantic_store

    def add_tag(
        self,
        memory_id: str,
        tag: str,
        memory_type: str = "episode",
    ) -> bool:
        """
        Add a tag to a memory.

        Args:
            memory_id: Memory ID
            tag: Tag to add
            memory_type: "episode" or "fact"

        Returns:
            True if successful
        """
        tag = self._normalize_tag(tag)

        if memory_type == "episode":
            memory = self.episodic_store.get(memory_id)
            if not memory:
                return False
            if tag not in memory.tags:
                memory.tags.append(tag)
                self.episodic_store.save(memory)
            return True
        else:
            memory = self.semantic_store.get(memory_id)
            if not memory:
                return False
            if tag not in memory.tags:
                memory.tags.append(tag)
                self.semantic_store.save(memory)
            return True

    def remove_tag(
        self,
        memory_id: str,
        tag: str,
        memory_type: str = "episode",
    ) -> bool:
        """
        Remove a tag from a memory.

        Args:
            memory_id: Memory ID
            tag: Tag to remove
            memory_type: "episode" or "fact"

        Returns:
            True if successful
        """
        tag = self._normalize_tag(tag)

        if memory_type == "episode":
            memory = self.episodic_store.get(memory_id)
            if not memory:
                return False
            if tag in memory.tags:
                memory.tags.remove(tag)
                self.episodic_store.save(memory)
            return True
        else:
            memory = self.semantic_store.get(memory_id)
            if not memory:
                return False
            if tag in memory.tags:
                memory.tags.remove(tag)
                self.semantic_store.save(memory)
            return True

    def get_tags(
        self,
        memory_id: str,
        memory_type: str = "episode",
    ) -> List[str]:
        """
        Get tags for a memory.

        Args:
            memory_id: Memory ID
            memory_type: "episode" or "fact"

        Returns:
            List of tags
        """
        if memory_type == "episode":
            memory = self.episodic_store.get(memory_id)
            return memory.tags if memory else []
        else:
            memory = self.semantic_store.get(memory_id)
            return memory.tags if memory else []

    def set_tags(
        self,
        memory_id: str,
        tags: List[str],
        memory_type: str = "episode",
    ) -> bool:
        """
        Set all tags for a memory (replaces existing).

        Args:
            memory_id: Memory ID
            tags: List of tags
            memory_type: "episode" or "fact"

        Returns:
            True if successful
        """
        tags = [self._normalize_tag(t) for t in tags]

        if memory_type == "episode":
            memory = self.episodic_store.get(memory_id)
            if not memory:
                return False
            memory.tags = tags
            self.episodic_store.save(memory)
            return True
        else:
            memory = self.semantic_store.get(memory_id)
            if not memory:
                return False
            memory.tags = tags
            self.semantic_store.save(memory)
            return True

    def find_by_tag(
        self,
        tag: str,
        memory_type: Optional[str] = None,
        limit: int = 100,
    ) -> Dict[str, List]:
        """
        Find memories by tag.

        Args:
            tag: Tag to search for
            memory_type: "episode", "fact", or None for both
            limit: Max results per type

        Returns:
            Dict with 'episodes' and/or 'facts' lists
        """
        tag = self._normalize_tag(tag)
        result = {}

        if memory_type in (None, "episode"):
            episodes = []
            all_episodes = self.episodic_store.list_all(limit=10000)
            for ep in all_episodes:
                if tag in ep.tags:
                    episodes.append(ep)
                    if len(episodes) >= limit:
                        break
            result["episodes"] = episodes

        if memory_type in (None, "fact"):
            facts = []
            all_facts = self.semantic_store.list_all(limit=10000)
            for fact in all_facts:
                if tag in fact.tags:
                    facts.append(fact)
                    if len(facts) >= limit:
                        break
            result["facts"] = facts

        return result

    def find_by_tags(
        self,
        tags: List[str],
        match_all: bool = False,
        memory_type: Optional[str] = None,
        limit: int = 100,
    ) -> Dict[str, List]:
        """
        Find memories by multiple tags.

        Args:
            tags: Tags to search for
            match_all: If True, require all tags; if False, any tag
            memory_type: "episode", "fact", or None for both
            limit: Max results per type

        Returns:
            Dict with 'episodes' and/or 'facts' lists
        """
        tags = {self._normalize_tag(t) for t in tags}
        result = {}

        if memory_type in (None, "episode"):
            episodes = []
            all_episodes = self.episodic_store.list_all(limit=10000)
            for ep in all_episodes:
                ep_tags = set(ep.tags)
                if match_all:
                    if tags.issubset(ep_tags):
                        episodes.append(ep)
                else:
                    if tags & ep_tags:
                        episodes.append(ep)
                if len(episodes) >= limit:
                    break
            result["episodes"] = episodes

        if memory_type in (None, "fact"):
            facts = []
            all_facts = self.semantic_store.list_all(limit=10000)
            for fact in all_facts:
                fact_tags = set(fact.tags)
                if match_all:
                    if tags.issubset(fact_tags):
                        facts.append(fact)
                else:
                    if tags & fact_tags:
                        facts.append(fact)
                if len(facts) >= limit:
                    break
            result["facts"] = facts

        return result

    def list_all_tags(self) -> List[Tuple[str, int]]:
        """
        List all tags with counts.

        Returns:
            List of (tag, count) tuples sorted by count
        """
        tag_counts = Counter()

        all_episodes = self.episodic_store.list_all(limit=10000)
        for ep in all_episodes:
            tag_counts.update(ep.tags)

        all_facts = self.semantic_store.list_all(limit=10000)
        for fact in all_facts:
            tag_counts.update(fact.tags)

        return tag_counts.most_common()

    def get_stats(self) -> TagStats:
        """Get tag statistics."""
        all_episodes = self.episodic_store.list_all(limit=10000)
        all_facts = self.semantic_store.list_all(limit=10000)

        tag_counts = Counter()
        episodes_with_tags = 0
        facts_with_tags = 0
        total_episode_tags = 0
        total_fact_tags = 0

        for ep in all_episodes:
            if ep.tags:
                episodes_with_tags += 1
                total_episode_tags += len(ep.tags)
            tag_counts.update(ep.tags)

        for fact in all_facts:
            if fact.tags:
                facts_with_tags += 1
                total_fact_tags += len(fact.tags)
            tag_counts.update(fact.tags)

        total_tags = sum(tag_counts.values())
        unique_tags = len(tag_counts)

        return TagStats(
            total_tags=total_tags,
            unique_tags=unique_tags,
            most_common=tag_counts.most_common(10),
            episodes_tagged=episodes_with_tags,
            facts_tagged=facts_with_tags,
            avg_tags_per_episode=total_episode_tags / len(all_episodes) if all_episodes else 0,
            avg_tags_per_fact=total_fact_tags / len(all_facts) if all_facts else 0,
        )

    def _normalize_tag(self, tag: str) -> str:
        """Normalize a tag (lowercase, strip whitespace)."""
        return tag.lower().strip()


class AutoTagger:
    """
    Automatically tags memories based on content analysis.

    Uses keyword extraction and pattern matching to suggest tags.
    """

    # Common topic categories
    TOPIC_KEYWORDS = {
        "programming": ["code", "programming", "function", "variable", "class", "python", "javascript", "api"],
        "work": ["project", "meeting", "deadline", "task", "team", "manager", "client"],
        "personal": ["family", "friend", "relationship", "birthday", "vacation", "hobby"],
        "learning": ["learn", "study", "course", "tutorial", "book", "lesson"],
        "health": ["exercise", "workout", "diet", "sleep", "health", "doctor", "medicine"],
        "finance": ["money", "budget", "investment", "salary", "expense", "savings"],
        "creative": ["art", "music", "writing", "design", "creative", "project"],
        "tech": ["computer", "software", "hardware", "app", "tool", "technology"],
    }

    # Sentiment/emotion tags
    EMOTION_KEYWORDS = {
        "positive": ["happy", "great", "love", "excited", "wonderful", "amazing", "excellent"],
        "negative": ["sad", "angry", "frustrated", "annoyed", "disappointed", "worried"],
        "important": ["important", "critical", "urgent", "priority", "essential"],
    }

    def __init__(self, min_confidence: float = 0.3):
        """
        Initialize auto-tagger.

        Args:
            min_confidence: Minimum confidence to apply a tag
        """
        self.min_confidence = min_confidence

    def suggest_tags(
        self,
        text: str,
        existing_tags: Optional[List[str]] = None,
        max_tags: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Suggest tags for text content.

        Args:
            text: Text to analyze
            existing_tags: Existing tags to avoid duplicating
            max_tags: Maximum tags to suggest

        Returns:
            List of (tag, confidence) tuples
        """
        existing = set(existing_tags or [])
        text_lower = text.lower()

        suggestions = []

        # Check topic keywords
        for topic, keywords in self.TOPIC_KEYWORDS.items():
            matches = sum(1 for kw in keywords if kw in text_lower)
            if matches > 0:
                confidence = min(1.0, matches / 3)  # 3+ matches = full confidence
                if confidence >= self.min_confidence and topic not in existing:
                    suggestions.append((topic, confidence))

        # Check emotion keywords
        for emotion, keywords in self.EMOTION_KEYWORDS.items():
            matches = sum(1 for kw in keywords if kw in text_lower)
            if matches > 0:
                confidence = min(1.0, matches / 2)
                if confidence >= self.min_confidence and emotion not in existing:
                    suggestions.append((emotion, confidence))

        # Extract potential custom tags from the text
        custom_tags = self._extract_custom_tags(text)
        for tag in custom_tags:
            if tag not in existing and tag not in [s[0] for s in suggestions]:
                suggestions.append((tag, 0.5))

        # Sort by confidence and limit
        suggestions.sort(key=lambda x: x[1], reverse=True)
        return suggestions[:max_tags]

    def auto_tag_episode(
        self,
        episode: Episode,
        max_tags: int = 5,
    ) -> List[str]:
        """
        Auto-tag an episode based on content.

        Args:
            episode: Episode to tag
            max_tags: Maximum tags to add

        Returns:
            List of suggested tags
        """
        # Build text from episode
        text_parts = []
        if episode.topic:
            text_parts.append(episode.topic)
        if episode.summary:
            text_parts.append(episode.summary)
        for msg in episode.messages[:10]:  # First 10 messages
            text_parts.append(msg.get("content", "")[:200])

        text = " ".join(text_parts)
        suggestions = self.suggest_tags(text, episode.tags, max_tags)

        return [tag for tag, _ in suggestions]

    def auto_tag_fact(
        self,
        fact: Fact,
        max_tags: int = 3,
    ) -> List[str]:
        """
        Auto-tag a fact based on content.

        Args:
            fact: Fact to tag
            max_tags: Maximum tags to add

        Returns:
            List of suggested tags
        """
        text = f"{fact.content} {fact.fact_type}"
        suggestions = self.suggest_tags(text, fact.tags, max_tags)

        # Also add fact_type as a tag if it's meaningful
        if fact.fact_type and fact.fact_type not in (FactType.GENERAL, "unknown"):
            if fact.fact_type not in fact.tags:
                suggestions.insert(0, (fact.fact_type, 0.9))

        return [tag for tag, _ in suggestions[:max_tags]]

    def _extract_custom_tags(self, text: str) -> List[str]:
        """Extract potential custom tags from text."""
        # Look for hashtag-style tags
        hashtags = re.findall(r'#(\w+)', text.lower())

        # Look for quoted important terms
        quoted = re.findall(r'"([^"]+)"', text)
        quoted_tags = [q.lower() for q in quoted if len(q) < 20]

        # Extract capitalized compound words (e.g., "ProjectAlpha")
        compounds = re.findall(r'\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b', text)
        compound_tags = [c.lower() for c in compounds]

        return list(set(hashtags + quoted_tags + compound_tags))[:5]
