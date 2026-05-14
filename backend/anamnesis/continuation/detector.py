"""
Conversation continuation detection.

Detects when new conversations continue previous topics
and manages episode chains.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

from ..models import Episode
from ..stores.episodic import EpisodicStore


@dataclass
class ContinuationResult:
    """Result of continuation detection."""
    is_continuation: bool
    source_episode_id: Optional[str]
    confidence: float  # 0.0 to 1.0
    reason: str
    shared_keywords: List[str] = field(default_factory=list)
    topic_similarity: float = 0.0
    temporal_proximity_hours: Optional[float] = None


@dataclass
class ContinuationLink:
    """Link between two episodes in a chain."""
    source_id: str
    target_id: str
    link_type: str  # "continues", "related", "references"
    confidence: float
    created_at: datetime = field(default_factory=datetime.now)


class ContinuationDetector:
    """
    Detects conversation continuations and manages episode chains.

    Uses multiple signals:
    - Topic/keyword overlap
    - Temporal proximity
    - Explicit references ("as we discussed", "continuing from")
    - Entity overlap
    """

    # Continuation phrases that indicate explicit reference
    CONTINUATION_PHRASES = [
        r"as we discussed",
        r"continuing from",
        r"going back to",
        r"about what we said",
        r"regarding our conversation",
        r"you mentioned",
        r"we talked about",
        r"following up on",
        r"as I was saying",
        r"back to the topic of",
        r"remember when we",
        r"you said earlier",
        r"previously we",
        r"last time we",
    ]

    def __init__(
        self,
        episodic_store: EpisodicStore,
        keyword_threshold: float = 0.3,
        temporal_window_hours: float = 72,
        min_confidence: float = 0.5,
    ):
        """
        Initialize continuation detector.

        Args:
            episodic_store: Episode store for retrieval
            keyword_threshold: Min keyword overlap ratio for detection
            temporal_window_hours: Max hours between related episodes
            min_confidence: Minimum confidence for a continuation
        """
        self.episodic_store = episodic_store
        self.keyword_threshold = keyword_threshold
        self.temporal_window = timedelta(hours=temporal_window_hours)
        self.min_confidence = min_confidence

        # Episode chains stored in metadata
        self._chains: Dict[str, Set[str]] = {}

    def detect_continuation(
        self,
        messages: List[Dict[str, Any]],
        topic: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        candidate_limit: int = 10,
    ) -> ContinuationResult:
        """
        Detect if messages continue a previous conversation.

        Args:
            messages: New conversation messages
            topic: Optional topic hint
            timestamp: Conversation timestamp (default: now)
            candidate_limit: Max recent episodes to check

        Returns:
            ContinuationResult with detection details
        """
        if not messages:
            return ContinuationResult(
                is_continuation=False,
                source_episode_id=None,
                confidence=0.0,
                reason="No messages provided",
            )

        timestamp = timestamp or datetime.now()

        # Extract text from messages
        text = self._extract_text(messages)
        keywords = self._extract_keywords(text)

        # Check for explicit continuation phrases
        explicit_ref = self._check_explicit_reference(text)

        # Get recent episodes as candidates
        recent = self.episodic_store.get_recent(limit=candidate_limit)

        if not recent:
            return ContinuationResult(
                is_continuation=False,
                source_episode_id=None,
                confidence=0.0,
                reason="No previous episodes to compare",
            )

        # Score each candidate
        best_match: Optional[Tuple[Episode, float, str, List[str]]] = None

        for ep in recent:
            score, reason, shared = self._score_continuation(
                ep, keywords, topic, timestamp, explicit_ref
            )

            if best_match is None or score > best_match[1]:
                best_match = (ep, score, reason, shared)

        if best_match is None or best_match[1] < self.min_confidence:
            return ContinuationResult(
                is_continuation=False,
                source_episode_id=None,
                confidence=best_match[1] if best_match else 0.0,
                reason="No strong continuation match found",
            )

        ep, confidence, reason, shared_keywords = best_match

        # Calculate temporal proximity
        temporal_hours = None
        if ep.ended_at and timestamp:
            delta = timestamp - ep.ended_at
            temporal_hours = delta.total_seconds() / 3600

        # Calculate topic similarity for the result
        topic_sim = 0.0
        if topic and ep.topic:
            topic_sim = self._keyword_overlap(
                self._extract_keywords(topic),
                self._extract_keywords(ep.topic)
            )

        return ContinuationResult(
            is_continuation=True,
            source_episode_id=ep.id,
            confidence=confidence,
            reason=reason,
            shared_keywords=shared_keywords,
            topic_similarity=topic_sim,
            temporal_proximity_hours=temporal_hours,
        )

    def link_episodes(
        self,
        source_id: str,
        target_id: str,
        link_type: str = "continues",
        confidence: float = 1.0,
    ) -> ContinuationLink:
        """
        Create a continuation link between episodes.

        Args:
            source_id: Earlier episode ID
            target_id: Later episode ID
            link_type: Type of link
            confidence: Link confidence

        Returns:
            ContinuationLink object
        """
        link = ContinuationLink(
            source_id=source_id,
            target_id=target_id,
            link_type=link_type,
            confidence=confidence,
        )

        # Update chain membership
        chain_id = self._get_chain_id(source_id) or source_id
        if chain_id not in self._chains:
            self._chains[chain_id] = {source_id}
        self._chains[chain_id].add(target_id)

        # Update episode metadata
        self._update_episode_chain_metadata(source_id, target_id, link)

        return link

    def get_episode_chain(self, episode_id: str) -> List[Episode]:
        """
        Get the full chain of related episodes.

        Args:
            episode_id: Any episode in the chain

        Returns:
            List of episodes in chronological order
        """
        chain_id = self._get_chain_id(episode_id)
        if not chain_id:
            # Single episode, no chain
            ep = self.episodic_store.get(episode_id)
            return [ep] if ep else []

        # Get all episodes in chain
        episode_ids = self._chains.get(chain_id, {episode_id})
        episodes = []

        for eid in episode_ids:
            ep = self.episodic_store.get(eid)
            if ep:
                episodes.append(ep)

        # Sort by start time
        episodes.sort(key=lambda e: e.started_at or datetime.min)

        return episodes

    def find_related_episodes(
        self,
        episode_id: str,
        limit: int = 5,
    ) -> List[Tuple[Episode, float]]:
        """
        Find episodes related to the given episode.

        Args:
            episode_id: Episode to find relations for
            limit: Max related episodes to return

        Returns:
            List of (Episode, similarity_score) tuples
        """
        source = self.episodic_store.get(episode_id)
        if not source:
            return []

        # Get source keywords
        source_text = f"{source.topic or ''} {source.summary or ''}"
        for msg in source.messages[:5]:
            source_text += " " + msg.get("content", "")[:200]
        source_keywords = self._extract_keywords(source_text)

        # Compare with all episodes
        all_episodes = self.episodic_store.list_all(limit=100)
        scored = []

        for ep in all_episodes:
            if ep.id == episode_id:
                continue

            ep_text = f"{ep.topic or ''} {ep.summary or ''}"
            for msg in ep.messages[:5]:
                ep_text += " " + msg.get("content", "")[:200]
            ep_keywords = self._extract_keywords(ep_text)

            overlap = self._keyword_overlap(source_keywords, ep_keywords)
            if overlap > 0.1:  # Minimum threshold
                scored.append((ep, overlap))

        # Sort by similarity
        scored.sort(key=lambda x: x[1], reverse=True)

        return scored[:limit]

    def auto_detect_chains(self, threshold: float = 0.4) -> int:
        """
        Automatically detect and link episode chains.

        Args:
            threshold: Minimum similarity for auto-linking

        Returns:
            Number of links created
        """
        episodes = self.episodic_store.list_all(limit=1000)

        # Sort by time
        episodes.sort(key=lambda e: e.started_at or datetime.min)

        links_created = 0

        for i, ep in enumerate(episodes):
            if i == 0:
                continue

            # Check against previous episodes (within temporal window)
            for prev in reversed(episodes[:i]):
                if prev.ended_at and ep.started_at:
                    time_gap = ep.started_at - prev.ended_at
                    if time_gap > self.temporal_window:
                        break  # Outside window, stop checking

                # Calculate similarity
                score, _, shared = self._score_continuation(
                    prev,
                    self._extract_keywords(
                        f"{ep.topic or ''} {ep.summary or ''}"
                    ),
                    ep.topic,
                    ep.started_at,
                    False,
                )

                if score >= threshold and shared:
                    self.link_episodes(prev.id, ep.id, "continues", score)
                    links_created += 1
                    break  # Only link to most recent match

        return links_created

    def _extract_text(self, messages: List[Dict[str, Any]]) -> str:
        """Extract text content from messages."""
        parts = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and "text" in item:
                        parts.append(item["text"])
        return " ".join(parts)

    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract keywords from text."""
        # Simple keyword extraction
        words = re.findall(r'\b[a-z]{3,}\b', text.lower())

        # Filter stop words
        stop_words = {
            "the", "and", "but", "for", "are", "was", "were", "been",
            "being", "have", "has", "had", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "can",
            "this", "that", "these", "those", "with", "from", "into",
            "what", "when", "where", "which", "who", "how", "why",
            "you", "your", "you're", "i'm", "it's", "we're", "they",
            "not", "just", "also", "very", "really", "some", "any",
            "all", "more", "most", "other", "than", "then", "now",
            "about", "know", "think", "want", "like", "make", "time",
        }

        keywords = {w for w in words if w not in stop_words}
        return keywords

    def _keyword_overlap(self, kw1: Set[str], kw2: Set[str]) -> float:
        """Calculate Jaccard similarity of keyword sets."""
        if not kw1 or not kw2:
            return 0.0

        intersection = kw1 & kw2
        union = kw1 | kw2

        return len(intersection) / len(union) if union else 0.0

    def _check_explicit_reference(self, text: str) -> bool:
        """Check for explicit continuation phrases."""
        text_lower = text.lower()
        for phrase in self.CONTINUATION_PHRASES:
            if re.search(phrase, text_lower):
                return True
        return False

    def _score_continuation(
        self,
        episode: Episode,
        new_keywords: Set[str],
        new_topic: Optional[str],
        timestamp: datetime,
        has_explicit_ref: bool,
    ) -> Tuple[float, str, List[str]]:
        """
        Score how likely an episode is the source of a continuation.

        Returns: (score, reason, shared_keywords)
        """
        score = 0.0
        reasons = []

        # Extract episode keywords
        ep_text = f"{episode.topic or ''} {episode.summary or ''}"
        for msg in episode.messages[:5]:
            ep_text += " " + msg.get("content", "")[:200]
        ep_keywords = self._extract_keywords(ep_text)

        # Keyword overlap
        shared = new_keywords & ep_keywords
        overlap = self._keyword_overlap(new_keywords, ep_keywords)

        if overlap >= self.keyword_threshold:
            score += overlap * 0.5
            reasons.append(f"keyword overlap ({overlap:.2f})")

        # Topic match
        if new_topic and episode.topic:
            topic_keywords = self._extract_keywords(new_topic)
            ep_topic_keywords = self._extract_keywords(episode.topic)
            topic_overlap = self._keyword_overlap(topic_keywords, ep_topic_keywords)

            if topic_overlap > 0.3:
                score += topic_overlap * 0.3
                reasons.append(f"topic match ({topic_overlap:.2f})")

        # Temporal proximity
        if episode.ended_at and timestamp:
            time_delta = timestamp - episode.ended_at
            if time_delta <= self.temporal_window:
                # Higher score for more recent
                recency = 1.0 - (time_delta / self.temporal_window)
                score += recency * 0.2
                reasons.append(f"recent ({time_delta.total_seconds()/3600:.1f}h ago)")

        # Explicit reference bonus
        if has_explicit_ref:
            score += 0.3
            reasons.append("explicit reference")

        # Cap at 1.0
        score = min(1.0, score)

        reason = "; ".join(reasons) if reasons else "weak match"

        return score, reason, list(shared)

    def _get_chain_id(self, episode_id: str) -> Optional[str]:
        """Get the chain ID for an episode."""
        for chain_id, members in self._chains.items():
            if episode_id in members:
                return chain_id
        return None

    def _update_episode_chain_metadata(
        self,
        source_id: str,
        target_id: str,
        link: ContinuationLink,
    ):
        """Update episode metadata with chain information."""
        # Update source episode
        source = self.episodic_store.get(source_id)
        if source:
            continues_to = source.metadata.get("continues_to", [])
            if target_id not in continues_to:
                continues_to.append(target_id)
            source.metadata["continues_to"] = continues_to
            self.episodic_store.save(source)

        # Update target episode
        target = self.episodic_store.get(target_id)
        if target:
            target.metadata["continues_from"] = source_id
            target.metadata["continuation_confidence"] = link.confidence
            self.episodic_store.save(target)
