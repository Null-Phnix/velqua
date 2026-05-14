"""
Conversation continuity detection.

Identifies when conversations are continuations of previous ones
and builds chains of related discussions.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set

from ..models import Episode


@dataclass
class ContinuityResult:
    """Result of continuity analysis."""
    is_continuation: bool = False
    continued_episode_id: Optional[str] = None
    continued_episode: Optional[Episode] = None
    confidence: float = 0.0
    signals: List[str] = field(default_factory=list)  # Why we think it's a continuation


@dataclass
class ConversationChain:
    """A chain of related conversations."""
    chain_id: str
    episodes: List[Episode] = field(default_factory=list)
    topic: Optional[str] = None
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    total_messages: int = 0

    @property
    def duration_days(self) -> Optional[float]:
        """Get chain duration in days."""
        if self.started_at and self.ended_at:
            return (self.ended_at - self.started_at).total_seconds() / 86400
        return None

    @property
    def episode_count(self) -> int:
        """Get number of episodes in chain."""
        return len(self.episodes)


class ContinuityDetector:
    """
    Detects conversation continuity.

    Identifies when a new conversation is a continuation of a
    previous one based on:
    - Topic similarity
    - Temporal proximity
    - Explicit references ("as we discussed", "continuing from")
    - Shared entities/keywords
    """

    # Continuation phrases
    CONTINUATION_PHRASES = [
        r'\b(as|like) (we|I) (mentioned|discussed|said|talked about)',
        r'\bcontinuing (from|our)',
        r'\bpicking up (from|where)',
        r'\bgoing back to (what|our)',
        r'\bremember (when|what) (we|I)',
        r'\bearlier (we|I|you)',
        r'\blast time (we|I)',
        r'\bbefore[,.]? (we|I)',
        r'\bwhat (we|I) were (talking|discussing)',
        r'\bfollowing up',
        r'\bto follow up',
        r'\bback to (our|the|that)',
    ]

    def __init__(
        self,
        topic_similarity_threshold: float = 0.4,
        temporal_window_hours: int = 72,
        min_keyword_overlap: int = 2,
    ):
        """
        Initialize continuity detector.

        Args:
            topic_similarity_threshold: Min similarity for topic match
            temporal_window_hours: Hours to consider for recency
            min_keyword_overlap: Min shared keywords for topic match
        """
        self.topic_similarity_threshold = topic_similarity_threshold
        self.temporal_window_hours = temporal_window_hours
        self.min_keyword_overlap = min_keyword_overlap

    def detect(
        self,
        current_messages: List[Dict[str, str]],
        recent_episodes: List[Episode],
        current_topic: Optional[str] = None,
    ) -> ContinuityResult:
        """
        Detect if current conversation continues a previous one.

        Args:
            current_messages: Messages in current conversation
            recent_episodes: Recent episodes to check against
            current_topic: Optional topic hint

        Returns:
            ContinuityResult with detection info
        """
        if not current_messages or not recent_episodes:
            return ContinuityResult()

        # Combine message content for analysis
        current_text = " ".join(
            m.get("content", "") for m in current_messages
        ).lower()

        signals = []
        best_match = None
        best_confidence = 0.0

        # Check for explicit continuation phrases
        for pattern in self.CONTINUATION_PHRASES:
            if re.search(pattern, current_text, re.IGNORECASE):
                signals.append(f"Explicit phrase: {pattern[:30]}...")
                # Boost confidence for explicit references
                best_confidence = max(best_confidence, 0.6)

        # Check each recent episode
        for episode in recent_episodes:
            confidence = 0.0
            episode_signals = []

            # Topic similarity
            if current_topic and episode.topic:
                topic_sim = self._topic_similarity(current_topic, episode.topic)
                if topic_sim > self.topic_similarity_threshold:
                    confidence += 0.3
                    episode_signals.append(f"Topic match: {topic_sim:.2f}")

            # Keyword overlap
            episode_text = self._get_episode_text(episode)
            keyword_overlap = self._keyword_overlap(current_text, episode_text)
            if keyword_overlap >= self.min_keyword_overlap:
                confidence += 0.2 * min(keyword_overlap / 5, 1.0)
                episode_signals.append(f"Keyword overlap: {keyword_overlap}")

            # Temporal proximity
            if episode.started_at:
                hours_ago = (datetime.now() - episode.started_at).total_seconds() / 3600
                if hours_ago <= self.temporal_window_hours:
                    temporal_boost = 1.0 - (hours_ago / self.temporal_window_hours)
                    confidence += 0.2 * temporal_boost
                    episode_signals.append(f"Recent: {hours_ago:.1f}h ago")

            # Entity mentions
            entities = self._extract_entities(current_text)
            episode_entities = self._extract_entities(episode_text)
            shared_entities = entities & episode_entities
            if shared_entities:
                confidence += 0.15 * min(len(shared_entities) / 3, 1.0)
                episode_signals.append(f"Shared entities: {len(shared_entities)}")

            # Check if this is best match
            if confidence > best_confidence:
                best_confidence = confidence
                best_match = episode
                signals = episode_signals

        # Determine if it's a continuation
        is_continuation = best_confidence >= 0.4

        return ContinuityResult(
            is_continuation=is_continuation,
            continued_episode_id=best_match.id if best_match else None,
            continued_episode=best_match,
            confidence=best_confidence,
            signals=signals,
        )

    def build_chain(
        self,
        episodes: List[Episode],
        max_gap_hours: int = 168,  # 1 week
    ) -> List[ConversationChain]:
        """
        Build chains of related conversations.

        Groups episodes that are likely part of the same ongoing
        discussion across multiple sessions.

        Args:
            episodes: Episodes to group
            max_gap_hours: Max hours between episodes in same chain

        Returns:
            List of conversation chains
        """
        if not episodes:
            return []

        # Sort by time
        sorted_eps = sorted(
            [ep for ep in episodes if ep.started_at],
            key=lambda ep: ep.started_at,
        )

        if not sorted_eps:
            return []

        chains: List[ConversationChain] = []
        current_chain: List[Episode] = [sorted_eps[0]]

        for ep in sorted_eps[1:]:
            prev_ep = current_chain[-1]

            # Check if should continue chain
            should_continue = False

            # Temporal check
            if prev_ep.started_at and ep.started_at:
                gap_hours = (ep.started_at - prev_ep.started_at).total_seconds() / 3600
                if gap_hours <= max_gap_hours:
                    # Topic similarity check
                    if prev_ep.topic and ep.topic:
                        sim = self._topic_similarity(prev_ep.topic, ep.topic)
                        if sim > self.topic_similarity_threshold:
                            should_continue = True

                    # Keyword overlap check
                    if not should_continue:
                        prev_text = self._get_episode_text(prev_ep)
                        curr_text = self._get_episode_text(ep)
                        overlap = self._keyword_overlap(prev_text, curr_text)
                        if overlap >= self.min_keyword_overlap:
                            should_continue = True

            if should_continue:
                current_chain.append(ep)
            else:
                # Finish current chain
                if len(current_chain) > 1:  # Only save multi-episode chains
                    chains.append(self._create_chain(current_chain))
                current_chain = [ep]

        # Don't forget the last chain
        if len(current_chain) > 1:
            chains.append(self._create_chain(current_chain))

        return chains

    def find_related(
        self,
        episode: Episode,
        all_episodes: List[Episode],
        limit: int = 5,
    ) -> List[Episode]:
        """
        Find episodes related to a given episode.

        Args:
            episode: Episode to find related for
            all_episodes: All episodes to search
            limit: Maximum related episodes to return

        Returns:
            List of related episodes
        """
        if not all_episodes:
            return []

        # Score each candidate
        scored = []
        episode_text = self._get_episode_text(episode)

        for candidate in all_episodes:
            if candidate.id == episode.id:
                continue

            score = 0.0

            # Topic similarity
            if episode.topic and candidate.topic:
                sim = self._topic_similarity(episode.topic, candidate.topic)
                score += sim * 0.4

            # Keyword overlap
            cand_text = self._get_episode_text(candidate)
            overlap = self._keyword_overlap(episode_text, cand_text)
            score += min(overlap / 10, 0.3)

            # Temporal proximity bonus
            if episode.started_at and candidate.started_at:
                gap_days = abs((episode.started_at - candidate.started_at).total_seconds()) / 86400
                if gap_days <= 7:
                    score += 0.2 * (1 - gap_days / 7)
                elif gap_days <= 30:
                    score += 0.1 * (1 - gap_days / 30)

            if score > 0:
                scored.append((candidate, score))

        # Sort by score and return top
        scored.sort(key=lambda x: x[1], reverse=True)
        return [ep for ep, _ in scored[:limit]]

    def _topic_similarity(self, topic1: str, topic2: str) -> float:
        """Calculate simple topic similarity."""
        words1 = set(topic1.lower().split())
        words2 = set(topic2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union)

    def _keyword_overlap(self, text1: str, text2: str) -> int:
        """Count significant keyword overlap."""
        # Extract significant words (>4 chars, not common)
        STOPWORDS = {'about', 'after', 'because', 'before', 'could', 'during',
                     'going', 'would', 'should', 'their', 'there', 'these',
                     'thing', 'think', 'those', 'through', 'where', 'which',
                     'while', 'being', 'having', 'other'}

        def get_keywords(text: str) -> Set[str]:
            words = re.findall(r'\b\w{4,}\b', text.lower())
            return {w for w in words if w not in STOPWORDS}

        kw1 = get_keywords(text1)
        kw2 = get_keywords(text2)

        return len(kw1 & kw2)

    def _extract_entities(self, text: str) -> Set[str]:
        """Extract potential named entities (simple heuristic)."""
        # Find capitalized words (rough entity extraction)
        entities = set()

        # Proper nouns (capitalized words not at sentence start)
        words = text.split()
        for i, word in enumerate(words):
            if i > 0 and word[0:1].isupper() and len(word) > 2:
                # Clean punctuation
                clean = re.sub(r'[^\w]', '', word)
                if clean:
                    entities.add(clean.lower())

        return entities

    def _get_episode_text(self, episode: Episode) -> str:
        """Get searchable text from episode."""
        parts = []
        if episode.topic:
            parts.append(episode.topic)
        if episode.summary:
            parts.append(episode.summary)
        for msg in episode.messages[:5]:  # First 5 messages
            if isinstance(msg, dict):
                parts.append(msg.get("content", ""))
            elif hasattr(msg, "content"):
                parts.append(msg.content)
        return " ".join(parts).lower()

    def _create_chain(self, episodes: List[Episode]) -> ConversationChain:
        """Create a chain from episodes."""
        import uuid

        # Determine chain topic
        topics = [ep.topic for ep in episodes if ep.topic]
        topic = topics[0] if topics else None

        # Calculate total messages
        total = sum(len(ep.messages) for ep in episodes)

        # Get time bounds
        times = [ep.started_at for ep in episodes if ep.started_at]

        return ConversationChain(
            chain_id=f"chain-{uuid.uuid4().hex[:8]}",
            episodes=episodes,
            topic=topic,
            started_at=min(times) if times else None,
            ended_at=max(times) if times else None,
            total_messages=total,
        )
