"""
Conversation chunking for long conversations.

Detects topic shifts and splits conversations into
coherent chunks for better retrieval.
"""

import re
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from ..models import EmotionalValence, Episode


@dataclass
class ChunkBoundary:
    """Boundary between conversation chunks."""
    message_index: int
    confidence: float  # 0.0 to 1.0
    reason: str
    previous_topic_keywords: List[str]
    new_topic_keywords: List[str]


@dataclass
class ChunkResult:
    """Result of chunking a conversation."""
    chunks: List[Episode]
    boundaries: List[ChunkBoundary]
    original_message_count: int
    chunk_count: int


class ConversationChunker:
    """
    Chunks long conversations by detecting topic shifts.

    Uses multiple signals to detect topic boundaries:
    - Keyword distribution changes
    - Explicit topic markers ("Let's talk about...", "Switching to...")
    - Long pauses in timestamp (if available)
    - Dramatic shift in entities/concepts
    """

    # Phrases that indicate topic change
    TOPIC_CHANGE_PHRASES = [
        r"let'?s talk about",
        r"moving on to",
        r"switching to",
        r"changing topic",
        r"on a different note",
        r"anyway,? (?:so )?",
        r"by the way",
        r"speaking of",
        r"now,? about",
        r"can we discuss",
        r"i want to ask about",
        r"another question",
        r"different topic",
        r"unrelated,? but",
    ]

    def __init__(
        self,
        min_chunk_size: int = 4,
        max_chunk_size: int = 50,
        similarity_threshold: float = 0.3,
        window_size: int = 5,
    ):
        """
        Initialize chunker.

        Args:
            min_chunk_size: Minimum messages per chunk
            max_chunk_size: Maximum messages before forcing split
            similarity_threshold: Min similarity to same topic (0-1)
            window_size: Messages to consider for topic comparison
        """
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.similarity_threshold = similarity_threshold
        self.window_size = window_size

    def chunk_conversation(
        self,
        messages: List[Dict[str, Any]],
        source_id: Optional[str] = None,
        base_timestamp: Optional[datetime] = None,
    ) -> ChunkResult:
        """
        Chunk a conversation into topic-based episodes.

        Args:
            messages: List of message dicts with 'role' and 'content'
            source_id: Optional source identifier
            base_timestamp: Base timestamp for episodes

        Returns:
            ChunkResult with episodes and boundary info
        """
        if len(messages) <= self.min_chunk_size:
            # Too short to chunk
            episode = self._create_episode(
                messages, 0, source_id, base_timestamp
            )
            return ChunkResult(
                chunks=[episode],
                boundaries=[],
                original_message_count=len(messages),
                chunk_count=1,
            )

        # Detect boundaries
        boundaries = self._detect_boundaries(messages)

        # Create chunks from boundaries
        chunks = self._create_chunks(
            messages, boundaries, source_id, base_timestamp
        )

        # Link chunks with FOLLOWS relationship
        self._link_chunks(chunks)

        return ChunkResult(
            chunks=chunks,
            boundaries=boundaries,
            original_message_count=len(messages),
            chunk_count=len(chunks),
        )

    def _detect_boundaries(
        self,
        messages: List[Dict[str, Any]],
    ) -> List[ChunkBoundary]:
        """
        Detect topic shift boundaries in messages.

        Returns list of boundaries (not including start/end).
        """
        boundaries = []
        current_chunk_start = 0

        for i in range(self.min_chunk_size, len(messages)):
            # Don't check too frequently
            chunk_len = i - current_chunk_start
            if chunk_len < self.min_chunk_size:
                continue

            # Force split at max size
            if chunk_len >= self.max_chunk_size:
                boundary = ChunkBoundary(
                    message_index=i,
                    confidence=1.0,
                    reason="max chunk size reached",
                    previous_topic_keywords=[],
                    new_topic_keywords=[],
                )
                boundaries.append(boundary)
                current_chunk_start = i
                continue

            # Check for topic shift
            shift_info = self._check_topic_shift(messages, i)

            if shift_info["is_shift"]:
                boundary = ChunkBoundary(
                    message_index=i,
                    confidence=shift_info["confidence"],
                    reason=shift_info["reason"],
                    previous_topic_keywords=shift_info["prev_keywords"],
                    new_topic_keywords=shift_info["new_keywords"],
                )
                boundaries.append(boundary)
                current_chunk_start = i

        return boundaries

    def _check_topic_shift(
        self,
        messages: List[Dict[str, Any]],
        index: int,
    ) -> Dict[str, Any]:
        """
        Check if there's a topic shift at given index.

        Returns dict with is_shift, confidence, reason, keywords.
        """
        result = {
            "is_shift": False,
            "confidence": 0.0,
            "reason": "",
            "prev_keywords": [],
            "new_keywords": [],
        }

        # Get text windows
        prev_start = max(0, index - self.window_size)
        prev_text = self._extract_text(messages[prev_start:index])
        next_text = self._extract_text(messages[index:index + self.window_size])

        if not prev_text or not next_text:
            return result

        # Check for explicit topic change phrases
        current_msg = self._get_content(messages[index])
        if self._has_topic_change_phrase(current_msg):
            result["is_shift"] = True
            result["confidence"] = 0.9
            result["reason"] = "explicit topic change phrase"
            return result

        # Compare keyword distributions
        prev_keywords = self._extract_keywords(prev_text)
        next_keywords = self._extract_keywords(next_text)

        result["prev_keywords"] = list(prev_keywords)[:5]
        result["new_keywords"] = list(next_keywords)[:5]

        similarity = self._jaccard_similarity(prev_keywords, next_keywords)

        # Low similarity = likely topic shift
        if similarity < self.similarity_threshold:
            result["is_shift"] = True
            result["confidence"] = 1.0 - similarity
            result["reason"] = f"topic divergence (sim={similarity:.2f})"

        return result

    def _create_chunks(
        self,
        messages: List[Dict[str, Any]],
        boundaries: List[ChunkBoundary],
        source_id: Optional[str],
        base_timestamp: Optional[datetime],
    ) -> List[Episode]:
        """Create Episode objects from message chunks."""
        chunks = []

        # Build boundary indices
        indices = [0] + [b.message_index for b in boundaries] + [len(messages)]

        for i in range(len(indices) - 1):
            start = indices[i]
            end = indices[i + 1]

            chunk_messages = messages[start:end]
            if not chunk_messages:
                continue

            episode = self._create_episode(
                chunk_messages,
                i,
                source_id,
                base_timestamp,
            )
            chunks.append(episode)

        return chunks

    def _create_episode(
        self,
        messages: List[Dict[str, Any]],
        chunk_index: int,
        source_id: Optional[str],
        base_timestamp: Optional[datetime],
    ) -> Episode:
        """Create an Episode from messages."""
        # Generate ID
        episode_id = f"chunk-{uuid.uuid4().hex[:12]}"

        # Extract topic
        topic = self._infer_topic(messages)

        # Generate summary
        summary = self._generate_summary(messages)

        # Analyze emotional valence
        valence = self._analyze_valence(messages)

        # Calculate importance
        importance = self._calculate_importance(messages)

        return Episode(
            id=episode_id,
            messages=messages,
            summary=summary,
            topic=topic,
            started_at=base_timestamp,
            ended_at=base_timestamp,
            overall_valence=valence,
            importance=importance,
            source_id=source_id,
            metadata={
                "chunk_index": chunk_index,
                "is_chunk": True,
                "message_count": len(messages),
            },
        )

    def _link_chunks(self, chunks: List[Episode]):
        """Link chunks with FOLLOWS relationship in metadata."""
        for i, chunk in enumerate(chunks):
            if i > 0:
                chunk.metadata["follows"] = chunks[i - 1].id
                chunk.metadata["previous_chunk"] = chunks[i - 1].id

            if i < len(chunks) - 1:
                chunk.metadata["followed_by"] = chunks[i + 1].id
                chunk.metadata["next_chunk"] = chunks[i + 1].id

    def _extract_text(self, messages: List[Dict[str, Any]]) -> str:
        """Extract text from messages."""
        parts = []
        for msg in messages:
            content = self._get_content(msg)
            if content:
                parts.append(content)
        return " ".join(parts)

    def _get_content(self, message: Dict[str, Any]) -> str:
        """Get text content from a message."""
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    parts.append(item["text"])
                elif isinstance(item, str):
                    parts.append(item)
            return " ".join(parts)
        return ""

    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract keywords from text."""
        words = re.findall(r'\b[a-z]{3,}\b', text.lower())

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
            "help", "need", "please", "thanks", "sure", "okay", "yes",
        }

        return {w for w in words if w not in stop_words}

    def _jaccard_similarity(self, set1: Set[str], set2: Set[str]) -> float:
        """Calculate Jaccard similarity between sets."""
        if not set1 or not set2:
            return 0.0

        intersection = set1 & set2
        union = set1 | set2

        return len(intersection) / len(union) if union else 0.0

    def _has_topic_change_phrase(self, text: str) -> bool:
        """Check for explicit topic change phrases."""
        text_lower = text.lower()
        for pattern in self.TOPIC_CHANGE_PHRASES:
            if re.search(pattern, text_lower):
                return True
        return False

    def _infer_topic(self, messages: List[Dict[str, Any]]) -> str:
        """Infer topic from messages."""
        # Get first user message as topic hint
        for msg in messages:
            if msg.get("role") == "user":
                content = self._get_content(msg)
                if content:
                    # Take first sentence or up to 50 chars
                    first_sentence = content.split('.')[0][:50]
                    return first_sentence.strip()

        # Fallback: use most common keywords
        text = self._extract_text(messages)
        keywords = self._extract_keywords(text)
        if keywords:
            counter = Counter(keywords)
            top = counter.most_common(3)
            return " ".join(w for w, _ in top).title()

        return "Untitled Chunk"

    def _generate_summary(self, messages: List[Dict[str, Any]]) -> str:
        """Generate a brief summary of messages."""
        # Simple approach: first user message + first assistant response
        user_msg = None
        assistant_msg = None

        for msg in messages:
            content = self._get_content(msg)
            if not content:
                continue

            if msg.get("role") == "user" and not user_msg:
                user_msg = content[:100]
            elif msg.get("role") == "assistant" and not assistant_msg:
                assistant_msg = content[:100]

            if user_msg and assistant_msg:
                break

        parts = []
        if user_msg:
            parts.append(f"User asked: {user_msg}")
        if assistant_msg:
            parts.append(f"Response: {assistant_msg}")

        if parts:
            return " | ".join(parts)

        return f"Conversation chunk with {len(messages)} messages"

    def _analyze_valence(
        self,
        messages: List[Dict[str, Any]],
    ) -> EmotionalValence:
        """Analyze emotional valence of messages."""
        text = self._extract_text(messages).lower()

        positive_words = {
            "great", "good", "excellent", "wonderful", "amazing", "love",
            "happy", "helpful", "thanks", "thank", "appreciate", "perfect",
            "awesome", "fantastic", "brilliant", "glad", "pleased",
        }
        negative_words = {
            "bad", "terrible", "awful", "hate", "angry", "frustrated",
            "annoyed", "disappointed", "wrong", "error", "problem",
            "issue", "fail", "failed", "broken", "stuck", "confused",
        }

        words = set(text.split())
        pos_count = len(words & positive_words)
        neg_count = len(words & negative_words)

        if pos_count > neg_count + 2:
            return EmotionalValence.POSITIVE
        elif pos_count > neg_count:
            return EmotionalValence.POSITIVE
        elif neg_count > pos_count + 2:
            return EmotionalValence.NEGATIVE
        elif neg_count > pos_count:
            return EmotionalValence.NEGATIVE

        return EmotionalValence.NEUTRAL

    def _calculate_importance(self, messages: List[Dict[str, Any]]) -> float:
        """Calculate importance score for chunk."""
        # Base importance
        importance = 0.5

        # More messages = potentially more important
        msg_count = len(messages)
        if msg_count > 10:
            importance += 0.1
        if msg_count > 20:
            importance += 0.1

        # Check for importance signals in text
        text = self._extract_text(messages).lower()

        important_signals = [
            "important", "critical", "remember", "note", "key",
            "crucial", "essential", "must", "urgent", "priority",
        ]

        for signal in important_signals:
            if signal in text:
                importance += 0.05

        return min(1.0, importance)


def chunk_conversation(
    messages: List[Dict[str, Any]],
    min_chunk_size: int = 4,
    max_chunk_size: int = 50,
) -> ChunkResult:
    """
    Convenience function to chunk a conversation.

    Args:
        messages: List of messages
        min_chunk_size: Minimum messages per chunk
        max_chunk_size: Maximum messages per chunk

    Returns:
        ChunkResult with episodes and boundaries
    """
    chunker = ConversationChunker(
        min_chunk_size=min_chunk_size,
        max_chunk_size=max_chunk_size,
    )
    return chunker.chunk_conversation(messages)
