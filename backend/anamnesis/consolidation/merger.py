"""
Memory merger for consolidating similar/duplicate episodes.

Detects and merges near-duplicate episodes to reduce redundancy
while preserving important information from both.
"""

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from ..models import EmotionalValence, Episode


@dataclass
class MergeCandidate:
    """A pair of episodes that may be mergeable."""
    episode1: Episode
    episode2: Episode
    similarity: float
    merge_reason: str
    common_keywords: List[str]


@dataclass
class MergeResult:
    """Result of a merge operation."""
    merged_episode: Episode
    source_ids: List[str]  # IDs of merged episodes
    preserved_messages: int
    removed_duplicates: int


class EpisodeMerger:
    """
    Merges similar or duplicate episodes.

    Uses multiple signals for similarity:
    - Topic overlap
    - Keyword similarity (Jaccard)
    - Temporal proximity
    - Embedding similarity (if available)
    """

    # Filler words to ignore in comparisons
    STOP_WORDS = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "must", "shall",
        "can", "to", "of", "in", "for", "on", "with", "at", "by",
        "from", "as", "into", "through", "during", "before", "after",
        "i", "you", "he", "she", "it", "we", "they", "me", "him",
        "her", "us", "them", "my", "your", "his", "its", "our",
        "their", "this", "that", "these", "those", "and", "but",
        "or", "if", "because", "while", "what", "when", "where",
        "who", "which", "how", "just", "also", "very", "really",
    }

    def __init__(
        self,
        similarity_threshold: float = 0.6,
        temporal_window_hours: float = 24.0,
    ):
        """
        Initialize merger.

        Args:
            similarity_threshold: Minimum similarity (0-1) to consider merging
            temporal_window_hours: Episodes within this window are more likely merges
        """
        self.similarity_threshold = similarity_threshold
        self.temporal_window_hours = temporal_window_hours

    def find_merge_candidates(
        self,
        episodes: List[Episode],
        use_embeddings: bool = False,
    ) -> List[MergeCandidate]:
        """
        Find pairs of episodes that could be merged.

        Args:
            episodes: List of episodes to analyze
            use_embeddings: Whether to use embedding similarity

        Returns:
            List of MergeCandidate objects sorted by similarity
        """
        candidates = []

        for i, ep1 in enumerate(episodes):
            for ep2 in episodes[i+1:]:
                similarity, reason, keywords = self._calculate_similarity(
                    ep1, ep2, use_embeddings
                )

                if similarity >= self.similarity_threshold:
                    candidates.append(MergeCandidate(
                        episode1=ep1,
                        episode2=ep2,
                        similarity=similarity,
                        merge_reason=reason,
                        common_keywords=keywords,
                    ))

        # Sort by similarity descending
        candidates.sort(key=lambda c: c.similarity, reverse=True)
        return candidates

    def _calculate_similarity(
        self,
        ep1: Episode,
        ep2: Episode,
        use_embeddings: bool,
    ) -> Tuple[float, str, List[str]]:
        """Calculate similarity between two episodes."""
        scores = {}
        reasons = []

        # 1. Topic similarity
        if ep1.topic and ep2.topic:
            topic_sim = self._text_similarity(ep1.topic, ep2.topic)
            scores["topic"] = topic_sim
            if topic_sim > 0.5:
                reasons.append(f"similar topics ({topic_sim:.0%})")

        # 2. Keyword similarity
        kw1 = self._extract_keywords(ep1)
        kw2 = self._extract_keywords(ep2)
        keyword_sim = self._jaccard_similarity(kw1, kw2)
        scores["keywords"] = keyword_sim
        common_keywords = list(kw1 & kw2)
        if keyword_sim > 0.3:
            reasons.append(f"shared keywords: {', '.join(common_keywords[:3])}")

        # 3. Summary similarity
        if ep1.summary and ep2.summary:
            summary_sim = self._text_similarity(ep1.summary, ep2.summary)
            scores["summary"] = summary_sim
            if summary_sim > 0.5:
                reasons.append(f"similar summaries ({summary_sim:.0%})")

        # 4. Temporal proximity bonus
        if ep1.started_at and ep2.started_at:
            hours_apart = abs((ep1.started_at - ep2.started_at).total_seconds() / 3600)
            if hours_apart <= self.temporal_window_hours:
                temporal_bonus = 1.0 - (hours_apart / self.temporal_window_hours)
                scores["temporal"] = temporal_bonus * 0.3
                if temporal_bonus > 0.5:
                    reasons.append(f"close in time ({hours_apart:.1f}h apart)")

        # 5. Embedding similarity (if available)
        if use_embeddings:
            emb1 = ep1.metadata.get("embedding")
            emb2 = ep2.metadata.get("embedding")
            if emb1 and emb2:
                embedding_sim = self._cosine_similarity(emb1, emb2)
                scores["embedding"] = embedding_sim
                if embedding_sim > 0.7:
                    reasons.append(f"semantic similarity ({embedding_sim:.0%})")

        # Calculate weighted average
        weights = {
            "topic": 0.25,
            "keywords": 0.25,
            "summary": 0.2,
            "temporal": 0.1,
            "embedding": 0.2,
        }

        total_weight = sum(weights.get(k, 0) for k in scores.keys())
        if total_weight == 0:
            return 0.0, "", []

        weighted_sim = sum(
            scores.get(k, 0) * weights.get(k, 0)
            for k in scores.keys()
        ) / total_weight

        reason_str = "; ".join(reasons) if reasons else "general similarity"

        return weighted_sim, reason_str, common_keywords

    def _extract_keywords(self, episode: Episode) -> Set[str]:
        """Extract keywords from episode."""
        text = ""
        if episode.topic:
            text += episode.topic + " "
        if episode.summary:
            text += episode.summary + " "
        for msg in episode.messages[:5]:  # Sample first few messages
            text += msg.get("content", "") + " "

        # Tokenize and filter
        words = re.findall(r'\b[a-z]{3,}\b', text.lower())
        keywords = {w for w in words if w not in self.STOP_WORDS}

        return keywords

    def _jaccard_similarity(self, set1: Set[str], set2: Set[str]) -> float:
        """Calculate Jaccard similarity between two sets."""
        if not set1 or not set2:
            return 0.0

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        return intersection / union if union > 0 else 0.0

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using word overlap."""
        words1 = set(re.findall(r'\b[a-z]{3,}\b', text1.lower()))
        words2 = set(re.findall(r'\b[a-z]{3,}\b', text2.lower()))

        # Remove stop words
        words1 = words1 - self.STOP_WORDS
        words2 = words2 - self.STOP_WORDS

        return self._jaccard_similarity(words1, words2)

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def merge_episodes(
        self,
        episode1: Episode,
        episode2: Episode,
        strategy: str = "combine",
    ) -> MergeResult:
        """
        Merge two episodes into one.

        Args:
            episode1: First episode
            episode2: Second episode
            strategy: Merge strategy
                - "combine": Keep all unique information
                - "newer": Prefer newer episode's content
                - "important": Prefer higher importance episode

        Returns:
            MergeResult with merged episode
        """
        # Determine which episode is primary
        if strategy == "newer":
            primary, secondary = (
                (episode2, episode1) if episode2.started_at > episode1.started_at
                else (episode1, episode2)
            )
        elif strategy == "important":
            primary, secondary = (
                (episode1, episode2) if episode1.importance >= episode2.importance
                else (episode2, episode1)
            )
        else:  # combine
            # Primary is more important, or if equal, the older one
            if episode1.importance > episode2.importance:
                primary, secondary = episode1, episode2
            elif episode2.importance > episode1.importance:
                primary, secondary = episode2, episode1
            elif episode1.started_at and episode2.started_at:
                primary, secondary = (
                    (episode1, episode2) if episode1.started_at < episode2.started_at
                    else (episode2, episode1)
                )
            else:
                primary, secondary = episode1, episode2

        # Create merged episode
        merged = Episode(
            id=primary.id,  # Keep primary's ID
            summary=self._merge_summaries(primary.summary, secondary.summary),
            messages=self._merge_messages(primary.messages, secondary.messages),
            started_at=min(
                ep.started_at for ep in [primary, secondary]
                if ep.started_at
            ) if any(ep.started_at for ep in [primary, secondary]) else None,
            ended_at=max(
                ep.ended_at for ep in [primary, secondary]
                if ep.ended_at
            ) if any(ep.ended_at for ep in [primary, secondary]) else None,
            topic=self._merge_topics(primary.topic, secondary.topic),
            participants=list(set(primary.participants + secondary.participants)),
            overall_valence=self._merge_valence(
                primary.overall_valence, secondary.overall_valence
            ),
            emotional_moments=primary.emotional_moments + secondary.emotional_moments,
            extracted_facts=list(set(primary.extracted_facts + secondary.extracted_facts)),
            importance=max(primary.importance, secondary.importance),
            source_id=primary.source_id,
            metadata=self._merge_metadata(primary.metadata, secondary.metadata),
        )

        # Add merge history to metadata
        merged.metadata["merged_from"] = [primary.id, secondary.id]
        merged.metadata["merge_timestamp"] = datetime.now().isoformat()

        # Count removed duplicates
        original_msg_count = len(primary.messages) + len(secondary.messages)
        removed = original_msg_count - len(merged.messages)

        return MergeResult(
            merged_episode=merged,
            source_ids=[primary.id, secondary.id],
            preserved_messages=len(merged.messages),
            removed_duplicates=removed,
        )

    def _merge_summaries(self, summary1: str, summary2: str) -> str:
        """Merge two summaries."""
        if not summary1:
            return summary2
        if not summary2:
            return summary1
        if summary1 == summary2:
            return summary1

        # Check if one contains the other
        if summary2 in summary1:
            return summary1
        if summary1 in summary2:
            return summary2

        # Combine them
        return f"{summary1} Additionally: {summary2}"

    def _merge_topics(self, topic1: Optional[str], topic2: Optional[str]) -> Optional[str]:
        """Merge topics."""
        if not topic1:
            return topic2
        if not topic2:
            return topic1
        if topic1 == topic2:
            return topic1

        # Combine if different
        if topic2.lower() in topic1.lower():
            return topic1
        if topic1.lower() in topic2.lower():
            return topic2

        return f"{topic1} / {topic2}"

    def _merge_messages(
        self,
        messages1: List[Dict[str, str]],
        messages2: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        """Merge messages, removing duplicates."""
        # Create content hashes for deduplication
        seen_content = set()
        merged = []

        # Add messages in chronological order (approximately)
        all_messages = messages1 + messages2

        for msg in all_messages:
            content = msg.get("content", "").strip()
            if not content:
                continue

            # Normalize for comparison
            content_key = content.lower()[:200]  # First 200 chars

            if content_key not in seen_content:
                seen_content.add(content_key)
                merged.append(msg)

        return merged

    def _merge_valence(
        self,
        valence1: EmotionalValence,
        valence2: EmotionalValence,
    ) -> EmotionalValence:
        """Merge emotional valence (weighted average toward extremes)."""
        # Prefer non-neutral
        if valence1 == EmotionalValence.NEUTRAL:
            return valence2
        if valence2 == EmotionalValence.NEUTRAL:
            return valence1

        # Average, preferring stronger emotions
        avg = (valence1.value + valence2.value) / 2

        # Find closest valence
        for valence in EmotionalValence:
            if abs(valence.value - avg) < 0.5:
                return valence

        return EmotionalValence.NEUTRAL

    def _merge_metadata(
        self,
        meta1: Dict[str, Any],
        meta2: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Merge metadata dicts."""
        merged = meta1.copy()

        for key, value in meta2.items():
            if key not in merged:
                merged[key] = value
            elif isinstance(value, list) and isinstance(merged.get(key), list):
                merged[key] = list(set(merged[key] + value))
            elif isinstance(value, (int, float)) and isinstance(merged.get(key), (int, float)):
                merged[key] = max(merged[key], value)

        return merged

    def auto_merge(
        self,
        episodes: List[Episode],
        max_merges: int = 10,
    ) -> List[MergeResult]:
        """
        Automatically find and merge similar episodes.

        Args:
            episodes: List of episodes to consider
            max_merges: Maximum number of merges to perform

        Returns:
            List of MergeResult objects
        """
        results = []
        merged_ids = set()

        candidates = self.find_merge_candidates(episodes)

        for candidate in candidates:
            if len(results) >= max_merges:
                break

            # Skip if either episode was already merged
            if candidate.episode1.id in merged_ids or candidate.episode2.id in merged_ids:
                continue

            result = self.merge_episodes(candidate.episode1, candidate.episode2)
            results.append(result)

            # Mark both as merged
            merged_ids.add(candidate.episode1.id)
            merged_ids.add(candidate.episode2.id)

        return results


def find_duplicates(
    episodes: List[Episode],
    threshold: float = 0.6,
) -> List[MergeCandidate]:
    """Convenience function to find duplicate episodes."""
    merger = EpisodeMerger(similarity_threshold=threshold)
    return merger.find_merge_candidates(episodes)


def merge_episodes(
    episode1: Episode,
    episode2: Episode,
    strategy: str = "combine",
) -> MergeResult:
    """Convenience function to merge two episodes."""
    merger = EpisodeMerger()
    return merger.merge_episodes(episode1, episode2, strategy)
