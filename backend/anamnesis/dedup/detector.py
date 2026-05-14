"""
Duplicate memory detection.

Finds and flags duplicate or near-duplicate memories in the system.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from ..models import Episode, Fact
from ..stores.episodic import EpisodicStore
from ..stores.semantic import SemanticStore
from .similarity import quick_similarity


@dataclass
class DuplicateCandidate:
    """A pair of potentially duplicate memories."""
    memory1_id: str
    memory2_id: str
    memory_type: str  # "episode" or "fact"
    similarity: float  # 0.0 to 1.0
    match_reasons: List[str] = field(default_factory=list)
    keep_recommendation: Optional[str] = None  # ID of memory to keep
    recommendation_reason: str = ""


@dataclass
class DuplicateStats:
    """Statistics about duplicate detection."""
    episodes_scanned: int
    facts_scanned: int
    episode_duplicates: int
    fact_duplicates: int
    highest_similarity: float
    avg_similarity: float


class DuplicateDetector:
    """
    Detects duplicate or near-duplicate memories.

    Uses multiple signals to identify duplicates:
    - Content similarity (text matching)
    - Topic matching
    - Temporal proximity
    - Source ID matching
    """

    def __init__(
        self,
        episodic_store: Optional[EpisodicStore] = None,
        semantic_store: Optional[SemanticStore] = None,
        similarity_threshold: float = 0.7,
        temporal_window_hours: float = 24.0,
    ):
        """
        Initialize detector.

        Args:
            episodic_store: Episode store for lookups
            semantic_store: Fact store for lookups
            similarity_threshold: Min similarity to consider duplicate (0-1)
            temporal_window_hours: Time window for temporal proximity bonus
        """
        self.episodic_store = episodic_store
        self.semantic_store = semantic_store
        self.similarity_threshold = similarity_threshold
        self.temporal_window_hours = temporal_window_hours

    def find_episode_duplicates(
        self,
        episodes: Optional[List[Episode]] = None,
        limit: int = 100,
    ) -> List[DuplicateCandidate]:
        """
        Find duplicate episodes.

        Args:
            episodes: Episodes to check (or load from store)
            limit: Max duplicates to return

        Returns:
            List of duplicate candidates
        """
        if episodes is None and self.episodic_store:
            episodes = self.episodic_store.list_all(limit=1000)

        if not episodes:
            return []

        duplicates = []
        checked_pairs: Set[Tuple[str, str]] = set()

        for i, ep1 in enumerate(episodes):
            for ep2 in episodes[i + 1:]:
                # Skip if already checked
                pair = tuple(sorted([ep1.id, ep2.id]))
                if pair in checked_pairs:
                    continue
                checked_pairs.add(pair)

                similarity, reasons = self._compare_episodes(ep1, ep2)

                if similarity >= self.similarity_threshold:
                    keep_id, keep_reason = self._recommend_episode_keep(ep1, ep2)

                    duplicates.append(DuplicateCandidate(
                        memory1_id=ep1.id,
                        memory2_id=ep2.id,
                        memory_type="episode",
                        similarity=similarity,
                        match_reasons=reasons,
                        keep_recommendation=keep_id,
                        recommendation_reason=keep_reason,
                    ))

                    if len(duplicates) >= limit:
                        break

            if len(duplicates) >= limit:
                break

        # Sort by similarity (highest first)
        duplicates.sort(key=lambda d: d.similarity, reverse=True)
        return duplicates

    def find_fact_duplicates(
        self,
        facts: Optional[List[Fact]] = None,
        limit: int = 100,
    ) -> List[DuplicateCandidate]:
        """
        Find duplicate facts.

        Args:
            facts: Facts to check (or load from store)
            limit: Max duplicates to return

        Returns:
            List of duplicate candidates
        """
        if facts is None and self.semantic_store:
            facts = self.semantic_store.list_all(limit=1000)

        if not facts:
            return []

        duplicates = []
        checked_pairs: Set[Tuple[str, str]] = set()

        for i, fact1 in enumerate(facts):
            for fact2 in facts[i + 1:]:
                # Skip if already checked
                pair = tuple(sorted([fact1.id, fact2.id]))
                if pair in checked_pairs:
                    continue
                checked_pairs.add(pair)

                similarity, reasons = self._compare_facts(fact1, fact2)

                if similarity >= self.similarity_threshold:
                    keep_id, keep_reason = self._recommend_fact_keep(fact1, fact2)

                    duplicates.append(DuplicateCandidate(
                        memory1_id=fact1.id,
                        memory2_id=fact2.id,
                        memory_type="fact",
                        similarity=similarity,
                        match_reasons=reasons,
                        keep_recommendation=keep_id,
                        recommendation_reason=keep_reason,
                    ))

                    if len(duplicates) >= limit:
                        break

            if len(duplicates) >= limit:
                break

        # Sort by similarity (highest first)
        duplicates.sort(key=lambda d: d.similarity, reverse=True)
        return duplicates

    def find_all_duplicates(
        self,
        limit: int = 100,
    ) -> Dict[str, List[DuplicateCandidate]]:
        """
        Find all duplicates in the system.

        Returns:
            Dict with 'episodes' and 'facts' lists
        """
        return {
            "episodes": self.find_episode_duplicates(limit=limit),
            "facts": self.find_fact_duplicates(limit=limit),
        }

    def check_is_duplicate(
        self,
        new_episode: Episode,
        existing_episodes: Optional[List[Episode]] = None,
    ) -> Optional[DuplicateCandidate]:
        """
        Check if a new episode is a duplicate of any existing episode.

        Args:
            new_episode: Episode to check
            existing_episodes: Episodes to compare against (or load from store)

        Returns:
            DuplicateCandidate if duplicate found, None otherwise
        """
        if existing_episodes is None and self.episodic_store:
            existing_episodes = self.episodic_store.list_all(limit=1000)

        if not existing_episodes:
            return None

        best_match = None
        best_similarity = 0.0

        for existing in existing_episodes:
            if existing.id == new_episode.id:
                continue

            similarity, reasons = self._compare_episodes(new_episode, existing)

            if similarity >= self.similarity_threshold and similarity > best_similarity:
                keep_id, keep_reason = self._recommend_episode_keep(new_episode, existing)

                best_match = DuplicateCandidate(
                    memory1_id=new_episode.id,
                    memory2_id=existing.id,
                    memory_type="episode",
                    similarity=similarity,
                    match_reasons=reasons,
                    keep_recommendation=keep_id,
                    recommendation_reason=keep_reason,
                )
                best_similarity = similarity

        return best_match

    def check_fact_is_duplicate(
        self,
        new_fact: Fact,
        existing_facts: Optional[List[Fact]] = None,
    ) -> Optional[DuplicateCandidate]:
        """
        Check if a new fact is a duplicate of any existing fact.

        Args:
            new_fact: Fact to check
            existing_facts: Facts to compare against (or load from store)

        Returns:
            DuplicateCandidate if duplicate found, None otherwise
        """
        if existing_facts is None and self.semantic_store:
            existing_facts = self.semantic_store.list_all(limit=1000)

        if not existing_facts:
            return None

        best_match = None
        best_similarity = 0.0

        for existing in existing_facts:
            if existing.id == new_fact.id:
                continue

            similarity, reasons = self._compare_facts(new_fact, existing)

            if similarity >= self.similarity_threshold and similarity > best_similarity:
                keep_id, keep_reason = self._recommend_fact_keep(new_fact, existing)

                best_match = DuplicateCandidate(
                    memory1_id=new_fact.id,
                    memory2_id=existing.id,
                    memory_type="fact",
                    similarity=similarity,
                    match_reasons=reasons,
                    keep_recommendation=keep_id,
                    recommendation_reason=keep_reason,
                )
                best_similarity = similarity

        return best_match

    def get_stats(
        self,
        episodes: Optional[List[Episode]] = None,
        facts: Optional[List[Fact]] = None,
    ) -> DuplicateStats:
        """
        Get duplicate detection statistics.

        Returns:
            DuplicateStats with counts and averages
        """
        episode_dups = self.find_episode_duplicates(episodes, limit=1000)
        fact_dups = self.find_fact_duplicates(facts, limit=1000)

        all_similarities = [d.similarity for d in episode_dups + fact_dups]

        if episodes is None and self.episodic_store:
            episodes = self.episodic_store.list_all(limit=1000)
        if facts is None and self.semantic_store:
            facts = self.semantic_store.list_all(limit=1000)

        return DuplicateStats(
            episodes_scanned=len(episodes) if episodes else 0,
            facts_scanned=len(facts) if facts else 0,
            episode_duplicates=len(episode_dups),
            fact_duplicates=len(fact_dups),
            highest_similarity=max(all_similarities) if all_similarities else 0.0,
            avg_similarity=sum(all_similarities) / len(all_similarities) if all_similarities else 0.0,
        )

    def _compare_episodes(
        self,
        ep1: Episode,
        ep2: Episode,
    ) -> Tuple[float, List[str]]:
        """
        Compare two episodes for similarity.

        Returns:
            (similarity_score, list_of_match_reasons)
        """
        scores = []
        reasons = []

        # Same source ID is a strong signal
        if ep1.source_id and ep2.source_id and ep1.source_id == ep2.source_id:
            scores.append(1.0)
            reasons.append("same_source_id")

        # Topic similarity
        if ep1.topic and ep2.topic:
            topic_sim = self._text_similarity(ep1.topic, ep2.topic)
            if topic_sim > 0.6:
                scores.append(topic_sim)
                reasons.append(f"topic_match ({topic_sim:.0%})")

        # Summary similarity
        if ep1.summary and ep2.summary:
            summary_sim = self._text_similarity(ep1.summary, ep2.summary)
            if summary_sim > 0.5:
                scores.append(summary_sim)
                reasons.append(f"summary_match ({summary_sim:.0%})")

        # Message overlap
        msg_sim = self._message_overlap(ep1.messages, ep2.messages)
        if msg_sim > 0.4:
            scores.append(msg_sim)
            reasons.append(f"message_overlap ({msg_sim:.0%})")

        # Temporal proximity
        if ep1.started_at and ep2.started_at:
            hours_diff = abs((ep1.started_at - ep2.started_at).total_seconds() / 3600)
            if hours_diff <= self.temporal_window_hours:
                temporal_score = 1.0 - (hours_diff / self.temporal_window_hours)
                if temporal_score > 0.5:
                    scores.append(temporal_score * 0.3)  # Lower weight for time
                    reasons.append(f"temporal_proximity ({hours_diff:.1f}h)")

        if not scores:
            return 0.0, []

        # Weighted average (emphasize higher scores)
        final_score = max(scores) * 0.6 + (sum(scores) / len(scores)) * 0.4
        return min(1.0, final_score), reasons

    def _compare_facts(
        self,
        fact1: Fact,
        fact2: Fact,
    ) -> Tuple[float, List[str]]:
        """
        Compare two facts for similarity.

        Returns:
            (similarity_score, list_of_match_reasons)
        """
        scores = []
        reasons = []

        # Content similarity is primary
        content_sim = self._text_similarity(fact1.content, fact2.content)
        if content_sim > 0.5:
            scores.append(content_sim)
            reasons.append(f"content_match ({content_sim:.0%})")

        # Same fact type is a bonus
        if fact1.fact_type and fact2.fact_type and fact1.fact_type == fact2.fact_type:
            scores.append(0.3)  # Small bonus
            reasons.append("same_fact_type")

        # Shared source episodes
        if fact1.source_episodes and fact2.source_episodes:
            shared = set(fact1.source_episodes) & set(fact2.source_episodes)
            if shared:
                source_ratio = len(shared) / max(len(fact1.source_episodes), len(fact2.source_episodes))
                scores.append(source_ratio * 0.5)
                reasons.append(f"shared_sources ({len(shared)})")

        if not scores:
            return 0.0, []

        # Weighted average
        final_score = max(scores) * 0.7 + (sum(scores) / len(scores)) * 0.3
        return min(1.0, final_score), reasons

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using TF-IDF cosine similarity."""
        return quick_similarity(text1, text2)

    def _message_overlap(
        self,
        msgs1: List[Dict[str, str]],
        msgs2: List[Dict[str, str]],
    ) -> float:
        """Calculate message overlap between two message lists."""
        if not msgs1 or not msgs2:
            return 0.0

        # Extract content from messages
        content1 = set()
        content2 = set()

        for m in msgs1:
            content = m.get("content", "")
            if content:
                # Hash first 100 chars for quick comparison
                content1.add(content[:100].lower())

        for m in msgs2:
            content = m.get("content", "")
            if content:
                content2.add(content[:100].lower())

        if not content1 or not content2:
            return 0.0

        # Jaccard similarity
        intersection = len(content1 & content2)
        union = len(content1 | content2)

        return intersection / union if union > 0 else 0.0

    def _recommend_episode_keep(
        self,
        ep1: Episode,
        ep2: Episode,
    ) -> Tuple[str, str]:
        """
        Recommend which episode to keep.

        Returns:
            (id_to_keep, reason)
        """
        score1 = 0
        score2 = 0
        reasons1 = []
        reasons2 = []

        # Higher importance
        if ep1.importance > ep2.importance:
            score1 += 2
            reasons1.append("higher importance")
        elif ep2.importance > ep1.importance:
            score2 += 2
            reasons2.append("higher importance")

        # More messages (more complete)
        if len(ep1.messages) > len(ep2.messages):
            score1 += 1
            reasons1.append("more messages")
        elif len(ep2.messages) > len(ep1.messages):
            score2 += 1
            reasons2.append("more messages")

        # Has summary
        if ep1.summary and not ep2.summary:
            score1 += 1
            reasons1.append("has summary")
        elif ep2.summary and not ep1.summary:
            score2 += 1
            reasons2.append("has summary")

        # More recent access
        access1 = ep1.access_count
        access2 = ep2.access_count
        if access1 > access2:
            score1 += 1
            reasons1.append("more accesses")
        elif access2 > access1:
            score2 += 1
            reasons2.append("more accesses")

        # Has topic
        if ep1.topic and not ep2.topic:
            score1 += 1
            reasons1.append("has topic")
        elif ep2.topic and not ep1.topic:
            score2 += 1
            reasons2.append("has topic")

        # Has tags
        if ep1.tags and not ep2.tags:
            score1 += 1
            reasons1.append("has tags")
        elif ep2.tags and not ep1.tags:
            score2 += 1
            reasons2.append("has tags")

        if score1 > score2:
            return ep1.id, ", ".join(reasons1)
        elif score2 > score1:
            return ep2.id, ", ".join(reasons2)
        else:
            # Default to first (or more recent)
            if ep1.started_at and ep2.started_at:
                if ep1.started_at > ep2.started_at:
                    return ep1.id, "more recent"
                else:
                    return ep2.id, "more recent"
            return ep1.id, "first encountered"

    def _recommend_fact_keep(
        self,
        fact1: Fact,
        fact2: Fact,
    ) -> Tuple[str, str]:
        """
        Recommend which fact to keep.

        Returns:
            (id_to_keep, reason)
        """
        score1 = 0
        score2 = 0
        reasons1 = []
        reasons2 = []

        # Higher confidence
        if fact1.confidence > fact2.confidence:
            score1 += 2
            reasons1.append("higher confidence")
        elif fact2.confidence > fact1.confidence:
            score2 += 2
            reasons2.append("higher confidence")

        # More confirmations
        if fact1.confirmation_count > fact2.confirmation_count:
            score1 += 2
            reasons1.append("more confirmations")
        elif fact2.confirmation_count > fact1.confirmation_count:
            score2 += 2
            reasons2.append("more confirmations")

        # Higher importance
        if fact1.importance > fact2.importance:
            score1 += 1
            reasons1.append("higher importance")
        elif fact2.importance > fact1.importance:
            score2 += 1
            reasons2.append("higher importance")

        # More source episodes
        if len(fact1.source_episodes) > len(fact2.source_episodes):
            score1 += 1
            reasons1.append("more sources")
        elif len(fact2.source_episodes) > len(fact1.source_episodes):
            score2 += 1
            reasons2.append("more sources")

        # Not superseded
        if not fact1.is_superseded and fact2.is_superseded:
            score1 += 3
            reasons1.append("not superseded")
        elif not fact2.is_superseded and fact1.is_superseded:
            score2 += 3
            reasons2.append("not superseded")

        # More recent confirmation
        if fact1.last_confirmed and fact2.last_confirmed:
            if fact1.last_confirmed > fact2.last_confirmed:
                score1 += 1
                reasons1.append("more recently confirmed")
            elif fact2.last_confirmed > fact1.last_confirmed:
                score2 += 1
                reasons2.append("more recently confirmed")

        # Has tags
        if fact1.tags and not fact2.tags:
            score1 += 1
            reasons1.append("has tags")
        elif fact2.tags and not fact1.tags:
            score2 += 1
            reasons2.append("has tags")

        if score1 > score2:
            return fact1.id, ", ".join(reasons1)
        elif score2 > score1:
            return fact2.id, ", ".join(reasons2)
        else:
            # Default to older (first learned)
            if fact1.first_learned and fact2.first_learned:
                if fact1.first_learned < fact2.first_learned:
                    return fact1.id, "older (original)"
                else:
                    return fact2.id, "older (original)"
            return fact1.id, "first encountered"

    def dedupe_episodes(
        self,
        episodes: Optional[List[Episode]] = None,
        dry_run: bool = True,
    ) -> Dict[str, Any]:
        """
        Deduplicate episodes by removing duplicates.

        Args:
            episodes: Episodes to dedupe (or load from store)
            dry_run: If True, don't actually delete

        Returns:
            Dict with results
        """
        duplicates = self.find_episode_duplicates(episodes)

        results = {
            "duplicates_found": len(duplicates),
            "removed": [],
            "kept": [],
            "dry_run": dry_run,
        }

        for dup in duplicates:
            keep_id = dup.keep_recommendation
            remove_id = dup.memory2_id if dup.memory1_id == keep_id else dup.memory1_id

            results["kept"].append(keep_id)
            results["removed"].append(remove_id)

            if not dry_run and self.episodic_store:
                self.episodic_store.delete(remove_id, hard=True)

        return results

    def dedupe_facts(
        self,
        facts: Optional[List[Fact]] = None,
        dry_run: bool = True,
    ) -> Dict[str, Any]:
        """
        Deduplicate facts by removing duplicates.

        Args:
            facts: Facts to dedupe (or load from store)
            dry_run: If True, don't actually delete

        Returns:
            Dict with results
        """
        duplicates = self.find_fact_duplicates(facts)

        results = {
            "duplicates_found": len(duplicates),
            "removed": [],
            "kept": [],
            "dry_run": dry_run,
        }

        for dup in duplicates:
            keep_id = dup.keep_recommendation
            remove_id = dup.memory2_id if dup.memory1_id == keep_id else dup.memory1_id

            results["kept"].append(keep_id)
            results["removed"].append(remove_id)

            if not dry_run and self.semantic_store:
                self.semantic_store.delete(remove_id, hard=True)

        return results
