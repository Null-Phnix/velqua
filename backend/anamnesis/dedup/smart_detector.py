"""
Smart duplicate detection using embedding-based cosine similarity.

Upgrades the TF-IDF dedup to use sentence-transformer embeddings for
semantic similarity, with intelligent merge logic that keeps the higher
quality version and combines metadata.

Falls back to TF-IDF when no embedder is available.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..models import Fact
from .similarity import quick_similarity

logger = logging.getLogger(__name__)

# Thresholds
EMBEDDING_DUPLICATE_THRESHOLD = 0.92  # Cosine similarity for embedding-based dedup
TFIDF_DUPLICATE_THRESHOLD = 0.75      # Fallback TF-IDF threshold


@dataclass
class DuplicateMatch:
    """Result of a duplicate check."""
    is_duplicate: bool
    existing_fact: Optional[Fact] = None
    similarity: float = 0.0
    method: str = "none"  # "embedding" or "tfidf"


def _fact_quality_score(fact: Fact) -> float:
    """
    Score a fact's overall quality for keeper selection.

    Higher score = better candidate to keep.

    Factors:
    - Content length (longer, more specific facts are better)
    - Confidence
    - Confirmation count (community validation)
    - Metadata richness (topic, emotion, tags)
    - Importance
    - Not superseded
    """
    score = 0.0

    # Content quality: prefer longer, more specific facts (up to a point)
    content_len = len(fact.content)
    if 30 <= content_len <= 200:
        score += 2.0  # Sweet spot
    elif content_len > 200:
        score += 1.0  # Still informative
    else:
        score += 0.5  # Too short

    # Confidence is a direct quality signal
    score += fact.confidence * 3.0

    # Confirmation count — logarithmic (diminishing returns)
    import math
    score += math.log1p(fact.confirmation_count) * 2.0

    # Metadata richness
    meta = fact.metadata or {}
    if meta.get("topic"):
        score += 0.5
    if meta.get("category"):
        score += 0.3
    if meta.get("emotion"):
        score += 0.2
    if meta.get("sentiment_score") is not None:
        score += 0.2

    # Tags
    if fact.tags:
        score += min(1.0, len(fact.tags) * 0.2)

    # Importance
    score += fact.importance * 1.5

    # Superseded facts are penalized heavily
    if fact.is_superseded:
        score -= 5.0

    return score


def _merge_metadata(keeper: Fact, duplicate: Fact) -> Dict[str, Any]:
    """
    Merge metadata from duplicate into keeper, preserving richer data.

    Strategy:
    - Keep keeper's values by default
    - Fill in missing fields from duplicate
    - Merge tags (union, deduplicated)
    - Merge source episodes (union)
    """
    merged = dict(keeper.metadata or {})
    dup_meta = duplicate.metadata or {}

    # Fill missing fields from duplicate
    for key in ("topic", "category", "emotion", "sentiment_score", "source"):
        if not merged.get(key) and dup_meta.get(key):
            merged[key] = dup_meta[key]

    # Track merge history
    merge_history = merged.get("merged_from", [])
    merge_history.append(duplicate.id)
    merged["merged_from"] = merge_history

    return merged


def _merge_facts(keeper: Fact, duplicate: Fact) -> Fact:
    """
    Merge a duplicate fact into the keeper.

    - Keeps the higher-quality version's content
    - Combines metadata
    - Merges source episodes and tags
    - Increments confirmation count
    - Boosts confidence
    """
    # Combine source episodes (union, no duplicates)
    all_episodes = list(keeper.source_episodes or [])
    for ep in (duplicate.source_episodes or []):
        if ep and ep not in all_episodes:
            all_episodes.append(ep)
    keeper.source_episodes = all_episodes

    # Merge tags (union)
    all_tags = list(keeper.tags or [])
    for tag in (duplicate.tags or []):
        if tag and tag not in all_tags:
            all_tags.append(tag)
    keeper.tags = all_tags

    # Merge metadata
    keeper.metadata = _merge_metadata(keeper, duplicate)

    # Confirm: bump count and confidence
    keeper.confirm()

    # Take the higher importance
    keeper.importance = max(keeper.importance, duplicate.importance)

    return keeper


class SmartDuplicateDetector:
    """
    Embedding-based duplicate detector with TF-IDF fallback.

    Uses sentence-transformer cosine similarity (threshold 0.92) for
    precise semantic matching. Falls back to TF-IDF (threshold 0.75)
    when no embedder is available.
    """

    def __init__(
        self,
        embedder=None,
        embedding_threshold: float = EMBEDDING_DUPLICATE_THRESHOLD,
        tfidf_threshold: float = TFIDF_DUPLICATE_THRESHOLD,
    ):
        """
        Args:
            embedder: Optional Embedder instance (from retrieval.embedder).
                      When None, falls back to TF-IDF.
            embedding_threshold: Cosine similarity threshold for embeddings.
            tfidf_threshold: Cosine similarity threshold for TF-IDF fallback.
        """
        self.embedder = embedder
        self.embedding_threshold = embedding_threshold
        self.tfidf_threshold = tfidf_threshold

    def check_duplicate(
        self,
        new_content: str,
        candidates: List[Fact],
    ) -> DuplicateMatch:
        """
        Check if new_content is a duplicate of any candidate fact.

        Tries embedding similarity first, falls back to TF-IDF.

        Args:
            new_content: The new fact content to check.
            candidates: Existing facts to compare against.

        Returns:
            DuplicateMatch with the best matching fact, or is_duplicate=False.
        """
        if not candidates or not new_content:
            return DuplicateMatch(is_duplicate=False)

        # Try embedding-based detection first
        if self.embedder is not None:
            match = self._check_embedding(new_content, candidates)
            if match.is_duplicate:
                return match

        # Fall back to TF-IDF
        return self._check_tfidf(new_content, candidates)

    def find_and_merge(
        self,
        new_content: str,
        new_fact: Fact,
        candidates: List[Fact],
    ) -> Optional[Fact]:
        """
        Check for duplicates and merge if found.

        If a duplicate is found:
        - Determines which version (new or existing) is higher quality
        - Merges metadata from the lower-quality version into the keeper
        - Returns the merged keeper fact

        If no duplicate:
        - Returns None (caller should save the new fact)

        Args:
            new_content: Content of the new fact.
            new_fact: The new Fact object (not yet saved).
            candidates: Existing facts to compare against.

        Returns:
            Merged Fact if duplicate found, None otherwise.
        """
        match = self.check_duplicate(new_content, candidates)

        if not match.is_duplicate or match.existing_fact is None:
            return None

        existing = match.existing_fact

        # Determine keeper: compare quality scores
        existing_score = _fact_quality_score(existing)
        new_score = _fact_quality_score(new_fact)

        if new_score > existing_score:
            # New fact is higher quality — it becomes the keeper
            # but we merge existing's history into it
            keeper = new_fact
            duplicate = existing
        else:
            # Existing fact is higher quality (or equal) — keep it
            keeper = existing
            duplicate = new_fact

        merged = _merge_facts(keeper, duplicate)

        logger.info(
            "Merged duplicate (%.3f %s): kept '%s' [%s], merged from '%s' [%s]",
            match.similarity,
            match.method,
            keeper.content[:50],
            keeper.id[:8],
            duplicate.content[:50],
            duplicate.id[:8],
        )

        return merged

    def _check_embedding(
        self,
        new_content: str,
        candidates: List[Fact],
    ) -> DuplicateMatch:
        """Check duplicates using embedding cosine similarity."""
        try:
            new_embedding = self.embedder.embed(new_content)
        except Exception as e:
            logger.warning("Embedding failed, will fall back to TF-IDF: %s", e)
            return DuplicateMatch(is_duplicate=False)

        best_sim = 0.0
        best_fact = None

        for fact in candidates:
            if fact.is_superseded:
                continue
            try:
                fact_embedding = self.embedder.embed(fact.content)
            except Exception:
                continue

            # Cosine similarity (embeddings are pre-normalized by sentence-transformers)
            import numpy as np
            sim = float(np.dot(new_embedding, fact_embedding))

            if sim > best_sim:
                best_sim = sim
                best_fact = fact

        if best_sim >= self.embedding_threshold and best_fact is not None:
            return DuplicateMatch(
                is_duplicate=True,
                existing_fact=best_fact,
                similarity=best_sim,
                method="embedding",
            )

        return DuplicateMatch(is_duplicate=False, similarity=best_sim, method="embedding")

    def _check_tfidf(
        self,
        new_content: str,
        candidates: List[Fact],
    ) -> DuplicateMatch:
        """Check duplicates using TF-IDF cosine similarity (fallback)."""
        best_sim = 0.0
        best_fact = None

        for fact in candidates:
            if fact.is_superseded:
                continue
            sim = quick_similarity(new_content, fact.content)
            if sim > best_sim:
                best_sim = sim
                best_fact = fact

        if best_sim >= self.tfidf_threshold and best_fact is not None:
            return DuplicateMatch(
                is_duplicate=True,
                existing_fact=best_fact,
                similarity=best_sim,
                method="tfidf",
            )

        return DuplicateMatch(is_duplicate=False, similarity=best_sim, method="tfidf")
