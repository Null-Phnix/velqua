"""
Hybrid retrieval combining text search and vector similarity.

Provides the best of both worlds:
- Text search: Exact keyword matching, good for specific terms
- Vector search: Semantic similarity, good for concepts
"""

import logging
import math
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

from ..models import Episode, Fact
from ..stores.sqlite_backend import SQLiteBackend
from .embedder import Embedder, get_default_embedder
from .synonyms import expand_terms
from .vector_store import ChromaVectorStore, VectorStore

logger = logging.getLogger(__name__)


class SearchMode(Enum):
    """Search mode for hybrid retrieval."""
    TEXT_ONLY = "text"
    VECTOR_ONLY = "vector"
    HYBRID = "hybrid"


@dataclass
class HybridSearchResult:
    """Result from hybrid search."""
    id: str
    content: str
    text_score: float
    vector_score: float
    combined_score: float
    source_type: str  # "episode" or "fact"
    metadata: Dict[str, Any]


class HybridRetriever:
    """
    Combines text (FTS) and vector search for optimal retrieval.

    Strategy:
    1. Run both searches in parallel (conceptually)
    2. Normalize scores to [0, 1] range
    3. Combine with configurable weights
    4. Re-rank by combined score
    """

    def __init__(
        self,
        sqlite_backend: SQLiteBackend,
        vector_store: Optional[VectorStore] = None,
        embedder: Optional[Embedder] = None,
        text_weight: float = 0.2,
        vector_weight: float = 0.8,
        decay_lambda: float = 0.01,
        decay_floor: float = 0.1,
        episode_decay_lambda: float = 0.099,
        episode_decay_floor: float = 0.02,
        episode_emotional_boost: float = 1.5,
        mmr_lambda: float = 0.5,
        mmr_diversity_threshold: float = 0.85,
    ):
        """
        Initialize hybrid retriever.

        Args:
            sqlite_backend: SQLite backend for text search
            vector_store: Vector store for similarity search
            embedder: Embedder for creating query embeddings
            text_weight: Weight for text search scores (default 0.2)
            vector_weight: Weight for vector search scores (default 0.8)
            decay_lambda: Exponential decay rate for facts (default 0.01, ~70-day half-life)
            decay_floor: Minimum decay multiplier for facts (never fully forget)
            episode_decay_lambda: Decay rate for episodes (default 0.099, ~7-day half-life)
            episode_decay_floor: Minimum decay for episodes (episodes can nearly vanish)
            episode_emotional_boost: How much emotional valence slows episode decay
            mmr_lambda: MMR trade-off (1.0 = pure relevance, 0.0 = pure diversity)
            mmr_diversity_threshold: Cosine similarity above which an extra penalty is applied
        """
        self.sqlite = sqlite_backend
        self.vector_store = vector_store
        self.embedder = embedder
        self.text_weight = text_weight
        self.vector_weight = vector_weight
        self.decay_lambda = decay_lambda
        self.decay_floor = decay_floor
        self.episode_decay_lambda = episode_decay_lambda
        self.episode_decay_floor = episode_decay_floor
        self.episode_emotional_boost = episode_emotional_boost
        self.mmr_lambda = mmr_lambda
        self.mmr_diversity_threshold = mmr_diversity_threshold

        # Lazy initialization
        self._embedder_initialized = False
        self._vector_store_initialized = False

    def _ensure_embedder(self):
        """Ensure embedder is initialized."""
        if self.embedder is None and not self._embedder_initialized:
            try:
                self.embedder = get_default_embedder()
                self._embedder_initialized = True
            except (ImportError, OSError, RuntimeError) as e:
                logger.warning("Could not initialize embedder: %s", e)
                self._embedder_initialized = True  # Mark as tried

    def _ensure_vector_store(self, persist_path: Optional[str] = None):
        """Ensure vector store is initialized."""
        if self.vector_store is None and not self._vector_store_initialized:
            try:
                self.vector_store = ChromaVectorStore(
                    collection_name="anamnesis_memories",
                    persist_directory=persist_path,
                )
                self._vector_store_initialized = True
            except (ImportError, OSError, RuntimeError) as e:
                logger.warning("Could not initialize vector store: %s", e)
                self._vector_store_initialized = True

    def _compute_decay(self, metadata: Dict[str, Any], source_type: str) -> float:
        """
        Compute adaptive decay multiplier for a memory.

        Models the forgetting curve: multiplier = exp(-lambda * days_since_last_access)

        Facts: confirmation count reduces effective lambda (confirmed facts resist decay).
        Episodes: decay faster by default; emotional valence and access count slow decay.
                  Metadata ``decay_rate`` overrides the base episode lambda.

        Result is clamped to [floor, 1.0].
        """
        now = datetime.now()

        # Determine the most recent temporal anchor
        last_access_str = metadata.get("last_accessed") or metadata.get("last_confirmed")
        if source_type == "episode" and not last_access_str:
            last_access_str = metadata.get("started_at")
        if source_type == "fact" and not last_access_str:
            last_access_str = metadata.get("first_learned")

        if not last_access_str:
            return 1.0  # No temporal info — no decay applied

        try:
            # Handle both ISO formats (with and without fractional seconds)
            last_access = datetime.fromisoformat(last_access_str)
        except (ValueError, TypeError):
            return 1.0

        days_elapsed = max(0.0, (now - last_access).total_seconds() / 86400)

        if source_type == "episode":
            # Episode-specific decay: faster base rate, emotional modulation
            base_lambda = metadata.get("decay_rate", self.episode_decay_lambda)
            floor = self.episode_decay_floor

            # Emotional valence slows decay (absolute intensity matters)
            valence = metadata.get("overall_valence", 0) or 0
            effective_lambda = base_lambda / (
                1.0 + self.episode_emotional_boost * abs(valence)
            )

            # Access count slows decay (analogous to confirmation for facts)
            access_count = metadata.get("access_count", 0) or 0
            effective_lambda = effective_lambda / (
                1.0 + 0.5 * math.log1p(access_count)
            )
        else:
            # Fact decay: confirmation count reduces effective lambda
            floor = self.decay_floor
            confirmation_count = metadata.get("confirmation_count", 0) or 0
            effective_lambda = self.decay_lambda / (
                1.0 + 0.5 * math.log1p(confirmation_count)
            )

        decay_multiplier = math.exp(-effective_lambda * days_elapsed)
        return max(floor, decay_multiplier)

    @staticmethod
    def _cosine_similarity_matrix(embeddings: List[List[float]]) -> np.ndarray:
        """Compute pairwise cosine similarity matrix."""
        matrix = np.array(embeddings, dtype=np.float64)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        normalized = matrix / norms
        return normalized @ normalized.T

    def _mmr_rerank(
        self,
        results: List[HybridSearchResult],
        embeddings: List[List[float]],
        limit: int,
    ) -> List[HybridSearchResult]:
        """
        Maximal Marginal Relevance reranking for result diversity.

        Greedy selection: at each step, pick the candidate that maximizes
        MMR(d) = λ * relevance(d) - (1-λ) * max_sim(d, selected).

        An additional penalty is applied when a candidate's cosine similarity
        to any already-selected result exceeds ``mmr_diversity_threshold``.
        """
        if len(results) <= 1:
            return results[:limit]

        sim_matrix = self._cosine_similarity_matrix(embeddings)

        # Normalize relevance scores to [0, 1] for balanced MMR
        scores = np.array([r.combined_score for r in results])
        max_score = scores.max()
        norm_scores = scores / max_score if max_score > 0 else scores

        selected: List[int] = []
        remaining = set(range(len(results)))

        # Seed with highest-relevance result
        first = int(np.argmax(norm_scores))
        selected.append(first)
        remaining.discard(first)

        while len(selected) < limit and remaining:
            best_mmr = -float("inf")
            best_idx = -1

            for idx in remaining:
                max_sim = max(float(sim_matrix[idx][s]) for s in selected)

                mmr = (
                    self.mmr_lambda * float(norm_scores[idx])
                    - (1 - self.mmr_lambda) * max_sim
                )

                # Extra penalty for near-duplicates above threshold
                if max_sim > self.mmr_diversity_threshold:
                    mmr -= (max_sim - self.mmr_diversity_threshold)

                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = idx

            if best_idx >= 0:
                selected.append(best_idx)
                remaining.discard(best_idx)
            else:
                break

        return [results[i] for i in selected]

    def search(
        self,
        query: str,
        limit: int = 10,
        mode: SearchMode = SearchMode.HYBRID,
        search_episodes: bool = True,
        search_facts: bool = True,
    ) -> List[HybridSearchResult]:
        """
        Search for relevant memories.

        Args:
            query: Search query
            limit: Maximum results to return
            mode: Search mode (text, vector, or hybrid)
            search_episodes: Include episodes in search
            search_facts: Include facts in search

        Returns:
            List of HybridSearchResult sorted by relevance
        """
        results = []

        # Expand query with synonyms for FTS search
        expanded_words = expand_terms(query.split())
        fts_query = " ".join(expanded_words)

        # Text search
        text_results = {}
        if mode in (SearchMode.TEXT_ONLY, SearchMode.HYBRID):
            if search_episodes:
                episodes = self.sqlite.search_episodes(fts_query, limit * 2)
                for i, ep in enumerate(episodes):
                    score = 1.0 - (i / (len(episodes) + 1))  # Rank-based score
                    text_results[f"episode:{ep['id']}"] = {
                        "id": ep["id"],
                        "content": ep.get("summary", ""),
                        "score": score,
                        "type": "episode",
                        "metadata": ep,
                    }

            if search_facts:
                facts = self.sqlite.search_facts(fts_query, limit * 2)
                for i, fact in enumerate(facts):
                    score = 1.0 - (i / (len(facts) + 1))
                    text_results[f"fact:{fact['id']}"] = {
                        "id": fact["id"],
                        "content": fact.get("content", ""),
                        "score": score,
                        "type": "fact",
                        "metadata": fact,
                    }

        # Vector search
        vector_results = {}
        if mode in (SearchMode.VECTOR_ONLY, SearchMode.HYBRID):
            self._ensure_embedder()
            self._ensure_vector_store()

            if self.embedder and self.vector_store:
                query_embedding = self.embedder.embed(query)
                vector_hits = self.vector_store.search(
                    query_embedding,
                    limit=limit * 2,
                )

                for hit in vector_hits:
                    key = f"{hit.metadata.get('type', 'unknown')}:{hit.id}"
                    vector_results[key] = {
                        "id": hit.id,
                        "content": hit.content,
                        "score": hit.score,
                        "type": hit.metadata.get("type", "unknown"),
                        "metadata": hit.metadata,
                    }

        # Combine results
        all_keys = set(text_results.keys()) | set(vector_results.keys())

        for key in all_keys:
            text_hit = text_results.get(key, {})
            vector_hit = vector_results.get(key, {})

            text_score = text_hit.get("score", 0)
            vector_score = vector_hit.get("score", 0)

            # Calculate combined score
            if mode == SearchMode.TEXT_ONLY:
                combined = text_score
            elif mode == SearchMode.VECTOR_ONLY:
                combined = vector_score
            else:
                combined = (
                    self.text_weight * text_score +
                    self.vector_weight * vector_score
                )

            # Get content and metadata from whichever has it
            content = text_hit.get("content") or vector_hit.get("content", "")
            metadata = text_hit.get("metadata") or vector_hit.get("metadata", {})
            source_type = text_hit.get("type") or vector_hit.get("type", "unknown")
            item_id = text_hit.get("id") or vector_hit.get("id", "")

            # Apply adaptive temporal decay
            decay = self._compute_decay(metadata, source_type)
            combined *= decay

            results.append(HybridSearchResult(
                id=item_id,
                content=content,
                text_score=text_score,
                vector_score=vector_score,
                combined_score=combined,
                source_type=source_type,
                metadata=metadata,
            ))

        # Sort by combined score
        results.sort(key=lambda x: x.combined_score, reverse=True)

        # MMR diversification (requires embedder, skipped when λ ≥ 1.0)
        if len(results) > 1 and self.mmr_lambda < 1.0:
            self._ensure_embedder()
            if self.embedder:
                pool_size = min(len(results), limit * 3)
                candidates = results[:pool_size]
                try:
                    contents = [r.content or "" for r in candidates]
                    embeddings = self.embedder.embed_batch(contents)
                    return self._mmr_rerank(candidates, embeddings, limit)
                except Exception as e:
                    logger.warning(
                        "MMR reranking failed, falling back to score order: %s", e
                    )

        return results[:limit]

    def index_episode(self, episode: Episode):
        """Index an episode for vector search."""
        self._ensure_embedder()
        self._ensure_vector_store()

        if not self.embedder or not self.vector_store:
            return

        # Create embedding from summary
        text = episode.summary or episode.topic or ""
        if not text:
            return

        embedding = self.embedder.embed(text)

        self.vector_store.add(
            id=episode.id,
            embedding=embedding,
            content=text,
            metadata={
                "type": "episode",
                "topic": episode.topic,
                "importance": episode.importance,
            },
        )

    def index_fact(self, fact: Fact):
        """Index a fact for vector search."""
        self._ensure_embedder()
        self._ensure_vector_store()

        if not self.embedder or not self.vector_store:
            return

        text = fact.content
        if not text:
            return

        embedding = self.embedder.embed(text)

        self.vector_store.add(
            id=fact.id,
            embedding=embedding,
            content=text,
            metadata={
                "type": "fact",
                "fact_type": fact.fact_type,
                "confidence": fact.confidence,
                "confirmation_count": fact.confirmation_count,
            },
        )

    def index_all(
        self,
        episodes: List[Episode],
        facts: List[Fact],
        batch_size: int = 100,
        progress_callback: Optional[callable] = None,
    ):
        """
        Index all memories for vector search.

        Args:
            episodes: List of episodes to index
            facts: List of facts to index
            batch_size: Batch size for embedding
            progress_callback: Optional callback(current, total)
        """
        self._ensure_embedder()
        self._ensure_vector_store()

        if not self.embedder or not self.vector_store:
            return

        total = len(episodes) + len(facts)
        current = 0

        # Index episodes in batches
        for i in range(0, len(episodes), batch_size):
            batch = episodes[i:i + batch_size]

            ids = [ep.id for ep in batch]
            texts = [ep.summary or ep.topic or "" for ep in batch]
            metadatas = [
                {"type": "episode", "topic": ep.topic, "importance": ep.importance}
                for ep in batch
            ]

            # Filter out empty texts
            valid = [(id, t, m) for id, t, m in zip(ids, texts, metadatas) if t]
            if valid:
                ids, texts, metadatas = zip(*valid)
                embeddings = self.embedder.embed_batch(list(texts))
                self.vector_store.add_batch(
                    list(ids),
                    embeddings,
                    list(texts),
                    list(metadatas),
                )

            current += len(batch)
            if progress_callback:
                progress_callback(current, total)

        # Index facts in batches
        for i in range(0, len(facts), batch_size):
            batch = facts[i:i + batch_size]

            ids = [f.id for f in batch]
            texts = [f.content for f in batch]
            metadatas = [
                {
                    "type": "fact",
                    "fact_type": f.fact_type,
                    "confidence": f.confidence,
                    "confirmation_count": f.confirmation_count,
                }
                for f in batch
            ]

            # Filter out empty texts
            valid = [(id, t, m) for id, t, m in zip(ids, texts, metadatas) if t]
            if valid:
                ids, texts, metadatas = zip(*valid)
                embeddings = self.embedder.embed_batch(list(texts))
                self.vector_store.add_batch(
                    list(ids),
                    embeddings,
                    list(texts),
                    list(metadatas),
                )

            current += len(batch)
            if progress_callback:
                progress_callback(current, total)

    def get_stats(self) -> Dict[str, Any]:
        """Get retriever statistics."""
        stats = {
            "text_search": "available",
            "vector_search": "unavailable",
            "vector_count": 0,
        }

        self._ensure_vector_store()
        if self.vector_store:
            stats["vector_search"] = "available"
            stats["vector_count"] = self.vector_store.count()

        return stats
