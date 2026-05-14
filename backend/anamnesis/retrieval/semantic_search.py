"""
Semantic search service.

Combines vector similarity search with traditional FTS
for hybrid retrieval.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..models import Episode, Fact
from ..stores.episodic import EpisodicStore
from ..stores.semantic import SemanticStore
from .embeddings import get_embedder
from .vector_store import ChromaVectorStore, InMemoryVectorStore


@dataclass
class HybridSearchResult:
    """Result from hybrid search."""
    id: str
    content: str
    score: float  # Combined score
    fts_score: float
    vector_score: float
    source_type: str  # "episode" or "fact"


class SemanticSearchService:
    """
    Semantic search service with hybrid retrieval.

    Combines:
    - Full-text search (FTS) from SQLite
    - Vector similarity search from ChromaDB
    - Weighted combination for best results
    """

    def __init__(
        self,
        episodic_store: EpisodicStore,
        semantic_store: SemanticStore,
        vector_persist_path: Optional[str] = None,
        use_transformers: bool = True,
        fts_weight: float = 0.3,
        vector_weight: float = 0.7,
    ):
        """
        Initialize semantic search service.

        Args:
            episodic_store: Episodic memory store
            semantic_store: Semantic fact store
            vector_persist_path: Path for vector DB persistence
            use_transformers: Use sentence-transformers if available
            fts_weight: Weight for FTS results (0-1)
            vector_weight: Weight for vector results (0-1)
        """
        self.episodic_store = episodic_store
        self.semantic_store = semantic_store
        self.fts_weight = fts_weight
        self.vector_weight = vector_weight

        # Initialize embedder
        self.embedder = get_embedder(use_transformers=use_transformers)

        # Initialize vector stores
        if vector_persist_path:
            # Separate collections for episodes and facts
            base_path = Path(vector_persist_path)
            self.episode_vectors = ChromaVectorStore(
                collection_name="anamnesis_episodes",
                persist_directory=str(base_path / "episodes"),
            )
            self.fact_vectors = ChromaVectorStore(
                collection_name="anamnesis_facts",
                persist_directory=str(base_path / "facts"),
            )
        else:
            # In-memory for testing
            self.episode_vectors = InMemoryVectorStore()
            self.fact_vectors = InMemoryVectorStore()

        self._indexed = False

    def index_episode(self, episode: Episode):
        """Index a single episode for vector search."""
        # Create searchable content
        content = self._episode_to_content(episode)

        # Generate embedding
        embedding = self.embedder.embed(content)

        # Store in vector DB
        self.episode_vectors.add(
            id=episode.id,
            embedding=embedding,
            content=content,
            metadata={
                "topic": episode.topic or "",
                "importance": episode.importance,
                "valence": episode.overall_valence.value if hasattr(episode.overall_valence, 'value') else 0,
            }
        )

    def index_fact(self, fact: Fact):
        """Index a single fact for vector search."""
        # Generate embedding
        embedding = self.embedder.embed(fact.content)

        # Store in vector DB
        self.fact_vectors.add(
            id=fact.id,
            embedding=embedding,
            content=fact.content,
            metadata={
                "fact_type": fact.fact_type,
                "confidence": fact.confidence,
                "importance": fact.importance,
            }
        )

    def index_all(self, batch_size: int = 50):
        """Index all episodes and facts."""
        # Index episodes
        episodes = self.episodic_store.list_all(limit=10000)
        for i in range(0, len(episodes), batch_size):
            batch = episodes[i:i + batch_size]
            contents = [self._episode_to_content(ep) for ep in batch]
            embeddings = self.embedder.embed_batch(contents)

            ids = [ep.id for ep in batch]
            metadatas = [{
                "topic": ep.topic or "",
                "importance": ep.importance,
                "valence": ep.overall_valence.value if hasattr(ep.overall_valence, 'value') else 0,
            } for ep in batch]

            self.episode_vectors.add_batch(ids, embeddings, contents, metadatas)

        # Index facts
        facts = self.semantic_store.list_all(limit=10000)
        for i in range(0, len(facts), batch_size):
            batch = facts[i:i + batch_size]
            contents = [f.content for f in batch]
            embeddings = self.embedder.embed_batch(contents)

            ids = [f.id for f in batch]
            metadatas = [{
                "fact_type": f.fact_type,
                "confidence": f.confidence,
                "importance": f.importance,
            } for f in batch]

            self.fact_vectors.add_batch(ids, embeddings, contents, metadatas)

        self._indexed = True

    def search_episodes(
        self,
        query: str,
        limit: int = 10,
        use_hybrid: bool = True,
    ) -> List[Tuple[Episode, float]]:
        """
        Search episodes using hybrid retrieval.

        Args:
            query: Search query
            limit: Max results
            use_hybrid: Combine FTS and vector search

        Returns:
            List of (Episode, score) tuples
        """
        results: Dict[str, float] = {}

        # FTS search
        fts_results = self.episodic_store.search(query, limit=limit * 2)
        for i, ep in enumerate(fts_results):
            # Score based on position (higher position = higher score)
            fts_score = 1.0 - (i / len(fts_results)) if fts_results else 0
            results[ep.id] = fts_score * self.fts_weight

        # Vector search
        if use_hybrid and self.episode_vectors.count() > 0:
            query_embedding = self.embedder.embed(query)
            vector_results = self.episode_vectors.search(query_embedding, limit=limit * 2)

            for vr in vector_results:
                if vr.id in results:
                    results[vr.id] += vr.score * self.vector_weight
                else:
                    results[vr.id] = vr.score * self.vector_weight

        # Sort by combined score
        sorted_ids = sorted(results.keys(), key=lambda x: results[x], reverse=True)

        # Fetch full episodes
        output = []
        for ep_id in sorted_ids[:limit]:
            episode = self.episodic_store.get(ep_id)
            if episode:
                output.append((episode, results[ep_id]))

        return output

    def search_facts(
        self,
        query: str,
        limit: int = 20,
        use_hybrid: bool = True,
    ) -> List[Tuple[Fact, float]]:
        """
        Search facts using hybrid retrieval.

        Args:
            query: Search query
            limit: Max results
            use_hybrid: Combine FTS and vector search

        Returns:
            List of (Fact, score) tuples
        """
        results: Dict[str, float] = {}

        # FTS search
        fts_results = self.semantic_store.search(query, limit=limit * 2)
        for i, fact in enumerate(fts_results):
            fts_score = 1.0 - (i / len(fts_results)) if fts_results else 0
            results[fact.id] = fts_score * self.fts_weight

        # Vector search
        if use_hybrid and self.fact_vectors.count() > 0:
            query_embedding = self.embedder.embed(query)
            vector_results = self.fact_vectors.search(query_embedding, limit=limit * 2)

            for vr in vector_results:
                if vr.id in results:
                    results[vr.id] += vr.score * self.vector_weight
                else:
                    results[vr.id] = vr.score * self.vector_weight

        # Sort and fetch
        sorted_ids = sorted(results.keys(), key=lambda x: results[x], reverse=True)

        output = []
        for fact_id in sorted_ids[:limit]:
            fact = self.semantic_store.get(fact_id)
            if fact:
                output.append((fact, results[fact_id]))

        return output

    def hybrid_search(
        self,
        query: str,
        max_episodes: int = 5,
        max_facts: int = 10,
    ) -> Dict[str, Any]:
        """
        Perform hybrid search across both episodes and facts.

        Returns:
            Dict with 'episodes' and 'facts' lists
        """
        return {
            "episodes": self.search_episodes(query, limit=max_episodes),
            "facts": self.search_facts(query, limit=max_facts),
        }

    def _episode_to_content(self, episode: Episode) -> str:
        """Convert episode to searchable content."""
        parts = []

        if episode.topic:
            parts.append(f"Topic: {episode.topic}")

        if episode.summary:
            parts.append(f"Summary: {episode.summary}")

        # Include a sample of messages
        for msg in episode.messages[:5]:
            content = msg.get("content", "")[:200]
            if content:
                parts.append(content)

        return " ".join(parts)

    def get_stats(self) -> Dict[str, Any]:
        """Get indexing statistics."""
        return {
            "episodes_indexed": self.episode_vectors.count(),
            "facts_indexed": self.fact_vectors.count(),
            "embedder": self.embedder.model_name,
            "embedding_dimension": self.embedder.dimension,
            "indexed": self._indexed,
        }
