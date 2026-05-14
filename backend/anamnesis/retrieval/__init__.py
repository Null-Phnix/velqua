"""
Retrieval enhancement modules.
"""

from .embeddings import (
    EmbeddingGenerator,
    SentenceTransformerEmbedder,
    SimpleTFIDFEmbedder,
    get_embedder,
)
from .query_expansion import (
    ExpansionResult,
    QueryExpander,
    expand_query,
)
from .synonyms import (
    SYNONYM_MAP,
    expand_terms,
    get_synonyms,
)
from .semantic_search import (
    HybridSearchResult,
    SemanticSearchService,
)
from .vector_store import (
    ChromaVectorStore,
    InMemoryVectorStore,
    SearchResult,
    VectorStore,
)

__all__ = [
    # Query expansion
    "QueryExpander",
    "ExpansionResult",
    "expand_query",
    # Vector store
    "VectorStore",
    "ChromaVectorStore",
    "InMemoryVectorStore",
    "SearchResult",
    # Embeddings
    "EmbeddingGenerator",
    "SentenceTransformerEmbedder",
    "SimpleTFIDFEmbedder",
    "get_embedder",
    # Synonyms
    "SYNONYM_MAP",
    "get_synonyms",
    "expand_terms",
    # Semantic search
    "SemanticSearchService",
    "HybridSearchResult",
]
