"""
Vector storage for semantic search.

Supports ChromaDB for persistent vector storage with
efficient similarity search.
"""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class SearchResult:
    """Result from vector search."""
    id: str
    content: str
    score: float  # Similarity score (higher = more similar)
    metadata: Dict[str, Any]


class VectorStore(ABC):
    """Abstract base class for vector stores."""

    @abstractmethod
    def add(
        self,
        id: str,
        embedding: List[float],
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Add a vector to the store."""
        pass

    @abstractmethod
    def add_batch(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        contents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ):
        """Add multiple vectors efficiently."""
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: List[float],
        limit: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Search for similar vectors."""
        pass

    @abstractmethod
    def delete(self, id: str):
        """Delete a vector by ID."""
        pass

    @abstractmethod
    def get(self, id: str) -> Optional[SearchResult]:
        """Get a specific vector by ID."""
        pass

    @abstractmethod
    def count(self) -> int:
        """Return total number of vectors."""
        pass


class ChromaVectorStore(VectorStore):
    """
    Vector store using ChromaDB.

    ChromaDB provides:
    - Persistent storage
    - Efficient similarity search
    - Metadata filtering
    - Easy to use
    """

    def __init__(
        self,
        collection_name: str = "anamnesis_memories",
        persist_directory: Optional[str] = None,
    ):
        """
        Initialize ChromaDB vector store.

        Args:
            collection_name: Name of the collection
            persist_directory: Directory for persistence (None for in-memory)
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self._client = None
        self._collection = None

    def _init_db(self):
        """Initialize ChromaDB connection."""
        if self._client is not None:
            return

        try:
            import chromadb

            if self.persist_directory:
                os.makedirs(self.persist_directory, exist_ok=True)
                self._client = chromadb.PersistentClient(
                    path=self.persist_directory,
                )
            else:
                self._client = chromadb.Client()

            # Get or create collection
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},  # Use cosine similarity
            )

        except ImportError:
            raise RuntimeError(
                "chromadb required. Install with: pip install chromadb"
            )

    def add(
        self,
        id: str,
        embedding: List[float],
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Add a single vector."""
        self._init_db()

        # ChromaDB doesn't like None metadata
        meta = metadata or {}
        meta["content"] = content[:1000]  # Store content preview

        self._collection.upsert(
            ids=[id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[meta],
        )

    def add_batch(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        contents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ):
        """Add multiple vectors efficiently."""
        self._init_db()

        if not ids:
            return

        # Prepare metadata
        if metadatas is None:
            metadatas = [{}] * len(ids)

        # Add content preview to metadata
        for i, content in enumerate(contents):
            metadatas[i]["content"] = content[:1000]

        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=contents,
            metadatas=metadatas,
        )

    def search(
        self,
        query_embedding: List[float],
        limit: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Search for similar vectors."""
        self._init_db()

        # Build where clause for metadata filtering
        where = None
        if filter_metadata:
            where = filter_metadata

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        # Convert to SearchResult objects
        search_results = []

        if results["ids"] and results["ids"][0]:
            ids = results["ids"][0]
            documents = results["documents"][0] if results["documents"] else [""] * len(ids)
            metadatas = results["metadatas"][0] if results["metadatas"] else [{}] * len(ids)
            distances = results["distances"][0] if results["distances"] else [0] * len(ids)

            for i, id in enumerate(ids):
                # Convert distance to similarity score (1 - distance for cosine)
                score = 1 - distances[i] if distances[i] else 0

                search_results.append(SearchResult(
                    id=id,
                    content=documents[i] if i < len(documents) else "",
                    score=score,
                    metadata=metadatas[i] if i < len(metadatas) else {},
                ))

        return search_results

    def delete(self, id: str):
        """Delete a vector by ID."""
        self._init_db()
        self._collection.delete(ids=[id])

    def get(self, id: str) -> Optional[SearchResult]:
        """Get a specific vector by ID."""
        self._init_db()

        result = self._collection.get(
            ids=[id],
            include=["documents", "metadatas"],
        )

        if result["ids"]:
            return SearchResult(
                id=id,
                content=result["documents"][0] if result["documents"] else "",
                score=1.0,  # Perfect match
                metadata=result["metadatas"][0] if result["metadatas"] else {},
            )

        return None

    def count(self) -> int:
        """Return total number of vectors."""
        self._init_db()
        return self._collection.count()

    def clear(self):
        """Clear all vectors from the collection."""
        self._init_db()
        # Delete and recreate collection
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )


class InMemoryVectorStore(VectorStore):
    """
    Simple in-memory vector store using numpy.

    Useful for testing or when ChromaDB isn't available.
    """

    def __init__(self):
        self._vectors: Dict[str, Dict[str, Any]] = {}

    def add(
        self,
        id: str,
        embedding: List[float],
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Add a single vector."""
        import numpy as np

        self._vectors[id] = {
            "embedding": np.array(embedding),
            "content": content,
            "metadata": metadata or {},
        }

    def add_batch(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        contents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ):
        """Add multiple vectors."""
        if metadatas is None:
            metadatas = [{}] * len(ids)

        for id, emb, content, meta in zip(ids, embeddings, contents, metadatas):
            self.add(id, emb, content, meta)

    def search(
        self,
        query_embedding: List[float],
        limit: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Search using brute-force cosine similarity."""
        import numpy as np

        query = np.array(query_embedding)
        query_norm = np.linalg.norm(query)

        scores = []
        for id, data in self._vectors.items():
            # Apply metadata filter
            if filter_metadata:
                match = all(
                    data["metadata"].get(k) == v
                    for k, v in filter_metadata.items()
                )
                if not match:
                    continue

            # Calculate cosine similarity
            emb = data["embedding"]
            emb_norm = np.linalg.norm(emb)
            if query_norm > 0 and emb_norm > 0:
                sim = np.dot(query, emb) / (query_norm * emb_norm)
            else:
                sim = 0

            scores.append((id, data, sim))

        # Sort by similarity
        scores.sort(key=lambda x: x[2], reverse=True)

        # Return top results
        results = []
        for id, data, score in scores[:limit]:
            results.append(SearchResult(
                id=id,
                content=data["content"],
                score=float(score),
                metadata=data["metadata"],
            ))

        return results

    def delete(self, id: str):
        """Delete a vector."""
        if id in self._vectors:
            del self._vectors[id]

    def get(self, id: str) -> Optional[SearchResult]:
        """Get a vector by ID."""
        if id not in self._vectors:
            return None

        data = self._vectors[id]
        return SearchResult(
            id=id,
            content=data["content"],
            score=1.0,
            metadata=data["metadata"],
        )

    def count(self) -> int:
        """Return count."""
        return len(self._vectors)

    def clear(self):
        """Clear all vectors."""
        self._vectors.clear()
