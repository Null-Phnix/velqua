"""
Embedding generation for semantic search.

Uses sentence-transformers for efficient, high-quality embeddings
that work well on consumer hardware.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np


class Embedder(ABC):
    """Abstract base class for embedders."""

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """Embed a single text string."""
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts efficiently."""
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """Return the embedding dimension."""
        pass


class SentenceTransformerEmbedder(Embedder):
    """
    Embedder using sentence-transformers models.

    Default model: all-MiniLM-L6-v2
    - 384 dimensions
    - Fast and lightweight
    - Good for semantic similarity
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        normalize: bool = True,
    ):
        """
        Initialize embedder.

        Args:
            model_name: HuggingFace model name or path
            device: Device to use (None for auto-detect)
            normalize: Whether to normalize embeddings
        """
        self.model_name = model_name
        self.normalize = normalize
        self._model = None
        self._device = device

    def _load_model(self):
        """Lazy load the model."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.model_name,
                device=self._device,
            )
        except ImportError:
            raise RuntimeError(
                "sentence-transformers required. Install with: "
                "pip install sentence-transformers"
            )

    def embed(self, text: str) -> List[float]:
        """Embed a single text."""
        self._load_model()

        embedding = self._model.encode(
            text,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )

        return embedding.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts efficiently."""
        self._load_model()

        if not texts:
            return []

        embeddings = self._model.encode(
            texts,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 10,
        )

        return embeddings.tolist()

    def get_dimension(self) -> int:
        """Return embedding dimension."""
        self._load_model()
        return self._model.get_sentence_embedding_dimension()

    def similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts."""
        emb1 = np.array(self.embed(text1))
        emb2 = np.array(self.embed(text2))

        if self.normalize:
            # For normalized vectors, dot product = cosine similarity
            return float(np.dot(emb1, emb2))
        else:
            # Compute cosine similarity
            return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))

    def unload(self):
        """Unload model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None

            import gc
            gc.collect()

            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass


class CachedEmbedder(Embedder):
    """
    Wrapper that caches embeddings to avoid re-computation.
    """

    def __init__(
        self,
        base_embedder: Embedder,
        cache_size: int = 10000,
    ):
        self.base = base_embedder
        self.cache_size = cache_size
        self._cache: dict = {}

    def embed(self, text: str) -> List[float]:
        """Embed with caching."""
        if text in self._cache:
            return self._cache[text]

        embedding = self.base.embed(text)

        # Add to cache, evict oldest if full
        if len(self._cache) >= self.cache_size:
            # Remove oldest entry (simple FIFO)
            oldest = next(iter(self._cache))
            del self._cache[oldest]

        self._cache[text] = embedding
        return embedding

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed batch with caching."""
        results = []
        uncached = []
        uncached_indices = []

        # Check cache first
        for i, text in enumerate(texts):
            if text in self._cache:
                results.append(self._cache[text])
            else:
                results.append(None)
                uncached.append(text)
                uncached_indices.append(i)

        # Embed uncached
        if uncached:
            new_embeddings = self.base.embed_batch(uncached)
            for idx, emb, text in zip(uncached_indices, new_embeddings, uncached):
                results[idx] = emb
                # Add to cache
                if len(self._cache) < self.cache_size:
                    self._cache[text] = emb

        return results

    def get_dimension(self) -> int:
        return self.base.get_dimension()

    def clear_cache(self):
        """Clear the embedding cache."""
        self._cache.clear()


# Convenience function
def get_default_embedder() -> Embedder:
    """Get the default embedder (cached MiniLM)."""
    return CachedEmbedder(
        SentenceTransformerEmbedder(),
        cache_size=5000,
    )
