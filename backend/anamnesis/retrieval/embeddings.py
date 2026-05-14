"""
Embedding generation for semantic search.

Supports:
- Sentence transformers (best quality, requires torch)
- Simple TF-IDF-like embeddings (no external deps)
"""

import math
import re
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class EmbeddingResult:
    """Result of embedding generation."""
    embedding: List[float]
    model: str
    dimension: int


class EmbeddingGenerator(ABC):
    """Abstract base for embedding generators."""

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """Generate embedding for text."""
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return embedding dimension."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return model name."""
        pass


class SentenceTransformerEmbedder(EmbeddingGenerator):
    """
    Embedding generator using sentence-transformers.

    Requires: pip install sentence-transformers
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize with a sentence-transformers model.

        Args:
            model_name: Model ID from HuggingFace
                - all-MiniLM-L6-v2: Fast, good quality (384d)
                - all-mpnet-base-v2: Better quality, slower (768d)
                - paraphrase-MiniLM-L6-v2: Good for paraphrase tasks
        """
        self._model_name = model_name
        self._model = None
        self._dimension = None

    def _load_model(self):
        """Lazy load the model."""
        if self._model is not None:
            return

        try:
            import logging
            _log = logging.getLogger(__name__)
            from sentence_transformers import SentenceTransformer
            _log.info(
                "Loading sentence-transformers model '%s' — first run downloads ~90MB to ~/.cache/huggingface/",
                self._model_name,
            )
            self._model = SentenceTransformer(self._model_name)
            self._dimension = self._model.get_sentence_embedding_dimension()
            _log.info("Sentence-transformers model ready (%dd embeddings)", self._dimension)
        except ImportError:
            raise RuntimeError(
                "sentence-transformers required. Install with: pip install sentence-transformers"
            )

    def embed(self, text: str) -> List[float]:
        """Generate embedding for single text."""
        self._load_model()
        embedding = self._model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts efficiently."""
        self._load_model()
        if not texts:
            return []
        embeddings = self._model.encode(texts, convert_to_numpy=True)
        return [emb.tolist() for emb in embeddings]

    @property
    def dimension(self) -> int:
        self._load_model()
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name


class SimpleTFIDFEmbedder(EmbeddingGenerator):
    """
    Simple TF-IDF-like embedding generator.

    No external dependencies required.
    Lower quality than neural embeddings but useful for testing
    or when transformers aren't available.
    """

    # Common English stop words
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
        "not", "no", "yes", "so", "than", "too", "only", "own",
        "same", "such", "very", "each", "few", "more", "most",
        "other", "some", "any", "all", "both", "many", "much",
    }

    def __init__(self, vocab_size: int = 512):
        """
        Initialize TF-IDF embedder.

        Args:
            vocab_size: Size of vocabulary (embedding dimension)
        """
        self._vocab_size = vocab_size
        self._vocab: Dict[str, int] = {}
        self._idf: Dict[str, float] = {}
        self._doc_count = 0

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        words = re.findall(r'\b[a-z]+\b', text.lower())
        return [w for w in words if w not in self.STOP_WORDS and len(w) > 2]

    def _update_vocab(self, tokens: List[str]):
        """Update vocabulary with new tokens."""
        for token in tokens:
            if token not in self._vocab:
                if len(self._vocab) < self._vocab_size:
                    self._vocab[token] = len(self._vocab)

    def _compute_tf(self, tokens: List[str]) -> Dict[str, float]:
        """Compute term frequency."""
        counts = Counter(tokens)
        total = len(tokens)
        return {t: c / total for t, c in counts.items()} if total > 0 else {}

    def embed(self, text: str) -> List[float]:
        """Generate TF-IDF-like embedding."""
        tokens = self._tokenize(text)
        self._update_vocab(tokens)

        tf = self._compute_tf(tokens)

        # Create sparse embedding
        embedding = [0.0] * self._vocab_size

        for token, freq in tf.items():
            if token in self._vocab:
                idx = self._vocab[token]
                # TF-IDF-like weighting
                idf = self._idf.get(token, 1.0)
                embedding[idx] = freq * idf

        # Normalize
        norm = math.sqrt(sum(x * x for x in embedding))
        if norm > 0:
            embedding = [x / norm for x in embedding]

        return embedding

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        # First pass: build vocabulary
        all_tokens = []
        for text in texts:
            tokens = self._tokenize(text)
            all_tokens.append(tokens)
            self._update_vocab(tokens)

        # Compute IDF
        self._doc_count = len(texts)
        doc_freq: Dict[str, int] = Counter()
        for tokens in all_tokens:
            for token in set(tokens):
                doc_freq[token] += 1

        for token, df in doc_freq.items():
            self._idf[token] = math.log(self._doc_count / (df + 1)) + 1

        # Generate embeddings
        embeddings = []
        for text in texts:
            embeddings.append(self.embed(text))

        return embeddings

    @property
    def dimension(self) -> int:
        return self._vocab_size

    @property
    def model_name(self) -> str:
        return f"simple-tfidf-{self._vocab_size}"


def get_embedder(use_transformers: bool = True) -> EmbeddingGenerator:
    """
    Get an appropriate embedding generator.

    Args:
        use_transformers: Prefer sentence-transformers if available

    Returns:
        EmbeddingGenerator instance
    """
    if use_transformers:
        try:
            return SentenceTransformerEmbedder()
        except RuntimeError:
            pass  # Fall through to simple embedder

    return SimpleTFIDFEmbedder()
