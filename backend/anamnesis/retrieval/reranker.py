"""
Cross-encoder re-ranker for breaking metric alignment bias.

The retrieval pipeline uses bge-m3 / MiniLM bi-encoder embeddings for both
indexing and scoring.  This creates a metric alignment confound: the same
embedding space decides what's retrieved AND what's "most relevant."

A cross-encoder scores (query, passage) pairs directly — no shared embedding
space — so it breaks the confound and produces genuinely independent
relevance judgments.

Default model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - ~22 MB, runs on CPU in <50 ms for 20 candidates
  - Trained on MS MARCO passage ranking
  - Good general-purpose relevance scoring
"""

import logging
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Re-ranks retrieval candidates using a cross-encoder model.

    Lazy-loads the model on first use to avoid startup cost when
    reranking is disabled.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self._device = device
        self._model = None

    def _load_model(self):
        """Lazy load the cross-encoder model."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, device=self._device)
            logger.info("Cross-encoder reranker loaded: %s", self.model_name)
        except ImportError:
            raise RuntimeError(
                "sentence-transformers required for reranking. "
                "Install with: pip install sentence-transformers"
            )

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[str, float]],
        top_k: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        """
        Re-rank candidates by cross-encoder relevance.

        Args:
            query: The user's search query.
            candidates: List of (content, original_score) tuples from
                        hybrid retrieval.
            top_k: How many to return after re-ranking. None = return all.

        Returns:
            Re-ranked list of (content, cross_encoder_score) tuples,
            sorted by cross-encoder score descending.
        """
        if not candidates:
            return []

        self._load_model()

        # Build (query, passage) pairs for the cross-encoder
        pairs = [(query, content) for content, _ in candidates]

        # Score all pairs — cross-encoder produces a single relevance score
        scores = self._model.predict(pairs)

        # Pair scores back with content
        scored = list(zip(
            [content for content, _ in candidates],
            [float(s) for s in scores],
        ))

        # Sort by cross-encoder score (higher = more relevant)
        scored.sort(key=lambda x: x[1], reverse=True)

        if top_k is not None:
            scored = scored[:top_k]

        return scored

    def unload(self):
        """Unload model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None

            import gc
            gc.collect()
