"""
Unit tests for the cross-encoder reranker module.

Tests the CrossEncoderReranker in isolation with mocked cross-encoder model,
verifying that re-ranking logic correctly reorders candidates by cross-encoder
score and respects top_k limits.
"""
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from backend.anamnesis.retrieval.reranker import CrossEncoderReranker


class TestCrossEncoderReranker:
    """Tests for CrossEncoderReranker."""

    def _make_reranker(self):
        """Create a reranker with a mocked model."""
        reranker = CrossEncoderReranker(model_name="test-model")
        mock_model = MagicMock()
        reranker._model = mock_model
        return reranker, mock_model

    def test_empty_candidates(self):
        reranker, _ = self._make_reranker()
        result = reranker.rerank("query", [])
        assert result == []

    def test_reranks_by_cross_encoder_score(self):
        reranker, mock_model = self._make_reranker()

        # Cross-encoder scores: B > A > C (different order than input)
        mock_model.predict.return_value = np.array([0.3, 0.9, 0.1])

        candidates = [
            ("Fact A", 0.8),
            ("Fact B", 0.6),
            ("Fact C", 0.7),
        ]

        result = reranker.rerank("test query", candidates)

        assert len(result) == 3
        assert result[0][0] == "Fact B"   # Highest CE score
        assert result[1][0] == "Fact A"
        assert result[2][0] == "Fact C"   # Lowest CE score

        # Verify (query, passage) pairs were sent to model
        pairs = mock_model.predict.call_args[0][0]
        assert pairs == [
            ("test query", "Fact A"),
            ("test query", "Fact B"),
            ("test query", "Fact C"),
        ]

    def test_top_k_limits_output(self):
        reranker, mock_model = self._make_reranker()
        mock_model.predict.return_value = np.array([0.5, 0.9, 0.1, 0.7])

        candidates = [
            ("A", 0.0),
            ("B", 0.0),
            ("C", 0.0),
            ("D", 0.0),
        ]

        result = reranker.rerank("query", candidates, top_k=2)

        assert len(result) == 2
        assert result[0][0] == "B"  # score 0.9
        assert result[1][0] == "D"  # score 0.7

    def test_top_k_none_returns_all(self):
        reranker, mock_model = self._make_reranker()
        mock_model.predict.return_value = np.array([0.5, 0.3])

        candidates = [("A", 0.0), ("B", 0.0)]
        result = reranker.rerank("q", candidates, top_k=None)
        assert len(result) == 2

    def test_scores_are_floats(self):
        reranker, mock_model = self._make_reranker()
        # numpy float64 should be converted to Python float
        mock_model.predict.return_value = np.array([0.42])

        result = reranker.rerank("q", [("fact", 0.0)])
        assert isinstance(result[0][1], float)
        assert result[0][1] == pytest.approx(0.42)

    def test_unload_clears_model(self):
        reranker, _ = self._make_reranker()
        assert reranker._model is not None
        reranker.unload()
        assert reranker._model is None

    def test_lazy_load_import_error(self):
        """Reranker raises RuntimeError if sentence-transformers missing."""
        reranker = CrossEncoderReranker(model_name="test-model")
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            with patch(
                "backend.anamnesis.retrieval.reranker.CrossEncoderReranker._load_model",
                side_effect=RuntimeError("sentence-transformers required"),
            ):
                with pytest.raises(RuntimeError, match="sentence-transformers"):
                    reranker.rerank("q", [("fact", 0.0)])

    def test_single_candidate(self):
        reranker, mock_model = self._make_reranker()
        mock_model.predict.return_value = np.array([0.75])

        result = reranker.rerank("q", [("only fact", 1.0)])
        assert len(result) == 1
        assert result[0] == ("only fact", pytest.approx(0.75))

    def test_preserves_original_content_exactly(self):
        """Content strings pass through without modification."""
        reranker, mock_model = self._make_reranker()
        mock_model.predict.return_value = np.array([0.5])

        content = "User's favorite IDE is VSCode — uses it daily"
        result = reranker.rerank("IDE preference", [(content, 0.0)])
        assert result[0][0] == content
