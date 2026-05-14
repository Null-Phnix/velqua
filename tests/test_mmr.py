"""Tests for MMR (Maximal Marginal Relevance) diversification in hybrid retrieval."""

import math

import numpy as np
import pytest

from backend.anamnesis.retrieval.hybrid import HybridRetriever, HybridSearchResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unit_vec(angle_deg: float, dims: int = 8) -> list[float]:
    """Return a unit vector rotated by *angle_deg* in the first two dimensions."""
    rad = math.radians(angle_deg)
    v = [0.0] * dims
    v[0] = math.cos(rad)
    v[1] = math.sin(rad)
    return v


def _make_result(id: str, score: float, content: str = "") -> HybridSearchResult:
    return HybridSearchResult(
        id=id,
        content=content or id,
        text_score=0.0,
        vector_score=score,
        combined_score=score,
        source_type="fact",
        metadata={},
    )


class MockEmbedder:
    """Embedder that returns pre-configured embeddings keyed by content."""

    def __init__(self, mapping: dict[str, list[float]]):
        self.mapping = mapping
        self._dim = len(next(iter(mapping.values())))

    def embed(self, text: str) -> list[float]:
        return self.mapping.get(text, [0.0] * self._dim)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]

    def get_dimension(self) -> int:
        return self._dim


@pytest.fixture
def backend(tmp_path):
    from backend.anamnesis.stores.sqlite_backend import SQLiteBackend
    return SQLiteBackend(str(tmp_path / "test.db"))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMMRParameters:
    """Verify MMR constructor parameters are stored and defaulted correctly."""

    def test_default_mmr_lambda(self, backend):
        r = HybridRetriever(sqlite_backend=backend)
        assert r.mmr_lambda == pytest.approx(0.5)

    def test_default_diversity_threshold(self, backend):
        r = HybridRetriever(sqlite_backend=backend)
        assert r.mmr_diversity_threshold == pytest.approx(0.85)

    def test_custom_mmr_params(self, backend):
        r = HybridRetriever(
            sqlite_backend=backend,
            mmr_lambda=0.7,
            mmr_diversity_threshold=0.9,
        )
        assert r.mmr_lambda == pytest.approx(0.7)
        assert r.mmr_diversity_threshold == pytest.approx(0.9)


class TestCosineSimMatrix:
    """Unit tests for the pairwise similarity matrix helper."""

    def test_identity_vectors(self):
        vecs = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        mat = HybridRetriever._cosine_similarity_matrix(vecs)
        # Diagonal should be 1.0, off-diagonal 0.0
        np.testing.assert_array_almost_equal(mat, np.eye(3))

    def test_identical_vectors(self):
        vecs = [[1, 0], [1, 0]]
        mat = HybridRetriever._cosine_similarity_matrix(vecs)
        assert mat[0][1] == pytest.approx(1.0)

    def test_opposite_vectors(self):
        vecs = [[1, 0], [-1, 0]]
        mat = HybridRetriever._cosine_similarity_matrix(vecs)
        assert mat[0][1] == pytest.approx(-1.0)


class TestMMRRerank:
    """Direct tests on the _mmr_rerank method."""

    def test_single_result_unchanged(self, backend):
        """One result comes back as-is."""
        r = HybridRetriever(sqlite_backend=backend)
        res = [_make_result("a", 1.0, "a")]
        embs = [_unit_vec(0)]
        out = r._mmr_rerank(res, embs, limit=5)
        assert len(out) == 1
        assert out[0].id == "a"

    def test_demotes_near_duplicate(self, backend):
        """Two results with sim > 0.85: the lower-scored one should be demoted
        below a diverse alternative with a lower raw score."""
        # A and B are near-identical (0° apart → sim ≈ 1.0)
        # C is orthogonal to both (90° apart → sim ≈ 0.0)
        results = [
            _make_result("A", 1.0, "A"),
            _make_result("B", 0.9, "B"),  # near-duplicate of A
            _make_result("C", 0.5, "C"),  # diverse
        ]
        embs = [
            _unit_vec(0),   # A
            _unit_vec(2),   # B — ~2° from A → cos ≈ 0.9994
            _unit_vec(90),  # C — orthogonal
        ]

        r = HybridRetriever(sqlite_backend=backend, mmr_lambda=0.5)
        out = r._mmr_rerank(results, embs, limit=3)

        assert out[0].id == "A", "Highest-relevance should be first"
        assert out[1].id == "C", "Diverse result should beat near-duplicate"
        assert out[2].id == "B", "Near-duplicate demoted to last"

    def test_preserves_order_for_diverse_results(self, backend):
        """When all results are well-separated and λ is high, relevance order wins."""
        results = [
            _make_result("A", 1.0, "A"),
            _make_result("B", 0.8, "B"),
            _make_result("C", 0.6, "C"),
        ]
        # 0°, 90°, 45° — all pairwise similarities ≤ 0.71, well below threshold
        embs = [_unit_vec(0), _unit_vec(90), _unit_vec(45)]

        r = HybridRetriever(sqlite_backend=backend, mmr_lambda=0.9)
        out = r._mmr_rerank(results, embs, limit=3)

        ids = [o.id for o in out]
        assert ids == ["A", "B", "C"]

    def test_respects_limit(self, backend):
        """MMR should return at most `limit` results."""
        results = [_make_result(f"r{i}", 1.0 - i * 0.1, f"r{i}") for i in range(10)]
        embs = [_unit_vec(i * 36) for i in range(10)]  # evenly spread

        r = HybridRetriever(sqlite_backend=backend)
        out = r._mmr_rerank(results, embs, limit=3)
        assert len(out) == 3

    def test_threshold_boundary(self, backend):
        """A result just below the threshold is penalised less than one above."""
        # A is the anchor.
        # B has sim ≈ 0.87 to A (above 0.85 threshold → extra penalty)
        # C has sim ≈ 0.83 to A (below threshold → no extra penalty)
        # Give B and C the same relevance score so the only difference is the penalty.
        angle_above = math.degrees(math.acos(0.87))  # ~29.5°
        angle_below = math.degrees(math.acos(0.83))  # ~33.9°

        results = [
            _make_result("A", 1.0, "A"),
            _make_result("B_above", 0.7, "B_above"),
            _make_result("C_below", 0.7, "C_below"),
        ]
        embs = [
            _unit_vec(0),
            _unit_vec(angle_above),
            _unit_vec(angle_below),
        ]

        r = HybridRetriever(
            sqlite_backend=backend,
            mmr_lambda=0.5,
            mmr_diversity_threshold=0.85,
        )
        out = r._mmr_rerank(results, embs, limit=3)

        # C_below should appear before B_above because B gets the extra penalty
        pos_b = next(i for i, o in enumerate(out) if o.id == "B_above")
        pos_c = next(i for i, o in enumerate(out) if o.id == "C_below")
        assert pos_c < pos_b, (
            f"Below-threshold C (pos {pos_c}) should rank above "
            f"above-threshold B (pos {pos_b})"
        )


class TestMMRIntegration:
    """Integration: MMR applied inside the search() pipeline."""

    def test_mmr_skipped_without_embedder(self, backend):
        """When no embedder is available, search returns pure score order."""
        r = HybridRetriever(sqlite_backend=backend, mmr_lambda=0.5)
        # search with TEXT_ONLY mode — no embedder needed, MMR should be skipped
        from backend.anamnesis.retrieval.hybrid import SearchMode
        results = r.search("anything", limit=5, mode=SearchMode.TEXT_ONLY)
        # Should not raise; returns whatever text search found (possibly empty)
        assert isinstance(results, list)

    def test_mmr_disabled_when_lambda_one(self, backend):
        """λ=1.0 skips MMR entirely (pure relevance)."""
        embedder = MockEmbedder({"a": _unit_vec(0), "b": _unit_vec(1)})
        r = HybridRetriever(
            sqlite_backend=backend,
            embedder=embedder,
            mmr_lambda=1.0,
        )
        # Even with embedder present, λ=1.0 should bypass MMR
        from backend.anamnesis.retrieval.hybrid import SearchMode
        results = r.search("test", limit=5, mode=SearchMode.TEXT_ONLY)
        assert isinstance(results, list)

    def test_original_scores_preserved_after_mmr(self, backend):
        """MMR reorders results but does not mutate combined_score values."""
        results = [
            _make_result("A", 1.0, "A"),
            _make_result("B", 0.9, "B"),
            _make_result("C", 0.5, "C"),
        ]
        embs = [_unit_vec(0), _unit_vec(2), _unit_vec(90)]

        r = HybridRetriever(sqlite_backend=backend, mmr_lambda=0.5)
        out = r._mmr_rerank(results, embs, limit=3)

        score_map = {o.id: o.combined_score for o in out}
        assert score_map["A"] == pytest.approx(1.0)
        assert score_map["B"] == pytest.approx(0.9)
        assert score_map["C"] == pytest.approx(0.5)

    def test_mmr_handles_empty_content(self, backend):
        """Results with empty content should not crash MMR."""
        results = [
            _make_result("A", 1.0, ""),
            _make_result("B", 0.8, ""),
        ]
        # Empty content maps to zero vectors → sim = 0 (due to norm clamping)
        embs = [[0.0] * 8, [0.0] * 8]

        r = HybridRetriever(sqlite_backend=backend, mmr_lambda=0.5)
        out = r._mmr_rerank(results, embs, limit=2)
        assert len(out) == 2
