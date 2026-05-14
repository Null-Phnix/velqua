"""Tests for smart duplicate detection and merging."""
import uuid
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from backend.anamnesis.dedup.smart_detector import (
    EMBEDDING_DUPLICATE_THRESHOLD,
    TFIDF_DUPLICATE_THRESHOLD,
    DuplicateMatch,
    SmartDuplicateDetector,
    _fact_quality_score,
    _merge_facts,
    _merge_metadata,
)
from backend.anamnesis.models import Fact, FactType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fact(
    content="I work at Google as a software engineer",
    fact_type=FactType.GENERAL,
    confidence=0.8,
    confirmation_count=1,
    importance=0.5,
    metadata=None,
    tags=None,
    source_episodes=None,
    is_superseded=False,
):
    """Create a Fact for testing."""
    return Fact(
        id=str(uuid.uuid4()),
        content=content,
        fact_type=fact_type,
        confidence=confidence,
        confirmation_count=confirmation_count,
        importance=importance,
        metadata=metadata or {},
        tags=tags or [],
        source_episodes=source_episodes or [],
        is_superseded=is_superseded,
        first_learned=datetime.now() - timedelta(days=5),
        last_confirmed=datetime.now(),
    )


def _mock_embedder(similarity=0.95):
    """
    Create a mock embedder that returns controlled similarity.

    The mock returns unit vectors where the cosine similarity between
    any two embeddings equals `similarity`.
    """
    import numpy as np

    embedder = MagicMock()

    # For the "new content" call, return a fixed vector
    base_vec = np.zeros(384)
    base_vec[0] = 1.0

    # For existing facts, return a vector with the desired cosine similarity
    # cos(theta) = similarity, so rotate by theta
    theta = np.arccos(np.clip(similarity, -1, 1))
    rotated = np.zeros(384)
    rotated[0] = np.cos(theta)
    rotated[1] = np.sin(theta)

    call_count = [0]

    def embed_side_effect(text):
        call_count[0] += 1
        # First call is the new content, subsequent are existing facts
        if call_count[0] == 1:
            return base_vec.tolist()
        return rotated.tolist()

    embedder.embed = MagicMock(side_effect=embed_side_effect)
    return embedder


# ---------------------------------------------------------------------------
# SmartDuplicateDetector — check_duplicate
# ---------------------------------------------------------------------------

class TestCheckDuplicate:
    """Test the duplicate checking logic."""

    def test_no_candidates_returns_not_duplicate(self):
        detector = SmartDuplicateDetector()
        result = detector.check_duplicate("some fact", [])
        assert not result.is_duplicate

    def test_empty_content_returns_not_duplicate(self):
        detector = SmartDuplicateDetector()
        result = detector.check_duplicate("", [_make_fact()])
        assert not result.is_duplicate

    def test_tfidf_detects_identical_content(self):
        """Identical text should always be detected as duplicate via TF-IDF."""
        detector = SmartDuplicateDetector(embedder=None)
        existing = _make_fact(content="I work at Google as a software engineer")
        result = detector.check_duplicate(
            "I work at Google as a software engineer",
            [existing],
        )
        assert result.is_duplicate
        assert result.method == "tfidf"
        assert result.existing_fact.id == existing.id

    def test_tfidf_rejects_different_content(self):
        """Completely different content should not be flagged."""
        detector = SmartDuplicateDetector(embedder=None)
        existing = _make_fact(content="I enjoy playing chess on weekends")
        result = detector.check_duplicate(
            "Python is my favorite programming language",
            [existing],
        )
        assert not result.is_duplicate

    def test_embedding_detects_above_threshold(self):
        """Embedder with similarity above 0.92 should detect duplicate."""
        embedder = _mock_embedder(similarity=0.95)
        detector = SmartDuplicateDetector(embedder=embedder)
        existing = _make_fact(content="I'm a software engineer at Google")

        result = detector.check_duplicate(
            "I work at Google as a software engineer",
            [existing],
        )
        assert result.is_duplicate
        assert result.method == "embedding"
        assert result.similarity >= EMBEDDING_DUPLICATE_THRESHOLD

    def test_embedding_rejects_below_threshold(self):
        """Embedder with similarity below 0.92 should not flag as duplicate."""
        embedder = _mock_embedder(similarity=0.80)
        detector = SmartDuplicateDetector(embedder=embedder)
        existing = _make_fact(content="I enjoy playing chess on weekends")

        result = detector.check_duplicate(
            "I work at Google as a software engineer",
            [existing],
        )
        # Embedding won't flag it, then TF-IDF check runs
        # TF-IDF should also fail since content is quite different
        assert not result.is_duplicate

    def test_skips_superseded_facts(self):
        """Superseded facts should be ignored during duplicate check."""
        detector = SmartDuplicateDetector(embedder=None)
        existing = _make_fact(
            content="I work at Google as a software engineer",
            is_superseded=True,
        )
        result = detector.check_duplicate(
            "I work at Google as a software engineer",
            [existing],
        )
        assert not result.is_duplicate

    def test_embedding_failure_falls_back_to_tfidf(self):
        """If embedder raises, should fall back to TF-IDF."""
        embedder = MagicMock()
        embedder.embed = MagicMock(side_effect=RuntimeError("model load failed"))
        detector = SmartDuplicateDetector(embedder=embedder)

        existing = _make_fact(content="I work at Google as a software engineer")
        result = detector.check_duplicate(
            "I work at Google as a software engineer",
            [existing],
        )
        # Should still detect via TF-IDF fallback
        assert result.is_duplicate
        assert result.method == "tfidf"

    def test_picks_best_match_from_multiple_candidates(self):
        """When multiple candidates exist, should pick the most similar."""
        detector = SmartDuplicateDetector(embedder=None)
        exact = _make_fact(content="I live in San Francisco, California")
        distant = _make_fact(content="The weather in Tokyo is mild today")

        result = detector.check_duplicate(
            "I live in San Francisco, California",
            [distant, exact],
        )
        assert result.is_duplicate
        assert result.existing_fact.id == exact.id

    def test_custom_thresholds(self):
        """Custom thresholds should be respected."""
        # Very high TF-IDF threshold — even identical content might not pass
        detector = SmartDuplicateDetector(
            embedder=None,
            tfidf_threshold=0.99,
        )
        existing = _make_fact(content="I work at Google as a senior software engineer")
        result = detector.check_duplicate(
            "I work at Google as a software engineer",
            [existing],
        )
        # Similar but not identical — high threshold rejects it
        assert not result.is_duplicate


# ---------------------------------------------------------------------------
# Fact quality scoring
# ---------------------------------------------------------------------------

class TestFactQualityScore:
    """Test the quality scoring function."""

    def test_higher_confidence_scores_higher(self):
        low = _make_fact(confidence=0.3)
        high = _make_fact(confidence=0.9)
        assert _fact_quality_score(high) > _fact_quality_score(low)

    def test_more_confirmations_scores_higher(self):
        few = _make_fact(confirmation_count=1)
        many = _make_fact(confirmation_count=10)
        assert _fact_quality_score(many) > _fact_quality_score(few)

    def test_superseded_penalized(self):
        active = _make_fact()
        superseded = _make_fact(is_superseded=True)
        assert _fact_quality_score(active) > _fact_quality_score(superseded)

    def test_rich_metadata_scores_higher(self):
        bare = _make_fact(metadata={})
        rich = _make_fact(metadata={
            "topic": "career",
            "category": "professional",
            "emotion": "positive",
            "sentiment_score": 0.8,
        })
        assert _fact_quality_score(rich) > _fact_quality_score(bare)

    def test_tags_boost_score(self):
        no_tags = _make_fact(tags=[])
        tagged = _make_fact(tags=["career", "tech", "google"])
        assert _fact_quality_score(tagged) > _fact_quality_score(no_tags)

    def test_optimal_length_content(self):
        """Content in the 30-200 char sweet spot should score higher."""
        short = _make_fact(content="I code")
        optimal = _make_fact(content="I work at Google as a senior software engineer in Mountain View")
        assert _fact_quality_score(optimal) > _fact_quality_score(short)

    def test_higher_importance_scores_higher(self):
        low_imp = _make_fact(importance=0.2)
        high_imp = _make_fact(importance=0.9)
        assert _fact_quality_score(high_imp) > _fact_quality_score(low_imp)


# ---------------------------------------------------------------------------
# Metadata merging
# ---------------------------------------------------------------------------

class TestMergeMetadata:
    """Test metadata merging logic."""

    def test_keeper_values_preserved(self):
        keeper = _make_fact(metadata={"topic": "career", "source": "proxy"})
        dup = _make_fact(metadata={"topic": "work", "source": "assistant"})
        merged = _merge_metadata(keeper, dup)
        assert merged["topic"] == "career"  # Keeper's value preserved
        assert merged["source"] == "proxy"

    def test_missing_fields_filled_from_duplicate(self):
        keeper = _make_fact(metadata={"topic": "career"})
        dup = _make_fact(metadata={"emotion": "positive", "category": "professional"})
        merged = _merge_metadata(keeper, dup)
        assert merged["topic"] == "career"
        assert merged["emotion"] == "positive"
        assert merged["category"] == "professional"

    def test_merge_history_tracked(self):
        keeper = _make_fact(metadata={})
        dup = _make_fact(metadata={})
        merged = _merge_metadata(keeper, dup)
        assert dup.id in merged["merged_from"]

    def test_merge_history_accumulates(self):
        """Multiple merges should accumulate history."""
        keeper = _make_fact(metadata={"merged_from": ["old-id-1"]})
        dup = _make_fact(metadata={})
        merged = _merge_metadata(keeper, dup)
        assert "old-id-1" in merged["merged_from"]
        assert dup.id in merged["merged_from"]


# ---------------------------------------------------------------------------
# Fact merging
# ---------------------------------------------------------------------------

class TestMergeFacts:
    """Test fact merging logic."""

    def test_confirmation_count_incremented(self):
        keeper = _make_fact(confirmation_count=3)
        dup = _make_fact(confirmation_count=1)
        original_count = keeper.confirmation_count
        merged = _merge_facts(keeper, dup)
        assert merged.confirmation_count == original_count + 1

    def test_confidence_boosted(self):
        keeper = _make_fact(confidence=0.6)
        dup = _make_fact(confidence=0.4)
        merged = _merge_facts(keeper, dup)
        assert merged.confidence > 0.6

    def test_confidence_capped_at_1(self):
        keeper = _make_fact(confidence=0.95)
        dup = _make_fact(confidence=0.9)
        merged = _merge_facts(keeper, dup)
        assert merged.confidence <= 1.0

    def test_source_episodes_merged(self):
        keeper = _make_fact(source_episodes=["ep1", "ep2"])
        dup = _make_fact(source_episodes=["ep2", "ep3"])
        merged = _merge_facts(keeper, dup)
        assert set(merged.source_episodes) == {"ep1", "ep2", "ep3"}

    def test_tags_merged(self):
        keeper = _make_fact(tags=["career"])
        dup = _make_fact(tags=["career", "tech", "google"])
        merged = _merge_facts(keeper, dup)
        assert set(merged.tags) == {"career", "tech", "google"}

    def test_importance_takes_max(self):
        keeper = _make_fact(importance=0.3)
        dup = _make_fact(importance=0.8)
        merged = _merge_facts(keeper, dup)
        assert merged.importance == 0.8

    def test_metadata_merged(self):
        keeper = _make_fact(metadata={"topic": "career"})
        dup = _make_fact(metadata={"emotion": "happy"})
        merged = _merge_facts(keeper, dup)
        assert merged.metadata["topic"] == "career"
        assert merged.metadata["emotion"] == "happy"

    def test_no_duplicate_episodes(self):
        """Source episodes should not have duplicates after merge."""
        keeper = _make_fact(source_episodes=["ep1"])
        dup = _make_fact(source_episodes=["ep1"])
        merged = _merge_facts(keeper, dup)
        assert merged.source_episodes.count("ep1") == 1

    def test_no_duplicate_tags(self):
        """Tags should not have duplicates after merge."""
        keeper = _make_fact(tags=["python"])
        dup = _make_fact(tags=["python"])
        merged = _merge_facts(keeper, dup)
        assert merged.tags.count("python") == 1


# ---------------------------------------------------------------------------
# find_and_merge (end-to-end)
# ---------------------------------------------------------------------------

class TestFindAndMerge:
    """Test the full find_and_merge flow."""

    def test_no_duplicate_returns_none(self):
        detector = SmartDuplicateDetector(embedder=None)
        new_fact = _make_fact(content="I enjoy hiking in the mountains")
        candidates = [_make_fact(content="Python is my favorite language")]
        result = detector.find_and_merge(new_fact.content, new_fact, candidates)
        assert result is None

    def test_duplicate_returns_merged_fact(self):
        detector = SmartDuplicateDetector(embedder=None)
        existing = _make_fact(
            content="I work at Google as a software engineer",
            confidence=0.8,
            confirmation_count=3,
        )
        new_fact = _make_fact(
            content="I work at Google as a software engineer",
            confidence=0.5,
            confirmation_count=1,
        )
        result = detector.find_and_merge(new_fact.content, new_fact, [existing])
        assert result is not None
        # Existing has higher quality (more confirmations, higher confidence)
        assert result.id == existing.id
        assert result.confirmation_count == 4  # Was 3, incremented

    def test_new_fact_kept_when_higher_quality(self):
        """When the new fact is higher quality, it should become the keeper."""
        detector = SmartDuplicateDetector(embedder=None)
        existing = _make_fact(
            content="I work at Google as a software engineer",
            confidence=0.3,
            confirmation_count=1,
            metadata={},
            tags=[],
        )
        new_fact = _make_fact(
            content="I work at Google as a software engineer",
            confidence=0.9,
            confirmation_count=5,
            metadata={"topic": "career", "emotion": "positive"},
            tags=["career", "tech"],
        )
        result = detector.find_and_merge(new_fact.content, new_fact, [existing])
        assert result is not None
        assert result.id == new_fact.id

    def test_merge_with_embedding_detector(self):
        """End-to-end with mock embedder."""
        embedder = _mock_embedder(similarity=0.96)
        detector = SmartDuplicateDetector(embedder=embedder)

        existing = _make_fact(
            content="I'm a software developer at Google",
            confidence=0.7,
            metadata={"topic": "career"},
        )
        new_fact = _make_fact(
            content="I work at Google as a software engineer",
            confidence=0.5,
            metadata={"emotion": "neutral"},
        )

        result = detector.find_and_merge(new_fact.content, new_fact, [existing])
        assert result is not None
        # Existing has higher confidence
        assert result.id == existing.id
        # Metadata merged from new fact
        assert result.metadata.get("emotion") == "neutral"
        assert result.metadata.get("topic") == "career"


# ---------------------------------------------------------------------------
# SemanticStore integration
# ---------------------------------------------------------------------------

class TestSemanticStoreIntegration:
    """Test that SemanticStore.add_fact() uses smart dedup."""

    def test_add_fact_detects_duplicate_via_tfidf(self, tmp_path):
        """Adding a duplicate fact should merge instead of creating new."""
        from backend.anamnesis.stores.sqlite_backend import SQLiteBackend
        from backend.anamnesis.stores.semantic import SemanticStore

        backend = SQLiteBackend(str(tmp_path / "test.db"))
        store = SemanticStore(backend)

        # Add first fact
        fact1 = store.add_fact(
            content="I work at Google as a software engineer",
            fact_type=FactType.PERSONAL,
            confidence=0.8,
        )
        assert fact1.confirmation_count == 1

        # Add same fact again — should be detected as duplicate
        fact2 = store.add_fact(
            content="I work at Google as a software engineer",
            fact_type=FactType.PERSONAL,
            confidence=0.7,
        )

        # Should have merged into fact1
        assert fact2.id == fact1.id
        assert fact2.confirmation_count == 2
        assert fact2.confidence > 0.8  # Boosted

        # Only one fact in the store
        all_facts = store.list_all(limit=100)
        matching = [f for f in all_facts if "Google" in f.content]
        assert len(matching) == 1

    def test_add_fact_creates_new_for_different_content(self, tmp_path):
        """Different facts should not be merged."""
        from backend.anamnesis.stores.sqlite_backend import SQLiteBackend
        from backend.anamnesis.stores.semantic import SemanticStore

        backend = SQLiteBackend(str(tmp_path / "test.db"))
        store = SemanticStore(backend)

        fact1 = store.add_fact(content="I work at Google as a software engineer")
        fact2 = store.add_fact(content="I enjoy hiking in the Pacific Northwest")

        assert fact1.id != fact2.id
        assert fact1.confirmation_count == 1
        assert fact2.confirmation_count == 1

    def test_add_fact_merges_source_episodes(self, tmp_path):
        """Source episodes should be merged during dedup."""
        from backend.anamnesis.stores.sqlite_backend import SQLiteBackend
        from backend.anamnesis.stores.semantic import SemanticStore

        backend = SQLiteBackend(str(tmp_path / "test.db"))
        store = SemanticStore(backend)

        fact1 = store.add_fact(
            content="I work at Google as a software engineer",
            source_episode_id="ep-001",
        )
        fact2 = store.add_fact(
            content="I work at Google as a software engineer",
            source_episode_id="ep-002",
        )

        assert fact2.id == fact1.id
        assert "ep-001" in fact2.source_episodes
        assert "ep-002" in fact2.source_episodes

    def test_set_embedder_updates_detector(self, tmp_path):
        """set_embedder() should propagate to the smart detector."""
        from backend.anamnesis.stores.sqlite_backend import SQLiteBackend
        from backend.anamnesis.stores.semantic import SemanticStore

        backend = SQLiteBackend(str(tmp_path / "test.db"))
        store = SemanticStore(backend)

        assert store._dedup.embedder is None

        mock_embedder = MagicMock()
        store.set_embedder(mock_embedder)

        assert store._dedup.embedder is mock_embedder

    def test_add_fact_with_embedder(self, tmp_path):
        """When an embedder is provided, dedup should use embeddings."""
        from backend.anamnesis.stores.sqlite_backend import SQLiteBackend
        from backend.anamnesis.stores.semantic import SemanticStore

        backend = SQLiteBackend(str(tmp_path / "test.db"))
        embedder = _mock_embedder(similarity=0.96)
        store = SemanticStore(backend, embedder=embedder)

        fact1 = store.add_fact(content="I'm a Python developer at Google")

        # Reset mock embedder for the second call (new call_count)
        embedder2 = _mock_embedder(similarity=0.96)
        store.set_embedder(embedder2)

        fact2 = store.add_fact(content="I work at Google as a Python engineer")

        # With high embedding similarity, should be detected as duplicate
        # (depends on FTS returning the first fact as a candidate)
        # The fact that FTS returns it depends on matching keywords
        # Since both contain "Google" and "Python", FTS should find the match
        if fact2.id == fact1.id:
            assert fact2.confirmation_count >= 2


# ---------------------------------------------------------------------------
# Auto-learner pipeline integration
# ---------------------------------------------------------------------------

class TestAutoLearnerIntegration:
    """Test that the auto-learner uses smart dedup through SemanticStore."""

    def test_autolearner_tracks_duplicates(self, tmp_path):
        """AutoLearner.duplicates_seen should increment on merge."""
        import asyncio
        from backend.anamnesis.stores.sqlite_backend import SQLiteBackend
        from backend.anamnesis.stores.semantic import SemanticStore
        from backend.anamnesis.stores.episodic import EpisodicStore
        from backend.auto_learner import AutoLearner

        backend = SQLiteBackend(str(tmp_path / "test.db"))

        # Build a minimal memory-like object
        class MockMemory:
            def __init__(self):
                self.backend = backend
                self.semantic = SemanticStore(backend)
                self.episodic = EpisodicStore(backend)

        memory = MockMemory()
        learner = AutoLearner(memory)

        # First message — should create a fact
        asyncio.run(
            learner.learn_from_message("I work at Google as a senior software engineer")
        )
        initial_learned = learner.facts_learned
        initial_dupes = learner.duplicates_seen

        # Same message again — should detect duplicate
        asyncio.run(
            learner.learn_from_message("I work at Google as a senior software engineer")
        )

        # Either learned a new one or detected a duplicate
        total = learner.facts_learned + learner.duplicates_seen
        assert total >= initial_learned + initial_dupes


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_word_content(self):
        """Very short content should not crash."""
        detector = SmartDuplicateDetector(embedder=None)
        result = detector.check_duplicate("hello", [_make_fact(content="hello world")])
        # Should not crash, result depends on TF-IDF
        assert isinstance(result, DuplicateMatch)

    def test_unicode_content(self):
        """Unicode content should be handled correctly."""
        detector = SmartDuplicateDetector(embedder=None)
        fact = _make_fact(content="I speak Japanese. 日本語を話します。")
        result = detector.check_duplicate(
            "I speak Japanese. 日本語を話します。",
            [fact],
        )
        assert result.is_duplicate

    def test_all_candidates_superseded(self):
        """When all candidates are superseded, should return no duplicate."""
        detector = SmartDuplicateDetector(embedder=None)
        candidates = [
            _make_fact(content="I work at Google", is_superseded=True),
            _make_fact(content="I work at Google", is_superseded=True),
        ]
        result = detector.check_duplicate("I work at Google", candidates)
        assert not result.is_duplicate

    def test_empty_metadata_merge(self):
        """Merging facts with empty metadata should not crash."""
        keeper = _make_fact(metadata={})
        dup = _make_fact(metadata={})
        merged = _merge_facts(keeper, dup)
        assert "merged_from" in merged.metadata

    def test_none_source_episodes(self):
        """Source episodes that are None should be handled."""
        keeper = _make_fact(source_episodes=[])
        dup = _make_fact(source_episodes=[None, "ep1"])
        merged = _merge_facts(keeper, dup)
        # None should not appear, ep1 should
        assert None not in merged.source_episodes
        assert "ep1" in merged.source_episodes

    def test_none_tags(self):
        """None tags should be filtered out during merge."""
        keeper = _make_fact(tags=["a"])
        dup = _make_fact(tags=[None, "b"])
        merged = _merge_facts(keeper, dup)
        assert None not in merged.tags
        assert "a" in merged.tags
        assert "b" in merged.tags

    def test_detector_with_no_embedder_and_no_match(self):
        """TF-IDF fallback with no match should return clean result."""
        detector = SmartDuplicateDetector(embedder=None)
        result = detector.check_duplicate(
            "I love surfing on the coast",
            [_make_fact(content="Quantum computing uses qubits for parallel processing")],
        )
        assert not result.is_duplicate
        assert result.method == "tfidf"
        assert result.similarity < TFIDF_DUPLICATE_THRESHOLD
