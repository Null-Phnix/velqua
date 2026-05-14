"""Tests for adaptive retrieval strategy selection."""

import pytest

from backend.anamnesis.retrieval.strategy import (
    QueryType,
    RetrievalStrategy,
    apply_strategy,
    classify_query,
    restore_strategy,
)


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def retriever(tmp_path):
    """HybridRetriever with default weights for apply/restore tests."""
    from backend.anamnesis.stores.sqlite_backend import SQLiteBackend
    from backend.anamnesis.retrieval.hybrid import HybridRetriever

    backend = SQLiteBackend(str(tmp_path / "test.db"))
    return HybridRetriever(sqlite_backend=backend)


# ── Named-entity detection ────────────────────────────────────────────

class TestNamedEntityDetection:
    """Queries mentioning specific gods/figures → FTS boosted to 40%."""

    def test_single_deity_name(self):
        strategy = classify_query("Tell me about Odin")
        assert strategy.query_type == QueryType.NAMED_ENTITY
        assert strategy.text_weight == pytest.approx(0.4)
        assert strategy.vector_weight == pytest.approx(0.6)

    def test_multiple_deity_names_same_culture(self):
        strategy = classify_query("What is the relationship between Thor and Loki?")
        assert strategy.query_type == QueryType.NAMED_ENTITY
        assert strategy.confidence > 0.5

    def test_greek_deity(self):
        strategy = classify_query("Who was Athena?")
        assert strategy.query_type == QueryType.NAMED_ENTITY
        assert strategy.text_weight == pytest.approx(0.4)

    def test_egyptian_deity(self):
        strategy = classify_query("Describe Anubis role in the afterlife")
        # "anubis" is entity, "afterlife" is thematic — entity should win
        # because entity_score = 1*0.35 = 0.35, thematic = 1*0.3 = 0.30
        assert strategy.query_type == QueryType.NAMED_ENTITY

    def test_synonym_names_detected(self):
        """Names that are synonyms of culture-tagged names should also count."""
        strategy = classify_query("Tell me about Wodan")
        assert strategy.query_type == QueryType.NAMED_ENTITY


# ── Thematic / emotional detection ────────────────────────────────────

class TestThematicDetection:
    """Abstract theme/emotion queries → vector boosted to 90%."""

    def test_single_theme(self):
        strategy = classify_query("What role does sacrifice play in mythology?")
        assert strategy.query_type == QueryType.THEMATIC
        assert strategy.vector_weight == pytest.approx(0.9)
        assert strategy.text_weight == pytest.approx(0.1)

    def test_emotional_query(self):
        strategy = classify_query("stories about grief and redemption")
        assert strategy.query_type == QueryType.THEMATIC
        assert strategy.confidence >= 0.5

    def test_multiple_themes(self):
        strategy = classify_query("the duality of chaos and order in creation myths")
        assert strategy.query_type == QueryType.THEMATIC
        assert strategy.confidence >= 0.5

    def test_abstract_concept(self):
        strategy = classify_query("How is immortal transcendence portrayed?")
        assert strategy.query_type == QueryType.THEMATIC


# ── Cross-cultural detection ──────────────────────────────────────────

class TestCrossCulturalDetection:
    """Queries comparing traditions → MMR diversity enabled."""

    def test_two_cultures_with_compare(self):
        strategy = classify_query("Compare Norse and Greek creation myths")
        assert strategy.query_type == QueryType.CROSS_CULTURAL
        assert strategy.mmr_lambda is not None
        assert strategy.mmr_lambda < 0.5  # stronger diversity
        assert strategy.mmr_diversity_threshold == pytest.approx(0.7)

    def test_two_cultures_without_compare_keyword(self):
        """Two culture references alone should trigger cross-cultural."""
        strategy = classify_query("Zeus and Odin as sky fathers")
        assert strategy.query_type == QueryType.CROSS_CULTURAL

    def test_vs_keyword(self):
        strategy = classify_query("Shiva vs Dionysus")
        assert strategy.query_type == QueryType.CROSS_CULTURAL

    def test_equivalent_keyword(self):
        strategy = classify_query("Roman equivalent of Athena")
        # athena=greek, roman=roman → 2 cultures + comparison signal
        assert strategy.query_type == QueryType.CROSS_CULTURAL


# ── Follow-up detection ───────────────────────────────────────────────

class TestFollowUpDetection:
    """Conversational follow-ups → boost confirmed facts."""

    def test_you_said_pattern(self):
        strategy = classify_query("You said Loki had children with a giant?")
        assert strategy.query_type == QueryType.FOLLOW_UP
        assert strategy.boost_confirmed is True

    def test_earlier_pattern(self):
        strategy = classify_query("Earlier you mentioned something about creation")
        assert strategy.query_type == QueryType.FOLLOW_UP

    def test_remember_when(self):
        strategy = classify_query("Remember when we discussed the flood myths?")
        assert strategy.query_type == QueryType.FOLLOW_UP

    def test_last_time(self):
        strategy = classify_query("Last time we talked about the trickster archetype")
        assert strategy.query_type == QueryType.FOLLOW_UP

    def test_previously(self):
        strategy = classify_query("You previously covered Norse cosmology")
        assert strategy.query_type == QueryType.FOLLOW_UP


# ── General / fallback detection ──────────────────────────────────────

class TestGeneralFallback:
    """Ambiguous or empty queries → GENERAL (default weights)."""

    def test_empty_string(self):
        strategy = classify_query("")
        assert strategy.query_type == QueryType.GENERAL

    def test_whitespace_only(self):
        strategy = classify_query("   ")
        assert strategy.query_type == QueryType.GENERAL

    def test_no_signal_query(self):
        strategy = classify_query("hello there")
        assert strategy.query_type == QueryType.GENERAL

    def test_general_has_no_overrides(self):
        strategy = classify_query("what is this about")
        assert strategy.text_weight is None
        assert strategy.vector_weight is None
        assert strategy.mmr_lambda is None
        assert strategy.boost_confirmed is False


# ── apply_strategy / restore_strategy ─────────────────────────────────

class TestApplyRestore:
    """Verify strategy application and rollback on HybridRetriever."""

    def test_apply_changes_weights(self, retriever):
        strategy = RetrievalStrategy(
            query_type=QueryType.NAMED_ENTITY,
            text_weight=0.4,
            vector_weight=0.6,
        )
        original = apply_strategy(retriever, strategy)
        assert retriever.text_weight == pytest.approx(0.4)
        assert retriever.vector_weight == pytest.approx(0.6)
        assert original["text_weight"] == pytest.approx(0.2)
        assert original["vector_weight"] == pytest.approx(0.8)

    def test_restore_reverts_weights(self, retriever):
        strategy = RetrievalStrategy(
            query_type=QueryType.NAMED_ENTITY,
            text_weight=0.4,
            vector_weight=0.6,
        )
        original = apply_strategy(retriever, strategy)
        restore_strategy(retriever, original)
        assert retriever.text_weight == pytest.approx(0.2)
        assert retriever.vector_weight == pytest.approx(0.8)

    def test_apply_mmr_overrides(self, retriever):
        strategy = RetrievalStrategy(
            query_type=QueryType.CROSS_CULTURAL,
            mmr_lambda=0.3,
            mmr_diversity_threshold=0.7,
        )
        original = apply_strategy(retriever, strategy)
        assert retriever.mmr_lambda == pytest.approx(0.3)
        assert retriever.mmr_diversity_threshold == pytest.approx(0.7)
        restore_strategy(retriever, original)
        assert retriever.mmr_lambda == pytest.approx(0.5)
        assert retriever.mmr_diversity_threshold == pytest.approx(0.85)

    def test_apply_none_fields_untouched(self, retriever):
        """Fields left as None in the strategy should not be changed."""
        strategy = RetrievalStrategy(
            query_type=QueryType.FOLLOW_UP,
            boost_confirmed=True,
        )
        original = apply_strategy(retriever, strategy)
        # Weights should remain at defaults
        assert retriever.text_weight == pytest.approx(0.2)
        assert retriever.vector_weight == pytest.approx(0.8)
        assert original == {}


# ── RetrievalStrategy.describe() ──────────────────────────────────────

class TestDescribe:
    """Test human-readable strategy description."""

    def test_named_entity_describe(self):
        s = RetrievalStrategy(
            query_type=QueryType.NAMED_ENTITY,
            text_weight=0.4,
            vector_weight=0.6,
            confidence=0.7,
        )
        desc = s.describe()
        assert "NAMED_ENTITY" in desc
        assert "fts=40%" in desc
        assert "vec=60%" in desc

    def test_followup_describe_includes_boost(self):
        s = RetrievalStrategy(
            query_type=QueryType.FOLLOW_UP,
            boost_confirmed=True,
            confidence=0.8,
        )
        desc = s.describe()
        assert "boost_confirmed" in desc

    def test_general_describe_minimal(self):
        s = RetrievalStrategy(query_type=QueryType.GENERAL)
        desc = s.describe()
        assert "GENERAL" in desc
        # No weight overrides
        assert "fts=" not in desc


# ── Edge cases ────────────────────────────────────────────────────────

class TestEdgeCases:
    """Edge cases and priority resolution."""

    def test_entity_with_thematic_word_entity_wins(self):
        """Entity + 1 thematic term — entity has higher per-hit weight."""
        strategy = classify_query("Odin's sacrifice")
        # odin → entity (0.35), sacrifice → thematic (0.30)
        assert strategy.query_type == QueryType.NAMED_ENTITY

    def test_followup_overrides_entity(self):
        """Follow-up patterns take priority over entity detection."""
        strategy = classify_query("You said earlier that Odin lost his eye")
        # "you said" + "earlier" = 2 follow-up hits → 0.8 score
        assert strategy.query_type == QueryType.FOLLOW_UP

    def test_cross_cultural_overrides_single_entity(self):
        """Two culture names + comparison signal → cross-cultural."""
        strategy = classify_query("Compare Odin and Zeus")
        assert strategy.query_type == QueryType.CROSS_CULTURAL

    def test_heavily_thematic_beats_single_entity(self):
        """Many thematic keywords should outweigh a single entity mention."""
        strategy = classify_query(
            "themes of death sacrifice rebirth and redemption in Osiris mythology"
        )
        # thematic: death, sacrifice, rebirth, redemption = 4 * 0.3 = 1.0 (capped)
        # entity: osiris = 1 * 0.35 = 0.35
        assert strategy.query_type == QueryType.THEMATIC

    def test_case_insensitivity(self):
        """Query classification should be case-insensitive."""
        s1 = classify_query("ODIN")
        s2 = classify_query("odin")
        assert s1.query_type == s2.query_type

    def test_confidence_increases_with_more_signals(self):
        """More matching keywords → higher confidence."""
        s1 = classify_query("death")
        s2 = classify_query("death sacrifice rebirth")
        assert s2.confidence > s1.confidence
