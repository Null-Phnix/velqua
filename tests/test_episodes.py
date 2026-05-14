"""
Unit tests for episode ingestion and retrieval.

Tests the episode import endpoint, episode scoring (temporal decay +
emotional weighting), and the modified memory context builder that
interleaves facts and episodes.

These tests verify the cognitive features that make Velqua different
from vanilla RAG: recency weighting, emotional relevance, and temporal
decay for episodic memories.
"""
import os
import tempfile
import importlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

import pytest

# Temp DB setup — must happen before proxy imports
_tmpdir = tempfile.mkdtemp()
os.environ["VELQUA_DB_PATH"] = os.path.join(_tmpdir, "test_episodes.db")

import backend.config
importlib.reload(backend.config)

from backend.proxy import (
    _build_memory_context,
    config,
    score_episode_freshness,
    score_fact_freshness,
)
from anamnesis.models import EmotionalValence


# -- Fake objects for testing --

@dataclass
class FakeEpisode:
    """Minimal episode-like object for testing score_episode_freshness."""
    summary: str = "User discussed deployment issues"
    started_at: Optional[datetime] = None
    overall_valence: EmotionalValence = EmotionalValence.NEUTRAL
    importance: float = 0.5
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FakeFact:
    """Minimal fact-like object for testing."""
    content: str = "User works as a developer"
    last_confirmed: Optional[datetime] = None
    first_learned: Optional[datetime] = None
    confirmation_count: int = 1
    importance: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)


# ==================================================================
# score_episode_freshness
# ==================================================================

class TestScoreEpisodeFreshness:
    """Episode-specific decay scoring with emotional weighting."""

    def test_recent_episode_scores_high(self):
        """An episode from just now should score near its importance."""
        ep = FakeEpisode(started_at=datetime.now(), importance=0.8)
        score = score_episode_freshness(ep)
        assert score > 0.7, f"Recent episode should score high, got {score}"

    def test_old_episode_scores_low(self):
        """An episode from 60 days ago should have decayed significantly."""
        ep = FakeEpisode(
            started_at=datetime.now() - timedelta(days=60),
            importance=0.8,
        )
        score = score_episode_freshness(ep)
        assert score < 0.5, f"Old episode should decay, got {score}"

    def test_emotional_episode_decays_slower(self):
        """Emotionally charged episodes should persist longer than neutral ones."""
        base_time = datetime.now() - timedelta(days=14)

        neutral = FakeEpisode(
            started_at=base_time,
            overall_valence=EmotionalValence.NEUTRAL,
            importance=0.7,
        )
        emotional = FakeEpisode(
            started_at=base_time,
            overall_valence=EmotionalValence.VERY_POSITIVE,
            importance=0.7,
        )
        negative = FakeEpisode(
            started_at=base_time,
            overall_valence=EmotionalValence.VERY_NEGATIVE,
            importance=0.7,
        )

        neutral_score = score_episode_freshness(neutral)
        emotional_score = score_episode_freshness(emotional)
        negative_score = score_episode_freshness(negative)

        assert emotional_score > neutral_score, (
            f"Positive emotion should boost persistence: {emotional_score} vs {neutral_score}"
        )
        assert negative_score > neutral_score, (
            f"Negative emotion should also boost persistence: {negative_score} vs {neutral_score}"
        )

    def test_importance_affects_score(self):
        """Higher importance episodes should score higher."""
        base_time = datetime.now() - timedelta(days=7)

        low = FakeEpisode(started_at=base_time, importance=0.2)
        high = FakeEpisode(started_at=base_time, importance=0.9)

        assert score_episode_freshness(high) > score_episode_freshness(low)

    def test_no_timestamp_treated_as_fresh(self):
        """Episodes without timestamps default to age=0 (treat as fresh)."""
        ep = FakeEpisode(started_at=None, importance=0.5)
        score = score_episode_freshness(ep)
        assert score > 0.4, f"No-timestamp episode should be treated as fresh, got {score}"

    def test_score_bounded_zero_to_one(self):
        """Score should always be in [0, 1]."""
        for valence in EmotionalValence:
            for age_days in [0, 1, 7, 30, 365]:
                for importance in [0.0, 0.5, 1.0]:
                    ep = FakeEpisode(
                        started_at=datetime.now() - timedelta(days=age_days),
                        overall_valence=valence,
                        importance=importance,
                    )
                    score = score_episode_freshness(ep)
                    assert 0.0 <= score <= 1.0, f"Score out of bounds: {score}"

    def test_episodes_decay_faster_than_facts(self):
        """Episodes should decay faster than facts (shorter halflife)."""
        base_time = datetime.now() - timedelta(days=21)

        episode = FakeEpisode(started_at=base_time, importance=0.7)
        fact = FakeFact(last_confirmed=base_time, importance=0.7)

        episode_score = score_episode_freshness(episode)
        fact_score = score_fact_freshness(fact)

        assert episode_score < fact_score, (
            f"Episodes should decay faster: episode={episode_score} fact={fact_score}"
        )

    def test_integer_valence_from_metadata(self):
        """Episode with integer valence in metadata (from HybridSearchResult)."""
        ep = FakeEpisode(
            started_at=datetime.now() - timedelta(days=7),
            importance=0.6,
        )
        # Simulate a HybridSearchResult where valence is stored as int
        ep.overall_valence = 2  # VERY_POSITIVE as raw int
        score = score_episode_freshness(ep)
        assert score > 0.0

    def test_access_count_boosts_score(self):
        """Frequently accessed episodes should score slightly higher."""
        base_time = datetime.now() - timedelta(days=10)

        no_access = FakeEpisode(started_at=base_time, importance=0.5, access_count=0)
        many_access = FakeEpisode(started_at=base_time, importance=0.5, access_count=20)

        assert score_episode_freshness(many_access) > score_episode_freshness(no_access)


# ==================================================================
# _build_memory_context with episodes
# ==================================================================

class TestBuildMemoryContextWithEpisodes:
    """Context builder should interleave episodes and facts correctly."""

    def test_facts_only_backward_compatible(self):
        """Without episodes, context should work exactly as before."""
        facts = ["User is a developer", "User uses Python"]
        context, facts_used, episodes_used = _build_memory_context(facts)

        assert facts_used == 2
        assert episodes_used == 0
        assert "User is a developer" in context
        assert "User uses Python" in context
        assert "[Recent experiences:]" not in context

    def test_episodes_and_facts(self):
        """With both, context should have separate sections."""
        facts = ["User is a developer"]
        episodes = [("User discussed deployment issues", 0.8)]

        context, facts_used, episodes_used = _build_memory_context(
            facts, episode_contents=episodes
        )

        assert episodes_used == 1
        assert facts_used == 1
        assert "[Recent experiences:]" in context
        assert "[Known facts:]" in context
        assert "deployment issues" in context
        assert "developer" in context

    def test_episodes_only(self):
        """With only episodes and no facts, should still produce context."""
        episodes = [("User expressed frustration", 0.9)]

        context, facts_used, episodes_used = _build_memory_context(
            [], episode_contents=episodes
        )

        assert episodes_used == 1
        assert facts_used == 0
        assert "frustration" in context

    def test_empty_everything(self):
        """No facts, no episodes = empty context."""
        context, facts_used, episodes_used = _build_memory_context(
            [], episode_contents=[]
        )
        assert context == ""
        assert facts_used == 0
        assert episodes_used == 0

    def test_token_budget_respected(self):
        """Total context should not exceed the token budget."""
        # Set a small budget
        original_budget = config.max_tokens
        config.max_tokens = 30

        try:
            facts = [f"Fact number {i} with some extra words to fill budget" for i in range(20)]
            episodes = [(f"Episode {i} with details about what happened", 0.5) for i in range(10)]

            context, facts_used, episodes_used = _build_memory_context(
                facts, episode_contents=episodes
            )

            word_count = len(context.split())
            assert word_count <= 40, f"Context ({word_count} words) exceeds budget"
            # Should have used some but not all
            assert episodes_used + facts_used > 0
            assert episodes_used + facts_used < 30
        finally:
            config.max_tokens = original_budget

    def test_episodes_get_priority_share(self):
        """Episodes should get their configured share of the budget."""
        original_budget = config.max_tokens
        config.max_tokens = 50

        try:
            facts = [f"Important fact {i}" for i in range(10)]
            episodes = [(f"Critical episode {i}", 0.9) for i in range(10)]

            context, facts_used, episodes_used = _build_memory_context(
                facts, episode_contents=episodes
            )

            # Both should be present
            assert episodes_used > 0, "Episodes should have been included"
            assert facts_used > 0, "Facts should have been included"
        finally:
            config.max_tokens = original_budget

    def test_none_episodes_backward_compat(self):
        """Passing episode_contents=None should behave like facts-only."""
        facts = ["User likes coffee"]
        context, facts_used, episodes_used = _build_memory_context(
            facts, episode_contents=None
        )
        assert facts_used == 1
        assert episodes_used == 0
        assert "coffee" in context


# ==================================================================
# Episode route models (validation)
# ==================================================================

class TestEpisodeChunkValidation:
    """Test Pydantic validation for the episode import schema."""

    def test_valid_chunk(self):
        from backend.routes.episodes import EpisodeChunk
        chunk = EpisodeChunk(
            content="User discussed their deployment pipeline",
            timestamp="2026-03-20T14:30:00",
            emotional_valence="negative",
            source_agent="dev-assistant",
            decay_rate=0.15,
            importance=0.7,
            topic="devops",
            tags=["deployment"],
        )
        assert chunk.content == "User discussed their deployment pipeline"
        assert chunk.emotional_valence == "negative"
        assert chunk.importance == 0.7

    def test_minimal_chunk(self):
        from backend.routes.episodes import EpisodeChunk
        chunk = EpisodeChunk(content="A short but valid episode content")
        assert chunk.emotional_valence == "neutral"
        assert chunk.source_agent == "unknown"
        assert chunk.importance == 0.5

    def test_rejects_short_content(self):
        from backend.routes.episodes import EpisodeChunk
        with pytest.raises(Exception):  # Pydantic ValidationError
            EpisodeChunk(content="short")

    def test_rejects_invalid_valence(self):
        from backend.routes.episodes import EpisodeChunk
        with pytest.raises(Exception):
            EpisodeChunk(
                content="Valid content for an episode chunk",
                emotional_valence="ecstatic",
            )

    def test_all_valences_accepted(self):
        from backend.routes.episodes import EpisodeChunk
        for v in ["very_negative", "negative", "neutral", "positive", "very_positive"]:
            chunk = EpisodeChunk(
                content="Valid content for testing valences",
                emotional_valence=v,
            )
            assert chunk.emotional_valence == v


# ==================================================================
# Valence mapping
# ==================================================================

class TestValenceMapping:
    """Verify string-to-enum valence mapping."""

    def test_all_valences_mapped(self):
        from backend.routes.episodes import _VALENCE_MAP
        assert _VALENCE_MAP["very_negative"] == EmotionalValence.VERY_NEGATIVE
        assert _VALENCE_MAP["negative"] == EmotionalValence.NEGATIVE
        assert _VALENCE_MAP["neutral"] == EmotionalValence.NEUTRAL
        assert _VALENCE_MAP["positive"] == EmotionalValence.POSITIVE
        assert _VALENCE_MAP["very_positive"] == EmotionalValence.VERY_POSITIVE

    def test_coverage(self):
        """Every EmotionalValence enum value should be in the map."""
        from backend.routes.episodes import _VALENCE_MAP
        mapped_values = set(_VALENCE_MAP.values())
        for v in EmotionalValence:
            assert v in mapped_values, f"{v} not mapped"


# ==================================================================
# Episode serialization
# ==================================================================

class TestEpisodeSerialization:
    """Test the _serialize_episode helper."""

    def test_serialize_basic(self):
        from backend.routes.episodes import _serialize_episode
        from anamnesis.models import Episode

        ep = Episode(
            id="test-123",
            summary="User deployed successfully",
            topic="devops",
            started_at=datetime(2026, 3, 20, 14, 30),
            overall_valence=EmotionalValence.POSITIVE,
            importance=0.8,
            tags=["deployment", "success"],
            access_count=3,
            source_id="dev-agent",
        )

        result = _serialize_episode(ep)
        assert result["id"] == "test-123"
        assert result["summary"] == "User deployed successfully"
        assert result["topic"] == "devops"
        assert result["overall_valence"] == "positive"
        assert result["importance"] == 0.8
        assert result["tags"] == ["deployment", "success"]
        assert result["access_count"] == 3

    def test_serialize_neutral_valence(self):
        from backend.routes.episodes import _serialize_episode
        from anamnesis.models import Episode

        ep = Episode(summary="Neutral event")
        result = _serialize_episode(ep)
        assert result["overall_valence"] == "neutral"


# ==================================================================
# Config values
# ==================================================================

class TestEpisodeConfig:
    """Verify episode config values exist and are sensible."""

    def test_episode_retrieval_limit(self):
        from backend.config import VelquaConfig
        assert VelquaConfig.EPISODE_RETRIEVAL_LIMIT > 0
        assert VelquaConfig.EPISODE_RETRIEVAL_LIMIT <= 50

    def test_episode_decay_faster_than_facts(self):
        from backend.config import VelquaConfig
        assert VelquaConfig.EPISODE_DECAY_HALFLIFE_WEEKS < VelquaConfig.DECAY_HALFLIFE_WEEKS

    def test_emotional_boost_positive(self):
        from backend.config import VelquaConfig
        assert VelquaConfig.EPISODE_EMOTIONAL_BOOST >= 1.0

    def test_token_share_valid(self):
        from backend.config import VelquaConfig
        assert 0.0 < VelquaConfig.EPISODE_TOKEN_SHARE < 1.0


# ==================================================================
# Integration: episode decay model
# ==================================================================

class TestEpisodeDecayModel:
    """Test the episode-specific AdaptiveDecay instance."""

    def test_episode_decay_instance_exists(self):
        from backend.proxy import episode_decay
        assert episode_decay is not None

    def test_episode_decay_shorter_halflife(self):
        from backend.proxy import decay, episode_decay
        assert episode_decay.base_halflife < decay.base_halflife

    def test_episode_decay_stronger_emotion_factor(self):
        from backend.proxy import decay, episode_decay
        assert episode_decay.emotion_factor >= decay.emotion_factor
