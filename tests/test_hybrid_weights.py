"""Tests for hybrid retrieval weight ratio, env-var configurability, and episode-aware decay."""

import math
import os
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from backend.anamnesis.retrieval.hybrid import HybridRetriever


@pytest.fixture
def backend(tmp_path):
    """Create a minimal SQLite backend."""
    from backend.anamnesis.stores.sqlite_backend import SQLiteBackend

    return SQLiteBackend(str(tmp_path / "test.db"))


class TestDefaultWeights:
    """Verify HybridRetriever defaults match the 20/80 ratio."""

    def test_default_text_weight_is_020(self, backend):
        r = HybridRetriever(sqlite_backend=backend)
        assert r.text_weight == pytest.approx(0.2)

    def test_default_vector_weight_is_080(self, backend):
        r = HybridRetriever(sqlite_backend=backend)
        assert r.vector_weight == pytest.approx(0.8)

    def test_weights_sum_to_1(self, backend):
        r = HybridRetriever(sqlite_backend=backend)
        assert r.text_weight + r.vector_weight == pytest.approx(1.0)


class TestCustomWeights:
    """Verify explicit weight overrides work."""

    def test_custom_weights_applied(self, backend):
        r = HybridRetriever(
            sqlite_backend=backend,
            text_weight=0.3,
            vector_weight=0.7,
        )
        assert r.text_weight == pytest.approx(0.3)
        assert r.vector_weight == pytest.approx(0.7)

    def test_zero_text_weight(self, backend):
        """Zero FTS weight → pure vector scoring."""
        r = HybridRetriever(
            sqlite_backend=backend,
            text_weight=0.0,
            vector_weight=1.0,
        )
        assert r.text_weight == pytest.approx(0.0)
        assert r.vector_weight == pytest.approx(1.0)

    def test_zero_vector_weight(self, backend):
        """Zero vector weight → pure FTS scoring."""
        r = HybridRetriever(
            sqlite_backend=backend,
            text_weight=1.0,
            vector_weight=0.0,
        )
        assert r.text_weight == pytest.approx(1.0)
        assert r.vector_weight == pytest.approx(0.0)


class TestConfigEnvVars:
    """Verify VelquaConfig reads VELQUA_FTS_WEIGHT / VELQUA_VECTOR_WEIGHT."""

    @pytest.fixture(autouse=True)
    def _restore_config(self):
        """Reload config after each test to prevent class-attribute pollution."""
        yield
        # Restore config defaults after env-var tests
        from importlib import reload
        import backend.config as cfg
        reload(cfg)

    def test_config_default_fts_weight(self):
        """Without env vars, Config defaults to 0.2."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VELQUA_FTS_WEIGHT", None)
            # Re-import to pick up fresh env
            from importlib import reload
            import backend.config as cfg
            reload(cfg)
            assert cfg.VelquaConfig.FTS_WEIGHT == pytest.approx(0.2)

    def test_config_default_vector_weight(self):
        """Without env vars, Config defaults to 0.8."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VELQUA_VECTOR_WEIGHT", None)
            from importlib import reload
            import backend.config as cfg
            reload(cfg)
            assert cfg.VelquaConfig.VECTOR_WEIGHT == pytest.approx(0.8)

    def test_env_var_overrides_fts_weight(self):
        """VELQUA_FTS_WEIGHT env var overrides default."""
        with patch.dict(os.environ, {"VELQUA_FTS_WEIGHT": "0.35"}):
            from importlib import reload
            import backend.config as cfg
            reload(cfg)
            assert cfg.VelquaConfig.FTS_WEIGHT == pytest.approx(0.35)

    def test_env_var_overrides_vector_weight(self):
        """VELQUA_VECTOR_WEIGHT env var overrides default."""
        with patch.dict(os.environ, {"VELQUA_VECTOR_WEIGHT": "0.65"}):
            from importlib import reload
            import backend.config as cfg
            reload(cfg)
            assert cfg.VelquaConfig.VECTOR_WEIGHT == pytest.approx(0.65)

    def test_env_vars_flow_through_to_config(self):
        """Both env vars together produce the expected config values."""
        with patch.dict(os.environ, {
            "VELQUA_FTS_WEIGHT": "0.10",
            "VELQUA_VECTOR_WEIGHT": "0.90",
        }):
            from importlib import reload
            import backend.config as cfg
            reload(cfg)
            assert cfg.VelquaConfig.FTS_WEIGHT == pytest.approx(0.10)
            assert cfg.VelquaConfig.VECTOR_WEIGHT == pytest.approx(0.90)


class TestScoringFormula:
    """Verify the weighted combination produces correct scores."""

    def test_20_80_weighting(self, backend):
        """Default 20/80: text=1.0, vector=0.5 → 0.2*1.0 + 0.8*0.5 = 0.60."""
        r = HybridRetriever(sqlite_backend=backend)
        expected = 0.2 * 1.0 + 0.8 * 0.5
        assert expected == pytest.approx(0.60)
        # Verify weights stored correctly
        assert r.text_weight * 1.0 + r.vector_weight * 0.5 == pytest.approx(0.60)

    def test_higher_vector_score_dominates(self, backend):
        """With 80% vector weight, vector score has outsized influence."""
        r = HybridRetriever(sqlite_backend=backend)
        # text=0.9, vector=0.1 → 0.2*0.9 + 0.8*0.1 = 0.26
        low_vector = r.text_weight * 0.9 + r.vector_weight * 0.1
        # text=0.1, vector=0.9 → 0.2*0.1 + 0.8*0.9 = 0.74
        high_vector = r.text_weight * 0.1 + r.vector_weight * 0.9
        assert high_vector > low_vector
        assert high_vector == pytest.approx(0.74)
        assert low_vector == pytest.approx(0.26)

    def test_old_40_60_would_dilute_vector(self, backend):
        """Demonstrate the old 40/60 ratio gave FTS too much influence.

        With text=0.3, vector=0.9:
          Old (40/60): 0.4*0.3 + 0.6*0.9 = 0.66
          New (20/80): 0.2*0.3 + 0.8*0.9 = 0.78

        The new ratio better preserves the vector signal.
        """
        r = HybridRetriever(sqlite_backend=backend)  # 0.2/0.8
        new_score = r.text_weight * 0.3 + r.vector_weight * 0.9
        old_score = 0.4 * 0.3 + 0.6 * 0.9  # what 40/60 would give
        assert new_score > old_score
        assert new_score == pytest.approx(0.78)
        assert old_score == pytest.approx(0.66)


class TestEpisodeDecayDifferentiation:
    """Verify episodes decay faster than facts and emotional valence matters."""

    def test_episode_decay_params_stored(self, backend):
        """Episode-specific decay params should be stored on the retriever."""
        r = HybridRetriever(
            sqlite_backend=backend,
            episode_decay_lambda=0.1,
            episode_decay_floor=0.02,
            episode_emotional_boost=1.5,
        )
        assert r.episode_decay_lambda == pytest.approx(0.1)
        assert r.episode_decay_floor == pytest.approx(0.02)
        assert r.episode_emotional_boost == pytest.approx(1.5)

    def test_default_episode_decay_faster_than_facts(self, backend):
        """Default episode lambda should be higher (faster decay) than fact lambda."""
        r = HybridRetriever(sqlite_backend=backend)
        assert r.episode_decay_lambda > r.decay_lambda

    def test_episode_decays_faster_at_same_age(self, backend):
        """An episode and fact with the same age: episode should have lower multiplier."""
        r = HybridRetriever(sqlite_backend=backend)
        age_30_days = (datetime.now() - timedelta(days=30)).isoformat()

        fact_meta = {"last_confirmed": age_30_days, "confirmation_count": 1}
        episode_meta = {"started_at": age_30_days, "overall_valence": 0}

        fact_decay = r._compute_decay(fact_meta, "fact")
        episode_decay = r._compute_decay(episode_meta, "episode")

        assert episode_decay < fact_decay, (
            f"Episodes should decay faster: episode={episode_decay} fact={fact_decay}"
        )

    def test_emotional_episode_decays_slower_than_neutral(self, backend):
        """Emotionally charged episodes should resist decay more than neutral ones."""
        r = HybridRetriever(sqlite_backend=backend)
        age = (datetime.now() - timedelta(days=14)).isoformat()

        neutral = {"started_at": age, "overall_valence": 0}
        very_negative = {"started_at": age, "overall_valence": -2}
        very_positive = {"started_at": age, "overall_valence": 2}

        neutral_decay = r._compute_decay(neutral, "episode")
        neg_decay = r._compute_decay(very_negative, "episode")
        pos_decay = r._compute_decay(very_positive, "episode")

        assert neg_decay > neutral_decay, (
            f"Negative emotion should slow decay: {neg_decay} vs {neutral_decay}"
        )
        assert pos_decay > neutral_decay, (
            f"Positive emotion should slow decay: {pos_decay} vs {neutral_decay}"
        )
        # Positive and negative of same intensity should have same decay
        assert neg_decay == pytest.approx(pos_decay)

    def test_custom_decay_rate_from_metadata(self, backend):
        """Episode with custom decay_rate in metadata should use it."""
        r = HybridRetriever(sqlite_backend=backend)
        age = (datetime.now() - timedelta(days=7)).isoformat()

        # Very slow custom decay
        slow = {"started_at": age, "overall_valence": 0, "decay_rate": 0.001}
        # Very fast custom decay
        fast = {"started_at": age, "overall_valence": 0, "decay_rate": 0.5}

        slow_decay = r._compute_decay(slow, "episode")
        fast_decay = r._compute_decay(fast, "episode")

        assert slow_decay > fast_decay, (
            f"Lower decay_rate should mean slower decay: slow={slow_decay} fast={fast_decay}"
        )

    def test_episode_floor_lower_than_fact_floor(self, backend):
        """Episode decay floor should be lower — episodes can fade to near-zero."""
        r = HybridRetriever(sqlite_backend=backend)
        assert r.episode_decay_floor < r.decay_floor

    def test_very_old_episode_hits_floor(self, backend):
        """A very old episode should hit the episode floor, not the fact floor."""
        r = HybridRetriever(sqlite_backend=backend)
        old = (datetime.now() - timedelta(days=365)).isoformat()

        episode_meta = {"started_at": old, "overall_valence": 0}
        decay = r._compute_decay(episode_meta, "episode")

        assert decay == pytest.approx(r.episode_decay_floor), (
            f"Old episode should be at floor: {decay} != {r.episode_decay_floor}"
        )

    def test_very_old_fact_hits_fact_floor(self, backend):
        """A very old fact should hit the fact floor (higher than episode floor)."""
        r = HybridRetriever(sqlite_backend=backend)
        old = (datetime.now() - timedelta(days=365)).isoformat()

        fact_meta = {"last_confirmed": old, "confirmation_count": 1}
        decay = r._compute_decay(fact_meta, "fact")

        assert decay == pytest.approx(r.decay_floor), (
            f"Old fact should be at floor: {decay} != {r.decay_floor}"
        )

    def test_access_count_slows_episode_decay(self, backend):
        """Frequently accessed episodes should decay slower."""
        r = HybridRetriever(sqlite_backend=backend)
        age = (datetime.now() - timedelta(days=10)).isoformat()

        no_access = {"started_at": age, "overall_valence": 0, "access_count": 0}
        many_access = {"started_at": age, "overall_valence": 0, "access_count": 50}

        no_decay = r._compute_decay(no_access, "episode")
        many_decay = r._compute_decay(many_access, "episode")

        assert many_decay > no_decay, (
            f"More access should slow decay: many={many_decay} none={no_decay}"
        )

    def test_no_temporal_info_returns_1(self, backend):
        """Missing timestamps should return 1.0 (no decay) for both types."""
        r = HybridRetriever(sqlite_backend=backend)

        assert r._compute_decay({}, "episode") == 1.0
        assert r._compute_decay({}, "fact") == 1.0

    def test_fact_confirmation_still_works(self, backend):
        """Existing fact confirmation weighting should still function."""
        r = HybridRetriever(sqlite_backend=backend)
        age = (datetime.now() - timedelta(days=30)).isoformat()

        low_conf = {"last_confirmed": age, "confirmation_count": 1}
        high_conf = {"last_confirmed": age, "confirmation_count": 50}

        low_decay = r._compute_decay(low_conf, "fact")
        high_decay = r._compute_decay(high_conf, "fact")

        assert high_decay > low_decay, (
            f"Higher confirmation should resist decay: high={high_decay} low={low_decay}"
        )
