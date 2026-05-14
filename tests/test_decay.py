"""Tests for adaptive temporal decay in hybrid retrieval scoring."""

import math
from datetime import datetime, timedelta

import pytest

from backend.anamnesis.retrieval.hybrid import HybridRetriever


@pytest.fixture
def retriever(tmp_path):
    """Create a HybridRetriever with an in-memory SQLite backend."""
    from backend.anamnesis.stores.sqlite_backend import SQLiteBackend

    backend = SQLiteBackend(str(tmp_path / "test.db"))
    return HybridRetriever(
        sqlite_backend=backend,
        decay_lambda=0.01,
        decay_floor=0.1,
    )


class TestComputeDecay:
    """Unit tests for _compute_decay."""

    def test_no_temporal_info_returns_1(self, retriever):
        """No timestamps → no decay applied."""
        assert retriever._compute_decay({}, "fact") == 1.0

    def test_just_accessed_returns_near_1(self, retriever):
        """Just-accessed memory should have ~1.0 multiplier."""
        meta = {"last_accessed": datetime.now().isoformat()}
        decay = retriever._compute_decay(meta, "fact")
        assert decay > 0.99

    def test_70_days_halves_score(self, retriever):
        """Lambda 0.01 → half-life ~69.3 days ≈ 70 days."""
        past = datetime.now() - timedelta(days=70)
        meta = {"last_accessed": past.isoformat()}
        decay = retriever._compute_decay(meta, "fact")
        # exp(-0.01 * 70) ≈ 0.4966
        assert 0.45 < decay < 0.55

    def test_200_days_hits_floor(self, retriever):
        """After 200+ days with lambda=0.01, decay should hit the floor."""
        past = datetime.now() - timedelta(days=300)
        meta = {"last_accessed": past.isoformat()}
        decay = retriever._compute_decay(meta, "fact")
        # exp(-0.01 * 300) ≈ 0.0498, clamped to floor 0.1
        assert decay == pytest.approx(0.1)

    def test_confirmation_count_resists_decay(self, retriever):
        """High confirmation_count should slow effective decay."""
        past = datetime.now() - timedelta(days=70)
        meta_low = {"last_accessed": past.isoformat(), "confirmation_count": 0}
        meta_high = {"last_accessed": past.isoformat(), "confirmation_count": 20}

        decay_low = retriever._compute_decay(meta_low, "fact")
        decay_high = retriever._compute_decay(meta_high, "fact")

        # More confirmations → higher multiplier (slower decay)
        assert decay_high > decay_low

    def test_episode_falls_back_to_started_at(self, retriever):
        """Episodes without last_accessed use started_at."""
        past = datetime.now() - timedelta(days=30)
        meta = {"started_at": past.isoformat()}
        decay = retriever._compute_decay(meta, "episode")
        expected = max(
            retriever.episode_decay_floor,
            math.exp(-retriever.episode_decay_lambda * 30),
        )
        assert decay == pytest.approx(expected, abs=0.01)

    def test_fact_falls_back_to_first_learned(self, retriever):
        """Facts without last_accessed/last_confirmed use first_learned."""
        past = datetime.now() - timedelta(days=50)
        meta = {"first_learned": past.isoformat()}
        decay = retriever._compute_decay(meta, "fact")
        expected = math.exp(-0.01 * 50)
        assert decay == pytest.approx(expected, abs=0.01)

    def test_last_confirmed_preferred_over_first_learned(self, retriever):
        """last_confirmed takes priority (it's checked first in the or-chain)."""
        old = datetime.now() - timedelta(days=100)
        recent = datetime.now() - timedelta(days=5)
        meta = {
            "first_learned": old.isoformat(),
            "last_confirmed": recent.isoformat(),
        }
        decay = retriever._compute_decay(meta, "fact")
        # Should use last_confirmed (5 days ago), not first_learned (100 days)
        expected = math.exp(-0.01 * 5)
        assert decay == pytest.approx(expected, abs=0.01)

    def test_invalid_date_returns_1(self, retriever):
        """Malformed date strings should not crash, just skip decay."""
        meta = {"last_accessed": "not-a-date"}
        assert retriever._compute_decay(meta, "fact") == 1.0

    def test_custom_lambda(self, tmp_path):
        """Different lambda values change half-life."""
        from backend.anamnesis.stores.sqlite_backend import SQLiteBackend

        backend = SQLiteBackend(str(tmp_path / "test.db"))
        fast = HybridRetriever(backend, decay_lambda=0.1, decay_floor=0.01)
        slow = HybridRetriever(backend, decay_lambda=0.001, decay_floor=0.01)

        past = datetime.now() - timedelta(days=30)
        meta = {"last_accessed": past.isoformat()}

        assert fast._compute_decay(meta, "fact") < slow._compute_decay(meta, "fact")

    def test_floor_configurable(self, tmp_path):
        """decay_floor parameter is respected."""
        from backend.anamnesis.stores.sqlite_backend import SQLiteBackend

        backend = SQLiteBackend(str(tmp_path / "test.db"))
        r = HybridRetriever(backend, decay_lambda=0.01, decay_floor=0.5)

        past = datetime.now() - timedelta(days=300)
        meta = {"last_accessed": past.isoformat()}
        assert r._compute_decay(meta, "fact") == pytest.approx(0.5)


class TestDecayInSearch:
    """Integration: decay affects search result ordering."""

    def test_recent_fact_outranks_stale_fact(self, retriever):
        """A recently accessed fact should rank above an equally relevant stale one."""
        backend = retriever.sqlite

        now = datetime.now()
        old = now - timedelta(days=120)

        backend.save_fact({
            "id": "stale",
            "content": "The user likes Python programming",
            "fact_type": "preference",
            "confidence": 0.9,
            "importance": 0.8,
            "last_accessed": old.isoformat(),
            "last_confirmed": old.isoformat(),
            "confirmation_count": 1,
        })
        backend.save_fact({
            "id": "fresh",
            "content": "The user likes Python development",
            "fact_type": "preference",
            "confidence": 0.9,
            "importance": 0.8,
            "last_accessed": now.isoformat(),
            "last_confirmed": now.isoformat(),
            "confirmation_count": 1,
        })

        from backend.anamnesis.retrieval.hybrid import SearchMode

        results = retriever.search(
            "Python", limit=10, mode=SearchMode.TEXT_ONLY, search_episodes=False
        )

        assert len(results) >= 2
        ids = [r.id for r in results]
        assert ids.index("fresh") < ids.index("stale")

    def test_highly_confirmed_stale_fact_resists_decay(self, retriever):
        """A stale fact with many confirmations should decay slower."""
        backend = retriever.sqlite

        now = datetime.now()
        old = now - timedelta(days=90)

        backend.save_fact({
            "id": "confirmed_stale",
            "content": "The user enjoys Rust systems programming",
            "fact_type": "preference",
            "confidence": 0.9,
            "importance": 0.8,
            "last_accessed": old.isoformat(),
            "last_confirmed": old.isoformat(),
            "confirmation_count": 50,
        })
        backend.save_fact({
            "id": "unconfirmed_stale",
            "content": "The user enjoys Rust language development",
            "fact_type": "preference",
            "confidence": 0.9,
            "importance": 0.8,
            "last_accessed": old.isoformat(),
            "last_confirmed": old.isoformat(),
            "confirmation_count": 1,
        })

        from backend.anamnesis.retrieval.hybrid import SearchMode

        results = retriever.search(
            "Rust", limit=10, mode=SearchMode.TEXT_ONLY, search_episodes=False
        )

        assert len(results) >= 2
        ids = [r.id for r in results]
        assert ids.index("confirmed_stale") < ids.index("unconfirmed_stale")
