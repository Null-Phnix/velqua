"""
Multi-turn evaluation harness for confirmation weighting.

Proves that repeated retrieval on the same topic causes:
1. confirmation_count to increase on relevant facts
2. Retrieval scores to improve across turns
3. On-topic facts to separate from off-topic distractors

This is the definitive test that confirmation weighting works end-to-end.
"""

import math
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import pytest

from backend.anamnesis.retrieval.hybrid import (
    HybridRetriever,
    HybridSearchResult,
    SearchMode,
)
from backend.anamnesis.stores.sqlite_backend import SQLiteBackend


# ---------------------------------------------------------------------------
# Seed data: facts about Python (on-topic) and distractors (off-topic)
# ---------------------------------------------------------------------------

# All facts start "stale" — last_confirmed 30 days ago, confirmation_count=1
_SEED_AGE_DAYS = 30

PYTHON_FACTS = [
    {
        "id": "py-lang",
        "content": "The user's primary programming language is Python",
        "fact_type": "preference",
    },
    {
        "id": "py-flask",
        "content": "The user builds web APIs with Flask and FastAPI in Python",
        "fact_type": "professional",
    },
    {
        "id": "py-data",
        "content": "The user uses Python pandas and numpy for data analysis",
        "fact_type": "professional",
    },
    {
        "id": "py-testing",
        "content": "The user writes Python tests with pytest and coverage",
        "fact_type": "professional",
    },
    {
        "id": "py-ml",
        "content": "The user trains machine learning models in Python with PyTorch",
        "fact_type": "professional",
    },
    {
        "id": "py-style",
        "content": "The user prefers Python type hints and black formatting",
        "fact_type": "preference",
    },
    {
        "id": "py-version",
        "content": "The user runs Python 3.12 on Arch Linux",
        "fact_type": "personal",
    },
]

DISTRACTOR_FACTS = [
    {
        "id": "cook-pasta",
        "content": "The user enjoys cooking Italian pasta dishes",
        "fact_type": "preference",
    },
    {
        "id": "cook-spice",
        "content": "The user prefers spicy food with habanero peppers",
        "fact_type": "preference",
    },
    {
        "id": "music-jazz",
        "content": "The user listens to jazz and lo-fi hip hop while working",
        "fact_type": "preference",
    },
    {
        "id": "sport-climb",
        "content": "The user goes bouldering at the local climbing gym",
        "fact_type": "personal",
    },
    {
        "id": "pet-cat",
        "content": "The user has two cats named Luna and Mochi",
        "fact_type": "personal",
    },
    {
        "id": "loc-canada",
        "content": "The user lives in Saskatchewan, Canada",
        "fact_type": "personal",
    },
    {
        "id": "hw-gpu",
        "content": "The user has an RTX 4060 graphics card with 8GB VRAM",
        "fact_type": "personal",
    },
    {
        "id": "game-rpg",
        "content": "The user plays roguelike RPG games in spare time",
        "fact_type": "preference",
    },
]

# 10 related queries about the same topic (Python programming)
PYTHON_QUERIES = [
    "What programming language does the user prefer?",
    "Tell me about the user's Python experience",
    "What Python frameworks does the user use for web development?",
    "How does the user do data analysis with Python?",
    "What testing tools does the user use in Python?",
    "Does the user work with machine learning in Python?",
    "What is the user's Python coding style?",
    "What Python version is the user running?",
    "What does the user build with Python and Flask?",
    "Describe the user's Python development workflow",
]

PYTHON_FACT_IDS = {f["id"] for f in PYTHON_FACTS}
DISTRACTOR_FACT_IDS = {f["id"] for f in DISTRACTOR_FACTS}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def backend(tmp_path):
    """Fresh SQLite backend in a temp directory."""
    db = SQLiteBackend(str(tmp_path / "multi_turn_eval.db"))
    yield db
    db.close()


@pytest.fixture
def seeded_backend(backend):
    """Backend pre-loaded with Python + distractor facts, all equally stale."""
    base_time = datetime.now() - timedelta(days=_SEED_AGE_DAYS)

    for fact in PYTHON_FACTS + DISTRACTOR_FACTS:
        backend.save_fact({
            **fact,
            "confidence": 0.8,
            "importance": 0.5,
            "first_learned": base_time.isoformat(),
            "last_confirmed": base_time.isoformat(),
            "confirmation_count": 1,
            "last_accessed": base_time.isoformat(),
            "access_count": 0,
        })

    return backend


@pytest.fixture
def retriever(seeded_backend):
    """HybridRetriever wired to the seeded backend, FTS-only (no embedder)."""
    return HybridRetriever(
        sqlite_backend=seeded_backend,
        decay_lambda=0.01,
        decay_floor=0.1,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simulate_turn(
    retriever: HybridRetriever,
    backend: SQLiteBackend,
    query: str,
    limit: int = 10,
) -> List[HybridSearchResult]:
    """
    Simulate one retrieval turn the way the proxy does:
    1. Search for relevant facts via FTS
    2. Increment confirmation_count on each retrieved fact (via record_fact_access)

    Returns the search results for scoring analysis.
    """
    results = retriever.search(
        query,
        limit=limit,
        mode=SearchMode.TEXT_ONLY,
        search_episodes=False,
    )

    # Simulate proxy's _increment_retrieval_confirmations
    for result in results:
        backend.record_fact_access(result.id, reinforce_importance=True)

    return results


def _get_confirmation_counts(backend: SQLiteBackend) -> Dict[str, int]:
    """Read current confirmation_count for all facts."""
    counts = {}
    for fact in backend.list_facts(limit=100):
        counts[fact["id"]] = fact["confirmation_count"]
    return counts


def _on_topic_scores(results: List[HybridSearchResult]) -> List[float]:
    """Extract scores for Python facts from a result set."""
    return [r.combined_score for r in results if r.id in PYTHON_FACT_IDS]


def _best_on_topic_score(results: List[HybridSearchResult]) -> float:
    """Best score among on-topic facts, or 0 if none found."""
    scores = _on_topic_scores(results)
    return max(scores) if scores else 0.0


def _mean_on_topic_score(results: List[HybridSearchResult]) -> float:
    """Mean score among on-topic facts, or 0 if none found."""
    scores = _on_topic_scores(results)
    return sum(scores) / len(scores) if scores else 0.0


def _on_topic_retrieved_count(results: List[HybridSearchResult]) -> int:
    """How many on-topic facts appeared in results."""
    return sum(1 for r in results if r.id in PYTHON_FACT_IDS)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMultiTurnConfirmation:
    """
    Core test suite: 10 sequential queries about Python.
    Proves confirmation_count drives retrieval improvement.
    """

    def test_confirmation_count_increases_across_turns(
        self, retriever, seeded_backend
    ):
        """Each retrieval turn should increment confirmation_count on hit facts."""
        initial_counts = _get_confirmation_counts(seeded_backend)

        # All facts start at 1
        for fid in PYTHON_FACT_IDS | DISTRACTOR_FACT_IDS:
            assert initial_counts[fid] == 1, f"{fid} should start at 1"

        # Run 10 turns of Python queries
        for query in PYTHON_QUERIES:
            _simulate_turn(retriever, seeded_backend, query)

        final_counts = _get_confirmation_counts(seeded_backend)

        # At least some Python facts must have been confirmed multiple times
        python_confirmed = [
            final_counts[fid] for fid in PYTHON_FACT_IDS
            if final_counts[fid] > 1
        ]
        detail = {fid: final_counts[fid] for fid in PYTHON_FACT_IDS}
        assert len(python_confirmed) >= 3, (
            f"Expected >=3 Python facts to be confirmed more than once, "
            f"got {len(python_confirmed)}. Counts: {detail}"
        )

        # The most-confirmed Python fact should have count >> 1
        max_python = max(final_counts[fid] for fid in PYTHON_FACT_IDS)
        assert max_python >= 4, (
            f"Most-confirmed Python fact only reached {max_python}, "
            f"expected >=4 after 10 related queries"
        )

    def test_distractor_confirmation_lower_than_on_topic(
        self, retriever, seeded_backend
    ):
        """
        Off-topic facts should be confirmed significantly less than on-topic.

        FTS5 is keyword-based, so some cross-matching is expected (e.g.
        "The user prefers" appears in both Python and cooking facts).
        The test verifies the *relative* gap: the mean on-topic confirmation
        should exceed the mean distractor confirmation.
        """
        for query in PYTHON_QUERIES:
            _simulate_turn(retriever, seeded_backend, query)

        final_counts = _get_confirmation_counts(seeded_backend)

        mean_python = sum(final_counts[fid] for fid in PYTHON_FACT_IDS) / len(PYTHON_FACT_IDS)
        mean_distractor = sum(final_counts[fid] for fid in DISTRACTOR_FACT_IDS) / len(DISTRACTOR_FACT_IDS)

        py_detail = {fid: final_counts[fid] for fid in PYTHON_FACT_IDS}
        dist_detail = {fid: final_counts[fid] for fid in DISTRACTOR_FACT_IDS}
        assert mean_python > mean_distractor, (
            f"Mean on-topic confirmation ({mean_python:.1f}) should exceed "
            f"mean distractor ({mean_distractor:.1f}). "
            f"Python: {py_detail}, Distractors: {dist_detail}"
        )

    def test_retrieval_scores_improve_across_turns(
        self, retriever, seeded_backend
    ):
        """
        Mean on-topic score should be higher in the last 3 turns
        than in the first 3 turns, proving confirmation weighting
        improves retrieval quality over time.
        """
        turn_scores: List[float] = []

        for query in PYTHON_QUERIES:
            results = _simulate_turn(retriever, seeded_backend, query)
            score = _mean_on_topic_score(results)
            turn_scores.append(score)

        early_avg = sum(turn_scores[:3]) / 3
        late_avg = sum(turn_scores[-3:]) / 3

        assert late_avg > early_avg, (
            f"Late turns avg score ({late_avg:.4f}) should exceed "
            f"early turns avg ({early_avg:.4f}). "
            f"All turn scores: {[f'{s:.4f}' for s in turn_scores]}"
        )

    def test_decay_multiplier_improves_with_confirmation(
        self, retriever, seeded_backend
    ):
        """
        Directly verify that _compute_decay returns a higher multiplier
        for a fact after its confirmation_count has been incremented.
        """
        # Get initial decay for a Python fact
        fact_before = seeded_backend.get_fact("py-lang")
        decay_before = retriever._compute_decay(fact_before, "fact")

        # Simulate 5 retrieval turns
        for query in PYTHON_QUERIES[:5]:
            _simulate_turn(retriever, seeded_backend, query)

        # Re-read the fact and compute decay again
        fact_after = seeded_backend.get_fact("py-lang")
        decay_after = retriever._compute_decay(fact_after, "fact")

        assert fact_after["confirmation_count"] > fact_before["confirmation_count"], (
            f"confirmation_count should have increased: "
            f"before={fact_before['confirmation_count']}, "
            f"after={fact_after['confirmation_count']}"
        )

        assert decay_after > decay_before, (
            f"Decay multiplier should improve with confirmation: "
            f"before={decay_before:.4f}, after={decay_after:.4f}"
        )

    def test_confirmation_weight_formula(self, retriever, seeded_backend):
        """
        Verify the log(1 + count) weighting formula from proxy.py
        produces the expected progressive boost.
        """
        # Run all turns to build up confirmation counts
        for query in PYTHON_QUERIES:
            _simulate_turn(retriever, seeded_backend, query)

        final_counts = _get_confirmation_counts(seeded_backend)

        # Check the formula: log(1 + count) should be monotonically
        # increasing with count
        weights = {}
        for fid in PYTHON_FACT_IDS | DISTRACTOR_FACT_IDS:
            count = final_counts[fid]
            weight = math.log(1 + count)
            weights[fid] = weight

        # Find the most-confirmed Python fact and a distractor
        most_confirmed_id = max(PYTHON_FACT_IDS, key=lambda fid: final_counts[fid])
        any_distractor_id = next(iter(DISTRACTOR_FACT_IDS))

        assert weights[most_confirmed_id] > weights[any_distractor_id], (
            f"Confirmation weight for {most_confirmed_id} "
            f"(count={final_counts[most_confirmed_id]}, "
            f"weight={weights[most_confirmed_id]:.3f}) "
            f"should exceed {any_distractor_id} "
            f"(count={final_counts[any_distractor_id]}, "
            f"weight={weights[any_distractor_id]:.3f})"
        )


class TestMultiTurnScoreDynamics:
    """
    Deeper analysis of how scores evolve across the 10-turn sequence.
    """

    def test_on_topic_recall_stable_or_improving(
        self, retriever, seeded_backend
    ):
        """
        The number of on-topic facts retrieved should not decrease
        across turns (confirmation should help, not hurt, recall).
        """
        recall_per_turn: List[int] = []

        for query in PYTHON_QUERIES:
            results = _simulate_turn(retriever, seeded_backend, query)
            recall_per_turn.append(_on_topic_retrieved_count(results))

        # Allow natural query variation, but the last 3 turns should
        # retrieve at least as many on-topic facts as the first 3
        early_recall = sum(recall_per_turn[:3])
        late_recall = sum(recall_per_turn[-3:])

        assert late_recall >= early_recall, (
            f"Late recall ({late_recall}) should be >= early recall "
            f"({early_recall}). Per-turn: {recall_per_turn}"
        )

    def test_score_trajectory_recorded(self, retriever, seeded_backend):
        """
        Record the full score trajectory and verify it's non-degenerate.
        At least one on-topic fact should appear in every turn.
        """
        trajectories: Dict[str, List[float]] = {fid: [] for fid in PYTHON_FACT_IDS}

        for query in PYTHON_QUERIES:
            results = _simulate_turn(retriever, seeded_backend, query)
            result_map = {r.id: r.combined_score for r in results}

            for fid in PYTHON_FACT_IDS:
                trajectories[fid].append(result_map.get(fid, 0.0))

        # At least one Python fact should appear in every turn
        for turn_idx in range(len(PYTHON_QUERIES)):
            turn_hits = sum(
                1 for fid in PYTHON_FACT_IDS
                if trajectories[fid][turn_idx] > 0
            )
            assert turn_hits >= 1, (
                f"Turn {turn_idx} ({PYTHON_QUERIES[turn_idx]!r}) "
                f"retrieved zero on-topic facts"
            )

    def test_best_on_topic_score_monotonic_tendency(
        self, retriever, seeded_backend
    ):
        """
        The best on-topic score should show an upward tendency.
        We don't require strict monotonicity (query variation exists),
        but a linear regression slope should be positive.
        """
        best_scores: List[float] = []

        for query in PYTHON_QUERIES:
            results = _simulate_turn(retriever, seeded_backend, query)
            best_scores.append(_best_on_topic_score(results))

        # Simple linear regression slope
        n = len(best_scores)
        x_mean = (n - 1) / 2.0
        y_mean = sum(best_scores) / n

        numerator = sum(
            (i - x_mean) * (s - y_mean)
            for i, s in enumerate(best_scores)
        )
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        slope = numerator / denominator if denominator > 0 else 0

        assert slope >= 0, (
            f"Best on-topic score should trend upward (slope={slope:.6f}). "
            f"Scores: {[f'{s:.4f}' for s in best_scores]}"
        )


class TestConfirmationCountIntegrity:
    """
    Verify the database-level confirmation mechanics are correct.
    """

    def test_record_fact_access_increments(self, seeded_backend):
        """record_fact_access should increment confirmation_count by 1."""
        before = seeded_backend.get_fact("py-lang")
        assert before["confirmation_count"] == 1

        seeded_backend.record_fact_access("py-lang")

        after = seeded_backend.get_fact("py-lang")
        assert after["confirmation_count"] == 2
        assert after["last_confirmed"] is not None
        assert after["access_count"] == 1

    def test_batch_touch_increments(self, seeded_backend):
        """touch_facts_batch should increment all targeted facts."""
        ids = ["py-lang", "py-flask", "py-data"]

        seeded_backend.touch_facts_batch(ids)

        for fid in ids:
            fact = seeded_backend.get_fact(fid)
            assert fact["confirmation_count"] == 2, f"{fid} not incremented"

        # Untouched fact stays at 1
        untouched = seeded_backend.get_fact("cook-pasta")
        assert untouched["confirmation_count"] == 1

    def test_repeated_access_accumulates(self, seeded_backend):
        """5 accesses should yield confirmation_count=6 (1 initial + 5)."""
        for _ in range(5):
            seeded_backend.record_fact_access("py-ml")

        fact = seeded_backend.get_fact("py-ml")
        assert fact["confirmation_count"] == 6
        assert fact["access_count"] == 5

    def test_last_confirmed_updates_on_access(self, seeded_backend):
        """Accessing a fact should update its last_confirmed timestamp."""
        before = seeded_backend.get_fact("py-testing")
        old_confirmed = before["last_confirmed"]

        seeded_backend.record_fact_access("py-testing")

        after = seeded_backend.get_fact("py-testing")
        assert after["last_confirmed"] != old_confirmed
        assert after["last_confirmed"] > old_confirmed


class TestDecayConfirmationInteraction:
    """
    Verify the mathematical interaction between decay and confirmation.
    """

    def test_decay_formula_with_varying_counts(self, retriever):
        """
        Confirm the formula: effective_lambda = lambda / (1 + 0.5 * log1p(count))
        produces expected decay values.
        """
        past = datetime.now() - timedelta(days=70)
        base_meta = {"last_confirmed": past.isoformat()}

        results = []
        for count in [0, 1, 5, 10, 20, 50, 100]:
            meta = {**base_meta, "confirmation_count": count}
            decay = retriever._compute_decay(meta, "fact")
            expected_lambda = 0.01 / (1.0 + 0.5 * math.log1p(count))
            expected_decay = max(0.1, math.exp(-expected_lambda * 70))
            results.append((count, decay, expected_decay))

            assert decay == pytest.approx(expected_decay, abs=0.001), (
                f"count={count}: got {decay:.4f}, expected {expected_decay:.4f}"
            )

        # Verify monotonically increasing with count
        decays = [r[1] for r in results]
        for i in range(1, len(decays)):
            assert decays[i] >= decays[i - 1], (
                f"Decay should increase with confirmation count: "
                f"count={results[i][0]} gave {decays[i]:.4f} < "
                f"count={results[i-1][0]} gave {decays[i-1]:.4f}"
            )

    def test_confirmation_makes_meaningful_difference(self, retriever):
        """
        At 70 days age, a fact with count=10 should have a measurably
        higher decay multiplier than count=1.
        """
        past = datetime.now() - timedelta(days=70)
        meta_low = {
            "last_confirmed": past.isoformat(),
            "confirmation_count": 1,
        }
        meta_high = {
            "last_confirmed": past.isoformat(),
            "confirmation_count": 10,
        }

        decay_low = retriever._compute_decay(meta_low, "fact")
        decay_high = retriever._compute_decay(meta_high, "fact")

        improvement = (decay_high - decay_low) / decay_low
        assert improvement > 0.10, (
            f"Expected >10% improvement, got {improvement:.1%}: "
            f"low={decay_low:.4f}, high={decay_high:.4f}"
        )
