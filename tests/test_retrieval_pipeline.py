from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from backend import proxy


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """
    Temporary DB fixture for retrieval-pipeline tests.

    The current tests focus on retrieval helpers and scoring logic, so this
    fixture primarily guarantees filesystem isolation and mirrors the intended
    contract for future DB-backed retrieval tests.
    """
    db_path = tmp_path / "velqua-test.db"
    monkeypatch.setenv("VELQUA_DB_PATH", str(db_path))
    return db_path


def _result(
    content: str,
    score: float = 1.0,
    confirmation_count: int = 1,
    importance: float = 0.5,
    last_confirmed=None,
    first_learned=None,
    category: str = "",
    fact_id: str | None = None,
):
    meta = {
        "confirmation_count": confirmation_count,
        "importance": importance,
        "category": category,
    }
    if last_confirmed is not None:
        meta["last_confirmed"] = (
            last_confirmed.isoformat() if hasattr(last_confirmed, "isoformat") else last_confirmed
        )
    if first_learned is not None:
        meta["first_learned"] = (
            first_learned.isoformat() if hasattr(first_learned, "isoformat") else first_learned
        )
    if fact_id is not None:
        meta["id"] = fact_id

    return SimpleNamespace(
        id=fact_id,
        content=content,
        score=score,
        confirmation_count=confirmation_count,
        importance=importance,
        last_confirmed=last_confirmed,
        first_learned=first_learned,
        metadata=meta,
    )


def test_hybrid_scoring_formula_respects_different_base_ratios(temp_db):
    """
    Higher hybrid base scores should remain higher after proxy post-scoring,
    even when freshness/confirmation factors are identical.
    """
    now = datetime.now()
    high = _result(
        "Vector-dominant result",
        score=0.82,
        confirmation_count=2,
        importance=0.7,
        last_confirmed=now,
    )
    low = _result(
        "FTS-dominant result",
        score=0.31,
        confirmation_count=2,
        importance=0.7,
        last_confirmed=now,
    )

    ranked = proxy._score_ranked_fact_results([low, high], query_category="")
    assert ranked[0][0] == "Vector-dominant result"
    assert ranked[0][1] > ranked[1][1]

    expected_high = (
        high.score
        * proxy.score_fact_freshness(high)
        * proxy._topic_boost(high, "")
        * proxy._confirmation_weight(high)
    )
    expected_low = (
        low.score
        * proxy.score_fact_freshness(low)
        * proxy._topic_boost(low, "")
        * proxy._confirmation_weight(low)
    )

    assert ranked[0][1] == pytest.approx(expected_high, rel=1e-6)
    assert ranked[1][1] == pytest.approx(expected_low, rel=1e-6)


def test_mmr_dedupes_duplicate_results(temp_db):
    items = [
        ("I live in Vancouver", 0.95),
        ("I live in Vancouver", 0.91),
        ("  I   live in   Vancouver  ", 0.88),
        ("I work remotely", 0.70),
    ]

    deduped = proxy._dedupe_ranked_contents(items)

    assert len(deduped) == 2
    assert deduped[0][0] == "I live in Vancouver"
    assert deduped[1][0] == "I work remotely"


def test_confirmation_weighting_increases_scores_on_repeated_retrieval(temp_db):
    base = _result(
        "I work at DataForge",
        score=0.6,
        confirmation_count=1,
        importance=0.6,
        last_confirmed=datetime.now(),
    )
    reinforced = _result(
        "I work at DataForge",
        score=0.6,
        confirmation_count=5,
        importance=0.6,
        last_confirmed=datetime.now(),
    )

    base_score = proxy._score_ranked_fact_results([base], query_category="")[0][1]
    reinforced_score = proxy._score_ranked_fact_results([reinforced], query_category="")[0][1]

    assert proxy._confirmation_weight(reinforced) > proxy._confirmation_weight(base)
    assert reinforced_score > base_score


def test_adaptive_decay_reduces_scores_for_old_facts(temp_db):
    fresh = _result(
        "I recently switched to Linux",
        score=0.7,
        confirmation_count=1,
        importance=0.6,
        last_confirmed=datetime.now() - timedelta(hours=1),
    )
    old = _result(
        "I recently switched to Linux",
        score=0.7,
        confirmation_count=1,
        importance=0.6,
        last_confirmed=datetime.now() - timedelta(days=180),
    )

    fresh_decay = proxy._compute_fact_decay_multiplier(fresh)
    old_decay = proxy._compute_fact_decay_multiplier(old)
    fresh_score = proxy._score_ranked_fact_results([fresh], query_category="")[0][1]
    old_score = proxy._score_ranked_fact_results([old], query_category="")[0][1]

    assert old_decay < fresh_decay
    assert old_score < fresh_score


def test_query_expansion_finds_synonym_matches(temp_db, monkeypatch):
    matched_queries = []

    def fake_search(query, limit):
        matched_queries.append(query)
        if "jupiter" in query.lower():
            return [_result("Jupiter is the Roman counterpart of Zeus", score=1.0)]
        return []

    fake_memory = SimpleNamespace(
        semantic=SimpleNamespace(search=fake_search)
    )

    monkeypatch.setattr(proxy, "VECTOR_ENABLED", False)
    monkeypatch.setattr(proxy, "memory", fake_memory)

    facts, mode = proxy._retrieve_relevant_facts("Tell me about Zeus")

    assert mode == "fts"
    assert facts == ["Jupiter is the Roman counterpart of Zeus"]
    assert matched_queries, "Expected the fallback FTS search to be invoked"
    assert "jupiter" in matched_queries[0].lower()
