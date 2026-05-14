"""Tests for the fact feedback (thumbs up/down) system."""
import json
import os
import tempfile
import importlib

import pytest

# Set up temp DB before any server imports
_tmpdir = tempfile.mkdtemp()
os.environ["VELQUA_DB_PATH"] = os.path.join(_tmpdir, "test_feedback.db")

import backend.config
importlib.reload(backend.config)

from backend.server import app
from backend.routes._shared import get_memory
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def clean_db():
    yield
    try:
        mem = get_memory()
        rows = mem.backend.list_facts(limit=10000)
        for r in rows:
            mem.backend.delete_fact_feedback(r["id"])
            mem.backend.delete_fact(r["id"])
    except Exception:
        pass


def make_claude_memories(text: str) -> bytes:
    data = [{
        "conversations_memory": f"**Personal context**\n\n- {text}",
        "account_uuid": "test-uuid-123",
    }]
    return json.dumps(data).encode()


def seed_one_fact(client) -> str:
    """Import one fact and return its ID."""
    data = make_claude_memories("enjoys building custom mechanical keyboards")
    client.post(
        "/import/smart",
        files={"file": ("memories.json", data, "application/json")},
    )
    resp = client.get("/facts/list?limit=5")
    facts = resp.json()["facts"]
    assert len(facts) >= 1, "Seed failed — no facts imported"
    return facts[0]["id"]


# === Submit Feedback ===

class TestSubmitFeedback:
    def test_thumbs_up(self, client):
        fact_id = seed_one_fact(client)
        r = client.post(
            f"/facts/{fact_id}/feedback",
            json={"is_positive": True},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is True
        assert data["fact_id"] == fact_id
        assert data["feedback"]["thumbs_up"] == 1
        assert data["feedback"]["thumbs_down"] == 0

    def test_thumbs_down(self, client):
        fact_id = seed_one_fact(client)
        r = client.post(
            f"/facts/{fact_id}/feedback",
            json={"is_positive": False},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is True
        assert data["feedback"]["thumbs_up"] == 0
        assert data["feedback"]["thumbs_down"] == 1

    def test_nonexistent_fact(self, client):
        r = client.post(
            "/facts/nonexistent-id/feedback",
            json={"is_positive": True},
        )
        assert r.status_code == 404

    def test_missing_body(self, client):
        fact_id = seed_one_fact(client)
        r = client.post(f"/facts/{fact_id}/feedback", json={})
        assert r.status_code == 422

    def test_invalid_body(self, client):
        fact_id = seed_one_fact(client)
        r = client.post(
            f"/facts/{fact_id}/feedback",
            json={"is_positive": "banana"},
        )
        assert r.status_code == 422


# === Confidence Adjustment ===

class TestConfidenceAdjustment:
    def test_thumbs_up_increases_confidence(self, client):
        fact_id = seed_one_fact(client)
        # Get original confidence
        original = client.get(f"/facts/list?limit=5").json()["facts"]
        orig_conf = next(f["confidence"] for f in original if f["id"] == fact_id)

        r = client.post(
            f"/facts/{fact_id}/feedback",
            json={"is_positive": True},
        )
        new_conf = r.json()["new_confidence"]
        assert new_conf > orig_conf

    def test_thumbs_down_decreases_confidence(self, client):
        fact_id = seed_one_fact(client)
        original = client.get(f"/facts/list?limit=5").json()["facts"]
        orig_conf = next(f["confidence"] for f in original if f["id"] == fact_id)

        r = client.post(
            f"/facts/{fact_id}/feedback",
            json={"is_positive": False},
        )
        new_conf = r.json()["new_confidence"]
        assert new_conf < orig_conf

    def test_confidence_capped_at_1(self, client):
        fact_id = seed_one_fact(client)
        # Spam thumbs up to push confidence to max
        for _ in range(30):
            client.post(
                f"/facts/{fact_id}/feedback",
                json={"is_positive": True},
            )
        r = client.post(
            f"/facts/{fact_id}/feedback",
            json={"is_positive": True},
        )
        assert r.json()["new_confidence"] <= 1.0

    def test_confidence_floored_at_0(self, client):
        fact_id = seed_one_fact(client)
        # Spam thumbs down to push confidence to min
        for _ in range(30):
            client.post(
                f"/facts/{fact_id}/feedback",
                json={"is_positive": False},
            )
        r = client.post(
            f"/facts/{fact_id}/feedback",
            json={"is_positive": False},
        )
        assert r.json()["new_confidence"] >= 0.0


# === Feedback Accumulation ===

class TestFeedbackAccumulation:
    def test_multiple_feedback_accumulates(self, client):
        fact_id = seed_one_fact(client)
        client.post(f"/facts/{fact_id}/feedback", json={"is_positive": True})
        client.post(f"/facts/{fact_id}/feedback", json={"is_positive": True})
        client.post(f"/facts/{fact_id}/feedback", json={"is_positive": False})

        r = client.get(f"/facts/{fact_id}/feedback")
        assert r.status_code == 200
        fb = r.json()["feedback"]
        assert fb["thumbs_up"] == 2
        assert fb["thumbs_down"] == 1

    def test_get_feedback_nonexistent_fact(self, client):
        r = client.get("/facts/nonexistent-id/feedback")
        assert r.status_code == 404


# === Feedback in Listings ===

class TestFeedbackInListings:
    def test_list_facts_includes_feedback(self, client):
        fact_id = seed_one_fact(client)
        client.post(f"/facts/{fact_id}/feedback", json={"is_positive": True})

        r = client.get("/facts/list?limit=5")
        facts = r.json()["facts"]
        target = next((f for f in facts if f["id"] == fact_id), None)
        assert target is not None
        assert "feedback" in target
        assert target["feedback"]["thumbs_up"] == 1

    def test_search_results_include_feedback(self, client):
        fact_id = seed_one_fact(client)
        client.post(f"/facts/{fact_id}/feedback", json={"is_positive": False})

        r = client.get("/facts/search?q=keyboard")
        if r.json()["count"] > 0:
            for result in r.json()["results"]:
                assert "feedback" in result

    def test_fact_with_no_feedback_shows_zeros(self, client):
        fact_id = seed_one_fact(client)
        r = client.get("/facts/list?limit=5")
        facts = r.json()["facts"]
        target = next((f for f in facts if f["id"] == fact_id), None)
        assert target is not None
        assert target["feedback"] == {"thumbs_up": 0, "thumbs_down": 0}


# === Demotion via Negative Feedback ===

class TestDemotion:
    def test_repeated_negative_feedback_demotes_fact(self, client):
        """Facts with sustained negative feedback should have noticeably lower confidence."""
        fact_id = seed_one_fact(client)
        original = client.get("/facts/list?limit=5").json()["facts"]
        orig_conf = next(f["confidence"] for f in original if f["id"] == fact_id)

        # 5 thumbs down
        for _ in range(5):
            client.post(f"/facts/{fact_id}/feedback", json={"is_positive": False})

        updated = client.get("/facts/list?limit=5").json()["facts"]
        new_conf = next(f["confidence"] for f in updated if f["id"] == fact_id)
        # Should have dropped by at least 0.2 (5 * 0.05)
        assert new_conf <= orig_conf - 0.2


# === Database Layer ===

class TestDatabaseFeedback:
    def test_save_and_retrieve_feedback(self, client):
        mem = get_memory()
        fact_id = seed_one_fact(client)

        mem.backend.save_fact_feedback(fact_id, True)
        mem.backend.save_fact_feedback(fact_id, True)
        mem.backend.save_fact_feedback(fact_id, False)

        summary = mem.backend.get_fact_feedback_summary(fact_id)
        assert summary["thumbs_up"] == 2
        assert summary["thumbs_down"] == 1

    def test_batch_summaries(self, client):
        mem = get_memory()
        # Create two facts
        data1 = make_claude_memories("likes hiking in the mountains every summer")
        client.post(
            "/import/smart",
            files={"file": ("memories.json", data1, "application/json")},
        )
        data2 = make_claude_memories("plays chess competitively at local tournaments")
        client.post(
            "/import/smart",
            files={"file": ("memories.json", data2, "application/json")},
        )
        facts = client.get("/facts/list?limit=10").json()["facts"]
        assert len(facts) >= 2

        id1, id2 = facts[0]["id"], facts[1]["id"]
        mem.backend.save_fact_feedback(id1, True)
        mem.backend.save_fact_feedback(id2, False)

        summaries = mem.backend.get_fact_feedback_summaries([id1, id2])
        assert summaries[id1]["thumbs_up"] == 1
        assert summaries[id2]["thumbs_down"] == 1

    def test_delete_feedback_on_fact_delete(self, client):
        """Deleting a fact's feedback should work."""
        mem = get_memory()
        fact_id = seed_one_fact(client)
        mem.backend.save_fact_feedback(fact_id, True)
        mem.backend.save_fact_feedback(fact_id, False)

        deleted = mem.backend.delete_fact_feedback(fact_id)
        assert deleted == 2

        summary = mem.backend.get_fact_feedback_summary(fact_id)
        assert summary["thumbs_up"] == 0
        assert summary["thumbs_down"] == 0

    def test_summary_for_fact_with_no_feedback(self, client):
        mem = get_memory()
        fact_id = seed_one_fact(client)
        summary = mem.backend.get_fact_feedback_summary(fact_id)
        assert summary == {"thumbs_up": 0, "thumbs_down": 0}

    def test_batch_summaries_empty_list(self, client):
        mem = get_memory()
        result = mem.backend.get_fact_feedback_summaries([])
        assert result == {}
