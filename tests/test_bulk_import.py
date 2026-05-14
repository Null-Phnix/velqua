"""Tests for POST /facts/import/bulk endpoint."""
import os
import tempfile
import importlib

import pytest

# Temp DB before any server imports
_tmpdir = tempfile.mkdtemp()
os.environ["VELQUA_DB_PATH"] = os.path.join(_tmpdir, "test_bulk.db")

import backend.config
importlib.reload(backend.config)

from backend.server import app
from backend.routes._shared import get_memory
from backend.config import VelquaConfig as Config
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
            mem.backend.delete_fact(r["id"])
    except Exception:
        pass


_UNIQUE_SUBJECTS = [
    "enjoys playing piano on weekends regularly in the living room",
    "works at a bakery making sourdough bread every single morning",
    "studies marine biology at the coastal university campus daily",
    "volunteers at the local animal shelter walking dogs weekly",
    "collects vintage vinyl records from the nineteen seventies era",
    "practices hot yoga every morning before eating breakfast meals",
    "coaches a youth basketball team during Saturday afternoon games",
    "writes science fiction novels about intergalactic space travel",
    "maintains a rooftop garden growing tomatoes peppers and herbs",
    "teaches calculus at the community college downtown on Tuesdays",
]


def _make_facts(n, prefix="User"):
    """Generate n unique fact payloads that won't trigger dedup."""
    return [
        {"content": f"{prefix} {_UNIQUE_SUBJECTS[i % len(_UNIQUE_SUBJECTS)]}"}
        for i in range(n)
    ]


# === Happy path ===

class TestBulkImportSuccess:

    def test_basic_bulk_import(self, client):
        facts = _make_facts(5)
        resp = client.post("/facts/import/bulk", json={"facts": facts})
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["inserted"] == 5
        assert body["skipped"] == 0
        assert body["failed"] == 0
        assert len(body["fact_ids"]) == 5

    def test_facts_are_retrievable(self, client):
        facts = _make_facts(3)
        client.post("/facts/import/bulk", json={"facts": facts})
        resp = client.get("/facts/list?limit=10")
        stored = resp.json()["facts"]
        contents = [f["content"] for f in stored]
        for item in facts:
            assert item["content"] in contents

    def test_custom_fields(self, client):
        facts = [{
            "content": "User prefers dark mode in all applications and editors",
            "fact_type": "preference",
            "confidence": 0.95,
            "importance": 0.8,
            "tags": ["ui", "preference"],
            "metadata": {"source": "bulk-test"},
        }]
        resp = client.post("/facts/import/bulk", json={"facts": facts})
        assert resp.status_code == 200
        body = resp.json()
        assert body["inserted"] == 1

        # Verify the fact has the right type and confidence
        list_resp = client.get("/facts/list?limit=5")
        stored = list_resp.json()["facts"]
        match = [f for f in stored if f["content"] == facts[0]["content"]]
        assert len(match) == 1
        assert match[0]["type"] == "preference"
        assert match[0]["confidence"] == 0.95

    def test_defaults_applied(self, client):
        """Facts with only content should get default type/confidence."""
        facts = [{"content": "A bare fact with just content and nothing else specified"}]
        resp = client.post("/facts/import/bulk", json={"facts": facts})
        body = resp.json()
        assert body["inserted"] == 1

        list_resp = client.get("/facts/list?limit=5")
        stored = list_resp.json()["facts"]
        match = [f for f in stored if f["content"] == facts[0]["content"]]
        assert len(match) == 1
        assert match[0]["type"] == "general"
        assert match[0]["confidence"] == Config.DEFAULT_CONFIDENCE


# === Deduplication ===

class TestBulkImportDedup:

    def test_duplicates_skipped(self, client):
        facts = _make_facts(3)
        # Import once
        resp1 = client.post("/facts/import/bulk", json={"facts": facts})
        assert resp1.json()["inserted"] == 3
        # Import same facts again
        resp2 = client.post("/facts/import/bulk", json={"facts": facts})
        body2 = resp2.json()
        assert body2["skipped"] == 3
        assert body2["inserted"] == 0

    def test_mixed_new_and_duplicate(self, client):
        # Use indices 0-1 for originals, 5-6 for new ones (maximally distinct)
        original = [
            {"content": _UNIQUE_SUBJECTS[0]},
            {"content": _UNIQUE_SUBJECTS[1]},
        ]
        client.post("/facts/import/bulk", json={"facts": original})

        new_facts = [
            {"content": _UNIQUE_SUBJECTS[5]},
            {"content": _UNIQUE_SUBJECTS[6]},
        ]
        mixed = original + new_facts
        resp = client.post("/facts/import/bulk", json={"facts": mixed})
        body = resp.json()
        assert body["inserted"] == 2
        assert body["skipped"] == 2


# === Validation errors ===

class TestBulkImportValidation:

    def test_empty_array_rejected(self, client):
        resp = client.post("/facts/import/bulk", json={"facts": []})
        assert resp.status_code == 400

    def test_over_limit_rejected(self, client):
        facts = _make_facts(1001)
        resp = client.post("/facts/import/bulk", json={"facts": facts})
        assert resp.status_code == 400
        assert "1000" in resp.json()["detail"]

    def test_too_short_content(self, client):
        facts = [{"content": "short"}]
        resp = client.post("/facts/import/bulk", json={"facts": facts})
        assert resp.status_code == 422
        body = resp.json()["detail"]
        assert len(body["errors"]) == 1
        assert body["errors"][0]["index"] == 0
        assert "too short" in body["errors"][0]["error"].lower()

    def test_too_long_content(self, client):
        facts = [{"content": "x" * (Config.MAX_FACT_LENGTH + 1)}]
        resp = client.post("/facts/import/bulk", json={"facts": facts})
        assert resp.status_code == 422

    def test_invalid_fact_type(self, client):
        facts = [{
            "content": "A perfectly valid content string for fact type testing",
            "fact_type": "nonexistent_type",
        }]
        resp = client.post("/facts/import/bulk", json={"facts": facts})
        assert resp.status_code == 422
        assert "nonexistent_type" in str(resp.json())

    def test_confidence_out_of_range(self, client):
        facts = [{
            "content": "Content is fine but confidence value is way too high",
            "confidence": 1.5,
        }]
        resp = client.post("/facts/import/bulk", json={"facts": facts})
        assert resp.status_code == 422

    def test_importance_out_of_range(self, client):
        facts = [{
            "content": "Content is fine but importance value is negative",
            "importance": -0.1,
        }]
        resp = client.post("/facts/import/bulk", json={"facts": facts})
        assert resp.status_code == 422

    def test_multiple_validation_errors(self, client):
        """Multiple bad items should all be reported."""
        facts = [
            {"content": "short"},
            {"content": "also short"},
            {"content": "A valid fact content string that passes length check"},
        ]
        resp = client.post("/facts/import/bulk", json={"facts": facts})
        assert resp.status_code == 422
        errors = resp.json()["detail"]["errors"]
        # The two short facts should fail; the valid one should NOT appear in errors
        failed_indices = {e["index"] for e in errors}
        assert 0 in failed_indices
        assert 1 in failed_indices
        assert 2 not in failed_indices

    def test_nothing_inserted_on_validation_failure(self, client):
        """If validation fails, no facts should be inserted — even valid ones."""
        facts = [
            {"content": "A perfectly valid fact that should not be stored yet"},
            {"content": "bad"},  # too short
        ]
        resp = client.post("/facts/import/bulk", json={"facts": facts})
        assert resp.status_code == 422

        # The valid fact should NOT have been stored
        list_resp = client.get("/facts/list?limit=100")
        contents = [f["content"] for f in list_resp.json()["facts"]]
        assert "A perfectly valid fact that should not be stored yet" not in contents


# === Edge cases ===

class TestBulkImportEdgeCases:

    def test_single_fact(self, client):
        facts = [{"content": "A single fact imported via the bulk endpoint works fine"}]
        resp = client.post("/facts/import/bulk", json={"facts": facts})
        assert resp.status_code == 200
        assert resp.json()["inserted"] == 1

    def test_all_fact_types(self, client):
        """Every valid FactType should be accepted."""
        types = ["personal", "preference", "professional", "project",
                 "relationship", "world", "general"]
        # Each fact must be unique enough to avoid dedup across types
        facts = [
            {"content": _UNIQUE_SUBJECTS[i], "fact_type": t}
            for i, t in enumerate(types)
        ]
        resp = client.post("/facts/import/bulk", json={"facts": facts})
        assert resp.status_code == 200
        assert resp.json()["inserted"] == len(types)

    def test_whitespace_trimmed(self, client):
        """Leading/trailing whitespace should be stripped from content."""
        facts = [{"content": "   This fact has extra whitespace that should be trimmed   "}]
        resp = client.post("/facts/import/bulk", json={"facts": facts})
        assert resp.status_code == 200

        list_resp = client.get("/facts/list?limit=5")
        stored = list_resp.json()["facts"]
        match = [f for f in stored if "extra whitespace" in f["content"]]
        assert len(match) == 1
        assert not match[0]["content"].startswith(" ")
        assert not match[0]["content"].endswith(" ")
