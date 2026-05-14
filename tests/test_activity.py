"""Tests for the activity log system."""
import json
import os
import tempfile
import importlib

import pytest

# Set up temp DB before any server imports
_tmpdir = tempfile.mkdtemp()
os.environ["VELQUA_DB_PATH"] = os.path.join(_tmpdir, "test_activity.db")

import backend.config
importlib.reload(backend.config)

from backend.activity.db import (
    log_event,
    list_events,
    count_events,
    clear_events,
    set_db_path,
    close_conn,
    EVENT_TYPES,
)
from backend.server import app
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def clean_activity_db():
    """Reset activity DB before each test."""
    activity_db = os.path.join(_tmpdir, "test_activity_events.db")
    set_db_path(activity_db)
    yield
    clear_events()
    close_conn()


# ==================================================================
# Unit tests — direct DB functions
# ==================================================================

class TestLogEvent:
    def test_log_event_returns_id(self):
        event_id = log_event("fact_learned", "Learned something new")
        assert event_id
        assert isinstance(event_id, str)

    def test_log_event_with_detail(self):
        log_event("fact_learned", "Learned X", "Extra detail here")
        events = list_events(limit=1)
        assert events[0]["detail"] == "Extra detail here"

    def test_log_event_with_metadata(self):
        log_event("import_completed", "Imported file", metadata={"stored": 42})
        events = list_events(limit=1)
        assert events[0]["metadata"]["stored"] == 42

    def test_log_event_stores_timestamp(self):
        log_event("backup_created", "Backup done")
        events = list_events(limit=1)
        assert events[0]["timestamp"] > 0

    def test_log_event_stores_type(self):
        log_event("provider_changed", "Changed to ollama")
        events = list_events(limit=1)
        assert events[0]["event_type"] == "provider_changed"


class TestListEvents:
    def test_list_empty(self):
        events = list_events()
        assert events == []

    def test_list_returns_reverse_chronological(self):
        log_event("fact_learned", "First")
        log_event("fact_learned", "Second")
        log_event("fact_learned", "Third")
        events = list_events()
        assert events[0]["title"] == "Third"
        assert events[2]["title"] == "First"

    def test_list_respects_limit(self):
        for i in range(10):
            log_event("fact_learned", f"Event {i}")
        events = list_events(limit=3)
        assert len(events) == 3

    def test_list_respects_offset(self):
        for i in range(5):
            log_event("fact_learned", f"Event {i}")
        events = list_events(limit=2, offset=2)
        assert len(events) == 2
        assert events[0]["title"] == "Event 2"

    def test_filter_by_event_type(self):
        log_event("fact_learned", "A fact")
        log_event("import_completed", "An import")
        log_event("fact_deleted", "A deletion")

        facts_only = list_events(event_type="fact_learned")
        assert len(facts_only) == 1
        assert facts_only[0]["event_type"] == "fact_learned"

        imports_only = list_events(event_type="import_completed")
        assert len(imports_only) == 1


class TestCountEvents:
    def test_count_empty(self):
        assert count_events() == 0

    def test_count_all(self):
        log_event("fact_learned", "A")
        log_event("import_completed", "B")
        assert count_events() == 2

    def test_count_by_type(self):
        log_event("fact_learned", "A")
        log_event("fact_learned", "B")
        log_event("import_completed", "C")
        assert count_events(event_type="fact_learned") == 2
        assert count_events(event_type="import_completed") == 1


class TestClearEvents:
    def test_clear(self):
        log_event("fact_learned", "A")
        log_event("fact_learned", "B")
        deleted = clear_events()
        assert deleted == 2
        assert count_events() == 0


# ==================================================================
# API endpoint tests
# ==================================================================

class TestActivityEndpoint:
    def test_get_activity_empty(self, client):
        resp = client.get("/activity")
        assert resp.status_code == 200
        data = resp.json()
        assert data["events"] == []
        assert data["total"] == 0

    def test_get_activity_with_events(self, client):
        log_event("fact_learned", "Test fact")
        log_event("import_completed", "Test import")
        resp = client.get("/activity")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert len(data["events"]) == 2

    def test_get_activity_with_filter(self, client):
        log_event("fact_learned", "A fact")
        log_event("import_completed", "An import")
        resp = client.get("/activity?event_type=fact_learned")
        data = resp.json()
        assert data["total"] == 1
        assert data["events"][0]["event_type"] == "fact_learned"

    def test_get_activity_invalid_filter(self, client):
        resp = client.get("/activity?event_type=nonexistent")
        data = resp.json()
        assert data["total"] == 0
        assert "error" in data

    def test_get_activity_pagination(self, client):
        for i in range(5):
            log_event("fact_learned", f"Event {i}")
        resp = client.get("/activity?limit=2&offset=0")
        data = resp.json()
        assert len(data["events"]) == 2
        assert data["total"] == 5

    def test_get_event_types(self, client):
        resp = client.get("/activity/types")
        assert resp.status_code == 200
        data = resp.json()
        assert "types" in data
        assert "fact_learned" in data["types"]
        assert "import_completed" in data["types"]

    def test_delete_activity(self, client):
        log_event("fact_learned", "To be cleared")
        resp = client.delete("/activity")
        assert resp.status_code == 200
        data = resp.json()
        assert data["deleted"] == 1

        # Verify empty
        resp2 = client.get("/activity")
        assert resp2.json()["total"] == 0


class TestActivityEventTypes:
    """Verify EVENT_TYPES constant is complete."""

    def test_event_types_not_empty(self):
        assert len(EVENT_TYPES) > 0

    def test_known_types_present(self):
        expected = [
            "fact_learned", "fact_approved", "fact_rejected",
            "fact_deleted", "fact_merged", "fact_edited",
            "import_completed", "import_failed",
            "backup_created", "backup_restored",
            "provider_changed",
            "agent_connected", "agent_disconnected",
            "system_started",
        ]
        for t in expected:
            assert t in EVENT_TYPES, f"Missing event type: {t}"


class TestActivityMetadata:
    """Verify metadata serialization round-trips correctly."""

    def test_empty_metadata(self):
        log_event("fact_learned", "No meta")
        events = list_events(limit=1)
        assert events[0]["metadata"] == {}

    def test_nested_metadata(self):
        meta = {"fact_id": "abc-123", "scores": [0.8, 0.9]}
        log_event("fact_learned", "With meta", metadata=meta)
        events = list_events(limit=1)
        assert events[0]["metadata"]["fact_id"] == "abc-123"
        assert events[0]["metadata"]["scores"] == [0.8, 0.9]
