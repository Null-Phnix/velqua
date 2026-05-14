"""Integration tests: verify route handlers emit activity events."""
import io
import json
import os
import shutil
import tempfile
import importlib

import pytest

# Set up temp DB before any server imports
_tmpdir = tempfile.mkdtemp()
os.environ["VELQUA_DB_PATH"] = os.path.join(_tmpdir, "test_activity_integ.db")

import backend.config
importlib.reload(backend.config)

from backend.activity.db import (
    list_events,
    clear_events,
    set_db_path as set_activity_db_path,
    close_conn as close_activity_conn,
)
from backend.server import app
from backend.routes._shared import get_memory
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def clean_state():
    """Reset activity DB and fact store between tests."""
    activity_db = os.path.join(_tmpdir, "test_activity_integ_events.db")
    set_activity_db_path(activity_db)
    clear_events()
    yield
    clear_events()
    close_activity_conn()
    try:
        mem = get_memory()
        rows = mem.backend.list_facts(limit=10000)
        for r in rows:
            mem.backend.delete_fact(r["id"])
    except Exception:
        pass


# --- Helpers ---

def make_claude_memories(text: str = "User likes Python and has 3 cats") -> bytes:
    data = [{
        "conversations_memory": f"**Personal context**\n\n- {text}",
        "account_uuid": "test-uuid-123",
    }]
    return json.dumps(data).encode()


def upload_file(client, data: bytes, filename: str = "test.json"):
    return client.post(
        "/import/smart",
        files={"file": (filename, io.BytesIO(data), "application/json")},
    )


_DISTINCT_FACTS = [
    "User works as a software engineer specializing in distributed systems",
    "User has three cats named Pixel, Voxel, and Shader who are all orange",
    "User prefers dark roast Ethiopian coffee every single weekday morning",
    "User lives in a small apartment near the downtown waterfront district",
    "User plays classical piano and enjoys performing Chopin nocturnes regularly",
]


def seed_facts(client, count=3):
    """Import highly distinct facts and return them."""
    for i in range(min(count, len(_DISTINCT_FACTS))):
        data = make_claude_memories(_DISTINCT_FACTS[i])
        upload_file(client, data)
    clear_events()  # Clear import events so tests only see the action they test
    resp = client.get(f"/facts/list?limit={count + 10}")
    return resp.json()["facts"]


# ==================================================================
# Route → activity event integration tests
# ==================================================================


class TestFactDeleteEmitsEvent:
    def test_delete_fact_logs_activity(self, client):
        facts = seed_facts(client, 1)
        fact_id = facts[0]["id"]
        resp = client.delete(f"/facts/{fact_id}")
        assert resp.status_code == 200

        events = list_events(event_type="fact_deleted")
        assert len(events) >= 1
        assert events[0]["event_type"] == "fact_deleted"
        assert fact_id in events[0]["metadata"].get("fact_id", "")

    def test_bulk_delete_logs_activity(self, client):
        facts = seed_facts(client, 2)
        fact_ids = [f["id"] for f in facts]
        resp = client.post("/facts/bulk-delete", json={"fact_ids": fact_ids})
        assert resp.status_code == 200

        events = list_events(event_type="fact_deleted")
        assert len(events) >= 1
        assert events[0]["metadata"].get("count") == 2


class TestFactEditEmitsEvent:
    def test_patch_fact_logs_activity(self, client):
        facts = seed_facts(client, 1)
        fact_id = facts[0]["id"]
        resp = client.patch(f"/facts/{fact_id}", json={"content": "Updated content that is long enough for validation"})
        assert resp.status_code == 200

        events = list_events(event_type="fact_edited")
        assert len(events) >= 1
        assert events[0]["metadata"].get("fact_id") == fact_id


class TestFactMergeEmitsEvent:
    def test_merge_logs_activity(self, client):
        facts = seed_facts(client, 2)
        fact_ids = [f["id"] for f in facts[:2]]
        resp = client.post("/facts/merge", json={
            "fact_ids": fact_ids,
            "merged_content": "A merged fact that combines both original facts for testing",
        })
        assert resp.status_code == 200

        events = list_events(event_type="fact_merged")
        assert len(events) >= 1
        assert events[0]["metadata"].get("source_count") == 2


class TestImportEmitsEvent:
    def test_successful_import_logs_activity(self, client):
        data = make_claude_memories("User has a strong preference for dark chocolate ice cream flavors")
        resp = upload_file(client, data, "memories.json")
        assert resp.status_code == 200

        events = list_events(event_type="import_completed")
        assert len(events) >= 1
        assert "memories.json" in events[0]["title"]

    def test_failed_import_logs_activity(self, client):
        resp = client.post(
            "/import/smart",
            files={"file": ("bad.json", io.BytesIO(b"not json at all"), "application/json")},
        )
        # Should fail (400)
        assert resp.status_code == 400

        events = list_events(event_type="import_failed")
        assert len(events) >= 1


class TestBackupEmitsEvent:
    def test_create_backup_logs_activity(self, client):
        resp = client.post("/backup/create")
        assert resp.status_code == 200

        events = list_events(event_type="backup_created")
        assert len(events) >= 1
        assert "MB" in events[0]["title"]

    def test_restore_backup_logs_activity(self, client):
        # Create a backup first
        create_resp = client.post("/backup/create")
        assert create_resp.status_code == 200
        backup_path = create_resp.json()["backup_path"]
        filename = os.path.basename(backup_path)
        clear_events()

        # Restore it
        resp = client.post(f"/backup/restore/{filename}")
        assert resp.status_code == 200

        events = list_events(event_type="backup_restored")
        assert len(events) >= 1
        assert filename in events[0]["title"]


class TestProviderChangeEmitsEvent:
    def test_set_active_provider_logs_activity(self, client):
        resp = client.post("/settings/active-provider", json={"name": "ollama"})
        assert resp.status_code == 200

        events = list_events(event_type="provider_changed")
        assert len(events) >= 1
        assert "ollama" in events[0]["title"].lower()


class TestReviewEmitsEvent:
    def test_approve_logs_activity(self, client):
        # Seed a pending fact
        from backend.routes.review import get_pending_store
        store = get_pending_store()
        store.add("This is a test fact that should be approved by the user", 0.6, "test")
        pending = store.list_all()
        assert len(pending) > 0
        pending_id = pending[0]["id"]

        resp = client.post(f"/review/approve/{pending_id}")
        assert resp.status_code == 200

        events = list_events(event_type="fact_approved")
        assert len(events) >= 1

    def test_reject_logs_activity(self, client):
        from backend.routes.review import get_pending_store
        store = get_pending_store()
        store.add("This is a test fact that should be rejected by user review", 0.6, "test")
        pending = store.list_all()
        pending_id = pending[0]["id"]

        resp = client.post(f"/review/reject/{pending_id}")
        assert resp.status_code == 200

        events = list_events(event_type="fact_rejected")
        assert len(events) >= 1


class TestActivityEndpointIntegration:
    """Verify the GET /activity endpoint returns events emitted by routes."""

    def test_activity_reflects_fact_operations(self, client):
        facts = seed_facts(client, 1)
        fact_id = facts[0]["id"]

        # Edit
        client.patch(f"/facts/{fact_id}", json={"content": "Edited version of the fact with enough characters here"})
        # Delete
        client.delete(f"/facts/{fact_id}")

        # Fetch activity via API
        resp = client.get("/activity")
        assert resp.status_code == 200
        data = resp.json()
        types = [e["event_type"] for e in data["events"]]
        assert "fact_edited" in types
        assert "fact_deleted" in types

    def test_activity_filter_returns_only_matching_type(self, client):
        seed_facts(client, 1)
        facts = client.get("/facts/list?limit=10").json()["facts"]
        client.delete(f"/facts/{facts[0]['id']}")

        # Also create a backup so we have multiple event types
        client.post("/backup/create")

        # Filter for backup only
        resp = client.get("/activity?event_type=backup_created")
        data = resp.json()
        for event in data["events"]:
            assert event["event_type"] == "backup_created"
