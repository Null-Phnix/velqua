"""
Tests for Velqua Mesh — agent registry, shared memory, noteboard, and API routes.

These tests use an in-memory (temp) mesh DB so they're fully isolated from
production data. The DB is overridden before any mesh module is imported.
"""
import json
import os
import sys
import tempfile
import importlib
from pathlib import Path
from unittest.mock import patch

import pytest

# Point the server DB at a temp path (conftest sets sys.path)
_tmpdir = tempfile.mkdtemp()
_mesh_db = os.path.join(_tmpdir, "test_mesh.db")
os.environ["VELQUA_DB_PATH"] = os.path.join(_tmpdir, "test_mesh_server.db")

import backend.config
importlib.reload(backend.config)

# Now import mesh.db and redirect its DB path
from backend.mesh import db as _mesh_db_mod
_mesh_db_mod.set_db_path(Path(_mesh_db))

# Now import everything else
from backend.mesh.registry import AgentRegistry, detect_agent_id, ACTIVE_TIMEOUT_SECONDS
from backend.mesh.shared_memory import SharedMemoryPool
from backend.mesh.noteboard import Noteboard


# ==================================================================
# Fixtures
# ==================================================================

@pytest.fixture(autouse=True)
def clean_mesh_db():
    """Wipe all mesh tables between tests."""
    yield
    conn = _mesh_db_mod.get_conn()
    conn.executescript("""
        DELETE FROM mesh_agents;
        DELETE FROM mesh_memory;
        DELETE FROM mesh_notes;
    """)
    conn.commit()


@pytest.fixture
def reg():
    return AgentRegistry()


@pytest.fixture
def mem():
    return SharedMemoryPool()


@pytest.fixture
def nb():
    return Noteboard()


# ==================================================================
# Agent identity detection
# ==================================================================

class TestDetectAgentId:
    def test_explicit_header_wins(self):
        assert detect_agent_id(x_velqua_agent="blackreach") == "blackreach"

    def test_header_normalized_lowercase(self):
        assert detect_agent_id(x_velqua_agent="Blackreach") == "blackreach"

    def test_header_truncated_at_64(self):
        long_name = "a" * 100
        result = detect_agent_id(x_velqua_agent=long_name)
        assert len(result) <= 64

    def test_ua_heuristic_blackreach(self):
        assert detect_agent_id(user_agent="Blackreach/2.0 Python") == "blackreach"

    def test_ua_heuristic_open_webui(self):
        assert detect_agent_id(user_agent="open-webui/1.0") == "open-webui"

    def test_ua_heuristic_python_httpx(self):
        assert detect_agent_id(user_agent="python-httpx/0.24.0") == "python-script"

    def test_no_hints_returns_unknown(self):
        assert detect_agent_id() == "unknown"

    def test_empty_ua_returns_unknown(self):
        assert detect_agent_id(user_agent="") == "unknown"

    def test_header_takes_priority_over_ua(self):
        result = detect_agent_id(x_velqua_agent="mybot", user_agent="Blackreach/2.0")
        assert result == "mybot"


# ==================================================================
# Agent Registry
# ==================================================================

class TestAgentRegistry:
    def test_heartbeat_creates_agent(self, reg):
        reg.heartbeat("test-agent")
        agent = reg.get("test-agent")
        assert agent is not None
        assert agent["id"] == "test-agent"

    def test_heartbeat_updates_task(self, reg):
        reg.heartbeat("bot", task_hint="searching arxiv papers")
        agent = reg.get("bot")
        assert "searching arxiv" in agent["current_task"]

    def test_heartbeat_idempotent(self, reg):
        reg.heartbeat("bot", task_hint="task A")
        reg.heartbeat("bot", task_hint="task B")
        agent = reg.get("bot")
        assert "task B" in agent["current_task"]

    def test_list_active_returns_recent(self, reg):
        reg.heartbeat("bot")
        active = reg.list_active()
        ids = [a["id"] for a in active]
        assert "bot" in ids

    def test_list_all(self, reg):
        reg.heartbeat("bot1")
        reg.heartbeat("bot2")
        all_agents = reg.list_all()
        ids = [a["id"] for a in all_agents]
        assert "bot1" in ids
        assert "bot2" in ids

    def test_get_nonexistent_returns_none(self, reg):
        assert reg.get("nonexistent") is None

    def test_mark_inactive(self, reg):
        reg.heartbeat("bot")
        reg.mark_inactive("bot")
        agent = reg.get("bot")
        assert agent["status"] == "inactive"

    def test_is_active_flag_set(self, reg):
        reg.heartbeat("bot")
        agent = reg.get("bot")
        assert agent["is_active"] is True

    def test_last_seen_ago_populated(self, reg):
        reg.heartbeat("bot")
        agent = reg.get("bot")
        assert agent["last_seen_ago"] >= 0

    def test_task_hint_truncated_at_200(self, reg):
        reg.heartbeat("bot", task_hint="x" * 300)
        agent = reg.get("bot")
        assert len(agent["current_task"]) <= 200


# ==================================================================
# Shared Memory Pool
# ==================================================================

class TestSharedMemoryPool:
    def test_write_creates_entry(self, mem):
        entry = mem.write("blackreach", "Found 23 papers on RLHF")
        assert entry["id"]
        assert entry["agent_id"] == "blackreach"
        assert "23 papers" in entry["content"]

    def test_write_returns_tags(self, mem):
        entry = mem.write("bot", "result", tags=["research", "done"])
        assert "research" in entry["tags"]
        assert "done" in entry["tags"]

    def test_read_returns_entries(self, mem):
        mem.write("bot", "fact A")
        mem.write("bot", "fact B")
        entries = mem.read()
        contents = [e["content"] for e in entries]
        assert "fact A" in contents
        assert "fact B" in contents

    def test_read_filter_by_agent(self, mem):
        mem.write("bot1", "bot1 fact")
        mem.write("bot2", "bot2 fact")
        entries = mem.read(agent_id="bot1")
        assert all(e["agent_id"] == "bot1" for e in entries)

    def test_read_limit_respected(self, mem):
        for i in range(10):
            mem.write("bot", f"fact {i}")
        entries = mem.read(limit=3)
        assert len(entries) <= 3

    def test_search_keyword(self, mem):
        mem.write("bot", "mechanistic interpretability in attention heads")
        mem.write("bot", "data pipeline optimization")
        results = mem.search("interpretability")
        assert len(results) >= 1
        assert "interpretability" in results[0]["content"].lower()

    def test_delete_entry(self, mem):
        entry = mem.write("bot", "temporary result")
        deleted = mem.delete(entry["id"])
        assert deleted is True
        entries = mem.read()
        ids = [e["id"] for e in entries]
        assert entry["id"] not in ids

    def test_delete_nonexistent_returns_false(self, mem):
        assert mem.delete("fake-id-123") is False

    def test_count(self, mem):
        assert mem.count() == 0
        mem.write("bot", "item 1")
        mem.write("bot", "item 2")
        assert mem.count() == 2

    def test_empty_content_raises(self, mem):
        with pytest.raises(ValueError):
            mem.write("bot", "   ")

    def test_content_truncated_at_max(self, mem):
        entry = mem.write("bot", "x" * 2000)
        assert len(entry["content"]) <= 1000


# ==================================================================
# Noteboard
# ==================================================================

class TestNoteboard:
    def test_post_creates_note(self, nb):
        note = nb.post("blackreach", "planner", "Research complete — see /results/")
        assert note["id"]
        assert note["from_agent"] == "blackreach"
        assert note["to_agent"] == "planner"
        assert note["read"] is False

    def test_post_with_tags(self, nb):
        note = nb.post("bot", "any", "Done", tags=["complete", "urgent"])
        assert "complete" in note["tags"]

    def test_get_for_agent_targeted(self, nb):
        nb.post("bot1", "planner", "note for planner")
        nb.post("bot2", "coder", "note for coder")
        notes = nb.get_for_agent("planner")
        assert all(n["to_agent"] in ("planner", "any") for n in notes)

    def test_get_for_agent_broadcast(self, nb):
        nb.post("bot", "any", "broadcast message")
        notes = nb.get_for_agent("planner")
        contents = [n["content"] for n in notes]
        assert "broadcast message" in contents

    def test_unread_only_filter(self, nb):
        note = nb.post("bot", "planner", "unread note")
        nb.mark_read(note["id"])
        unread = nb.get_for_agent("planner", unread_only=True)
        ids = [n["id"] for n in unread]
        assert note["id"] not in ids

    def test_mark_read(self, nb):
        note = nb.post("bot", "planner", "test note")
        result = nb.mark_read(note["id"])
        assert result is True
        unread = nb.get_for_agent("planner", unread_only=True)
        ids = [n["id"] for n in unread]
        assert note["id"] not in ids

    def test_mark_read_nonexistent(self, nb):
        assert nb.mark_read("fake-id") is False

    def test_mark_all_read(self, nb):
        nb.post("bot", "planner", "note 1")
        nb.post("bot", "planner", "note 2")
        nb.post("bot", "any", "broadcast")
        count = nb.mark_all_read("planner")
        assert count >= 2

    def test_delete_note(self, nb):
        note = nb.post("bot", "any", "delete me")
        assert nb.delete(note["id"]) is True
        all_notes = nb.get_all()
        ids = [n["id"] for n in all_notes]
        assert note["id"] not in ids

    def test_delete_nonexistent(self, nb):
        assert nb.delete("fake-id") is False

    def test_count_unread(self, nb):
        nb.post("bot", "planner", "msg 1")
        nb.post("bot", "planner", "msg 2")
        assert nb.count_unread("planner") == 2

    def test_empty_content_raises(self, nb):
        with pytest.raises(ValueError):
            nb.post("bot", "any", "   ")

    def test_get_all(self, nb):
        nb.post("a", "b", "note 1")
        nb.post("c", "d", "note 2")
        all_notes = nb.get_all()
        assert len(all_notes) >= 2


# ==================================================================
# Mesh API Routes
# ==================================================================

from backend.server import app as _server_app
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def api_client():
    with TestClient(_server_app) as c:
        yield c


class TestMeshAgentRoutes:
    def test_list_agents_empty(self, api_client):
        r = api_client.get("/mesh/agents")
        assert r.status_code == 200
        data = r.json()
        assert "agents" in data
        assert "count" in data

    def test_get_nonexistent_agent(self, api_client):
        r = api_client.get("/mesh/agents/no-such-agent")
        assert r.status_code == 404

    def test_mesh_status(self, api_client):
        r = api_client.get("/mesh/status")
        assert r.status_code == 200
        data = r.json()
        assert "active_agents" in data
        assert "shared_memory_entries" in data


class TestMeshMemoryRoutes:
    def test_read_empty(self, api_client):
        r = api_client.get("/mesh/memory")
        assert r.status_code == 200
        data = r.json()
        assert "entries" in data

    def test_write_and_read(self, api_client):
        r = api_client.post("/mesh/memory", json={
            "agent_id": "test-bot",
            "content": "Found interesting result",
            "tags": ["test"],
        })
        assert r.status_code == 200
        entry = r.json()
        assert entry["agent_id"] == "test-bot"
        assert "interesting result" in entry["content"]

    def test_write_empty_content_rejected(self, api_client):
        r = api_client.post("/mesh/memory", json={
            "agent_id": "bot",
            "content": "   ",
        })
        assert r.status_code == 400

    def test_delete_entry(self, api_client):
        r = api_client.post("/mesh/memory", json={"agent_id": "bot", "content": "delete me"})
        entry_id = r.json()["id"]
        r2 = api_client.delete(f"/mesh/memory/{entry_id}")
        assert r2.status_code == 200
        assert r2.json()["success"] is True

    def test_delete_nonexistent(self, api_client):
        r = api_client.delete("/mesh/memory/fake-id")
        assert r.status_code == 404


class TestMeshNoteRoutes:
    def test_get_notes_empty(self, api_client):
        r = api_client.get("/mesh/notes")
        assert r.status_code == 200
        assert "notes" in r.json()

    def test_post_and_read_note(self, api_client):
        r = api_client.post("/mesh/notes", json={
            "from_agent": "bot1",
            "to_agent": "bot2",
            "content": "Task complete",
        })
        assert r.status_code == 200
        note = r.json()
        assert note["from_agent"] == "bot1"
        assert note["to_agent"] == "bot2"
        assert note["read"] is False

    def test_post_empty_content_rejected(self, api_client):
        r = api_client.post("/mesh/notes", json={
            "from_agent": "a",
            "to_agent": "b",
            "content": "  ",
        })
        assert r.status_code == 400

    def test_mark_note_read(self, api_client):
        r = api_client.post("/mesh/notes", json={
            "from_agent": "a", "to_agent": "b", "content": "test"
        })
        note_id = r.json()["id"]
        r2 = api_client.put(f"/mesh/notes/{note_id}/read")
        assert r2.status_code == 200
        assert r2.json()["success"] is True

    def test_mark_nonexistent_read(self, api_client):
        r = api_client.put("/mesh/notes/fake-note-id/read")
        assert r.status_code == 404

    def test_delete_note(self, api_client):
        r = api_client.post("/mesh/notes", json={
            "from_agent": "a", "to_agent": "b", "content": "delete me"
        })
        note_id = r.json()["id"]
        r2 = api_client.delete(f"/mesh/notes/{note_id}")
        assert r2.status_code == 200

    def test_filter_notes_by_agent(self, api_client):
        api_client.post("/mesh/notes", json={
            "from_agent": "a", "to_agent": "target", "content": "for target"
        })
        api_client.post("/mesh/notes", json={
            "from_agent": "a", "to_agent": "other", "content": "for other"
        })
        r = api_client.get("/mesh/notes?agent_id=target")
        notes = r.json()["notes"]
        for n in notes:
            assert n["to_agent"] in ("target", "any")


# ==================================================================
# Coverage patches — mesh/db.py and shared_memory.py
# ==================================================================

class TestMeshDbSetPathCloseExisting:
    """Cover mesh/db.py lines 24-28: set_db_path() closes existing cached connection."""

    def test_set_db_path_closes_cached_connection(self, tmp_path):
        """Calling set_db_path again closes and nulls the cached thread-local connection."""
        from backend.mesh import db as mesh_db
        from pathlib import Path

        # Force a connection to be opened on the current path
        conn = mesh_db.get_conn()
        assert conn is not None

        # Now call set_db_path with a new path — triggers close + _local.conn = None
        new_path = tmp_path / "new_mesh.db"
        mesh_db.set_db_path(new_path)

        # After set_db_path, get_conn() should open a new connection to the new path
        new_conn = mesh_db.get_conn()
        assert new_conn is not None

        # Restore original path so other tests continue working
        mesh_db.set_db_path(Path(_mesh_db))


class TestSharedMemorySinceFilter:
    """Cover mesh/shared_memory.py lines 80-81: read() with since parameter."""

    @pytest.fixture
    def pool(self):
        p = SharedMemoryPool()
        p._db_path = Path(_mesh_db)
        return p

    def test_read_with_since_filter(self, pool):
        """Passing since= filters out older entries."""
        import time
        pool.write(agent_id="agent-x", content="old entry", tags=[])
        time.sleep(0.01)
        cutoff = time.time()
        time.sleep(0.01)
        pool.write(agent_id="agent-x", content="new entry", tags=[])

        recent = pool.read(since=cutoff)
        contents = [e["content"] for e in recent]
        assert "new entry" in contents
        assert isinstance(recent, list)


# ==================================================================
# WebSocket + _ConnectionManager coverage
# ==================================================================

class TestConnectionManagerBroadcast:
    """Cover _ConnectionManager.broadcast() with dead connections (lines 52-55, 57)."""

    def test_broadcast_removes_dead_connections(self):
        """Dead connections are removed after a failed send_text call."""
        import asyncio
        from unittest.mock import AsyncMock
        from backend.routes.mesh import _ConnectionManager

        mgr = _ConnectionManager()

        dead_ws = AsyncMock()
        dead_ws.send_text = AsyncMock(side_effect=RuntimeError("connection closed"))
        mgr._connections = [dead_ws]

        asyncio.run(mgr.broadcast({"type": "test", "data": {}}))

        assert dead_ws not in mgr._connections

    def test_broadcast_succeeds_with_live_connection(self):
        """broadcast() sends to all live connections without removing them."""
        import asyncio
        from unittest.mock import AsyncMock
        from backend.routes.mesh import _ConnectionManager

        mgr = _ConnectionManager()

        live_ws = AsyncMock()
        live_ws.send_text = AsyncMock()
        mgr._connections = [live_ws]

        asyncio.run(mgr.broadcast({"type": "ping", "data": {}}))

        live_ws.send_text.assert_called_once()
        assert live_ws in mgr._connections

    def test_disconnect_ws_not_in_list(self):
        """disconnect() called on ws not in list hits ValueError handler (lines 43-44)."""
        from unittest.mock import MagicMock
        from backend.routes.mesh import _ConnectionManager

        mgr = _ConnectionManager()
        unknown_ws = MagicMock()

        # Should not raise — ValueError is silently caught
        mgr.disconnect(unknown_ws)

    def test_heartbeat_exception_breaks_loop(self):
        """send_text failure in heartbeat loop triggers except Exception: break (line 268)."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock
        from backend.routes.mesh import mesh_stream

        mock_ws = AsyncMock()

        # First send_text (snapshot) succeeds, second (ping) raises
        send_count = [0]

        async def _send_text(data):
            send_count[0] += 1
            if send_count[0] > 1:
                raise RuntimeError("connection lost")

        mock_ws.accept = AsyncMock()
        mock_ws.send_text = _send_text

        async def _instant_sleep(_):
            pass

        async def _run():
            with patch("backend.routes.mesh.asyncio.sleep", _instant_sleep):
                await mesh_stream(mock_ws)

        asyncio.run(_run())
        # At least 2 sends: snapshot + one ping attempt that failed
        assert send_count[0] >= 2


class TestMeshWebSocket:
    """Cover routes/mesh.py WebSocket endpoint and _ConnectionManager (lines 36-38, 41-45, 240-273)."""

    def test_websocket_snapshot_on_connect(self, api_client):
        """Connecting receives snapshot; sleep raises WebSocketDisconnect to exit loop (lines 36-38, 240-256, 270-273)."""
        from starlette.websockets import WebSocketDisconnect as _WSD

        async def _raise_wsd(_):
            raise _WSD()

        with patch("backend.routes.mesh.asyncio.sleep", _raise_wsd):
            with api_client.websocket_connect("/mesh/stream") as ws:
                data = json.loads(ws.receive_text())
                assert data["type"] == "snapshot"
                assert "agents" in data["data"]
                assert "memory" in data["data"]

    def test_websocket_heartbeat_path(self, api_client):
        """Heartbeat try block executes and ping is received (lines 255-267)."""
        async def _instant_sleep(_):
            pass

        with patch("backend.routes.mesh.asyncio.sleep", _instant_sleep):
            with api_client.websocket_connect("/mesh/stream") as ws:
                snapshot = json.loads(ws.receive_text())
                assert snapshot["type"] == "snapshot"
                # Receive the first heartbeat ping
                ping = json.loads(ws.receive_text())
                assert ping["type"] == "ping"
                assert "active_agents" in ping["data"]
            # Exiting 'with' closes the WS; server's next send_text raises → break

    def test_list_agents_all(self, api_client):
        """GET /mesh/agents?active_only=false covers line 98."""
        r = api_client.get("/mesh/agents?active_only=false")
        assert r.status_code == 200
        data = r.json()
        assert "agents" in data
        assert "count" in data

    def test_get_agent_success(self, api_client):
        """GET /mesh/agents/{id} returns agent data for a known agent (line 108)."""
        from backend.mesh.registry import AgentRegistry
        AgentRegistry().heartbeat("ws-get-test-agent")

        r = api_client.get("/mesh/agents/ws-get-test-agent")
        assert r.status_code == 200
        data = r.json()
        assert data["id"] == "ws-get-test-agent"

    def test_delete_note_404(self, api_client):
        """DELETE /mesh/notes/{nonexistent} returns 404 (line 204)."""
        r = api_client.delete("/mesh/notes/totally-fake-note-id-xyz")
        assert r.status_code == 404
