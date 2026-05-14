"""
Mesh API routes — agent registry, shared memory, noteboard, and real-time stream.

All endpoints under /mesh/*. The WebSocket at /mesh/stream pushes real-time
events to the dashboard whenever agents connect, memory is written, or notes
are posted.
"""
import asyncio
import json
import time
from typing import Optional

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from backend.mesh.registry import registry, detect_agent_id
from backend.mesh.shared_memory import pool
from backend.mesh.noteboard import noteboard
from backend.logging_config import get_logger

logger = get_logger("mesh.routes")
router = APIRouter(prefix="/mesh", tags=["mesh"])


# ============================================================
# WebSocket connection manager — broadcasts events to all
# connected dashboard clients
# ============================================================

class _ConnectionManager:
    def __init__(self):
        self._connections: list[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._connections.append(ws)
        logger.debug("Dashboard connected (%d total)", len(self._connections))

    def disconnect(self, ws: WebSocket) -> None:
        try:
            self._connections.remove(ws)
        except ValueError:
            pass
        logger.debug("Dashboard disconnected (%d total)", len(self._connections))

    async def broadcast(self, event: dict) -> None:
        """Send event to all connected dashboard clients."""
        message = json.dumps(event)
        dead = []
        for ws in list(self._connections):
            try:
                await ws.send_text(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

    def connection_count(self) -> int:
        return len(self._connections)


_manager = _ConnectionManager()


async def broadcast_event(event_type: str, data: dict) -> None:
    """Broadcast a mesh event to all connected dashboard clients."""
    await _manager.broadcast({"type": event_type, "data": data, "ts": time.time()})


# ============================================================
# Pydantic models
# ============================================================

class MemoryWriteRequest(BaseModel):
    agent_id: str
    content: str
    tags: Optional[list[str]] = None


class NoteRequest(BaseModel):
    from_agent: str
    to_agent: str
    content: str
    tags: Optional[list[str]] = None


# ============================================================
# Agent Registry
# ============================================================

@router.get("/agents")
async def list_agents(active_only: bool = True):
    """List all agents. active_only=true (default) returns only recently-seen agents."""
    if active_only:
        agents = registry.list_active()
    else:
        agents = registry.list_all()
    return {"agents": agents, "count": len(agents)}


@router.get("/agents/{agent_id}")
async def get_agent(agent_id: str):
    """Get details for a specific agent."""
    agent = registry.get(agent_id)
    if agent is None:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
    return agent


# ============================================================
# Shared Memory Pool
# ============================================================

@router.get("/memory")
async def read_shared_memory(
    limit: int = 50,
    agent_id: Optional[str] = None,
    since: Optional[float] = None,
):
    """Read the shared memory pool. Optionally filter by agent or timestamp."""
    entries = pool.read(limit=limit, agent_id=agent_id, since=since)
    return {"entries": entries, "count": len(entries), "total": pool.count()}


@router.post("/memory")
async def write_shared_memory(body: MemoryWriteRequest):
    """Write a finding to the shared memory pool."""
    try:
        entry = pool.write(
            agent_id=body.agent_id,
            content=body.content,
            tags=body.tags,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Broadcast to dashboard clients
    await broadcast_event("memory_written", entry)
    return entry


@router.delete("/memory/{entry_id}")
async def delete_memory_entry(entry_id: str):
    """Delete a shared memory entry."""
    if not pool.delete(entry_id):
        raise HTTPException(status_code=404, detail=f"Entry '{entry_id}' not found")
    await broadcast_event("memory_deleted", {"id": entry_id})
    return {"success": True, "id": entry_id}


# ============================================================
# Noteboard
# ============================================================

@router.get("/notes")
async def get_notes(
    agent_id: Optional[str] = None,
    unread_only: bool = False,
    limit: int = 50,
):
    """
    Read notes. If agent_id is provided, returns notes addressed to that agent.
    Without agent_id, returns all notes (for dashboard overview).
    """
    if agent_id:
        notes = noteboard.get_for_agent(agent_id, unread_only=unread_only, limit=limit)
    else:
        notes = noteboard.get_all(limit=limit)
    return {"notes": notes, "count": len(notes)}


@router.post("/notes")
async def post_note(body: NoteRequest):
    """Post a note from one agent to another (or to 'any' for broadcast)."""
    try:
        note = noteboard.post(
            from_agent=body.from_agent,
            to_agent=body.to_agent,
            content=body.content,
            tags=body.tags,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Broadcast to dashboard
    await broadcast_event("note_posted", note)
    return note


@router.put("/notes/{note_id}/read")
async def mark_note_read(note_id: str):
    """Mark a note as read."""
    if not noteboard.mark_read(note_id):
        raise HTTPException(status_code=404, detail=f"Note '{note_id}' not found")
    await broadcast_event("note_read", {"id": note_id})
    return {"success": True, "id": note_id}


@router.delete("/notes/{note_id}")
async def delete_note(note_id: str):
    """Delete a note."""
    if not noteboard.delete(note_id):
        raise HTTPException(status_code=404, detail=f"Note '{note_id}' not found")
    await broadcast_event("note_deleted", {"id": note_id})
    return {"success": True, "id": note_id}


# ============================================================
# Status
# ============================================================

@router.get("/status")
async def mesh_status():
    """Overall mesh status — agent counts, memory size, pending notes."""
    active = registry.list_active()
    return {
        "active_agents": len(active),
        "total_agents": len(registry.list_all()),
        "shared_memory_entries": pool.count(),
        "dashboard_connections": _manager.connection_count(),
        "agents": active,
    }


# ============================================================
# WebSocket — real-time dashboard stream
# ============================================================

@router.websocket("/stream")
async def mesh_stream(ws: WebSocket):
    """
    WebSocket endpoint for the real-time Mesh dashboard.

    On connection, sends a full state snapshot. Subsequently pushes events
    whenever agents connect, memory is written, or notes are posted.

    Message format: { "type": event_type, "data": {...}, "ts": unix_timestamp }
    """
    await _manager.connect(ws)
    try:
        # Send initial state snapshot
        snapshot = {
            "type": "snapshot",
            "data": {
                "agents": registry.list_active(),
                "memory": pool.read(limit=20),
                "notes": noteboard.get_all(limit=20),
            },
            "ts": time.time(),
        }
        await ws.send_text(json.dumps(snapshot))

        # Keep connection alive with periodic heartbeat pings
        while True:
            await asyncio.sleep(15)
            try:
                ping = {
                    "type": "ping",
                    "data": {
                        "active_agents": len(registry.list_active()),
                        "memory_count": pool.count(),
                    },
                    "ts": time.time(),
                }
                await ws.send_text(json.dumps(ping))
            except Exception:
                break

    except WebSocketDisconnect:
        pass
    finally:
        _manager.disconnect(ws)
