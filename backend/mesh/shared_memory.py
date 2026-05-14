"""
Shared Memory Pool — cross-agent knowledge store.

Agents can write findings here; other agents receive relevant entries
injected into their context alongside personal memory. Namespaced separately
from personal Anamnesis facts so they never mix.

Write explicitly via POST /mesh/memory, or Velqua auto-extracts from responses
when the proxy detects factual assertions worth sharing.
"""
import json
import time
import uuid
from typing import Optional

from backend.mesh.db import get_conn
from backend.logging_config import get_logger

logger = get_logger("mesh.shared_memory")

MAX_CONTENT_LENGTH = 1000
MAX_ENTRIES_LIMIT = 200


class SharedMemoryPool:
    """Thread-safe shared memory pool backed by SQLite mesh_memory table."""

    def write(
        self,
        agent_id: str,
        content: str,
        tags: list[str] | None = None,
    ) -> dict:
        """
        Write a finding to the shared pool.

        Returns the stored entry (with generated id and timestamp).
        """
        content = content.strip()[:MAX_CONTENT_LENGTH]
        if not content:
            raise ValueError("content cannot be empty")

        entry_id = str(uuid.uuid4())
        now = time.time()
        tags_json = json.dumps(tags or [])

        conn = get_conn()
        conn.execute(
            "INSERT INTO mesh_memory (id, agent_id, content, timestamp, tags) VALUES (?, ?, ?, ?, ?)",
            (entry_id, agent_id, content, now, tags_json),
        )
        conn.commit()
        logger.info("Shared memory write: [%s] %s…", agent_id, content[:60])
        return {"id": entry_id, "agent_id": agent_id, "content": content, "timestamp": now, "tags": tags or []}

    def read(
        self,
        limit: int = 50,
        agent_id: Optional[str] = None,
        since: Optional[float] = None,
    ) -> list[dict]:
        """
        Read recent entries from the shared pool.

        Args:
            limit: Max entries to return.
            agent_id: Filter to a specific agent's writes.
            since: Only return entries after this Unix timestamp.
        """
        limit = min(limit, MAX_ENTRIES_LIMIT)
        conn = get_conn()

        clauses = []
        params: list = []

        if agent_id:
            clauses.append("agent_id = ?")
            params.append(agent_id)
        if since is not None:
            clauses.append("timestamp > ?")
            params.append(since)

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(limit)

        rows = conn.execute(
            f"SELECT * FROM mesh_memory {where} ORDER BY timestamp DESC LIMIT ?",
            params,
        ).fetchall()
        return [_memory_row_to_dict(r) for r in rows]

    def search(self, query: str, limit: int = 10) -> list[dict]:
        """Simple keyword search over shared memory content."""
        limit = min(limit, MAX_ENTRIES_LIMIT)
        conn = get_conn()
        pattern = f"%{query.lower()}%"
        rows = conn.execute(
            "SELECT * FROM mesh_memory WHERE LOWER(content) LIKE ? ORDER BY timestamp DESC LIMIT ?",
            (pattern, limit),
        ).fetchall()
        return [_memory_row_to_dict(r) for r in rows]

    def delete(self, entry_id: str) -> bool:
        """Delete a specific entry. Returns True if found and deleted."""
        conn = get_conn()
        cursor = conn.execute("DELETE FROM mesh_memory WHERE id = ?", (entry_id,))
        conn.commit()
        return cursor.rowcount > 0

    def count(self) -> int:
        """Total entries in the shared pool."""
        conn = get_conn()
        return conn.execute("SELECT COUNT(*) FROM mesh_memory").fetchone()[0]


def _memory_row_to_dict(row) -> dict:
    d = dict(row)
    d["tags"] = json.loads(d.get("tags") or "[]")
    return d


# Module-level singleton
pool = SharedMemoryPool()
