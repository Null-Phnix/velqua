"""
Noteboard — structured inter-agent notes.

An agent can leave a note for another agent (or broadcast to "any").
When the target agent's next request passes through the proxy, unread notes
addressed to it are injected into context automatically.

Notes are simple: from, to, content, tags. Read state is tracked so the
same note isn't injected twice.
"""
import json
import time
import uuid
from typing import Optional

from backend.mesh.db import get_conn
from backend.logging_config import get_logger

logger = get_logger("mesh.noteboard")

MAX_NOTE_LENGTH = 2000
MAX_NOTES_LIMIT = 100


class Noteboard:
    """Thread-safe noteboard backed by SQLite mesh_notes table."""

    def post(
        self,
        from_agent: str,
        to_agent: str,
        content: str,
        tags: list[str] | None = None,
    ) -> dict:
        """
        Post a note from one agent to another (or to "any" for broadcast).

        Args:
            from_agent: Identity of the writing agent.
            to_agent: Target agent ID, or "any" for broadcast.
            content: Note content.
            tags: Optional labels for filtering.
        """
        content = content.strip()[:MAX_NOTE_LENGTH]
        if not content:
            raise ValueError("content cannot be empty")

        note_id = str(uuid.uuid4())
        now = time.time()
        tags_json = json.dumps(tags or [])

        conn = get_conn()
        conn.execute(
            "INSERT INTO mesh_notes (id, from_agent, to_agent, content, timestamp, read, tags) "
            "VALUES (?, ?, ?, ?, ?, 0, ?)",
            (note_id, from_agent.strip(), to_agent.strip(), content, now, tags_json),
        )
        conn.commit()
        logger.info("Note posted: %s → %s: %s…", from_agent, to_agent, content[:60])
        return {
            "id": note_id, "from_agent": from_agent, "to_agent": to_agent,
            "content": content, "timestamp": now, "read": False, "tags": tags or [],
        }

    def get_for_agent(
        self,
        agent_id: str,
        unread_only: bool = True,
        limit: int = 20,
    ) -> list[dict]:
        """
        Get notes addressed to agent_id or to "any".

        Args:
            agent_id: The reading agent's ID.
            unread_only: If True, only return notes not yet read.
            limit: Max notes to return.
        """
        limit = min(limit, MAX_NOTES_LIMIT)
        conn = get_conn()
        read_filter = "AND read = 0" if unread_only else ""
        rows = conn.execute(
            f"""SELECT * FROM mesh_notes
                WHERE (to_agent = ? OR to_agent = 'any') {read_filter}
                ORDER BY timestamp DESC LIMIT ?""",
            (agent_id, limit),
        ).fetchall()
        return [_note_row_to_dict(r) for r in rows]

    def get_all(self, limit: int = 50) -> list[dict]:
        """Get all notes (for dashboard display), most recent first."""
        limit = min(limit, MAX_NOTES_LIMIT)
        conn = get_conn()
        rows = conn.execute(
            "SELECT * FROM mesh_notes ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
        return [_note_row_to_dict(r) for r in rows]

    def mark_read(self, note_id: str) -> bool:
        """Mark a note as read. Returns True if found."""
        conn = get_conn()
        cursor = conn.execute("UPDATE mesh_notes SET read = 1 WHERE id = ?", (note_id,))
        conn.commit()
        return cursor.rowcount > 0

    def mark_all_read(self, agent_id: str) -> int:
        """Mark all unread notes for agent_id as read. Returns count."""
        conn = get_conn()
        cursor = conn.execute(
            "UPDATE mesh_notes SET read = 1 WHERE (to_agent = ? OR to_agent = 'any') AND read = 0",
            (agent_id,),
        )
        conn.commit()
        return cursor.rowcount

    def delete(self, note_id: str) -> bool:
        """Delete a note. Returns True if found."""
        conn = get_conn()
        cursor = conn.execute("DELETE FROM mesh_notes WHERE id = ?", (note_id,))
        conn.commit()
        return cursor.rowcount > 0

    def count_unread(self, agent_id: str) -> int:
        """Count unread notes for agent_id (including broadcasts)."""
        conn = get_conn()
        return conn.execute(
            "SELECT COUNT(*) FROM mesh_notes WHERE (to_agent = ? OR to_agent = 'any') AND read = 0",
            (agent_id,),
        ).fetchone()[0]


def _note_row_to_dict(row) -> dict:
    d = dict(row)
    d["tags"] = json.loads(d.get("tags") or "[]")
    d["read"] = bool(d["read"])
    return d


# Module-level singleton
noteboard = Noteboard()
