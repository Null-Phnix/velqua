"""
Agent Identity Registry — detect who is talking through the proxy and track them.

Detection priority (highest to lowest):
  1. X-Velqua-Agent header — explicit declaration by the agent
  2. User-Agent string — heuristic matching for known tools
  3. Fallback — "unknown" (grouped by session fingerprint if available)

The registry persists to SQLite so the dashboard sees history across restarts.
Active agents are those seen within the last ACTIVE_TIMEOUT_SECONDS seconds.
"""
import json
import re
import sqlite3
import time
import uuid
from typing import Optional

from backend.mesh.db import get_conn
from backend.logging_config import get_logger

logger = get_logger("mesh.registry")

ACTIVE_TIMEOUT_SECONDS = 120  # Agent considered "active" if seen in last 2 minutes

# User-agent pattern → friendly name mapping
_UA_PATTERNS = [
    (r"(?i)blackreach", "blackreach"),
    (r"(?i)open.?webui", "open-webui"),
    (r"(?i)continue\.dev", "continue"),
    (r"(?i)cursor", "cursor"),
    (r"(?i)anamnesis", "anamnesis"),
    (r"(?i)python-httpx", "python-script"),
    (r"(?i)python-requests", "python-script"),
]


def _detect_from_ua(user_agent: str) -> Optional[str]:
    """Guess an agent name from the User-Agent string."""
    if not user_agent:
        return None
    for pattern, name in _UA_PATTERNS:
        if re.search(pattern, user_agent):
            return name
    return None


def detect_agent_id(
    x_velqua_agent: Optional[str] = None,
    user_agent: Optional[str] = None,
) -> str:
    """
    Determine agent identity for an incoming request.

    Returns a stable, lowercase identifier like "blackreach" or "unknown-7a3f".
    Does NOT persist to DB — call heartbeat() to record activity.
    """
    # 1. Explicit header wins
    if x_velqua_agent:
        return x_velqua_agent.strip().lower()[:64]

    # 2. User-agent heuristic
    ua_name = _detect_from_ua(user_agent or "")
    if ua_name:
        return ua_name

    # 3. Fallback — anonymous but stable within session (not persisted)
    return "unknown"


class AgentRegistry:
    """Thread-safe agent registry backed by SQLite."""

    def heartbeat(
        self,
        agent_id: str,
        task_hint: str = "",
        metadata: dict | None = None,
    ) -> None:
        """Record that agent_id is alive. Call on every proxied request."""
        now = time.time()
        conn = get_conn()
        # Use the last N words of the first user message as the current task hint
        task = task_hint[:200] if task_hint else ""
        meta_json = json.dumps(metadata or {})
        conn.execute("""
            INSERT INTO mesh_agents (id, name, last_seen, current_task, status, metadata)
            VALUES (?, ?, ?, ?, 'active', ?)
            ON CONFLICT(id) DO UPDATE SET
                last_seen    = excluded.last_seen,
                current_task = CASE WHEN excluded.current_task != '' THEN excluded.current_task ELSE current_task END,
                status       = 'active',
                metadata     = excluded.metadata
        """, (agent_id, agent_id, now, task, meta_json))
        conn.commit()
        logger.debug("Heartbeat: %s — %s", agent_id, task[:60] if task else "(no task)")

    def list_active(self, timeout_seconds: int = ACTIVE_TIMEOUT_SECONDS) -> list[dict]:
        """Return agents seen within the last timeout_seconds."""
        cutoff = time.time() - timeout_seconds
        conn = get_conn()
        rows = conn.execute(
            "SELECT * FROM mesh_agents WHERE last_seen >= ? ORDER BY last_seen DESC",
            (cutoff,),
        ).fetchall()
        return [_agent_row_to_dict(r) for r in rows]

    def list_all(self, limit: int = 100) -> list[dict]:
        """Return all agents ever seen, most recent first."""
        conn = get_conn()
        rows = conn.execute(
            "SELECT * FROM mesh_agents ORDER BY last_seen DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [_agent_row_to_dict(r) for r in rows]

    def get(self, agent_id: str) -> dict | None:
        """Get a specific agent by ID."""
        conn = get_conn()
        row = conn.execute(
            "SELECT * FROM mesh_agents WHERE id = ?", (agent_id,)
        ).fetchone()
        return _agent_row_to_dict(row) if row else None

    def mark_inactive(self, agent_id: str) -> None:
        """Explicitly mark an agent as inactive."""
        conn = get_conn()
        conn.execute(
            "UPDATE mesh_agents SET status = 'inactive' WHERE id = ?",
            (agent_id,),
        )
        conn.commit()


def _agent_row_to_dict(row: sqlite3.Row) -> dict:  # type: ignore[name-defined]
    import sqlite3 as _sq
    d = dict(row)
    d["metadata"] = json.loads(d.get("metadata") or "{}")
    d["is_active"] = (time.time() - d["last_seen"]) < ACTIVE_TIMEOUT_SECONDS
    d["last_seen_ago"] = int(time.time() - d["last_seen"])
    return d


# Module-level singleton used by routes and proxy
registry = AgentRegistry()
