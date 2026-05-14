"""
Activity log SQLite storage.

Uses the same data directory as mesh.db but keeps its own file (activity.db)
so it doesn't interfere with other schemas. Thread-local connections.
"""
import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import List, Optional

from backend.config import VelquaConfig as Config

_db_path: Path = Config.DATA_DIR / "activity.db"
_local = threading.local()


def set_db_path(path: Path) -> None:
    """Override DB path (used in tests)."""
    global _db_path
    _db_path = Path(path)
    if hasattr(_local, "conn") and _local.conn:
        try:
            _local.conn.close()
        except Exception:
            pass
        _local.conn = None


def close_conn() -> None:
    """Close the thread-local connection (used in test teardown)."""
    if hasattr(_local, "conn") and _local.conn:
        try:
            _local.conn.close()
        except Exception:
            pass
        _local.conn = None


def get_conn() -> sqlite3.Connection:
    """Return a thread-local SQLite connection, creating the table on first access."""
    if not getattr(_local, "conn", None):
        _db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(_db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        _local.conn = conn
        _ensure_tables(conn)
    return _local.conn


def _ensure_tables(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS activity_events (
            id         TEXT PRIMARY KEY,
            event_type TEXT NOT NULL,
            title      TEXT NOT NULL,
            detail     TEXT DEFAULT '',
            metadata   TEXT DEFAULT '{}',
            timestamp  REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_activity_ts ON activity_events(timestamp DESC);
        CREATE INDEX IF NOT EXISTS idx_activity_type ON activity_events(event_type);
    """)
    conn.commit()


# ── Event types ──────────────────────────────────────────────────
EVENT_TYPES = {
    "fact_learned",
    "fact_approved",
    "fact_rejected",
    "fact_deleted",
    "fact_merged",
    "fact_edited",
    "import_completed",
    "import_failed",
    "backup_created",
    "backup_restored",
    "provider_changed",
    "agent_connected",
    "agent_disconnected",
    "system_started",
}


def log_event(
    event_type: str,
    title: str,
    detail: str = "",
    metadata: Optional[dict] = None,
) -> str:
    """Record an activity event. Returns the event ID."""
    event_id = str(uuid.uuid4())
    conn = get_conn()
    conn.execute(
        "INSERT INTO activity_events (id, event_type, title, detail, metadata, timestamp) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            event_id,
            event_type,
            title,
            detail,
            json.dumps(metadata or {}),
            time.time(),
        ),
    )
    conn.commit()
    return event_id


def list_events(
    limit: int = 50,
    offset: int = 0,
    event_type: Optional[str] = None,
) -> List[dict]:
    """Return events in reverse chronological order."""
    conn = get_conn()
    if event_type:
        rows = conn.execute(
            "SELECT * FROM activity_events WHERE event_type = ? "
            "ORDER BY timestamp DESC LIMIT ? OFFSET ?",
            (event_type, limit, offset),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM activity_events ORDER BY timestamp DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


def count_events(event_type: Optional[str] = None) -> int:
    conn = get_conn()
    if event_type:
        row = conn.execute(
            "SELECT COUNT(*) FROM activity_events WHERE event_type = ?",
            (event_type,),
        ).fetchone()
    else:
        row = conn.execute("SELECT COUNT(*) FROM activity_events").fetchone()
    return row[0]


def clear_events() -> int:
    """Delete all events. Returns count deleted."""
    conn = get_conn()
    count = count_events()
    conn.execute("DELETE FROM activity_events")
    conn.commit()
    return count


def _row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    try:
        d["metadata"] = json.loads(d["metadata"])
    except (json.JSONDecodeError, TypeError):
        d["metadata"] = {}
    return d
