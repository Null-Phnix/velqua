"""
Mesh SQLite database — tables for agent registry, shared memory, and noteboard.

Uses a separate DB file (data/mesh.db) so it doesn't touch the Anamnesis schema.
All tables are auto-created on first access.
"""
import json
import sqlite3
import threading
from pathlib import Path
from backend.config import VelquaConfig as Config


_db_path: Path = Config.DATA_DIR / "mesh.db"
_local = threading.local()


def set_db_path(path: Path) -> None:
    """Override DB path (used in tests)."""
    global _db_path
    _db_path = Path(path)
    # Close any cached connection so the next get_conn() opens the new path
    if hasattr(_local, "conn") and _local.conn:
        try:
            _local.conn.close()
        except Exception:
            pass
        _local.conn = None


def close_conn() -> None:
    """Close the thread-local connection if open (used in test teardown)."""
    if hasattr(_local, "conn") and _local.conn:
        try:
            _local.conn.close()
        except Exception:
            pass
        _local.conn = None


def get_conn() -> sqlite3.Connection:
    """Return a thread-local SQLite connection, creating tables on first access."""
    if not getattr(_local, "conn", None):
        _db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(_db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        _local.conn = conn
        _ensure_tables(conn)
    return _local.conn


def _ensure_tables(conn: sqlite3.Connection) -> None:
    """Create mesh tables if they don't exist yet."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS mesh_agents (
            id          TEXT PRIMARY KEY,
            name        TEXT NOT NULL,
            last_seen   REAL NOT NULL,
            current_task TEXT DEFAULT '',
            status      TEXT DEFAULT 'active',
            metadata    TEXT DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS mesh_memory (
            id          TEXT PRIMARY KEY,
            agent_id    TEXT NOT NULL,
            content     TEXT NOT NULL,
            timestamp   REAL NOT NULL,
            tags        TEXT DEFAULT '[]'
        );
        CREATE INDEX IF NOT EXISTS idx_mesh_memory_ts ON mesh_memory(timestamp DESC);
        CREATE INDEX IF NOT EXISTS idx_mesh_memory_agent ON mesh_memory(agent_id);

        CREATE TABLE IF NOT EXISTS mesh_notes (
            id          TEXT PRIMARY KEY,
            from_agent  TEXT NOT NULL,
            to_agent    TEXT NOT NULL,
            content     TEXT NOT NULL,
            timestamp   REAL NOT NULL,
            read        INTEGER DEFAULT 0,
            tags        TEXT DEFAULT '[]'
        );
        CREATE INDEX IF NOT EXISTS idx_mesh_notes_to ON mesh_notes(to_agent, read);
        CREATE INDEX IF NOT EXISTS idx_mesh_notes_ts ON mesh_notes(timestamp DESC);
    """)
    conn.commit()
