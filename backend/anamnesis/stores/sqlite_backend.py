"""
SQLite backend for memory storage.

Handles structured data persistence with full-text search support.
"""

import json
import logging
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "data/anamnesis.db"


class SQLiteBackend:
    """SQLite database backend for memory storage."""

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = os.environ.get("ANAMNESIS_DB", DEFAULT_DB_PATH)
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._persistent_conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def connect(self):
        """Open a persistent connection for batch operations."""
        if self._persistent_conn is None:
            self._persistent_conn = sqlite3.connect(self.db_path)
            self._persistent_conn.row_factory = sqlite3.Row

    def close(self):
        """Close the persistent connection."""
        if self._persistent_conn is not None:
            self._persistent_conn.close()
            self._persistent_conn = None

    @contextmanager
    def _get_conn(self):
        """Get a database connection with proper cleanup."""
        if self._persistent_conn is not None:
            yield self._persistent_conn
            self._persistent_conn.commit()
        else:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
                conn.commit()
            finally:
                conn.close()

    def _has_column(self, cursor: sqlite3.Cursor, table: str, column: str) -> bool:
        """Check if a column exists in a table."""
        cursor.execute(f"PRAGMA table_info({table})")
        return any(row[1] == column for row in cursor.fetchall())

    def _add_column_if_missing(
        self, cursor: sqlite3.Cursor, table: str, column: str, col_type: str
    ):
        """Add a column to a table if it doesn't exist."""
        if not self._has_column(cursor, table, column):
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")

    def _ensure_fts_tables(self, cursor: sqlite3.Cursor):
        """Create or migrate FTS5 tables to standalone (non-external-content) mode."""
        # Check if episodes_fts exists and is external-content (broken)
        # If so, drop and recreate as standalone
        needs_recreate = False
        try:
            cursor.execute("SELECT sql FROM sqlite_master WHERE name = 'episodes_fts'")
            row = cursor.fetchone()
            if row is None:
                needs_recreate = True
            elif "content=" in (row[0] or ""):
                needs_recreate = True
        except sqlite3.OperationalError:
            needs_recreate = True

        if needs_recreate:
            try:
                cursor.execute("DROP TABLE IF EXISTS episodes_fts")
            except sqlite3.DatabaseError:
                pass
            cursor.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5(
                    id, summary, topic
                )
            """)
            # Populate from episodes table
            cursor.execute("""
                INSERT OR IGNORE INTO episodes_fts (id, summary, topic)
                SELECT id, COALESCE(summary, ''), COALESCE(topic, '')
                FROM episodes
            """)

        # Same for facts_fts
        needs_recreate = False
        try:
            cursor.execute("SELECT sql FROM sqlite_master WHERE name = 'facts_fts'")
            row = cursor.fetchone()
            if row is None:
                needs_recreate = True
            elif "content=" in (row[0] or ""):
                needs_recreate = True
        except sqlite3.OperationalError:
            needs_recreate = True

        if needs_recreate:
            try:
                cursor.execute("DROP TABLE IF EXISTS facts_fts")
            except sqlite3.DatabaseError:
                pass
            cursor.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(
                    id, content, fact_type
                )
            """)
            cursor.execute("""
                INSERT OR IGNORE INTO facts_fts (id, content, fact_type)
                SELECT id, content, COALESCE(fact_type, 'general')
                FROM facts
            """)

    def _init_db(self):
        """Initialize database schema."""
        with self._get_conn() as conn:
            cursor = conn.cursor()

            # Episodes table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS episodes (
                    id TEXT PRIMARY KEY,
                    summary TEXT,
                    messages TEXT,  -- JSON
                    topic TEXT,
                    started_at TEXT,
                    ended_at TEXT,
                    overall_valence INTEGER DEFAULT 0,
                    importance REAL DEFAULT 0.5,
                    source_id TEXT,
                    metadata TEXT,  -- JSON
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    last_accessed TEXT,
                    access_count INTEGER DEFAULT 0
                )
            """)

            # Migrations for episodes
            self._add_column_if_missing(cursor, "episodes", "last_accessed", "TEXT")
            self._add_column_if_missing(cursor, "episodes", "access_count", "INTEGER DEFAULT 0")
            self._add_column_if_missing(cursor, "episodes", "tags", "TEXT DEFAULT '[]'")

            # Facts table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS facts (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    fact_type TEXT DEFAULT 'general',
                    confidence REAL DEFAULT 0.8,
                    source_episodes TEXT,  -- JSON array of episode IDs
                    first_learned TEXT,
                    last_confirmed TEXT,
                    confirmation_count INTEGER DEFAULT 1,
                    is_superseded INTEGER DEFAULT 0,
                    importance REAL DEFAULT 0.5,
                    metadata TEXT,  -- JSON
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    last_accessed TEXT,
                    access_count INTEGER DEFAULT 0
                )
            """)

            # Migrations for facts
            self._add_column_if_missing(cursor, "facts", "last_accessed", "TEXT")
            self._add_column_if_missing(cursor, "facts", "access_count", "INTEGER DEFAULT 0")
            self._add_column_if_missing(cursor, "facts", "tags", "TEXT DEFAULT '[]'")

            # Memories table (generic)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    memory_type TEXT DEFAULT 'episodic',
                    created_at TEXT,
                    last_accessed TEXT,
                    access_count INTEGER DEFAULT 0,
                    importance REAL DEFAULT 0.5,
                    decay_rate REAL DEFAULT 0.1,
                    valence INTEGER DEFAULT 0,
                    tags TEXT,  -- JSON array
                    source_conversation_id TEXT,
                    metadata TEXT,  -- JSON
                    embedding BLOB  -- Binary embedding vector
                )
            """)

            # Conversations table (raw import)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    summary TEXT,
                    messages TEXT,  -- JSON
                    created_at TEXT,
                    updated_at TEXT,
                    metadata TEXT,  -- JSON
                    processed INTEGER DEFAULT 0
                )
            """)

            # Create FTS5 tables for full-text search (standalone, not external content)
            # External content FTS5 causes desync/corruption on INSERT OR REPLACE.
            # Standalone FTS tables are self-contained and reliable.
            self._ensure_fts_tables(cursor)

            # Fact feedback table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS fact_feedback (
                    id TEXT PRIMARY KEY,
                    fact_id TEXT NOT NULL,
                    is_positive INTEGER NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_fact ON fact_feedback(fact_id)")

            # Fact relationships table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS fact_relationships (
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    relationship_type TEXT NOT NULL,
                    confidence REAL DEFAULT 0.5,
                    evidence TEXT DEFAULT '',
                    metadata TEXT DEFAULT '{}',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (source_id, target_id, relationship_type)
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_rel_source ON fact_relationships(source_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_rel_target ON fact_relationships(target_id)")

            # Indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_episodes_started_at ON episodes(started_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_episodes_importance ON episodes(importance)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_facts_type ON facts(fact_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_facts_confidence ON facts(confidence)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_conversations_processed ON conversations(processed)")

    # Episode operations
    def save_episode(self, episode: Dict[str, Any]) -> str:
        """Save an episode."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO episodes
                (id, summary, messages, topic, started_at, ended_at,
                 overall_valence, importance, source_id, metadata, tags)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                episode["id"],
                episode.get("summary", ""),
                json.dumps(episode.get("messages", [])),
                episode.get("topic"),
                episode.get("started_at"),
                episode.get("ended_at"),
                episode.get("overall_valence", 0),
                episode.get("importance", 0.5),
                episode.get("source_id"),
                json.dumps(episode.get("metadata", {})),
                json.dumps(episode.get("tags", [])),
            ))

            # Update FTS (delete-then-insert to prevent desync)
            cursor.execute("DELETE FROM episodes_fts WHERE id = ?", (episode["id"],))
            cursor.execute("""
                INSERT INTO episodes_fts (id, summary, topic)
                VALUES (?, ?, ?)
            """, (episode["id"], episode.get("summary", ""), episode.get("topic", "")))

            return episode["id"]

    def get_episode(self, episode_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve an episode by ID."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM episodes WHERE id = ?", (episode_id,))
            row = cursor.fetchone()
            if row:
                return self._row_to_episode(row)
            return None

    def _row_to_episode(self, row: sqlite3.Row) -> Dict[str, Any]:
        """Convert database row to episode dict."""
        return {
            "id": row["id"],
            "summary": row["summary"],
            "messages": json.loads(row["messages"]) if row["messages"] else [],
            "topic": row["topic"],
            "started_at": row["started_at"],
            "ended_at": row["ended_at"],
            "overall_valence": row["overall_valence"],
            "importance": row["importance"],
            "source_id": row["source_id"],
            "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
            "last_accessed": row["last_accessed"] if "last_accessed" in row.keys() else None,
            "access_count": row["access_count"] if "access_count" in row.keys() else 0,
            "tags": json.loads(row["tags"]) if "tags" in row.keys() and row["tags"] else [],
        }

    def delete_episode(self, episode_id: str) -> bool:
        """Permanently delete an episode from the database."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            # Delete from FTS first
            cursor.execute("DELETE FROM episodes_fts WHERE id = ?", (episode_id,))
            # Delete from main table
            cursor.execute("DELETE FROM episodes WHERE id = ?", (episode_id,))
            return cursor.rowcount > 0

    def search_episodes(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Full-text search for episodes."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            # Try FTS first with wildcard suffix for partial matching
            fts_query = " OR ".join(f'"{word}"*' for word in query.split() if word)
            try:
                cursor.execute("""
                    SELECT e.* FROM episodes e
                    JOIN episodes_fts fts ON e.id = fts.id
                    WHERE episodes_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                """, (fts_query, limit))
                results = cursor.fetchall()
                if results:
                    return [self._row_to_episode(row) for row in results]
            except sqlite3.OperationalError:
                logger.debug("FTS search failed for episodes, falling back to LIKE")

            # Fallback to LIKE search
            like_pattern = f"%{query}%"
            cursor.execute("""
                SELECT * FROM episodes
                WHERE summary LIKE ? OR topic LIKE ?
                ORDER BY importance DESC
                LIMIT ?
            """, (like_pattern, like_pattern, limit))
            return [self._row_to_episode(row) for row in cursor.fetchall()]

    def list_episodes(
        self,
        limit: int = 100,
        offset: int = 0,
        order_by: str = "started_at",
        descending: bool = True
    ) -> List[Dict[str, Any]]:
        """List episodes with pagination."""
        order = "DESC" if descending else "ASC"
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                SELECT * FROM episodes
                ORDER BY {order_by} {order}
                LIMIT ? OFFSET ?
            """, (limit, offset))
            return [self._row_to_episode(row) for row in cursor.fetchall()]

    # Fact operations
    def save_fact(self, fact: Dict[str, Any]) -> str:
        """Save a fact."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO facts
                (id, content, fact_type, confidence, source_episodes,
                 first_learned, last_confirmed, confirmation_count,
                 is_superseded, importance, metadata, tags)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                fact["id"],
                fact["content"],
                fact.get("fact_type", "general"),
                fact.get("confidence", 0.8),
                json.dumps(fact.get("source_episodes", [])),
                fact.get("first_learned"),
                fact.get("last_confirmed"),
                fact.get("confirmation_count", 1),
                1 if fact.get("is_superseded") else 0,
                fact.get("importance", 0.5),
                json.dumps(fact.get("metadata", {})),
                json.dumps(fact.get("tags", [])),
            ))

            # Update FTS (delete-then-insert to prevent desync)
            cursor.execute("DELETE FROM facts_fts WHERE id = ?", (fact["id"],))
            cursor.execute("""
                INSERT INTO facts_fts (id, content, fact_type)
                VALUES (?, ?, ?)
            """, (fact["id"], fact["content"], fact.get("fact_type", "general")))

            return fact["id"]

    def get_fact(self, fact_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a fact by ID."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM facts WHERE id = ?", (fact_id,))
            row = cursor.fetchone()
            if row:
                return self._row_to_fact(row)
            return None

    def _row_to_fact(self, row: sqlite3.Row) -> Dict[str, Any]:
        """Convert database row to fact dict."""
        return {
            "id": row["id"],
            "content": row["content"],
            "fact_type": row["fact_type"],
            "confidence": row["confidence"],
            "source_episodes": json.loads(row["source_episodes"]) if row["source_episodes"] else [],
            "first_learned": row["first_learned"],
            "last_confirmed": row["last_confirmed"],
            "confirmation_count": row["confirmation_count"],
            "is_superseded": bool(row["is_superseded"]),
            "importance": row["importance"],
            "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
            "last_accessed": row["last_accessed"] if "last_accessed" in row.keys() else None,
            "access_count": row["access_count"] if "access_count" in row.keys() else 0,
            "tags": json.loads(row["tags"]) if "tags" in row.keys() and row["tags"] else [],
        }

    def delete_fact(self, fact_id: str) -> bool:
        """Permanently delete a fact from the database."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            # Delete from FTS first
            cursor.execute("DELETE FROM facts_fts WHERE id = ?", (fact_id,))
            # Delete from main table
            cursor.execute("DELETE FROM facts WHERE id = ?", (fact_id,))
            return cursor.rowcount > 0

    def search_facts(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Full-text search for facts."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            # Try FTS first with wildcard suffix
            fts_query = " OR ".join(f'"{word}"*' for word in query.split() if word)
            try:
                cursor.execute("""
                    SELECT f.* FROM facts f
                    JOIN facts_fts fts ON f.id = fts.id
                    WHERE facts_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                """, (fts_query, limit))
                results = cursor.fetchall()
                if results:
                    return [self._row_to_fact(row) for row in results]
            except sqlite3.OperationalError:
                logger.debug("FTS search failed for facts, falling back to LIKE")

            # Fallback to LIKE search
            like_pattern = f"%{query}%"
            cursor.execute("""
                SELECT * FROM facts
                WHERE content LIKE ?
                ORDER BY importance DESC, confidence DESC
                LIMIT ?
            """, (like_pattern, limit))
            return [self._row_to_fact(row) for row in cursor.fetchall()]

    def list_facts(
        self,
        limit: int = 100,
        offset: int = 0,
        fact_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List facts with optional type filter."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            if fact_type:
                cursor.execute("""
                    SELECT * FROM facts WHERE fact_type = ?
                    ORDER BY importance DESC, confidence DESC
                    LIMIT ? OFFSET ?
                """, (fact_type, limit, offset))
            else:
                cursor.execute("""
                    SELECT * FROM facts
                    ORDER BY importance DESC, confidence DESC
                    LIMIT ? OFFSET ?
                """, (limit, offset))
            return [self._row_to_fact(row) for row in cursor.fetchall()]

    # Conversation operations (raw import)
    def save_conversation(self, convo: Dict[str, Any]) -> str:
        """Save a raw conversation."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO conversations
                (id, name, summary, messages, created_at, updated_at, metadata, processed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                convo["id"],
                convo.get("name"),
                convo.get("summary"),
                json.dumps(convo.get("messages", [])),
                convo.get("created_at"),
                convo.get("updated_at"),
                json.dumps(convo.get("metadata", {})),
                1 if convo.get("processed") else 0,
            ))
            return convo["id"]

    def get_unprocessed_conversations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get conversations that haven't been processed yet."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM conversations WHERE processed = 0
                ORDER BY created_at ASC
                LIMIT ?
            """, (limit,))
            return [self._row_to_conversation(row) for row in cursor.fetchall()]

    def mark_conversation_processed(self, convo_id: str):
        """Mark a conversation as processed."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE conversations SET processed = 1 WHERE id = ?",
                (convo_id,)
            )

    def _row_to_conversation(self, row: sqlite3.Row) -> Dict[str, Any]:
        """Convert database row to conversation dict."""
        return {
            "id": row["id"],
            "name": row["name"],
            "summary": row["summary"],
            "messages": json.loads(row["messages"]) if row["messages"] else [],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
            "processed": bool(row["processed"]),
        }

    # Access tracking
    def record_episode_access(
        self,
        episode_id: str,
        reinforce_importance: bool = True,
        importance_boost: float = 0.02
    ) -> bool:
        """
        Record that an episode was accessed.

        Updates last_accessed timestamp, increments access_count,
        and optionally boosts importance.

        Args:
            episode_id: ID of the episode
            reinforce_importance: Whether to boost importance on access
            importance_boost: How much to boost importance (capped at 1.0)

        Returns:
            True if episode was found and updated
        """
        with self._get_conn() as conn:
            cursor = conn.cursor()
            now = datetime.now().isoformat()

            if reinforce_importance:
                cursor.execute("""
                    UPDATE episodes
                    SET last_accessed = ?,
                        access_count = access_count + 1,
                        importance = MIN(1.0, importance + ?)
                    WHERE id = ?
                """, (now, importance_boost, episode_id))
            else:
                cursor.execute("""
                    UPDATE episodes
                    SET last_accessed = ?,
                        access_count = access_count + 1
                    WHERE id = ?
                """, (now, episode_id))

            return cursor.rowcount > 0

    def record_fact_access(
        self,
        fact_id: str,
        reinforce_importance: bool = True,
        importance_boost: float = 0.02
    ) -> bool:
        """
        Record that a fact was accessed.

        Updates last_accessed timestamp, increments access_count,
        and optionally boosts importance.

        Args:
            fact_id: ID of the fact
            reinforce_importance: Whether to boost importance on access
            importance_boost: How much to boost importance (capped at 1.0)

        Returns:
            True if fact was found and updated
        """
        with self._get_conn() as conn:
            cursor = conn.cursor()
            now = datetime.now().isoformat()

            if reinforce_importance:
                cursor.execute("""
                    UPDATE facts
                    SET last_accessed = ?,
                        access_count = access_count + 1,
                        confirmation_count = confirmation_count + 1,
                        last_confirmed = ?,
                        importance = MIN(1.0, importance + ?)
                    WHERE id = ?
                """, (now, now, importance_boost, fact_id))
            else:
                cursor.execute("""
                    UPDATE facts
                    SET last_accessed = ?,
                        access_count = access_count + 1,
                        confirmation_count = confirmation_count + 1,
                        last_confirmed = ?
                    WHERE id = ?
                """, (now, now, fact_id))

            return cursor.rowcount > 0

    def record_batch_access(
        self,
        episode_ids: List[str] = None,
        fact_ids: List[str] = None,
        reinforce_importance: bool = True,
        importance_boost: float = 0.02
    ) -> Dict[str, int]:
        """
        Record access for multiple memories at once.

        Args:
            episode_ids: List of episode IDs to mark as accessed
            fact_ids: List of fact IDs to mark as accessed
            reinforce_importance: Whether to boost importance
            importance_boost: How much to boost importance

        Returns:
            Dict with counts of episodes and facts updated
        """
        episode_ids = episode_ids or []
        fact_ids = fact_ids or []

        episodes_updated = 0
        facts_updated = 0

        for ep_id in episode_ids:
            if self.record_episode_access(ep_id, reinforce_importance, importance_boost):
                episodes_updated += 1

        for fact_id in fact_ids:
            if self.record_fact_access(fact_id, reinforce_importance, importance_boost):
                facts_updated += 1

        return {
            "episodes_updated": episodes_updated,
            "facts_updated": facts_updated,
        }

    def get_access_stats(self) -> Dict[str, Any]:
        """Get access statistics for memories."""
        with self._get_conn() as conn:
            cursor = conn.cursor()

            # Most accessed episodes
            cursor.execute("""
                SELECT id, topic, access_count, importance
                FROM episodes
                WHERE access_count > 0
                ORDER BY access_count DESC
                LIMIT 10
            """)
            most_accessed_episodes = [
                {"id": r["id"], "topic": r["topic"],
                 "access_count": r["access_count"], "importance": r["importance"]}
                for r in cursor.fetchall()
            ]

            # Most accessed facts
            cursor.execute("""
                SELECT id, content, access_count, importance
                FROM facts
                WHERE access_count > 0
                ORDER BY access_count DESC
                LIMIT 10
            """)
            most_accessed_facts = [
                {"id": r["id"], "content": r["content"][:50],
                 "access_count": r["access_count"], "importance": r["importance"]}
                for r in cursor.fetchall()
            ]

            # Total access counts
            cursor.execute("SELECT SUM(access_count) FROM episodes")
            total_episode_accesses = cursor.fetchone()[0] or 0

            cursor.execute("SELECT SUM(access_count) FROM facts")
            total_fact_accesses = cursor.fetchone()[0] or 0

            # Recently accessed
            cursor.execute("""
                SELECT COUNT(*) FROM episodes
                WHERE last_accessed > datetime('now', '-7 days')
            """)
            recent_episode_accesses = cursor.fetchone()[0]

            cursor.execute("""
                SELECT COUNT(*) FROM facts
                WHERE last_accessed > datetime('now', '-7 days')
            """)
            recent_fact_accesses = cursor.fetchone()[0]

            return {
                "total_episode_accesses": total_episode_accesses,
                "total_fact_accesses": total_fact_accesses,
                "recent_episode_accesses": recent_episode_accesses,
                "recent_fact_accesses": recent_fact_accesses,
                "most_accessed_episodes": most_accessed_episodes,
                "most_accessed_facts": most_accessed_facts,
            }

    # Stats
    def get_stats(self) -> Dict[str, int]:
        """Get counts for all tables."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            stats = {}
            for table in ["episodes", "facts", "memories", "conversations"]:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                stats[table] = cursor.fetchone()[0]
            return stats

    # === SQL-filtered queries (Phase 2) ===

    def get_episodes_by_timerange(
        self, start_iso: str, end_iso: str, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get episodes within a time range using SQL WHERE."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM episodes
                WHERE started_at >= ? AND started_at <= ?
                ORDER BY started_at DESC
                LIMIT ?
            """, (start_iso, end_iso, limit))
            return [self._row_to_episode(row) for row in cursor.fetchall()]

    def get_episodes_by_importance(
        self, threshold: float = 0.7, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Get episodes above importance threshold using SQL."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM episodes
                WHERE importance >= ?
                ORDER BY importance DESC
                LIMIT ?
            """, (threshold, limit))
            return [self._row_to_episode(row) for row in cursor.fetchall()]

    def get_episodes_by_valence(
        self, valence: int, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Get episodes with specific emotional valence using SQL."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM episodes
                WHERE overall_valence = ?
                ORDER BY importance DESC, started_at DESC
                LIMIT ?
            """, (valence, limit))
            return [self._row_to_episode(row) for row in cursor.fetchall()]

    def get_episodes_most_accessed(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most accessed episodes using SQL."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM episodes
                WHERE access_count > 0
                ORDER BY access_count DESC
                LIMIT ?
            """, (limit,))
            return [self._row_to_episode(row) for row in cursor.fetchall()]

    def get_episodes_recently_accessed(
        self, cutoff_iso: str, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Get recently accessed episodes using SQL."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM episodes
                WHERE last_accessed >= ?
                ORDER BY last_accessed DESC
                LIMIT ?
            """, (cutoff_iso, limit))
            return [self._row_to_episode(row) for row in cursor.fetchall()]

    def get_facts_by_confidence(
        self, threshold: float = 0.9, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get facts above confidence threshold using SQL."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM facts
                WHERE confidence >= ? AND is_superseded = 0
                ORDER BY confidence DESC
                LIMIT ?
            """, (threshold, limit))
            return [self._row_to_fact(row) for row in cursor.fetchall()]

    def get_facts_most_accessed(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most accessed facts using SQL."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM facts
                WHERE access_count > 0 AND is_superseded = 0
                ORDER BY access_count DESC
                LIMIT ?
            """, (limit,))
            return [self._row_to_fact(row) for row in cursor.fetchall()]

    def get_episode_count(self) -> int:
        """Get episode count without loading all records."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM episodes")
            return cursor.fetchone()[0]

    def get_fact_count(self) -> int:
        """Get fact count without loading all records."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM facts WHERE is_superseded = 0")
            return cursor.fetchone()[0]

    def touch_episodes_batch(
        self, episode_ids: List[str], importance_boost: float = 0.02
    ) -> int:
        """Touch multiple episodes in a single SQL statement."""
        if not episode_ids:
            return 0
        with self._get_conn() as conn:
            cursor = conn.cursor()
            now = datetime.now().isoformat()
            placeholders = ",".join("?" for _ in episode_ids)
            cursor.execute(f"""
                UPDATE episodes
                SET last_accessed = ?,
                    access_count = access_count + 1,
                    importance = MIN(1.0, importance + ?)
                WHERE id IN ({placeholders})
            """, [now, importance_boost] + list(episode_ids))
            return cursor.rowcount

    def touch_facts_batch(
        self, fact_ids: List[str], importance_boost: float = 0.02
    ) -> int:
        """Touch multiple facts in a single SQL statement."""
        if not fact_ids:
            return 0
        with self._get_conn() as conn:
            cursor = conn.cursor()
            now = datetime.now().isoformat()
            placeholders = ",".join("?" for _ in fact_ids)
            cursor.execute(f"""
                UPDATE facts
                SET last_accessed = ?,
                    access_count = access_count + 1,
                    confirmation_count = confirmation_count + 1,
                    last_confirmed = ?,
                    importance = MIN(1.0, importance + ?)
                WHERE id IN ({placeholders})
            """, [now, now, importance_boost] + list(fact_ids))
            return cursor.rowcount

    def rebuild_fts(self) -> Dict[str, int]:
        """
        Rebuild FTS5 indexes from content tables.

        Drops and recreates FTS tables as standalone (non-external-content),
        then populates from the content tables.

        Returns:
            Dict with counts of episodes and facts re-indexed
        """
        with self._get_conn() as conn:
            cursor = conn.cursor()

            # Drop and recreate episodes FTS as standalone
            try:
                cursor.execute("DROP TABLE IF EXISTS episodes_fts")
            except sqlite3.DatabaseError:
                # If corrupt, force drop via sqlite_master
                pass
            cursor.execute("""
                CREATE VIRTUAL TABLE episodes_fts USING fts5(
                    id, summary, topic
                )
            """)
            cursor.execute("""
                INSERT INTO episodes_fts (id, summary, topic)
                SELECT id, COALESCE(summary, ''), COALESCE(topic, '')
                FROM episodes
            """)
            cursor.execute("SELECT COUNT(*) FROM episodes")
            episode_count = cursor.fetchone()[0]

            # Drop and recreate facts FTS as standalone
            try:
                cursor.execute("DROP TABLE IF EXISTS facts_fts")
            except sqlite3.DatabaseError:
                pass
            cursor.execute("""
                CREATE VIRTUAL TABLE facts_fts USING fts5(
                    id, content, fact_type
                )
            """)
            cursor.execute("""
                INSERT INTO facts_fts (id, content, fact_type)
                SELECT id, content, COALESCE(fact_type, 'general')
                FROM facts
            """)
            cursor.execute("SELECT COUNT(*) FROM facts")
            fact_count = cursor.fetchone()[0]

            return {"episodes": episode_count, "facts": fact_count}

    # === Fact feedback ===

    def save_fact_feedback(self, fact_id: str, is_positive: bool) -> str:
        """Record a feedback event (thumbs up/down) for a fact."""
        import uuid as _uuid
        feedback_id = str(_uuid.uuid4())
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO fact_feedback (id, fact_id, is_positive, created_at) "
                "VALUES (?, ?, ?, ?)",
                (feedback_id, fact_id, 1 if is_positive else 0, datetime.now().isoformat()),
            )
        return feedback_id

    def get_fact_feedback_summary(self, fact_id: str) -> Dict[str, int]:
        """Return {"thumbs_up": N, "thumbs_down": N} for a fact."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT is_positive, COUNT(*) FROM fact_feedback "
                "WHERE fact_id = ? GROUP BY is_positive",
                (fact_id,),
            )
            counts = {"thumbs_up": 0, "thumbs_down": 0}
            for row in cursor.fetchall():
                if row[0]:
                    counts["thumbs_up"] = row[1]
                else:
                    counts["thumbs_down"] = row[1]
            return counts

    def get_fact_feedback_summaries(self, fact_ids: List[str]) -> Dict[str, Dict[str, int]]:
        """Batch fetch feedback summaries for multiple facts."""
        if not fact_ids:
            return {}
        result = {fid: {"thumbs_up": 0, "thumbs_down": 0} for fid in fact_ids}
        with self._get_conn() as conn:
            cursor = conn.cursor()
            placeholders = ",".join("?" for _ in fact_ids)
            cursor.execute(
                f"SELECT fact_id, is_positive, COUNT(*) FROM fact_feedback "
                f"WHERE fact_id IN ({placeholders}) GROUP BY fact_id, is_positive",
                list(fact_ids),
            )
            for row in cursor.fetchall():
                fid, is_pos, cnt = row[0], row[1], row[2]
                if fid in result:
                    if is_pos:
                        result[fid]["thumbs_up"] = cnt
                    else:
                        result[fid]["thumbs_down"] = cnt
        return result

    def delete_fact_feedback(self, fact_id: str) -> int:
        """Delete all feedback for a fact. Returns count deleted."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM fact_feedback WHERE fact_id = ?",
                (fact_id,),
            )
            count = cursor.fetchone()[0]
            cursor.execute("DELETE FROM fact_feedback WHERE fact_id = ?", (fact_id,))
            return count

    # === Fact relationships ===

    def save_fact_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str,
        confidence: float,
        evidence: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Save or upsert a fact relationship edge."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO fact_relationships "
                "(source_id, target_id, relationship_type, confidence, evidence, metadata, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(source_id, target_id, relationship_type) DO UPDATE SET "
                "confidence = excluded.confidence, evidence = excluded.evidence, "
                "metadata = excluded.metadata",
                (
                    source_id, target_id, relationship_type, confidence, evidence,
                    json.dumps(metadata or {}), datetime.now().isoformat(),
                ),
            )
            return f"{source_id}->{target_id}"

    def get_fact_relationships(
        self,
        fact_id: Optional[str] = None,
        relationship_type: Optional[str] = None,
        min_confidence: float = 0.0,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Query fact relationships with optional filters."""
        conditions = ["confidence >= ?"]
        params: list = [min_confidence]
        if fact_id:
            conditions.append("(source_id = ? OR target_id = ?)")
            params.extend([fact_id, fact_id])
        if relationship_type:
            conditions.append("relationship_type = ?")
            params.append(relationship_type)
        where = " AND ".join(conditions)
        params.extend([limit, offset])
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT * FROM fact_relationships WHERE {where} "
                "ORDER BY confidence DESC LIMIT ? OFFSET ?",
                params,
            )
            return [self._row_to_relationship(r) for r in cursor.fetchall()]

    def count_fact_relationships(
        self,
        fact_id: Optional[str] = None,
        relationship_type: Optional[str] = None,
        min_confidence: float = 0.0,
    ) -> int:
        """Count fact relationships matching filters."""
        conditions = ["confidence >= ?"]
        params: list = [min_confidence]
        if fact_id:
            conditions.append("(source_id = ? OR target_id = ?)")
            params.extend([fact_id, fact_id])
        if relationship_type:
            conditions.append("relationship_type = ?")
            params.append(relationship_type)
        where = " AND ".join(conditions)
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT COUNT(*) FROM fact_relationships WHERE {where}",
                params,
            )
            return cursor.fetchone()[0]

    def delete_fact_relationships(self, fact_id: str) -> int:
        """Delete all relationships involving a fact."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM fact_relationships WHERE source_id = ? OR target_id = ?",
                (fact_id, fact_id),
            )
            return cursor.rowcount

    def clear_fact_relationships(self) -> int:
        """Delete all relationships."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM fact_relationships")
            count = cursor.fetchone()[0]
            cursor.execute("DELETE FROM fact_relationships")
            return count

    def _row_to_relationship(self, row: sqlite3.Row) -> Dict[str, Any]:
        """Convert a fact_relationships row to dict."""
        return {
            "source_id": row["source_id"],
            "target_id": row["target_id"],
            "relationship_type": row["relationship_type"],
            "confidence": row["confidence"],
            "evidence": row["evidence"],
            "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
            "created_at": row["created_at"],
        }

    def clear_all(self):
        """Clear all data (use with caution!)."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            for table in ["episodes", "facts", "memories", "conversations", "episodes_fts", "facts_fts"]:
                cursor.execute(f"DELETE FROM {table}")
            # Also clear extension tables if they exist
            for table in ["fact_feedback", "fact_relationships"]:
                try:
                    cursor.execute(f"DELETE FROM {table}")
                except sqlite3.OperationalError:
                    pass
