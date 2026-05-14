"""
Memory graph implementation.

Stores and queries associative links between memories.
"""

import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class LinkType(Enum):
    """Types of links between memories."""
    RELATED = "related"       # Generally related topics
    FOLLOWS = "follows"       # Temporal sequence (A happened before B)
    CONFIRMS = "confirms"     # One memory confirms another
    CONTRADICTS = "contradicts"  # One memory contradicts another
    MENTIONS = "mentions"     # One memory mentions entity in another
    SIMILAR = "similar"       # High similarity score
    CUSTOM = "custom"         # User-defined link


@dataclass
class MemoryLink:
    """A link between two memories."""
    source_id: str      # Source memory ID
    target_id: str      # Target memory ID
    link_type: LinkType
    weight: float       # Link strength (0.0-1.0)
    metadata: Dict[str, Any]
    created_at: datetime


class MemoryGraph:
    """
    Graph structure for memory associations.

    Stores bidirectional links between memories with types and weights.
    Supports efficient traversal and querying.
    """

    def __init__(self, db_path: str = "anamnesis.db"):
        """
        Initialize memory graph.

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = Path(db_path)
        self._init_db()

    def _get_conn(self):
        """Get database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Initialize graph tables."""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memory_links (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                link_type TEXT NOT NULL,
                weight REAL DEFAULT 0.5,
                metadata TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(source_id, target_id, link_type)
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_links_source ON memory_links(source_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_links_target ON memory_links(target_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_links_type ON memory_links(link_type)
        """)

        conn.commit()
        conn.close()

    def add_link(
        self,
        source_id: str,
        target_id: str,
        link_type: LinkType,
        weight: float = 0.5,
        metadata: Optional[Dict] = None,
        bidirectional: bool = True,
    ) -> bool:
        """
        Add a link between two memories.

        Args:
            source_id: Source memory ID
            target_id: Target memory ID
            link_type: Type of relationship
            weight: Link strength (0.0-1.0)
            metadata: Optional metadata
            bidirectional: If True, also add reverse link

        Returns:
            True if link was added
        """
        if source_id == target_id:
            return False

        conn = self._get_conn()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                INSERT OR REPLACE INTO memory_links
                (source_id, target_id, link_type, weight, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                source_id,
                target_id,
                link_type.value,
                weight,
                json.dumps(metadata or {}),
                datetime.now().isoformat(),
            ))

            if bidirectional:
                # Add reverse link with same type (or appropriate reverse type)
                reverse_type = self._get_reverse_type(link_type)
                cursor.execute("""
                    INSERT OR REPLACE INTO memory_links
                    (source_id, target_id, link_type, weight, metadata, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    target_id,
                    source_id,
                    reverse_type.value,
                    weight,
                    json.dumps(metadata or {}),
                    datetime.now().isoformat(),
                ))

            conn.commit()
            return True

        except sqlite3.Error as e:
            logger.warning("Failed to add link %s -> %s: %s", source_id, target_id, e)
            conn.rollback()
            return False
        finally:
            conn.close()

    def _get_reverse_type(self, link_type: LinkType) -> LinkType:
        """Get the reverse link type."""
        reverse_map = {
            LinkType.FOLLOWS: LinkType.FOLLOWS,  # Both directions valid
            LinkType.CONFIRMS: LinkType.CONFIRMS,
            LinkType.CONTRADICTS: LinkType.CONTRADICTS,
            LinkType.RELATED: LinkType.RELATED,
            LinkType.MENTIONS: LinkType.MENTIONS,
            LinkType.SIMILAR: LinkType.SIMILAR,
            LinkType.CUSTOM: LinkType.CUSTOM,
        }
        return reverse_map.get(link_type, LinkType.RELATED)

    def remove_link(
        self,
        source_id: str,
        target_id: str,
        link_type: Optional[LinkType] = None,
    ) -> bool:
        """
        Remove a link between memories.

        Args:
            source_id: Source memory ID
            target_id: Target memory ID
            link_type: If provided, only remove this type

        Returns:
            True if link was removed
        """
        conn = self._get_conn()
        cursor = conn.cursor()

        try:
            if link_type:
                cursor.execute("""
                    DELETE FROM memory_links
                    WHERE source_id = ? AND target_id = ? AND link_type = ?
                """, (source_id, target_id, link_type.value))
            else:
                cursor.execute("""
                    DELETE FROM memory_links
                    WHERE source_id = ? AND target_id = ?
                """, (source_id, target_id))

            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def get_links(
        self,
        memory_id: str,
        direction: str = "both",
        link_type: Optional[LinkType] = None,
        min_weight: float = 0.0,
    ) -> List[MemoryLink]:
        """
        Get links for a memory.

        Args:
            memory_id: Memory ID to query
            direction: "outgoing", "incoming", or "both"
            link_type: Filter by link type
            min_weight: Minimum link weight

        Returns:
            List of MemoryLink objects
        """
        conn = self._get_conn()
        cursor = conn.cursor()

        links = []

        # Outgoing links
        if direction in ("outgoing", "both"):
            query = """
                SELECT * FROM memory_links
                WHERE source_id = ? AND weight >= ?
            """
            params = [memory_id, min_weight]

            if link_type:
                query += " AND link_type = ?"
                params.append(link_type.value)

            cursor.execute(query, params)
            for row in cursor.fetchall():
                links.append(self._row_to_link(row))

        # Incoming links
        if direction in ("incoming", "both"):
            query = """
                SELECT * FROM memory_links
                WHERE target_id = ? AND weight >= ?
            """
            params = [memory_id, min_weight]

            if link_type:
                query += " AND link_type = ?"
                params.append(link_type.value)

            cursor.execute(query, params)
            for row in cursor.fetchall():
                links.append(self._row_to_link(row))

        conn.close()
        return links

    def get_related(
        self,
        memory_id: str,
        depth: int = 1,
        min_weight: float = 0.3,
    ) -> List[str]:
        """
        Get related memory IDs via graph traversal.

        Args:
            memory_id: Starting memory ID
            depth: How many hops to traverse
            min_weight: Minimum link weight to follow

        Returns:
            List of related memory IDs
        """
        visited = {memory_id}
        frontier = [memory_id]

        for _ in range(depth):
            next_frontier = []

            for current_id in frontier:
                links = self.get_links(
                    current_id,
                    direction="both",
                    min_weight=min_weight,
                )

                for link in links:
                    # Determine the connected ID
                    connected_id = (
                        link.target_id if link.source_id == current_id
                        else link.source_id
                    )

                    if connected_id not in visited:
                        visited.add(connected_id)
                        next_frontier.append(connected_id)

            frontier = next_frontier

        visited.discard(memory_id)
        return list(visited)

    def _row_to_link(self, row: sqlite3.Row) -> MemoryLink:
        """Convert database row to MemoryLink."""
        return MemoryLink(
            source_id=row["source_id"],
            target_id=row["target_id"],
            link_type=LinkType(row["link_type"]),
            weight=row["weight"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM memory_links")
        total_links = cursor.fetchone()[0]

        cursor.execute("""
            SELECT link_type, COUNT(*) as count
            FROM memory_links
            GROUP BY link_type
        """)
        by_type = {row["link_type"]: row["count"] for row in cursor.fetchall()}

        cursor.execute("SELECT COUNT(DISTINCT source_id) FROM memory_links")
        unique_sources = cursor.fetchone()[0]

        conn.close()

        return {
            "total_links": total_links,
            "by_type": by_type,
            "unique_memories": unique_sources,
        }

    def auto_link_similar(
        self,
        memory_id: str,
        candidate_ids: List[str],
        similarity_fn,
        threshold: float = 0.7,
    ) -> int:
        """
        Automatically create SIMILAR links based on similarity function.

        Args:
            memory_id: Source memory ID
            candidate_ids: IDs to compare against
            similarity_fn: Function(id1, id2) -> float
            threshold: Minimum similarity to create link

        Returns:
            Number of links created
        """
        created = 0

        for candidate_id in candidate_ids:
            if candidate_id == memory_id:
                continue

            try:
                similarity = similarity_fn(memory_id, candidate_id)
                if similarity >= threshold:
                    if self.add_link(
                        memory_id,
                        candidate_id,
                        LinkType.SIMILAR,
                        weight=similarity,
                        bidirectional=True,
                    ):
                        created += 1
            except (ValueError, TypeError, KeyError) as e:
                logger.debug("Similarity check failed for %s vs %s: %s", memory_id, candidate_id, e)
                continue

        return created

    def clear_links(self, memory_id: Optional[str] = None) -> int:
        """
        Clear links.

        Args:
            memory_id: If provided, clear only links for this memory.
                      If None, clear all links.

        Returns:
            Number of links cleared
        """
        conn = self._get_conn()
        cursor = conn.cursor()

        if memory_id:
            cursor.execute("""
                DELETE FROM memory_links
                WHERE source_id = ? OR target_id = ?
            """, (memory_id, memory_id))
        else:
            cursor.execute("DELETE FROM memory_links")

        count = cursor.rowcount
        conn.commit()
        conn.close()

        return count
