"""
Memory linking system.

Provides explicit linking between related memories.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from ..models import Episode, Fact
from ..stores.episodic import EpisodicStore
from ..stores.semantic import SemanticStore
from ..stores.sqlite_backend import SQLiteBackend


class LinkType(Enum):
    """Types of links between memories."""
    RELATED_TO = "related_to"      # General relation
    CONTINUES = "continues"         # Conversation continuation
    CONTRADICTS = "contradicts"     # Factual contradiction
    SUPERSEDES = "supersedes"       # Newer version replaces older
    SEE_ALSO = "see_also"          # Suggested related reading
    DERIVED_FROM = "derived_from"   # Fact derived from episode
    REFERENCES = "references"       # Memory references another


@dataclass
class MemoryLink:
    """A link between two memories."""
    id: str
    source_id: str
    source_type: str  # "episode" or "fact"
    target_id: str
    target_type: str  # "episode" or "fact"
    link_type: LinkType
    strength: float = 1.0  # 0.0 to 1.0
    bidirectional: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    note: str = ""  # Optional description


@dataclass
class LinkStats:
    """Statistics about links in the system."""
    total_links: int
    links_by_type: Dict[str, int]
    most_linked_memories: List[Tuple[str, str, int]]  # (id, type, count)
    orphaned_links: int  # Links to non-existent memories


class LinkManager:
    """
    Manages links between memories.

    Links can connect:
    - Episode to Episode
    - Episode to Fact
    - Fact to Fact
    """

    def __init__(
        self,
        backend: SQLiteBackend,
        episodic_store: Optional[EpisodicStore] = None,
        semantic_store: Optional[SemanticStore] = None,
    ):
        """
        Initialize link manager.

        Args:
            backend: SQLite backend for storing links
            episodic_store: Episode store for lookups
            semantic_store: Fact store for lookups
        """
        self.backend = backend
        self.episodic_store = episodic_store
        self.semantic_store = semantic_store
        self._ensure_table()

    def _ensure_table(self):
        """Ensure the links table exists."""
        with self.backend._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_links (
                    id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    target_type TEXT NOT NULL,
                    link_type TEXT NOT NULL,
                    strength REAL DEFAULT 1.0,
                    bidirectional INTEGER DEFAULT 1,
                    created_at TEXT,
                    metadata TEXT,
                    note TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_links_source
                ON memory_links(source_id, source_type)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_links_target
                ON memory_links(target_id, target_type)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_links_type
                ON memory_links(link_type)
            """)

    def create_link(
        self,
        source_id: str,
        source_type: str,
        target_id: str,
        target_type: str,
        link_type: LinkType,
        strength: float = 1.0,
        bidirectional: bool = True,
        note: str = "",
        metadata: Optional[Dict] = None,
    ) -> MemoryLink:
        """
        Create a link between two memories.

        Args:
            source_id: ID of source memory
            source_type: "episode" or "fact"
            target_id: ID of target memory
            target_type: "episode" or "fact"
            link_type: Type of relationship
            strength: Link strength (0-1)
            bidirectional: If True, link works both ways
            note: Optional description
            metadata: Additional metadata

        Returns:
            Created MemoryLink
        """
        import uuid

        # Validate memories exist
        if self.episodic_store and source_type == "episode":
            if not self.episodic_store.get(source_id):
                raise ValueError(f"Source episode not found: {source_id}")
        if self.semantic_store and source_type == "fact":
            if not self.semantic_store.get(source_id):
                raise ValueError(f"Source fact not found: {source_id}")
        if self.episodic_store and target_type == "episode":
            if not self.episodic_store.get(target_id):
                raise ValueError(f"Target episode not found: {target_id}")
        if self.semantic_store and target_type == "fact":
            if not self.semantic_store.get(target_id):
                raise ValueError(f"Target fact not found: {target_id}")

        # Check for existing link
        existing = self.get_link(source_id, target_id)
        if existing:
            # Update existing link
            return self.update_link(
                existing.id,
                link_type=link_type,
                strength=strength,
                note=note,
            )

        link = MemoryLink(
            id=str(uuid.uuid4()),
            source_id=source_id,
            source_type=source_type,
            target_id=target_id,
            target_type=target_type,
            link_type=link_type,
            strength=strength,
            bidirectional=bidirectional,
            note=note,
            metadata=metadata or {},
        )

        self._save_link(link)
        return link

    def _save_link(self, link: MemoryLink):
        """Save a link to the database."""
        with self.backend._get_conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO memory_links
                (id, source_id, source_type, target_id, target_type,
                 link_type, strength, bidirectional, created_at, metadata, note)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                link.id,
                link.source_id,
                link.source_type,
                link.target_id,
                link.target_type,
                link.link_type.value,
                link.strength,
                1 if link.bidirectional else 0,
                link.created_at.isoformat(),
                json.dumps(link.metadata),
                link.note,
            ))

    def get_link(
        self,
        source_id: str,
        target_id: str,
    ) -> Optional[MemoryLink]:
        """Get a specific link between two memories."""
        with self.backend._get_conn() as conn:
            cursor = conn.execute("""
                SELECT * FROM memory_links
                WHERE (source_id = ? AND target_id = ?)
                   OR (bidirectional = 1 AND source_id = ? AND target_id = ?)
                LIMIT 1
            """, (source_id, target_id, target_id, source_id))

            row = cursor.fetchone()
            if row:
                return self._row_to_link(row)
        return None

    def get_link_by_id(self, link_id: str) -> Optional[MemoryLink]:
        """Get a link by its ID."""
        with self.backend._get_conn() as conn:
            cursor = conn.execute(
                "SELECT * FROM memory_links WHERE id = ?",
                (link_id,)
            )
            row = cursor.fetchone()
            if row:
                return self._row_to_link(row)
        return None

    def _row_to_link(self, row) -> MemoryLink:
        """Convert database row to MemoryLink."""
        return MemoryLink(
            id=row[0],
            source_id=row[1],
            source_type=row[2],
            target_id=row[3],
            target_type=row[4],
            link_type=LinkType(row[5]),
            strength=row[6],
            bidirectional=bool(row[7]),
            created_at=datetime.fromisoformat(row[8]) if row[8] else datetime.now(),
            metadata=json.loads(row[9]) if row[9] else {},
            note=row[10] or "",
        )

    def delete_link(self, link_id: str) -> bool:
        """Delete a link by ID."""
        with self.backend._get_conn() as conn:
            cursor = conn.execute(
                "DELETE FROM memory_links WHERE id = ?",
                (link_id,)
            )
            return cursor.rowcount > 0

    def delete_links_for_memory(
        self,
        memory_id: str,
        memory_type: Optional[str] = None,
    ) -> int:
        """Delete all links involving a memory."""
        with self.backend._get_conn() as conn:
            if memory_type:
                cursor = conn.execute("""
                    DELETE FROM memory_links
                    WHERE (source_id = ? AND source_type = ?)
                       OR (target_id = ? AND target_type = ?)
                """, (memory_id, memory_type, memory_id, memory_type))
            else:
                cursor = conn.execute("""
                    DELETE FROM memory_links
                    WHERE source_id = ? OR target_id = ?
                """, (memory_id, memory_id))
            return cursor.rowcount

    def update_link(
        self,
        link_id: str,
        link_type: Optional[LinkType] = None,
        strength: Optional[float] = None,
        note: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> Optional[MemoryLink]:
        """Update an existing link."""
        link = self.get_link_by_id(link_id)
        if not link:
            return None

        if link_type is not None:
            link.link_type = link_type
        if strength is not None:
            link.strength = strength
        if note is not None:
            link.note = note
        if metadata is not None:
            link.metadata.update(metadata)

        self._save_link(link)
        return link

    def get_links_from(
        self,
        memory_id: str,
        memory_type: Optional[str] = None,
        link_type: Optional[LinkType] = None,
        limit: int = 100,
    ) -> List[MemoryLink]:
        """Get all links originating from a memory."""
        with self.backend._get_conn() as conn:
            query = "SELECT * FROM memory_links WHERE source_id = ?"
            params = [memory_id]

            if memory_type:
                query += " AND source_type = ?"
                params.append(memory_type)

            if link_type:
                query += " AND link_type = ?"
                params.append(link_type.value)

            query += " ORDER BY strength DESC LIMIT ?"
            params.append(limit)

            cursor = conn.execute(query, params)
            return [self._row_to_link(row) for row in cursor.fetchall()]

    def get_links_to(
        self,
        memory_id: str,
        memory_type: Optional[str] = None,
        link_type: Optional[LinkType] = None,
        limit: int = 100,
    ) -> List[MemoryLink]:
        """Get all links pointing to a memory."""
        with self.backend._get_conn() as conn:
            query = "SELECT * FROM memory_links WHERE target_id = ?"
            params = [memory_id]

            if memory_type:
                query += " AND target_type = ?"
                params.append(memory_type)

            if link_type:
                query += " AND link_type = ?"
                params.append(link_type.value)

            query += " ORDER BY strength DESC LIMIT ?"
            params.append(limit)

            cursor = conn.execute(query, params)
            return [self._row_to_link(row) for row in cursor.fetchall()]

    def get_all_links(
        self,
        memory_id: str,
        memory_type: Optional[str] = None,
        include_bidirectional: bool = True,
    ) -> List[MemoryLink]:
        """Get all links connected to a memory."""
        links = []

        # Get outgoing links
        links.extend(self.get_links_from(memory_id, memory_type))

        # Get incoming links (for bidirectional)
        if include_bidirectional:
            incoming = self.get_links_to(memory_id, memory_type)
            # Filter to only bidirectional links
            links.extend([link for link in incoming if link.bidirectional])

        return links

    def get_linked_memories(
        self,
        memory_id: str,
        memory_type: str,
        link_type: Optional[LinkType] = None,
        depth: int = 1,
    ) -> Dict[str, List]:
        """
        Get memories linked to a given memory.

        Args:
            memory_id: Starting memory ID
            memory_type: "episode" or "fact"
            link_type: Filter by link type
            depth: How many hops to follow (1 = direct links only)

        Returns:
            Dict with 'episodes' and 'facts' lists
        """
        visited = set()
        result = {"episodes": [], "facts": []}

        def traverse(mid: str, mtype: str, current_depth: int):
            if current_depth > depth:
                return
            if (mid, mtype) in visited:
                return

            visited.add((mid, mtype))

            links = self.get_all_links(mid, mtype)

            for link in links:
                # Determine the other end of the link
                if link.source_id == mid:
                    other_id = link.target_id
                    other_type = link.target_type
                else:
                    other_id = link.source_id
                    other_type = link.source_type

                if (other_id, other_type) in visited:
                    continue

                # Filter by link type if specified
                if link_type and link.link_type != link_type:
                    continue

                # Add to result
                if other_type == "episode" and self.episodic_store:
                    ep = self.episodic_store.get(other_id)
                    if ep and ep not in result["episodes"]:
                        result["episodes"].append(ep)
                elif other_type == "fact" and self.semantic_store:
                    fact = self.semantic_store.get(other_id)
                    if fact and fact not in result["facts"]:
                        result["facts"].append(fact)

                # Recurse
                traverse(other_id, other_type, current_depth + 1)

        traverse(memory_id, memory_type, 1)
        return result

    def find_path(
        self,
        source_id: str,
        source_type: str,
        target_id: str,
        target_type: str,
        max_depth: int = 5,
    ) -> Optional[List[MemoryLink]]:
        """
        Find a path of links between two memories.

        Returns:
            List of links forming the path, or None if no path exists
        """
        from collections import deque

        # BFS to find shortest path
        queue = deque([(source_id, source_type, [])])
        visited = {(source_id, source_type)}

        while queue:
            current_id, current_type, path = queue.popleft()

            if len(path) >= max_depth:
                continue

            links = self.get_all_links(current_id, current_type)

            for link in links:
                # Determine other end
                if link.source_id == current_id:
                    other_id = link.target_id
                    other_type = link.target_type
                else:
                    other_id = link.source_id
                    other_type = link.source_type

                if (other_id, other_type) in visited:
                    continue

                new_path = path + [link]

                if other_id == target_id and other_type == target_type:
                    return new_path

                visited.add((other_id, other_type))
                queue.append((other_id, other_type, new_path))

        return None

    def list_all_links(
        self,
        link_type: Optional[LinkType] = None,
        limit: int = 100,
    ) -> List[MemoryLink]:
        """List all links in the system."""
        with self.backend._get_conn() as conn:
            if link_type:
                cursor = conn.execute(
                    "SELECT * FROM memory_links WHERE link_type = ? ORDER BY created_at DESC LIMIT ?",
                    (link_type.value, limit)
                )
            else:
                cursor = conn.execute(
                    "SELECT * FROM memory_links ORDER BY created_at DESC LIMIT ?",
                    (limit,)
                )
            return [self._row_to_link(row) for row in cursor.fetchall()]

    def get_stats(self) -> LinkStats:
        """Get statistics about links."""
        with self.backend._get_conn() as conn:
            # Total links
            cursor = conn.execute("SELECT COUNT(*) FROM memory_links")
            total = cursor.fetchone()[0]

            # Links by type
            cursor = conn.execute(
                "SELECT link_type, COUNT(*) FROM memory_links GROUP BY link_type"
            )
            by_type = {row[0]: row[1] for row in cursor.fetchall()}

            # Most linked (combine source and target counts)
            cursor = conn.execute("""
                SELECT id, type, SUM(cnt) as total FROM (
                    SELECT source_id as id, source_type as type, COUNT(*) as cnt
                    FROM memory_links GROUP BY source_id, source_type
                    UNION ALL
                    SELECT target_id as id, target_type as type, COUNT(*) as cnt
                    FROM memory_links GROUP BY target_id, target_type
                ) GROUP BY id, type ORDER BY total DESC LIMIT 10
            """)
            most_linked = [(row[0], row[1], row[2]) for row in cursor.fetchall()]

            # Check for orphaned links
            orphaned = 0
            all_links = self.list_all_links(limit=10000)
            for link in all_links:
                source_exists = True
                target_exists = True

                if self.episodic_store:
                    if link.source_type == "episode" and not self.episodic_store.get(link.source_id):
                        source_exists = False
                    if link.target_type == "episode" and not self.episodic_store.get(link.target_id):
                        target_exists = False

                if self.semantic_store:
                    if link.source_type == "fact" and not self.semantic_store.get(link.source_id):
                        source_exists = False
                    if link.target_type == "fact" and not self.semantic_store.get(link.target_id):
                        target_exists = False

                if not source_exists or not target_exists:
                    orphaned += 1

        return LinkStats(
            total_links=total,
            links_by_type=by_type,
            most_linked_memories=most_linked,
            orphaned_links=orphaned,
        )

    def cleanup_orphaned_links(self) -> int:
        """Remove links pointing to non-existent memories."""
        orphaned_ids = []
        all_links = self.list_all_links(limit=10000)

        for link in all_links:
            source_exists = True
            target_exists = True

            if self.episodic_store:
                if link.source_type == "episode" and not self.episodic_store.get(link.source_id):
                    source_exists = False
                if link.target_type == "episode" and not self.episodic_store.get(link.target_id):
                    target_exists = False

            if self.semantic_store:
                if link.source_type == "fact" and not self.semantic_store.get(link.source_id):
                    source_exists = False
                if link.target_type == "fact" and not self.semantic_store.get(link.target_id):
                    target_exists = False

            if not source_exists or not target_exists:
                orphaned_ids.append(link.id)

        for link_id in orphaned_ids:
            self.delete_link(link_id)

        return len(orphaned_ids)

    def auto_link_continuations(
        self,
        episodes: Optional[List[Episode]] = None,
    ) -> int:
        """Automatically create continuation links between episodes."""
        if episodes is None and self.episodic_store:
            episodes = self.episodic_store.list_all(limit=1000)

        if not episodes:
            return 0

        created = 0

        for ep in episodes:
            # Check for continues_from in metadata
            continues_from = ep.metadata.get("continues_from")
            if continues_from:
                try:
                    self.create_link(
                        source_id=ep.id,
                        source_type="episode",
                        target_id=continues_from,
                        target_type="episode",
                        link_type=LinkType.CONTINUES,
                        note="Auto-detected continuation",
                    )
                    created += 1
                except ValueError:
                    pass  # Source or target doesn't exist

        return created

    def auto_link_fact_sources(
        self,
        facts: Optional[List[Fact]] = None,
    ) -> int:
        """Automatically create derived_from links for facts with source episodes."""
        if facts is None and self.semantic_store:
            facts = self.semantic_store.list_all(limit=1000)

        if not facts:
            return 0

        created = 0

        for fact in facts:
            for source_ep_id in fact.source_episodes:
                try:
                    self.create_link(
                        source_id=fact.id,
                        source_type="fact",
                        target_id=source_ep_id,
                        target_type="episode",
                        link_type=LinkType.DERIVED_FROM,
                        note="Auto-linked from source_episodes",
                    )
                    created += 1
                except ValueError:
                    pass  # Episode doesn't exist

        return created
