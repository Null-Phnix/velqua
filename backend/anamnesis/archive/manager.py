"""
Memory archive manager.

Handles archiving, compression, and restoration of memories.
"""

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..models import EmotionalValence, Episode, Fact
from ..stores.episodic import EpisodicStore
from ..stores.semantic import SemanticStore


@dataclass
class ArchiveEntry:
    """An archived memory entry."""
    id: str
    entry_type: str  # "episode" or "fact"
    summary: str
    topic: Optional[str]
    importance: float
    archived_at: datetime
    original_date: Optional[datetime]
    archive_reason: str
    compressed_data: Dict[str, Any]  # Compressed version of original


@dataclass
class ArchiveRule:
    """Rule for auto-archiving."""
    name: str
    min_age_days: int = 90
    max_importance: float = 0.5
    max_access_count: int = 2
    enabled: bool = True


@dataclass
class ArchiveStats:
    """Statistics about the archive."""
    total_archived: int
    episodes_archived: int
    facts_archived: int
    oldest_entry: Optional[datetime]
    newest_entry: Optional[datetime]
    total_size_kb: float
    space_saved_estimate_kb: float


class ArchiveManager:
    """
    Manages memory archiving and restoration.

    Archives compress memories by:
    - Removing raw messages from episodes
    - Keeping only summaries and key metadata
    - Storing in separate archive database
    """

    def __init__(
        self,
        episodic_store: EpisodicStore,
        semantic_store: SemanticStore,
        archive_path: Optional[str] = None,
    ):
        """
        Initialize archive manager.

        Args:
            episodic_store: Episodic memory store
            semantic_store: Semantic fact store
            archive_path: Path for archive database
        """
        self.episodic_store = episodic_store
        self.semantic_store = semantic_store

        if archive_path:
            self.archive_path = Path(archive_path)
        else:
            # Default to same directory as main db
            self.archive_path = Path("anamnesis_archive.db")

        self._init_archive_db()

    def _init_archive_db(self):
        """Initialize archive database schema."""
        self.archive_path.parent.mkdir(parents=True, exist_ok=True)

        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS archive (
                    id TEXT PRIMARY KEY,
                    entry_type TEXT NOT NULL,
                    summary TEXT,
                    topic TEXT,
                    importance REAL,
                    archived_at TEXT NOT NULL,
                    original_date TEXT,
                    archive_reason TEXT,
                    compressed_data TEXT
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_archive_type ON archive(entry_type)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_archive_date ON archive(archived_at)
            """)
            conn.commit()

    @contextmanager
    def _get_conn(self):
        """Get archive database connection."""
        conn = sqlite3.connect(self.archive_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def archive_episode(
        self,
        episode_id: str,
        reason: str = "manual",
        delete_original: bool = True,
    ) -> Optional[ArchiveEntry]:
        """
        Archive an episode.

        Args:
            episode_id: Episode to archive
            reason: Reason for archiving
            delete_original: Whether to delete from main store

        Returns:
            ArchiveEntry if successful
        """
        episode = self.episodic_store.get(episode_id)
        if not episode:
            return None

        # Compress episode
        compressed = self._compress_episode(episode)

        entry = ArchiveEntry(
            id=episode.id,
            entry_type="episode",
            summary=episode.summary or "",
            topic=episode.topic,
            importance=episode.importance,
            archived_at=datetime.now(),
            original_date=episode.started_at,
            archive_reason=reason,
            compressed_data=compressed,
        )

        # Save to archive
        self._save_archive_entry(entry)

        # Delete from main store (hard delete)
        if delete_original:
            self.episodic_store.delete(episode_id, hard=True)

        return entry

    def archive_fact(
        self,
        fact_id: str,
        reason: str = "manual",
        delete_original: bool = True,
    ) -> Optional[ArchiveEntry]:
        """
        Archive a fact.

        Args:
            fact_id: Fact to archive
            reason: Reason for archiving
            delete_original: Whether to delete from main store

        Returns:
            ArchiveEntry if successful
        """
        fact = self.semantic_store.get(fact_id)
        if not fact:
            return None

        # Compress fact (facts are already small)
        compressed = {
            "content": fact.content,
            "fact_type": fact.fact_type,
            "confidence": fact.confidence,
            "source_episodes": fact.source_episodes,
            "first_learned": fact.first_learned.isoformat() if fact.first_learned else None,
            "confirmation_count": fact.confirmation_count,
        }

        entry = ArchiveEntry(
            id=fact.id,
            entry_type="fact",
            summary=fact.content[:100],
            topic=fact.fact_type,
            importance=fact.importance,
            archived_at=datetime.now(),
            original_date=fact.first_learned,
            archive_reason=reason,
            compressed_data=compressed,
        )

        # Save to archive
        self._save_archive_entry(entry)

        # Delete from main store (hard delete)
        if delete_original:
            self.semantic_store.delete(fact_id, hard=True)

        return entry

    def _compress_episode(self, episode: Episode) -> Dict[str, Any]:
        """
        Compress an episode for archiving.

        Removes raw messages, keeps only summary and metadata.
        """
        return {
            "summary": episode.summary,
            "topic": episode.topic,
            "message_count": len(episode.messages),
            "first_message_preview": (
                episode.messages[0].get("content", "")[:200]
                if episode.messages else None
            ),
            "valence": (
                episode.overall_valence.value
                if hasattr(episode.overall_valence, 'value')
                else episode.overall_valence
            ),
            "source_id": episode.source_id,
            "started_at": episode.started_at.isoformat() if episode.started_at else None,
            "ended_at": episode.ended_at.isoformat() if episode.ended_at else None,
            "key_metadata": {
                k: v for k, v in episode.metadata.items()
                if k in ["access_count", "continues_from", "continues_to"]
            },
        }

    def _save_archive_entry(self, entry: ArchiveEntry):
        """Save an archive entry to database."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO archive
                (id, entry_type, summary, topic, importance, archived_at,
                 original_date, archive_reason, compressed_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry.id,
                entry.entry_type,
                entry.summary,
                entry.topic,
                entry.importance,
                entry.archived_at.isoformat(),
                entry.original_date.isoformat() if entry.original_date else None,
                entry.archive_reason,
                json.dumps(entry.compressed_data),
            ))

    def restore_episode(self, archive_id: str) -> Optional[Episode]:
        """
        Restore an episode from archive.

        Note: Raw messages are lost, only summary is restored.
        """
        entry = self.get_archived(archive_id)
        if not entry or entry.entry_type != "episode":
            return None

        data = entry.compressed_data

        # Reconstruct episode (without original messages)
        episode = Episode(
            id=entry.id,
            messages=[],  # Messages are lost
            summary=data.get("summary"),
            topic=data.get("topic"),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            ended_at=datetime.fromisoformat(data["ended_at"]) if data.get("ended_at") else None,
            overall_valence=EmotionalValence(data.get("valence", 0)),
            importance=entry.importance,
            source_id=data.get("source_id"),
            metadata={
                "restored_from_archive": True,
                "archive_date": entry.archived_at.isoformat(),
                "original_message_count": data.get("message_count", 0),
                **data.get("key_metadata", {}),
            },
        )

        # Save to main store
        self.episodic_store.save(episode)

        # Remove from archive
        self._delete_archive_entry(archive_id)

        return episode

    def restore_fact(self, archive_id: str) -> Optional[Fact]:
        """Restore a fact from archive."""
        entry = self.get_archived(archive_id)
        if not entry or entry.entry_type != "fact":
            return None

        data = entry.compressed_data

        fact = Fact(
            id=entry.id,
            content=data.get("content", ""),
            fact_type=data.get("fact_type", "unknown"),
            confidence=data.get("confidence", 0.5),
            importance=entry.importance,
            source_episodes=data.get("source_episodes", []),
            first_learned=datetime.fromisoformat(data["first_learned"]) if data.get("first_learned") else None,
            confirmation_count=data.get("confirmation_count", 1),
            metadata={"restored_from_archive": True},
        )

        self.semantic_store.save(fact)
        self._delete_archive_entry(archive_id)

        return fact

    def get_archived(self, archive_id: str) -> Optional[ArchiveEntry]:
        """Get an archived entry by ID."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM archive WHERE id = ?", (archive_id,))
            row = cursor.fetchone()

            if not row:
                return None

            return ArchiveEntry(
                id=row["id"],
                entry_type=row["entry_type"],
                summary=row["summary"],
                topic=row["topic"],
                importance=row["importance"],
                archived_at=datetime.fromisoformat(row["archived_at"]),
                original_date=datetime.fromisoformat(row["original_date"]) if row["original_date"] else None,
                archive_reason=row["archive_reason"],
                compressed_data=json.loads(row["compressed_data"]) if row["compressed_data"] else {},
            )

    def list_archived(
        self,
        entry_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[ArchiveEntry]:
        """List archived entries."""
        with self._get_conn() as conn:
            cursor = conn.cursor()

            if entry_type:
                cursor.execute(
                    "SELECT * FROM archive WHERE entry_type = ? ORDER BY archived_at DESC LIMIT ?",
                    (entry_type, limit)
                )
            else:
                cursor.execute(
                    "SELECT * FROM archive ORDER BY archived_at DESC LIMIT ?",
                    (limit,)
                )

            entries = []
            for row in cursor.fetchall():
                entries.append(ArchiveEntry(
                    id=row["id"],
                    entry_type=row["entry_type"],
                    summary=row["summary"],
                    topic=row["topic"],
                    importance=row["importance"],
                    archived_at=datetime.fromisoformat(row["archived_at"]),
                    original_date=datetime.fromisoformat(row["original_date"]) if row["original_date"] else None,
                    archive_reason=row["archive_reason"],
                    compressed_data=json.loads(row["compressed_data"]) if row["compressed_data"] else {},
                ))

            return entries

    def _delete_archive_entry(self, archive_id: str):
        """Delete an entry from archive."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM archive WHERE id = ?", (archive_id,))

    def auto_archive(
        self,
        rule: Optional[ArchiveRule] = None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        Automatically archive memories based on rules.

        Args:
            rule: Archive rule to apply (default rule if None)
            dry_run: If True, don't actually archive, just report

        Returns:
            Dict with archive results
        """
        if rule is None:
            rule = ArchiveRule(name="default")

        results = {
            "rule": rule.name,
            "episodes_archived": 0,
            "facts_archived": 0,
            "candidates": [],
            "dry_run": dry_run,
        }

        cutoff_date = datetime.now() - timedelta(days=rule.min_age_days)

        # Check episodes
        episodes = self.episodic_store.list_all(limit=10000)
        for ep in episodes:
            if self._should_archive_episode(ep, rule, cutoff_date):
                results["candidates"].append({
                    "id": ep.id,
                    "type": "episode",
                    "topic": ep.topic,
                    "importance": ep.importance,
                    "age_days": (datetime.now() - ep.started_at).days if ep.started_at else None,
                })

                if not dry_run:
                    self.archive_episode(ep.id, reason=f"auto:{rule.name}")
                    results["episodes_archived"] += 1

        # Check facts
        facts = self.semantic_store.list_all(limit=10000)
        for fact in facts:
            if self._should_archive_fact(fact, rule, cutoff_date):
                results["candidates"].append({
                    "id": fact.id,
                    "type": "fact",
                    "content": fact.content[:50],
                    "importance": fact.importance,
                })

                if not dry_run:
                    self.archive_fact(fact.id, reason=f"auto:{rule.name}")
                    results["facts_archived"] += 1

        return results

    def _should_archive_episode(
        self,
        episode: Episode,
        rule: ArchiveRule,
        cutoff_date: datetime,
    ) -> bool:
        """Check if episode should be archived."""
        # Check age
        if episode.started_at and episode.started_at > cutoff_date:
            return False

        # Check importance
        if episode.importance > rule.max_importance:
            return False

        # Check access count
        access_count = episode.metadata.get("access_count", 0)
        if access_count > rule.max_access_count:
            return False

        return True

    def _should_archive_fact(
        self,
        fact: Fact,
        rule: ArchiveRule,
        cutoff_date: datetime,
    ) -> bool:
        """Check if fact should be archived."""
        # Check age
        if fact.first_learned and fact.first_learned > cutoff_date:
            return False

        # Check importance
        if fact.importance > rule.max_importance:
            return False

        # Don't archive confirmed facts
        if fact.confirmation_count > rule.max_access_count:
            return False

        return True

    def get_stats(self) -> ArchiveStats:
        """Get archive statistics."""
        with self._get_conn() as conn:
            cursor = conn.cursor()

            # Count entries
            cursor.execute("SELECT COUNT(*) as total FROM archive")
            total = cursor.fetchone()["total"]

            cursor.execute("SELECT COUNT(*) as count FROM archive WHERE entry_type = 'episode'")
            episodes = cursor.fetchone()["count"]

            cursor.execute("SELECT COUNT(*) as count FROM archive WHERE entry_type = 'fact'")
            facts = cursor.fetchone()["count"]

            # Date range
            cursor.execute("SELECT MIN(archived_at) as oldest, MAX(archived_at) as newest FROM archive")
            row = cursor.fetchone()
            oldest = datetime.fromisoformat(row["oldest"]) if row["oldest"] else None
            newest = datetime.fromisoformat(row["newest"]) if row["newest"] else None

        # Estimate file size
        size_kb = self.archive_path.stat().st_size / 1024 if self.archive_path.exists() else 0

        # Rough estimate of space saved (compressed is ~10% of original)
        space_saved = size_kb * 9  # Rough estimate

        return ArchiveStats(
            total_archived=total,
            episodes_archived=episodes,
            facts_archived=facts,
            oldest_entry=oldest,
            newest_entry=newest,
            total_size_kb=size_kb,
            space_saved_estimate_kb=space_saved,
        )
