"""
Shared state for route modules.

Holds the Anamnesis memory instance and import history store.
Initialized once by server.py at startup; accessed by routes via getters.
"""
import json
import os
import tempfile
import time
from pathlib import Path
from typing import List

from backend.config import VelquaConfig as Config


# Singleton memory instance, set by init_shared()
_memory = None


def init_shared(memory_instance):
    """Called once at startup to inject the Anamnesis instance."""
    global _memory
    _memory = memory_instance


def get_memory():
    """Access the shared Anamnesis instance from any route module."""
    return _memory


class ImportHistoryStore:
    """
    Persists import history to a JSON file so it survives server restarts.

    Each entry records what was imported, how many facts were stored vs
    deduplicated, and the IDs of stored facts (for undo).
    """

    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or Config.DATA_DIR
        self.file_path = self.data_dir / "import_history.json"
        self._history = self._load()

    def _load(self) -> List[dict]:
        if self.file_path.exists():
            try:
                with open(self.file_path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                return []
        return []

    def _save(self):
        # Atomic write: temp file + rename prevents corruption on crash
        fd, tmp_path = tempfile.mkstemp(
            dir=str(self.data_dir), suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(self._history, f, indent=2)
            os.replace(tmp_path, str(self.file_path))
        except BaseException:
            # Clean up temp file if rename failed
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def record(self, file_type: str, facts_stored: int, duplicates: int,
               filename: str = "", fact_ids: list = None) -> str:
        """Record an import event. Returns the batch ID."""
        batch_id = f"import-{int(time.time() * 1000)}"
        self._history.append({
            "batch_id": batch_id,
            "file_type": file_type,
            "filename": filename,
            "facts_stored": facts_stored,
            "duplicates_skipped": duplicates,
            "timestamp": time.time(),
            "fact_ids": fact_ids or [],
        })
        self._save()
        return batch_id

    def list_all(self) -> List[dict]:
        return list(reversed(self._history))

    def get_batch(self, batch_id: str) -> dict:
        for entry in self._history:
            if entry["batch_id"] == batch_id:
                return entry
        return None

    def mark_undone(self, batch_id: str, facts_deleted: int):
        """Mark a batch as undone after undo operation."""
        batch = self.get_batch(batch_id)
        if batch:
            batch["undone"] = True
            batch["facts_deleted"] = facts_deleted
            self._save()

    def count(self) -> int:
        return len(self._history)


# Singleton import history, shared across route modules
import_history = ImportHistoryStore()
