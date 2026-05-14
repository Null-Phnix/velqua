"""Tests for backend/routes/_shared.py — ImportHistoryStore edge cases."""
import json
import pytest

from backend.routes._shared import ImportHistoryStore


class TestImportHistoryStore:
    """Test ImportHistoryStore persistence and error handling."""

    @pytest.fixture
    def store(self, tmp_path):
        return ImportHistoryStore(data_dir=tmp_path)

    def test_record_and_list(self, store):
        batch_id = store.record("claude_memories", 5, 2, "test.json", ["id1", "id2"])
        assert batch_id.startswith("import-")
        history = store.list_all()
        assert len(history) == 1
        assert history[0]["facts_stored"] == 5

    def test_get_batch(self, store):
        batch_id = store.record("claude_memories", 3, 1, "file.json", ["id1"])
        batch = store.get_batch(batch_id)
        assert batch is not None
        assert batch["file_type"] == "claude_memories"

    def test_get_batch_nonexistent(self, store):
        assert store.get_batch("nonexistent-batch") is None

    def test_mark_undone(self, store):
        batch_id = store.record("claude_memories", 3, 0, "file.json", ["id1"])
        store.mark_undone(batch_id, facts_deleted=2)
        batch = store.get_batch(batch_id)
        assert batch["undone"] is True
        assert batch["facts_deleted"] == 2

    def test_count(self, store):
        assert store.count() == 0
        store.record("claude_memories", 1, 0)
        assert store.count() == 1
        store.record("chatgpt_conversations", 2, 0)
        assert store.count() == 2

    def test_load_corrupt_json(self, tmp_path):
        """Corrupt JSON file should be treated as empty."""
        history_file = tmp_path / "import_history.json"
        history_file.write_text("not valid json{{{")

        store = ImportHistoryStore(data_dir=tmp_path)
        assert store.count() == 0
        assert store.list_all() == []

    def test_persistence(self, tmp_path):
        """History should persist across instances."""
        store1 = ImportHistoryStore(data_dir=tmp_path)
        store1.record("claude_memories", 5, 0, "file.json", ["id1"])

        store2 = ImportHistoryStore(data_dir=tmp_path)
        assert store2.count() == 1

    def test_save_failure_cleans_up(self, tmp_path):
        """If save fails, temp file should be cleaned up."""
        store = ImportHistoryStore(data_dir=tmp_path)

        import json as json_mod
        original_dump = json_mod.dump

        def failing_dump(*args, **kwargs):
            raise IOError("disk full")

        json_mod.dump = failing_dump
        try:
            with pytest.raises(IOError):
                store.record("test", 1, 0)
        finally:
            json_mod.dump = original_dump

        # Verify no .tmp files left behind
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert len(tmp_files) == 0

    def test_list_all_reversed(self, store):
        """list_all should return newest first."""
        store.record("a", 1, 0)
        store.record("b", 2, 0)
        history = store.list_all()
        assert history[0]["file_type"] == "b"
        assert history[1]["file_type"] == "a"
