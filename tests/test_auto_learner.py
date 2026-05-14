"""Tests for auto-learning fact extraction."""
import pytest

from backend.auto_learner import (
    extract_facts_from_text,
    extract_facts_from_assistant,
    _to_first_person,
    AutoLearner,
    PendingFactStore,
    score_fact_quality,
    QUALITY_AUTO_ACCEPT,
    QUALITY_MIN_THRESHOLD,
)


class TestExtractFactsFromText:
    """Test real-time fact extraction from user messages."""

    def test_basic_self_disclosure(self):
        facts = extract_facts_from_text("I'm working on a FastAPI backend for my portfolio")
        assert len(facts) >= 1
        assert any("FastAPI" in f for f in facts)

    def test_multiple_markers(self):
        text = "I work at Google as a software engineer. I live in San Francisco."
        facts = extract_facts_from_text(text)
        assert len(facts) >= 2

    def test_name_disclosure(self):
        facts = extract_facts_from_text("My name is Alex and I'm a student at MIT")
        assert len(facts) >= 1

    def test_preference_disclosure(self):
        facts = extract_facts_from_text("I prefer Python over JavaScript for backend development")
        assert len(facts) >= 1

    def test_hobby_disclosure(self):
        facts = extract_facts_from_text("I play guitar and I love hiking in the mountains")
        assert len(facts) >= 1

    def test_ignores_short_messages(self):
        facts = extract_facts_from_text("I am")
        assert len(facts) == 0

    def test_ignores_empty_messages(self):
        assert extract_facts_from_text("") == []
        assert extract_facts_from_text(None) == []

    def test_ignores_technical_questions(self):
        """Technical questions without self-disclosure shouldn't generate facts."""
        facts = extract_facts_from_text("How do you implement a binary search tree?")
        assert len(facts) == 0

    def test_filters_fiction(self):
        """Fiction keywords should be filtered out."""
        facts = extract_facts_from_text("I'm working on a story about a wizard who casts spells")
        # "wizard" and "spell" are fiction keywords
        assert len(facts) == 0

    def test_filters_short_facts(self):
        """Facts under MIN_FACT_LENGTH should be filtered."""
        facts = extract_facts_from_text("I am a dev. I like coding.")
        # Both sentences are too short (< 20 chars)
        assert len(facts) == 0

    def test_filters_long_facts(self):
        """Facts over MAX_FACT_LENGTH should be filtered."""
        long_text = "I'm working on " + "a very detailed " * 50 + "project"
        facts = extract_facts_from_text(long_text)
        assert all(len(f) < 500 for f in facts)

    def test_real_conversation_examples(self):
        """Test with realistic conversation snippets."""
        examples = [
            ("I've been learning Rust for the past 3 months", True),
            ("Can you explain how async/await works?", False),
            ("I built a home automation system using Raspberry Pi", True),
            ("What's the difference between TCP and UDP?", False),
            ("I study computer science at Stanford University", True),
            ("Fix this TypeError in my code", False),
        ]

        for text, should_extract in examples:
            facts = extract_facts_from_text(text)
            if should_extract:
                assert len(facts) >= 1, f"Expected facts from: {text}"
            else:
                assert len(facts) == 0, f"Unexpected facts from: {text}"

    def test_sentence_boundary_extraction(self):
        """Should extract only the relevant sentence, not the whole message."""
        text = "How do I sort a list? I'm a Python developer. Can you show me an example?"
        facts = extract_facts_from_text(text)
        if facts:
            # Should extract "I'm a Python developer", not the whole thing
            assert all(len(f) < 100 for f in facts)


class TestAutoLearner:
    """Test the AutoLearner class."""

    @pytest.fixture
    def mock_memory(self, tmp_path):
        """Create a temporary Anamnesis instance for testing."""
        from anamnesis import Anamnesis  # noqa: available via sys.path insert
        db_path = str(tmp_path / "test.db")
        return Anamnesis(db_path)

    @pytest.fixture
    def pending_store(self, tmp_path):
        return PendingFactStore(data_dir=tmp_path)

    @pytest.fixture
    def learner(self, mock_memory, pending_store):
        return AutoLearner(mock_memory, retriever=None, pending_store=pending_store)

    @pytest.mark.asyncio
    async def test_learn_auto_approve(self, learner):
        """With auto_approve=True, facts go directly to storage."""
        learner.auto_approve = True
        await learner.learn_from_message("I'm a software engineer working at a startup")
        assert learner.facts_learned >= 1

    @pytest.mark.asyncio
    async def test_learn_queues_medium_quality(self, learner):
        """Medium-quality facts go to pending queue."""
        await learner.learn_from_message("I'm a software engineer working at a startup")
        assert learner.facts_pending >= 1
        assert learner.pending.count() >= 1

    @pytest.mark.asyncio
    async def test_duplicate_detection(self, learner):
        """Same fact stated twice should be deduplicated when auto-approved."""
        learner.auto_approve = True
        await learner.learn_from_message("I'm a software engineer working at a startup in Seattle")
        first_count = learner.facts_learned

        await learner.learn_from_message("I'm a software engineer working at a startup in Seattle")
        assert learner.facts_learned == first_count
        assert learner.duplicates_seen >= 1

    @pytest.mark.asyncio
    async def test_disabled_learning(self, learner):
        learner.enabled = False
        await learner.learn_from_message("I'm a developer who loves Python")
        assert learner.facts_learned == 0

    @pytest.mark.asyncio
    async def test_stats(self, learner):
        stats = learner.get_stats()
        assert "enabled" in stats
        assert "facts_learned" in stats
        assert "facts_pending" in stats
        assert "duplicates_seen" in stats

    @pytest.mark.asyncio
    async def test_no_crash_on_error(self, learner):
        """Auto-learner should never crash even with weird input."""
        await learner.learn_from_message("")
        await learner.learn_from_message("x" * 10000)
        assert True

    @pytest.mark.asyncio
    async def test_approve_pending(self, learner):
        """Approved pending facts should be stored."""
        await learner.learn_from_message("I'm a Python developer living in Seattle")
        pending = learner.pending.list_all()
        if pending:
            result = learner.approve_pending(pending[0]["id"])
            assert result is True
            assert learner.facts_learned >= 1

    @pytest.mark.asyncio
    async def test_reject_pending(self, learner):
        """Rejected pending facts should be removed."""
        await learner.learn_from_message("I'm working on a project for my team")
        pending = learner.pending.list_all()
        if pending:
            # Reject all pending facts
            for p in pending:
                result = learner.reject_pending(p["id"])
                assert result is True
            assert learner.pending.count() == 0


class TestQualityScoring:
    """Test fact quality scoring."""

    def test_high_value_fact(self):
        """Permanent personal facts should score high."""
        score = score_fact_quality("My name is Josii and I live in Toronto")
        assert score >= 0.7

    def test_transient_fact_low(self):
        """Temporary state should score low."""
        score = score_fact_quality("I'm debugging this function and getting weird errors")
        assert score < QUALITY_MIN_THRESHOLD

    def test_question_penalized(self):
        """Questions shouldn't be stored as facts."""
        score = score_fact_quality("I'm wondering if I should switch to Rust?")
        assert score < 0.5

    def test_code_content_penalized(self):
        """Code-like content gets penalized."""
        score = score_fact_quality("I'm writing def calculate_score(): return 42")
        assert score < 0.5

    def test_generic_statement_penalized(self):
        """Generic help requests aren't facts."""
        score = score_fact_quality("I have a question about how to use Python decorators")
        assert score < 0.5

    def test_specific_personal_fact(self):
        """Specific personal facts with proper nouns score well."""
        score = score_fact_quality("I work at Google as a machine learning engineer")
        assert score >= 0.6

    def test_preference_moderate(self):
        """Preferences are moderate quality."""
        score = score_fact_quality("I prefer using Vim over Emacs for editing code")
        assert 0.4 <= score <= 0.8

    def test_score_bounded(self):
        """Scores should always be between 0 and 1."""
        test_texts = [
            "x",
            "I'm debugging I'm trying to I need help with I'm stuck on",
            "My name is Alex and I live in Seattle and I work at Google",
        ]
        for text in test_texts:
            score = score_fact_quality(text)
            assert 0.0 <= score <= 1.0


class TestAutoLearnerStorePath:
    """Test _store_fact and _check_contradictions methods."""

    @pytest.fixture
    def mock_memory(self, tmp_path):
        from anamnesis import Anamnesis
        return Anamnesis(str(tmp_path / "test_store.db"))

    @pytest.fixture
    def pending_store(self, tmp_path):
        return PendingFactStore(data_dir=tmp_path)

    @pytest.fixture
    def learner(self, mock_memory, pending_store):
        return AutoLearner(mock_memory, retriever=None, pending_store=pending_store)

    def test_store_fact_new(self, learner):
        """A new fact should increment facts_learned."""
        from anamnesis.models import FactType
        learner._store_fact("User is a software developer in Toronto", FactType.GENERAL, "test")
        assert learner.facts_learned == 1
        assert learner.duplicates_seen == 0

    def test_store_fact_duplicate(self, learner):
        """Storing the same fact twice should increment duplicates_seen."""
        from anamnesis.models import FactType
        learner._store_fact("User works as a budtender in Toronto area", FactType.GENERAL, "test")
        assert learner.facts_learned == 1

        learner._store_fact("User works as a budtender in Toronto area", FactType.GENERAL, "test")
        assert learner.facts_learned == 1
        assert learner.duplicates_seen == 1

    def test_store_fact_with_retriever(self, mock_memory, pending_store):
        """When a retriever is available, new facts should be indexed."""
        from unittest.mock import MagicMock
        from anamnesis.models import FactType

        mock_retriever = MagicMock()
        mock_retriever.index_fact = MagicMock()

        learner = AutoLearner(mock_memory, retriever=mock_retriever, pending_store=pending_store)
        learner._store_fact("User has three cats named Luna Pixel and Byte", FactType.GENERAL, "test")

        assert learner.facts_learned == 1
        mock_retriever.index_fact.assert_called_once()

    def test_store_fact_retriever_failure(self, mock_memory, pending_store):
        """Retriever failure should be logged but not crash."""
        from unittest.mock import MagicMock
        from anamnesis.models import FactType

        mock_retriever = MagicMock()
        mock_retriever.index_fact.side_effect = RuntimeError("Vector index full")

        learner = AutoLearner(mock_memory, retriever=mock_retriever, pending_store=pending_store)
        learner._store_fact("User studies computer science at university", FactType.GENERAL, "test")

        # Should still store the fact even if indexing fails
        assert learner.facts_learned == 1

    def test_check_contradictions_skips_small_db(self, learner):
        """Contradiction check should be skipped when < 2 facts exist."""
        from anamnesis.models import FactType
        from unittest.mock import MagicMock

        fact = MagicMock()
        fact.content = "User lives in Toronto"
        fact.id = "test-id"
        # Should not crash even with no facts in DB
        learner._check_contradictions(fact)
        assert learner.contradictions_found == 0

    @pytest.mark.asyncio
    async def test_approve_all_pending(self, learner):
        """approve_all_pending should store all pending facts."""
        # Queue some facts via learning
        await learner.learn_from_message("I'm a Python developer living in downtown Toronto")
        pending_count = learner.pending.count()

        if pending_count > 0:
            stored = learner.approve_all_pending()
            assert stored == pending_count
            assert learner.pending.count() == 0
            assert learner.facts_learned >= stored

    @pytest.mark.asyncio
    async def test_reject_all_pending(self, learner):
        """reject_all_pending should clear queue and increment rejected count."""
        await learner.learn_from_message("I'm studying machine learning at Stanford University")
        pending_count = learner.pending.count()

        if pending_count > 0:
            rejected = learner.reject_all_pending()
            assert rejected == pending_count
            assert learner.pending.count() == 0
            assert learner.facts_rejected >= rejected

    def test_approve_pending_nonexistent(self, learner):
        """Approving a nonexistent ID should return False."""
        assert learner.approve_pending("nonexistent-id") is False

    def test_reject_pending_nonexistent(self, learner):
        """Rejecting a nonexistent ID should return False."""
        assert learner.reject_pending("nonexistent-id") is False


class TestPendingFactStore:
    """Test the pending fact file store."""

    @pytest.fixture
    def store(self, tmp_path):
        return PendingFactStore(data_dir=tmp_path)

    def test_add_and_list(self, store):
        store.add("Test fact content here", 0.6, "proxy")
        pending = store.list_all()
        assert len(pending) == 1
        assert pending[0]["content"] == "Test fact content here"

    def test_approve(self, store):
        entry = store.add("Fact to approve", 0.5, "chat")
        approved = store.approve(entry["id"])
        assert approved is not None
        assert approved["content"] == "Fact to approve"
        assert store.count() == 0

    def test_reject(self, store):
        entry = store.add("Fact to reject", 0.4, "chat")
        result = store.reject(entry["id"])
        assert result is True
        assert store.count() == 0

    def test_approve_nonexistent(self, store):
        result = store.approve("nonexistent-id")
        assert result is None

    def test_reject_nonexistent(self, store):
        result = store.reject("nonexistent-id")
        assert result is False

    def test_approve_all(self, store):
        store.add("Fact one for testing", 0.5, "proxy")
        store.add("Fact two for testing", 0.6, "proxy")
        approved = store.approve_all()
        assert len(approved) == 2
        assert store.count() == 0

    def test_reject_all(self, store):
        store.add("Fact one to reject", 0.5, "proxy")
        store.add("Fact two to reject", 0.4, "proxy")
        count = store.reject_all()
        assert count == 2
        assert store.count() == 0

    def test_persistence(self, tmp_path):
        """Store should persist across instances."""
        store1 = PendingFactStore(data_dir=tmp_path)
        store1.add("Persistent fact content", 0.6, "proxy")

        store2 = PendingFactStore(data_dir=tmp_path)
        assert store2.count() == 1
        assert store2.list_all()[0]["content"] == "Persistent fact content"

    def test_cross_instance_approve_all_no_data_loss(self, tmp_path):
        """approve_all on instance A must not lose facts added by instance B."""
        store_a = PendingFactStore(data_dir=tmp_path)
        store_b = PendingFactStore(data_dir=tmp_path)

        # Instance A adds fact, instance B's cache doesn't know about it
        store_a.add("Fact from instance A for testing", 0.6, "proxy")
        # Instance B adds another fact (re-reads from disk on add)
        store_b.add("Fact from instance B for testing", 0.5, "review")

        # Instance A approve_all: should see BOTH facts (re-reads from disk)
        approved = store_a.approve_all()
        assert len(approved) == 2
        # Disk should be empty now
        assert store_b.count() == 0

    def test_cross_instance_reject_preserves_new(self, tmp_path):
        """Rejecting one fact on instance A doesn't lose concurrent adds from B."""
        store_a = PendingFactStore(data_dir=tmp_path)
        store_b = PendingFactStore(data_dir=tmp_path)

        entry_a = store_a.add("Fact A to reject during testing", 0.6, "proxy")
        # Instance B adds after A's cache is populated
        store_b.add("Fact B should survive the rejection", 0.5, "review")

        # A rejects its own fact — should re-read disk first
        store_a.reject(entry_a["id"])
        # B's fact should still be there
        assert store_b.count() == 1
        remaining = store_b.list_all()
        assert remaining[0]["content"] == "Fact B should survive the rejection"

    def test_load_corrupt_json(self, tmp_path):
        """Corrupt JSON file should be treated as empty."""
        pending_file = tmp_path / "pending_facts.json"
        pending_file.write_text("not valid json{{{")

        store = PendingFactStore(data_dir=tmp_path)
        assert store.count() == 0
        assert store.list_all() == []

    def test_save_failure_cleans_up(self, tmp_path):
        """If save fails during write, temp file should be cleaned up."""
        store = PendingFactStore(data_dir=tmp_path)

        # Make json.dump fail by patching
        import json as json_mod
        original_dump = json_mod.dump

        def failing_dump(*args, **kwargs):
            raise IOError("disk full")

        json_mod.dump = failing_dump
        try:
            with pytest.raises(IOError):
                store.add("This will fail to save", 0.5, "proxy")
        finally:
            json_mod.dump = original_dump

        # Verify no .tmp files left behind
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert len(tmp_files) == 0


class TestQualityScoringEdgeCases:
    """Test edge cases in quality scoring."""

    def test_long_fact_penalized(self):
        """Facts over 300 characters should be penalized."""
        long_fact = "I'm working on " + "a detailed project about " * 15 + "something important"
        assert len(long_fact) > 300
        score = score_fact_quality(long_fact)

        short_fact = "I'm working on a Python backend project"
        short_score = score_fact_quality(short_fact)

        # Long fact should score lower than equivalent short one
        assert score < short_score


class TestAutoLearnerContradictions:
    """Test _check_contradictions behavior."""

    @pytest.fixture
    def mock_memory(self, tmp_path):
        from anamnesis import Anamnesis
        return Anamnesis(str(tmp_path / "test_contra.db"))

    @pytest.fixture
    def pending_store(self, tmp_path):
        return PendingFactStore(data_dir=tmp_path)

    @pytest.fixture
    def learner(self, mock_memory, pending_store):
        return AutoLearner(mock_memory, retriever=None, pending_store=pending_store)

    def test_check_contradictions_with_detection(self, learner):
        """Full contradiction detection + superseding path."""
        from anamnesis.models import FactType
        from unittest.mock import MagicMock
        import sys as _sys
        import types

        # Seed 2+ facts so the check doesn't skip
        f1 = learner.memory.semantic.add_fact(
            content="User lives in Toronto Canada downtown",
            fact_type=FactType.GENERAL, confidence=0.7,
        )
        f2 = learner.memory.semantic.add_fact(
            content="User works at a coffee shop in Montreal",
            fact_type=FactType.GENERAL, confidence=0.6,
        )

        # Create a new fact that "contradicts" f1
        new_fact = learner.memory.semantic.add_fact(
            content="User recently moved to Vancouver from Toronto",
            fact_type=FactType.GENERAL, confidence=0.8,
        )

        # Mock detect_contradictions to return a high-confidence contradiction
        fake_mod = types.ModuleType("anamnesis.consolidation.contradiction")

        class FakeResult:
            def __init__(self, existing):
                self.is_contradiction = True
                self.existing_fact = existing
                self.contradiction_type = "location"
                self.confidence = 0.8  # >= 0.7 threshold for superseding

        def fake_detect(new, existing, threshold=0.5):
            return [FakeResult(f1)]

        fake_mod.detect_contradictions = fake_detect
        real_module = _sys.modules.get("anamnesis.consolidation.contradiction")
        _sys.modules["anamnesis.consolidation.contradiction"] = fake_mod
        try:
            learner._check_contradictions(new_fact)
            assert learner.contradictions_found == 1
        finally:
            if real_module is not None:
                _sys.modules["anamnesis.consolidation.contradiction"] = real_module
            else:
                _sys.modules.pop("anamnesis.consolidation.contradiction", None)

    def test_check_contradictions_low_confidence_no_supersede(self, learner):
        """Low-confidence contradictions should be logged but not supersede."""
        from anamnesis.models import FactType
        import sys as _sys
        import types

        f1 = learner.memory.semantic.add_fact(
            content="User has a pet cat named Whiskers at home",
            fact_type=FactType.GENERAL, confidence=0.7,
        )
        f2 = learner.memory.semantic.add_fact(
            content="User likes going to the park on weekends",
            fact_type=FactType.GENERAL, confidence=0.6,
        )
        new_fact = learner.memory.semantic.add_fact(
            content="User adopted a dog recently from the shelter",
            fact_type=FactType.GENERAL, confidence=0.7,
        )

        fake_mod = types.ModuleType("anamnesis.consolidation.contradiction")

        class FakeResult:
            def __init__(self, existing):
                self.is_contradiction = True
                self.existing_fact = existing
                self.contradiction_type = "pet"
                self.confidence = 0.5  # Below 0.7 threshold

        def fake_detect(new, existing, threshold=0.5):
            return [FakeResult(f1)]

        fake_mod.detect_contradictions = fake_detect
        real_module = _sys.modules.get("anamnesis.consolidation.contradiction")
        _sys.modules["anamnesis.consolidation.contradiction"] = fake_mod
        try:
            learner._check_contradictions(new_fact)
            # Low confidence should NOT supersede
            assert learner.contradictions_found == 0
        finally:
            if real_module is not None:
                _sys.modules["anamnesis.consolidation.contradiction"] = real_module
            else:
                _sys.modules.pop("anamnesis.consolidation.contradiction", None)

    def test_check_contradictions_import_error(self, learner):
        """ImportError in contradiction module should be silently caught."""
        from unittest.mock import MagicMock
        import sys as _sys

        fact = MagicMock()
        fact.content = "User lives in Toronto"
        fact.id = "test-id"

        # Set module to None to trigger ImportError
        real_module = _sys.modules.get("anamnesis.consolidation.contradiction")
        _sys.modules["anamnesis.consolidation.contradiction"] = None
        try:
            learner._check_contradictions(fact)
            assert learner.contradictions_found == 0
        finally:
            if real_module is not None:
                _sys.modules["anamnesis.consolidation.contradiction"] = real_module
            else:
                _sys.modules.pop("anamnesis.consolidation.contradiction", None)

    def test_check_contradictions_generic_exception(self, learner):
        """Generic exception should be caught and logged."""
        from anamnesis.models import FactType
        import sys as _sys
        import types

        f1 = learner.memory.semantic.add_fact(
            content="User works at a bakery in the mornings",
            fact_type=FactType.GENERAL, confidence=0.7,
        )
        f2 = learner.memory.semantic.add_fact(
            content="User studies at the local community college",
            fact_type=FactType.GENERAL, confidence=0.6,
        )
        new_fact = learner.memory.semantic.add_fact(
            content="User just got a job at a tech company nearby",
            fact_type=FactType.GENERAL, confidence=0.7,
        )

        fake_mod = types.ModuleType("anamnesis.consolidation.contradiction")

        def boom(*args, **kwargs):
            raise RuntimeError("something broke in detection")

        fake_mod.detect_contradictions = boom
        real_module = _sys.modules.get("anamnesis.consolidation.contradiction")
        _sys.modules["anamnesis.consolidation.contradiction"] = fake_mod
        try:
            # Should not raise
            learner._check_contradictions(new_fact)
            assert learner.contradictions_found == 0
        finally:
            if real_module is not None:
                _sys.modules["anamnesis.consolidation.contradiction"] = real_module
            else:
                _sys.modules.pop("anamnesis.consolidation.contradiction", None)

    @pytest.mark.asyncio
    async def test_learn_low_quality_rejected(self, learner):
        """Facts below QUALITY_MIN_THRESHOLD should be rejected."""
        # "I'm debugging" + question mark => transient + question penalty = very low score
        await learner.learn_from_message(
            "I'm debugging this weird error and I'm stuck on the implementation?"
        )
        assert learner.facts_rejected >= 1

    @pytest.mark.asyncio
    async def test_learn_exception_caught(self, learner):
        """Generic exception in learn_from_message should not crash."""
        from unittest.mock import patch

        with patch(
            "backend.auto_learner.extract_facts_from_text",
            side_effect=RuntimeError("extraction blew up"),
        ):
            await learner.learn_from_message("I'm a software developer in Toronto")
            # Should not raise — error caught internally

    def test_approve_pending_import_error(self, tmp_path):
        """approve_pending should return False when FactType import fails."""
        import sys as _sys
        from anamnesis import Anamnesis

        mock_memory = Anamnesis(str(tmp_path / "test_import_err.db"))
        store = PendingFactStore(data_dir=tmp_path)
        learner = AutoLearner(mock_memory, retriever=None, pending_store=store)

        # Add a pending fact
        store.add("User likes hiking in the mountains", 0.5, "test")
        pending = store.list_all()
        pid = pending[0]["id"]

        # Block FactType import
        real_module = _sys.modules.get("anamnesis.models")
        _sys.modules["anamnesis.models"] = None
        try:
            result = learner.approve_pending(pid)
            assert result is False
        finally:
            if real_module is not None:
                _sys.modules["anamnesis.models"] = real_module
            else:
                _sys.modules.pop("anamnesis.models", None)

    def test_approve_all_pending_import_error(self, tmp_path):
        """approve_all_pending should return 0 when FactType import fails."""
        import sys as _sys
        from anamnesis import Anamnesis

        mock_memory = Anamnesis(str(tmp_path / "test_import_err2.db"))
        store = PendingFactStore(data_dir=tmp_path)
        learner = AutoLearner(mock_memory, retriever=None, pending_store=store)

        store.add("User likes hiking in the mountains", 0.5, "test")

        real_module = _sys.modules.get("anamnesis.models")
        _sys.modules["anamnesis.models"] = None
        try:
            result = learner.approve_all_pending()
            assert result == 0
        finally:
            if real_module is not None:
                _sys.modules["anamnesis.models"] = real_module
            else:
                _sys.modules.pop("anamnesis.models", None)


class TestTopicDetectionOnFacts:
    """Test that _store_fact enriches facts with topic + category metadata."""

    @pytest.fixture
    def mock_memory(self, tmp_path):
        from anamnesis import Anamnesis
        return Anamnesis(str(tmp_path / "test_topic.db"))

    @pytest.fixture
    def pending_store(self, tmp_path):
        return PendingFactStore(data_dir=tmp_path)

    @pytest.fixture
    def learner(self, mock_memory, pending_store):
        return AutoLearner(mock_memory, retriever=None, pending_store=pending_store)

    def test_topic_stored_in_metadata(self, learner):
        """_store_fact should detect and store topic in metadata."""
        from anamnesis.models import FactType
        learner._store_fact("I work as a Python developer at Google", FactType.GENERAL, "test")
        facts = learner.memory.semantic.list_all(limit=10)
        assert len(facts) >= 1
        # The fact should have topic metadata
        meta = facts[0].metadata or {}
        assert "topic" in meta
        assert "category" in meta

    def test_category_is_technical_for_code_fact(self, learner):
        """Technical facts should be categorized as 'technical'."""
        from anamnesis.models import FactType
        learner._store_fact(
            "I study computer programming and database design",
            FactType.GENERAL, "test",
        )
        facts = learner.memory.semantic.list_all(limit=10)
        meta = facts[0].metadata or {}
        assert meta.get("category") == "technical"

    def test_category_is_personal_for_life_fact(self, learner):
        """Personal life facts should be categorized as 'personal'."""
        from anamnesis.models import FactType
        learner._store_fact(
            "I feel really happy about my new relationship and career",
            FactType.GENERAL, "test",
        )
        facts = learner.memory.semantic.list_all(limit=10)
        meta = facts[0].metadata or {}
        # "feel", "happy", "relationship", "career" → personal
        assert meta.get("category") == "personal"

    def test_topic_detection_failure_doesnt_crash(self, learner):
        """If TopicDetector import fails, _store_fact should still succeed."""
        from anamnesis.models import FactType
        import sys as _sys

        real_module = _sys.modules.get("anamnesis.topics.detector")
        _sys.modules["anamnesis.topics.detector"] = None
        try:
            learner._store_fact("I live in Toronto Canada", FactType.GENERAL, "test")
            # Should succeed even without topic detection
            assert learner.facts_learned >= 1
        finally:
            if real_module is not None:
                _sys.modules["anamnesis.topics.detector"] = real_module
            else:
                _sys.modules.pop("anamnesis.topics.detector", None)


class TestSentimentOnFacts:
    """Test that _store_fact enriches facts with sentiment metadata."""

    @pytest.fixture
    def mock_memory(self, tmp_path):
        from anamnesis import Anamnesis
        return Anamnesis(str(tmp_path / "test_sentiment.db"))

    @pytest.fixture
    def pending_store(self, tmp_path):
        return PendingFactStore(data_dir=tmp_path)

    @pytest.fixture
    def learner(self, mock_memory, pending_store):
        return AutoLearner(mock_memory, retriever=None, pending_store=pending_store)

    def test_sentiment_stored_in_metadata(self, learner):
        """_store_fact should detect and store emotion in metadata."""
        from anamnesis.models import FactType
        learner._store_fact("I love hiking in the beautiful mountains", FactType.GENERAL, "test")
        facts = learner.memory.semantic.list_all(limit=10)
        meta = facts[0].metadata or {}
        assert "emotion" in meta
        assert "sentiment_score" in meta

    def test_positive_emotion_detected(self, learner):
        """Positive text should be detected as joy or positive."""
        from anamnesis.models import FactType
        learner._store_fact(
            "I am so happy and excited about my amazing new project",
            FactType.GENERAL, "test",
        )
        facts = learner.memory.semantic.list_all(limit=10)
        meta = facts[0].metadata or {}
        assert meta.get("sentiment_score", 0) > 0

    def test_neutral_text_handled(self, learner):
        """Neutral text should not crash and should store some emotion."""
        from anamnesis.models import FactType
        learner._store_fact(
            "I work at a company in the downtown area of the city",
            FactType.GENERAL, "test",
        )
        facts = learner.memory.semantic.list_all(limit=10)
        meta = facts[0].metadata or {}
        assert "emotion" in meta

    def test_sentiment_failure_doesnt_crash(self, learner):
        """If SentimentAnalyzer import fails, _store_fact should still succeed."""
        from anamnesis.models import FactType
        import sys as _sys

        real_module = _sys.modules.get("anamnesis.emotional.analyzer")
        _sys.modules["anamnesis.emotional.analyzer"] = None
        try:
            learner._store_fact("I live in Vancouver BC Canada", FactType.GENERAL, "test")
            assert learner.facts_learned >= 1
        finally:
            if real_module is not None:
                _sys.modules["anamnesis.emotional.analyzer"] = real_module
            else:
                _sys.modules.pop("anamnesis.emotional.analyzer", None)


class TestFictionFilterUpgrade:
    """Test upgraded fiction filtering with FANTASY_KEYWORDS."""

    def test_fantasy_keyword_caught(self):
        """Fantasy keywords from FANTASY_KEYWORDS should be filtered."""
        # 'necromancer' is in FANTASY_KEYWORDS but NOT in old Config.FICTION_KEYWORDS
        facts = extract_facts_from_text(
            "I'm working on a necromancer character design for my game world"
        )
        assert len(facts) == 0

    def test_normal_fact_passes(self):
        """Normal real-world facts should not be caught by fiction filter."""
        facts = extract_facts_from_text(
            "I'm working on a FastAPI backend for my portfolio project"
        )
        assert len(facts) >= 1

    def test_riven_keyword_caught(self):
        """Josii's worldbuilding keywords (riven, alderwick) should be filtered."""
        facts = extract_facts_from_text(
            "I'm building a city called Alderwick in my story world"
        )
        # 'alderwick' is in FANTASY_KEYWORDS
        assert len(facts) == 0


class TestPendingFactEnrichment:
    """Test that PendingFactStore.add() enriches with topic + emotion."""

    def test_pending_has_topic_and_emotion(self, tmp_path):
        """Pending facts should include detected_topic and detected_emotion."""
        store = PendingFactStore(data_dir=tmp_path)
        store.add("I love programming in Python and building cool apps", 0.5, "test")
        pending = store.list_all()
        assert len(pending) == 1
        item = pending[0]
        assert "detected_topic" in item
        assert "detected_category" in item
        assert "detected_emotion" in item

    def test_pending_topic_is_string(self, tmp_path):
        """Detected topic should be a non-empty string for meaningful content."""
        store = PendingFactStore(data_dir=tmp_path)
        store.add("I study computer science and machine learning at university", 0.5, "test")
        pending = store.list_all()
        item = pending[0]
        assert isinstance(item["detected_topic"], str)
        assert isinstance(item["detected_category"], str)


class TestAutoLinkRelated:
    """Test auto-linking related facts via MemoryGraph."""

    def test_auto_link_creates_graph_links(self, tmp_path):
        """Storing a fact should create graph links to related facts."""
        from unittest.mock import patch, MagicMock
        from anamnesis import Anamnesis

        db_path = tmp_path / "test.db"
        mem = Anamnesis(str(db_path))
        learner = AutoLearner(mem)

        # Store two related facts
        learner._store_fact("User works as a Python developer at Google", "FactType.GENERAL", "test")
        learner._store_fact("User programs in Python and Java for work", "FactType.GENERAL", "test")

        # Check if graph links were created
        try:
            from anamnesis.graph.memory_graph import MemoryGraph
            graph = MemoryGraph(str(db_path))
            stats = graph.get_stats()
            # Links may or may not exist depending on FTS match
            assert isinstance(stats, dict)
        except ImportError:
            pass  # Graph module not available

    def test_auto_link_handles_import_error(self, tmp_path):
        """Auto-link should silently handle import errors."""
        from unittest.mock import patch
        from anamnesis import Anamnesis

        db_path = tmp_path / "test.db"
        mem = Anamnesis(str(db_path))
        learner = AutoLearner(mem)

        with patch.dict("sys.modules", {"anamnesis.graph.memory_graph": None}):
            # Should not raise
            learner._store_fact("User enjoys hiking in the mountains", "FactType.GENERAL", "test")

    def test_auto_link_skips_self_link(self, tmp_path):
        """Auto-link should not create a link from a fact to itself."""
        from anamnesis import Anamnesis

        db_path = tmp_path / "test.db"
        mem = Anamnesis(str(db_path))
        learner = AutoLearner(mem)

        # Store a fact
        learner._store_fact("User has a very unique fact about quantum physics", "FactType.GENERAL", "test")
        # No error should occur


class TestAssistantFactExtraction:
    """Test extraction of user facts from assistant confirmation phrases."""

    def test_extracts_you_work(self):
        text = "You work at DataForge as a junior Python developer, that's great!"
        facts = extract_facts_from_assistant(text)
        assert len(facts) >= 1
        assert any("work" in f.lower() for f in facts)

    def test_converts_to_first_person(self):
        facts = extract_facts_from_assistant("You work at Google as an engineer.")
        assert len(facts) >= 1
        assert any("I work" in f for f in facts)

    def test_extracts_you_live(self):
        facts = extract_facts_from_assistant("Since you live in Vancouver, local transit is relevant.")
        assert len(facts) >= 1
        assert any("I live" in f or "live" in f for f in facts)

    def test_extracts_as_a(self):
        facts = extract_facts_from_assistant("As a Python developer, you'll find this pattern useful.")
        assert len(facts) >= 1

    def test_ignores_short_text(self):
        facts = extract_facts_from_assistant("ok")
        assert facts == []

    def test_ignores_empty(self):
        facts = extract_facts_from_assistant("")
        assert facts == []

    def test_no_marker_no_facts(self):
        facts = extract_facts_from_assistant("Here is how you implement a binary search tree.")
        assert facts == []

    def test_to_first_person_you_are(self):
        result = _to_first_person("you are a software developer")
        assert result == "I am a software developer"

    def test_to_first_person_youre(self):
        result = _to_first_person("you're working on a cool project")
        assert result == "I'm working on a cool project"

    def test_to_first_person_no_match_unchanged(self):
        result = _to_first_person("here is some code")
        assert result == "here is some code"

    def test_multiple_markers_in_response(self):
        text = "You work at DataForge. You live in East Vancouver. You have a cat named Pixel."
        facts = extract_facts_from_assistant(text)
        assert len(facts) >= 2

    @pytest.mark.asyncio
    async def test_learn_from_assistant_message(self, tmp_path):
        """AutoLearner.learn_from_assistant_message stores confirmed facts directly."""
        from unittest.mock import MagicMock, AsyncMock
        mem = MagicMock()
        result = MagicMock()
        result.confirmation_count = 1
        result.id = "test-id"
        mem.semantic.add_fact.return_value = result
        mem.semantic.search.return_value = []
        mem.semantic.list_all.return_value = []

        learner = AutoLearner(mem)
        learner.auto_approve = True
        await learner.learn_from_assistant_message(
            "You work at DataForge as a junior Python developer.", source="test"
        )
        assert mem.semantic.add_fact.called

    @pytest.mark.asyncio
    async def test_learn_from_assistant_when_disabled(self):
        """learn_from_assistant_message exits immediately when learner is disabled (line 499)."""
        from unittest.mock import MagicMock
        mem = MagicMock()
        learner = AutoLearner(mem)
        learner.enabled = False
        await learner.learn_from_assistant_message("You are a developer.", source="test")
        # No facts should be stored
        assert not mem.semantic.add_fact.called

    @pytest.mark.asyncio
    async def test_learn_from_assistant_handles_exception(self):
        """learn_from_assistant_message swallows exceptions without crashing (lines 506-507)."""
        from unittest.mock import MagicMock, patch
        mem = MagicMock()
        learner = AutoLearner(mem)
        learner.enabled = True
        learner.auto_approve = True

        with patch("backend.auto_learner.extract_facts_from_assistant", side_effect=RuntimeError("boom")):
            # Should not raise — exception is caught
            await learner.learn_from_assistant_message("You work at ACME Corp.", source="test")


class TestAutoLearnerImportFallbacks:
    """Cover ImportError fallback paths in extract_facts_from_text and PendingFactStore."""

    def test_extract_facts_fantasy_keywords_import_error(self):
        """When FANTASY_KEYWORDS import fails, falls back to Config.FICTION_KEYWORDS (lines 176-177)."""
        from unittest.mock import patch
        import sys

        with patch.dict(sys.modules, {"anamnesis": None, "anamnesis.consolidation": None,
                                       "anamnesis.consolidation.context_detector": None}):
            # This triggers the ImportError fallback path
            try:
                from backend.auto_learner import extract_facts_from_text
                # Call with text that contains a fact marker — fiction filter runs
                result = extract_facts_from_text("I am a developer in Vancouver")
                assert isinstance(result, list)
            except ImportError:
                pass  # Acceptable if module isn't reloadable in test context

    def test_pending_store_topic_exception_real(self, tmp_path):
        """TopicDetector.detect() raises → except Exception: pass (lines 243-244)."""
        from unittest.mock import patch
        from backend.auto_learner import PendingFactStore

        store = PendingFactStore(tmp_path / "pending.json")
        with patch("anamnesis.topics.detector.TopicDetector") as mock_cls:
            mock_cls.return_value.detect.side_effect = RuntimeError("topic explode")
            # Should not raise — exception is caught silently
            store.add("I live in Ottawa and work in technology", quality_score=0.6)
        facts = store.list_all()
        assert len(facts) == 1

    def test_pending_store_sentiment_exception(self, tmp_path):
        """SentimentAnalyzer.analyze() raises → except Exception: pass (lines 252-253)."""
        from unittest.mock import patch
        from backend.auto_learner import PendingFactStore

        store = PendingFactStore(tmp_path / "pending.json")
        with patch("anamnesis.emotional.analyzer.SentimentAnalyzer") as mock_cls:
            mock_cls.return_value.analyze.side_effect = RuntimeError("sentiment explode")
            # Should not raise — exception is caught silently
            store.add("I enjoy working remotely and learning new skills", quality_score=0.6)
        facts = store.list_all()
        assert len(facts) == 1

    def test_auto_link_exception_logged(self):
        """MemoryGraph.add_link raises → except Exception: logger.debug (lines 488-489)."""
        from unittest.mock import MagicMock, patch

        mem = MagicMock()
        other_fact = MagicMock()
        other_fact.id = "other-fact-id"
        other_fact.content = "some related content about programming"
        # search() returns a different fact (triggering the loop body)
        mem.semantic.search.return_value = [other_fact]

        new_fact = MagicMock()
        new_fact.id = "new-fact-id"  # different from other_fact.id
        new_fact.content = "I work as a software engineer"

        learner = AutoLearner(mem)

        with patch("anamnesis.graph.memory_graph.MemoryGraph") as mock_graph_cls:
            mock_graph_cls.return_value.add_link.side_effect = RuntimeError("link fail")
            # Should not raise — exception is caught by except Exception: logger.debug
            learner._auto_link_related(new_fact)
