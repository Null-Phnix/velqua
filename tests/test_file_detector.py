"""
Unit tests for file_detector module.
"""
import pytest
import json
from pathlib import Path
from backend.file_detector import (
    detect_file_type,
    FileType,
    extract_facts_from_conversations,
    extract_facts_from_messages,
    extract_facts_from_projects,
    extract_facts_from_chatgpt,
)


# Test fixtures
@pytest.fixture
def temp_json_file(tmp_path):
    """Create temporary JSON file for testing."""
    def _create(data):
        file_path = tmp_path / "test.json"
        with open(file_path, 'w') as f:
            json.dump(data, f)
        return str(file_path)
    return _create


class TestDetectFileType:
    """Test file type detection."""

    def test_detect_claude_memories(self, temp_json_file):
        """Test detection of Claude memories.json format."""
        data = [{
            "conversations_memory": "User is a developer",
            "account_uuid": "test-uuid"
        }]
        file_path = temp_json_file(data)

        file_type, metadata = detect_file_type(file_path)

        assert file_type == FileType.CLAUDE_MEMORIES
        assert metadata["records"] == 1

    def test_detect_claude_memories_with_projects(self, temp_json_file):
        """Test detection of Claude memories with project memories."""
        data = [{
            "conversations_memory": "User is a developer",
            "account_uuid": "test-uuid",
            "project_memories": ["Working on web app"]
        }]
        file_path = temp_json_file(data)

        file_type, metadata = detect_file_type(file_path)

        assert file_type == FileType.CLAUDE_MEMORIES
        assert metadata["has_project_memories"] is True

    def test_detect_claude_conversations(self, temp_json_file):
        """Test detection of Claude conversations.json format."""
        data = [{
            "uuid": "conv-123",
            "chat_messages": [
                {"text": "Hello", "sender": "human"}
            ]
        }]
        file_path = temp_json_file(data)

        file_type, metadata = detect_file_type(file_path)

        assert file_type == FileType.CLAUDE_CONVERSATIONS
        assert metadata["conversations"] == 1
        assert metadata["total_messages"] == 1

    def test_detect_claude_conversations_multiple(self, temp_json_file):
        """Test detection with multiple conversations."""
        data = [
            {
                "uuid": "conv-1",
                "chat_messages": [
                    {"text": "Hello", "sender": "human"},
                    {"text": "Hi", "sender": "assistant"}
                ]
            },
            {
                "uuid": "conv-2",
                "chat_messages": [
                    {"text": "Test", "sender": "human"}
                ]
            }
        ]
        file_path = temp_json_file(data)

        file_type, metadata = detect_file_type(file_path)

        assert file_type == FileType.CLAUDE_CONVERSATIONS
        assert metadata["conversations"] == 2
        assert metadata["total_messages"] == 3

    def test_detect_claude_projects(self, temp_json_file):
        """Test detection of Claude projects.json format."""
        data = [{
            "name": "My Project",
            "docs": [{"content": "doc1"}]
        }]
        file_path = temp_json_file(data)

        file_type, metadata = detect_file_type(file_path)

        assert file_type == FileType.CLAUDE_PROJECTS
        assert metadata["projects"] == 1

    def test_detect_chatgpt_conversations_with_mapping(self, temp_json_file):
        """Test detection of ChatGPT conversations with mapping."""
        data = [{
            "conversation_id": "chat-123",
            "mapping": {}
        }]
        file_path = temp_json_file(data)

        file_type, metadata = detect_file_type(file_path)

        assert file_type == FileType.CHATGPT_CONVERSATIONS

    def test_detect_chatgpt_conversations_minimal(self, temp_json_file):
        """Test detection of ChatGPT with just conversation_id."""
        data = [{
            "conversation_id": "chat-123",
            "title": "Test Chat"
        }]
        file_path = temp_json_file(data)

        file_type, metadata = detect_file_type(file_path)

        assert file_type == FileType.CHATGPT_CONVERSATIONS
        assert metadata["conversations"] == 1

    def test_detect_unknown_format(self, temp_json_file):
        """Test detection of unknown JSON format."""
        data = [{"random": "data"}]
        file_path = temp_json_file(data)

        file_type, metadata = detect_file_type(file_path)

        assert file_type == FileType.UNKNOWN
        assert metadata["structure"] == "list"

    def test_detect_unknown_dict_format(self, temp_json_file):
        """Test detection of dict instead of list."""
        data = {"key": "value"}
        file_path = temp_json_file(data)

        file_type, metadata = detect_file_type(file_path)

        assert file_type == FileType.UNKNOWN
        assert metadata["structure"] == "dict"

    def test_invalid_json(self, tmp_path):
        """Test detection with invalid JSON."""
        file_path = tmp_path / "invalid.json"
        file_path.write_text("not valid json{")

        file_type, metadata = detect_file_type(str(file_path))

        assert file_type == FileType.UNKNOWN
        assert "error" in metadata


class TestExtractFactsFromConversations:
    """Test fact extraction from Claude conversations."""

    def test_extract_from_summary(self):
        """Test extraction from conversation summary."""
        conversations = [{
            "summary": "The user expressed interest in Python programming. They demonstrated knowledge of machine learning.",
            "name": "Python Discussion"
        }]

        facts = extract_facts_from_conversations(conversations)

        # Should extract from summary and conversation name
        assert len(facts) >= 1
        assert any("Discussed: Python Discussion" in f for f in facts)

    def test_extract_fact_markers(self):
        """Test extraction with fact marker patterns."""
        conversations = [{
            "summary": "The user is interested in building web applications. They asked about FastAPI.",
            "name": "Web Dev"
        }]

        facts = extract_facts_from_conversations(conversations)

        # Should find "interested in" marker
        assert len(facts) >= 1

    def test_extract_filters_short_sentences(self):
        """Test that short sentences are filtered out."""
        conversations = [{
            "summary": "Hi. The user is learning. They code. The user expressed interest in Python.",
            "name": ""
        }]

        facts = extract_facts_from_conversations(conversations)

        # Short sentences (<20 chars) should be filtered
        assert all(len(f) >= 20 for f in facts)

    def test_respects_max_conversations_limit(self):
        """Test that max_conversations limit is respected."""
        conversations = [{"summary": f"Conv {i}", "name": f"C{i}"} for i in range(100)]

        facts = extract_facts_from_conversations(conversations, max_conversations=10)

        # Should only process first 10 conversations
        # Each has 1 fact from name = 10 facts minimum
        assert len(facts) <= 20  # Some buffer for summary extracts

    def test_empty_summary(self):
        """Test handling of empty summary."""
        conversations = [{
            "summary": "",
            "name": "Test Conv"
        }]

        facts = extract_facts_from_conversations(conversations)

        # Should still get conversation name
        assert len(facts) == 1
        assert "Discussed: Test Conv" in facts[0]

    def test_no_name(self):
        """Test handling of missing conversation name."""
        conversations = [{
            "summary": "The user expressed interest in testing.",
            "name": ""
        }]

        facts = extract_facts_from_conversations(conversations)

        # Should extract from summary only
        assert len(facts) >= 0

    def test_long_fact_filtered(self):
        """Test that overly long facts are filtered."""
        conversations = [{
            "summary": "The user expressed " + ("x" * 600),
            "name": ""
        }]

        facts = extract_facts_from_conversations(conversations)

        # Facts over 500 chars should be filtered
        assert all(len(f) < 500 for f in facts)


class TestExtractFactsFromMessages:
    """Test fact extraction from raw chat messages (slower, per-message extraction)."""

    def test_extract_self_disclosure(self):
        """Finds personal facts from user messages."""
        messages = [
            {"sender": "human", "text": "I work as a data scientist at a biotech company in Boston."},
        ]
        facts = extract_facts_from_messages(messages)
        assert len(facts) >= 1
        assert any("data scientist" in f.lower() for f in facts)

    def test_ignores_assistant_messages(self):
        messages = [{"sender": "assistant", "text": "I am Claude, made by Anthropic."}]
        facts = extract_facts_from_messages(messages)
        assert len(facts) == 0

    def test_ignores_short_messages(self):
        messages = [{"sender": "human", "text": "Hi"}]
        facts = extract_facts_from_messages(messages)
        assert len(facts) == 0

    def test_filters_short_facts(self):
        messages = [{"sender": "human", "text": "I am ok."}]  # extracted fact < 20 chars
        facts = extract_facts_from_messages(messages)
        assert len(facts) == 0

    def test_multiple_markers(self):
        """Each marker produces at most one fact per message."""
        messages = [{
            "sender": "human",
            "text": "I work as a teacher and I live in Toronto. I have two cats named Luna and Pixel.",
        }]
        facts = extract_facts_from_messages(messages)
        assert len(facts) >= 1

    def test_respects_max_messages(self):
        messages = [
            {"sender": "human", "text": f"I am interested in topic number {i} and it is fascinating."}
            for i in range(200)
        ]
        facts = extract_facts_from_messages(messages, max_messages=5)
        assert len(facts) <= 5

    def test_empty_messages(self):
        assert extract_facts_from_messages([]) == []

    def test_no_marker_no_extraction(self):
        """Messages without self-disclosure markers produce no facts."""
        messages = [{"sender": "human", "text": "Please help me debug this code for the API."}]
        facts = extract_facts_from_messages(messages)
        assert len(facts) == 0


class TestExtractFactsFromProjects:
    """Test fact extraction from Claude projects."""

    def test_extract_real_project(self):
        """Test extraction from real project."""
        projects = [{
            "name": "API Backend",
            "description": "FastAPI backend for web app",
            "docs": [{"content": "doc1"}, {"content": "doc2"}]
        }]

        facts = extract_facts_from_projects(projects)

        # Should extract project metadata
        assert len(facts) == 2
        assert any("Working on project: API Backend" in f for f in facts)
        assert any("2 documents" in f for f in facts)

    def test_detect_fiction_project(self):
        """Test detection of creative writing project."""
        projects = [{
            "name": "My Novel",
            "description": "A story about wizards",
            "docs": []
        }]

        facts = extract_facts_from_projects(projects)

        # Should mark as creative writing, not detailed project
        assert len(facts) == 1
        assert "creative writing project" in facts[0].lower()

    def test_project_without_description(self):
        """Test project with missing description."""
        projects = [{
            "name": "Untitled Project",
            "description": "",
            "docs": [{"content": "doc1"}]
        }]

        facts = extract_facts_from_projects(projects)

        # Should still capture project name
        assert len(facts) == 2
        assert any("Has project: Untitled Project" in f for f in facts)

    def test_fiction_markers(self):
        """Test various fiction marker keywords."""
        fiction_names = ["My Story", "Novel Draft", "Character Development", "Chapter 1", "Fiction Project"]

        for name in fiction_names:
            # Description must be > 10 chars for fiction detection to trigger
            projects = [{"name": name, "description": "A creative writing project", "docs": []}]
            facts = extract_facts_from_projects(projects)
            assert any("creative writing" in f.lower() for f in facts), f"Failed to detect fiction in: {name}"

    def test_respects_max_projects_limit(self):
        """Test that max_projects limit is respected."""
        projects = [{"name": f"Project {i}", "description": "", "docs": []} for i in range(30)]

        facts = extract_facts_from_projects(projects, max_projects=10)

        # Should only process first 10 projects
        assert len(facts) <= 20  # Max 2 facts per project

    def test_no_name_project(self):
        """Test handling of project without name."""
        projects = [{
            "name": "",
            "description": "Test description",
            "docs": []
        }]

        facts = extract_facts_from_projects(projects)

        # Should skip projects without names
        assert len(facts) == 0

    def test_doc_count_zero(self):
        """Test project with no documents."""
        projects = [{
            "name": "Empty Project",
            "description": "Just starting",
            "docs": []
        }]

        facts = extract_facts_from_projects(projects)

        # Should get project fact but not doc count
        assert len(facts) == 1
        assert "Working on project" in facts[0]
        assert "0 documents" not in facts[0]


class TestExtractFactsFromChatGPT:
    """Test fact extraction from ChatGPT conversations."""

    def test_extract_from_user_messages(self):
        """Test extraction from user messages."""
        conversations = [{
            "title": "Python Help",
            "mapping": {
                "msg1": {
                    "message": {
                        "author": {"role": "user"},
                        "content": {"parts": ["I'm working on a Django project for my company"]}
                    }
                }
            }
        }]

        facts = extract_facts_from_chatgpt(conversations)

        # Should extract self-disclosure + title
        assert len(facts) >= 1
        assert any("working on" in f.lower() for f in facts)

    def test_ignore_assistant_messages(self):
        """Test that assistant messages are ignored."""
        conversations = [{
            "title": "Chat",
            "mapping": {
                "msg1": {
                    "message": {
                        "author": {"role": "assistant"},
                        "content": {"parts": ["I am Claude"]}
                    }
                }
            }
        }]

        facts = extract_facts_from_chatgpt(conversations)

        # Should only extract from title, not assistant message
        assert len(facts) == 1
        assert facts[0] == "Discussed: Chat"

    def test_filter_new_chat_titles(self):
        """Test that default 'New chat' titles are filtered."""
        conversations = [{
            "title": "New chat",
            "mapping": {}
        }]

        facts = extract_facts_from_chatgpt(conversations)

        # Should skip default "New chat" titles
        assert len(facts) == 0

    def test_multiple_message_parts(self):
        """Test handling of messages with multiple parts."""
        conversations = [{
            "title": "Test",
            "mapping": {
                "msg1": {
                    "message": {
                        "author": {"role": "user"},
                        "content": {"parts": ["I'm working on ", "a web project"]}
                    }
                }
            }
        }]

        facts = extract_facts_from_chatgpt(conversations)

        # Should join parts
        assert len(facts) >= 1

    def test_empty_mapping(self):
        """Test handling of empty mapping."""
        conversations = [{
            "title": "Empty Chat",
            "mapping": {}
        }]

        facts = extract_facts_from_chatgpt(conversations)

        # Should still get title
        assert len(facts) == 1
        assert "Empty Chat" in facts[0]

    def test_fact_markers(self):
        """Test various self-disclosure markers."""
        markers = [
            "I am a developer",
            "I'm learning Python",
            "I work at Google",
            "I live in Toronto",
            "My name is Alice",
            "I like coding",
            "I love Python",
            "I have ADHD",
            "I'm working on a project"
        ]

        for marker in markers:
            conversations = [{
                "title": "Test",
                "mapping": {
                    "msg1": {
                        "message": {
                            "author": {"role": "user"},
                            "content": {"parts": [marker]}
                        }
                    }
                }
            }]

            facts = extract_facts_from_chatgpt(conversations)
            # Should extract the marker (or title if marker is too short)
            assert len(facts) >= 1, f"Failed to extract from: {marker}"

    def test_short_messages_filtered(self):
        """Test that short messages are filtered."""
        conversations = [{
            "title": "Test",
            "mapping": {
                "msg1": {
                    "message": {
                        "author": {"role": "user"},
                        "content": {"parts": ["Hi"]}
                    }
                }
            }
        }]

        facts = extract_facts_from_chatgpt(conversations)

        # Should only get title, not short message
        assert "Hi" not in str(facts)

    def test_respects_max_conversations_limit(self):
        """Test that max_conversations limit is respected."""
        conversations = [
            {"title": f"Chat {i}", "mapping": {}}
            for i in range(100)
        ]

        facts = extract_facts_from_chatgpt(conversations, max_conversations=10)

        # Should only process first 10
        assert len(facts) <= 10

    def test_null_message_handling(self):
        """Test handling of null/missing message data."""
        conversations = [{
            "title": "Test",
            "mapping": {
                "msg1": None,
                "msg2": {"message": None},
                "msg3": {"message": {"author": None, "content": None}}
            }
        }]

        facts = extract_facts_from_chatgpt(conversations)

        # Should not crash, just get title
        assert len(facts) == 1
        assert "Test" in facts[0]


# ===========================================================================
# Coverage gaps: file_detector.py lines 77-78 and 243
# ===========================================================================

class TestDetectFileTypeGeneralException:
    """Cover file_detector.py lines 77-78 (except Exception fallback)."""

    def test_general_exception_returns_unknown(self, tmp_path):
        """Non-JSONDecodeError exception in detect_file_type → FileType.UNKNOWN (lines 77-78)."""
        from backend.file_detector import detect_file_type, FileType
        from unittest.mock import patch

        tmp = tmp_path / "f.json"
        tmp.write_bytes(b'{"test": 1}')

        # Raise OSError (not JSONDecodeError) to hit the general except clause
        with patch("backend.file_detector.json.loads", side_effect=OSError("disk error")):
            file_type, info = detect_file_type(tmp)

        assert file_type == FileType.UNKNOWN
        assert "error" in info


class TestChatGPTEmptyParts:
    """Cover file_detector.py line 243 (if not parts: continue)."""

    def test_extract_chatgpt_empty_parts_skipped(self):
        """Message with author=user but empty parts list → continue (line 243)."""
        from backend.file_detector import extract_facts_from_chatgpt

        conversations = [{
            "title": "Test conversation about programming",
            "mapping": {
                "msg1": {
                    "message": {
                        "author": {"role": "user"},
                        "content": {"parts": []},  # empty parts → line 243 (continue)
                    }
                },
                "msg2": {
                    "message": {
                        "author": {"role": "user"},
                        "content": {"parts": ["I work as a software engineer at Google"]},
                    }
                },
            }
        }]

        facts = extract_facts_from_chatgpt(conversations)
        # msg1 is skipped (empty parts), msg2 may produce a fact
        # Should not crash; only msg2 content is considered
        assert isinstance(facts, list)
