"""
Smart file type detector for Claude exports.

Detects and routes different JSON formats to appropriate importers.
"""
import json
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from backend.auto_learner import FACT_MARKERS
from backend.config import VelquaConfig as Config


class FileType:
    """Detected file types."""
    CLAUDE_MEMORIES = "claude_memories"
    CLAUDE_CONVERSATIONS = "claude_conversations"
    CLAUDE_PROJECTS = "claude_projects"
    CHATGPT_CONVERSATIONS = "chatgpt_conversations"
    UNKNOWN = "unknown"


def detect_file_type(file_path: str) -> Tuple[str, Dict]:
    """
    Detect the type of JSON file.

    Returns:
        (file_type, metadata)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Claude memories.json format
        if isinstance(data, list) and len(data) > 0:
            first = data[0]
            if "conversations_memory" in first and "account_uuid" in first:
                return FileType.CLAUDE_MEMORIES, {
                    "records": len(data),
                    "has_project_memories": "project_memories" in first
                }

        # Claude projects.json format (check before conversations, more specific)
        if isinstance(data, list) and len(data) > 0:
            first = data[0]
            if "docs" in first and "name" in first and isinstance(first.get("docs"), list):
                return FileType.CLAUDE_PROJECTS, {
                    "projects": len(data)
                }

        # Claude conversations.json format
        if isinstance(data, list) and len(data) > 0:
            first = data[0]
            if "chat_messages" in first and "uuid" in first:
                total_messages = sum(len(c.get("chat_messages", [])) for c in data)
                return FileType.CLAUDE_CONVERSATIONS, {
                    "conversations": len(data),
                    "total_messages": total_messages
                }

        # ChatGPT conversations.json format
        if isinstance(data, list) and len(data) > 0:
            first = data[0]
            if "mapping" in first or "conversation_id" in first:
                return FileType.CHATGPT_CONVERSATIONS, {
                    "conversations": len(data)
                }

        # Unknown format
        return FileType.UNKNOWN, {
            "structure": "list" if isinstance(data, list) else "dict",
            "keys": list(data.keys())[:10] if isinstance(data, dict) else []
        }

    except json.JSONDecodeError as e:
        return FileType.UNKNOWN, {"error": f"Invalid JSON: {e}"}
    except Exception as e:
        return FileType.UNKNOWN, {"error": str(e)}


def extract_facts_from_conversations(
    conversations: List[Dict],
    max_conversations: int = 50
) -> List[str]:
    """
    Extract facts from Claude conversation summaries.

    Claude's summaries are LLM-generated and contain key facts.
    This is faster than processing every message.
    """
    facts = []

    for conv in conversations[:max_conversations]:
        # Extract from summary (LLM-generated, high quality)
        summary = conv.get("summary", "")
        if summary and len(summary) > 50:
            # Split into sentences
            sentences = summary.split(". ")
            for sentence in sentences:
                sentence = sentence.strip()
                # Look for fact patterns
                if any(marker in sentence.lower() for marker in [
                    "the user", "they expressed", "demonstrated",
                    "interested in", "working on", "asked about"
                ]):
                    # Extract the actual fact
                    fact = sentence.replace("The user ", "User ").replace("they ", "User ")
                    if Config.MIN_FACT_LENGTH < len(fact) < Config.MAX_FACT_LENGTH:
                        facts.append(fact)

        # Extract from conversation name (often reveals topics)
        name = conv.get("name", "")
        if name and len(name) > 3:
            facts.append(f"Discussed: {name}")

    return facts


def extract_facts_from_messages(
    messages: List[Dict],
    max_messages: int = 100
) -> List[str]:
    """
    Extract facts from raw chat messages.

    Slower but more thorough than summary extraction.
    """
    facts = []

    for msg in messages[:max_messages]:
        if msg.get("sender") != "human":
            continue

        text = msg.get("text", "")
        if not text or len(text) < 10:
            continue

        text_lower = text.lower()
        for marker in FACT_MARKERS:
            if marker in text_lower:
                # Extract the sentence containing the marker
                sentences = text.split(". ")
                for sentence in sentences:
                    if marker in sentence.lower():
                        cleaned = sentence.strip()
                        if Config.MIN_FACT_LENGTH < len(cleaned) < Config.MAX_FACT_LENGTH:
                            facts.append(cleaned)
                        break

    return facts


def extract_facts_from_projects(
    projects: List[Dict],
    max_projects: int = 20
) -> List[str]:
    """
    Extract facts from Claude projects.

    Projects contain user's work: novels, code, research, etc.
    Extract metadata but be careful with content (may be fiction).
    """
    facts = []

    for project in projects[:max_projects]:
        name = project.get("name", "")
        description = project.get("description", "")
        docs = project.get("docs", [])

        if not name:
            continue

        # Extract project metadata
        if description and len(description) > 10:
            # Check if it's a real project or fiction
            is_fiction = any(kw in name.lower() or kw in description.lower()
                           for kw in Config.FICTION_KEYWORDS)

            if is_fiction:
                # Still capture that they're working on it
                facts.append(f"Working on creative writing project: {name}")
            else:
                # Real project - capture details
                facts.append(f"Working on project: {name} - {description}")
        else:
            # No description, just capture project name
            facts.append(f"Has project: {name}")

        # Count docs as indicator of project size/importance
        doc_count = len(docs)
        if doc_count > 0:
            facts.append(f"Project '{name}' has {doc_count} documents")

    return facts


def extract_facts_from_chatgpt(
    conversations: List[Dict],
    max_conversations: int = 50
) -> List[str]:
    """
    Extract facts from ChatGPT conversations.json.

    ChatGPT format:
    [
        {
            "title": "Conversation name",
            "mapping": {
                "msg_id": {
                    "message": {
                        "author": {"role": "user" | "assistant"},
                        "content": {"parts": ["text"]}
                    }
                }
            }
        }
    ]
    """
    facts = []

    for conv in conversations[:max_conversations]:
        title = conv.get("title", "")
        mapping = conv.get("mapping", {})

        # Extract user messages
        for msg_id, msg_data in mapping.items():
            if not msg_data or "message" not in msg_data:
                continue

            message = msg_data["message"]
            if not message:
                continue

            author = message.get("author") or {}
            content = message.get("content") or {}

            # Only process user messages
            if not author or author.get("role") != "user":
                continue

            parts = content.get("parts", [])
            if not parts:
                continue

            text = " ".join(str(p) for p in parts if p)
            if len(text) < 10:
                continue

            text_lower = text.lower()
            for marker in FACT_MARKERS:
                if marker in text_lower:
                    sentences = text.split(". ")
                    for sentence in sentences:
                        if marker in sentence.lower():
                            cleaned = sentence.strip()
                            if Config.MIN_FACT_LENGTH < len(cleaned) < Config.MAX_FACT_LENGTH:
                                facts.append(cleaned)
                            break

        # Also extract from title if it reveals topics
        if title and len(title) > 3 and title != "New chat":
            facts.append(f"Discussed: {title}")

    return facts
