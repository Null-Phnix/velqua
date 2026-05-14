"""
Loader for Claude web export data.

Handles the JSON format from Claude's data export feature.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from ..models import Conversation, ConversationMessage, Fact, FactType


class ClaudeExportLoader:
    """
    Load conversations from Claude web export.

    Expected files:
    - conversations.json: Array of conversation objects
    - memories.json: Claude's extracted memories (optional)
    - projects.json: Project-specific conversations (optional)
    """

    def __init__(self, export_path: str):
        self.export_path = Path(export_path)
        self._conversations_cache: Optional[List[Dict]] = None
        self._memories_cache: Optional[List[Dict]] = None

    def _parse_timestamp(self, ts: str) -> datetime:
        """Parse ISO timestamp from Claude export."""
        if not ts:
            return datetime.now()
        try:
            # Handle various ISO formats
            ts = ts.replace("Z", "+00:00")
            if "." in ts:
                # Truncate microseconds if too long
                parts = ts.split(".")
                if len(parts[1]) > 6:
                    ts = parts[0] + "." + parts[1][:6] + parts[1][parts[1].find("+"):]
            return datetime.fromisoformat(ts.replace("+00:00", ""))
        except (ValueError, IndexError):
            return datetime.now()

    def load_conversations(self) -> List[Conversation]:
        """Load all conversations from export."""
        conversations_file = self.export_path / "conversations.json"
        if not conversations_file.exists():
            raise FileNotFoundError(f"No conversations.json in {self.export_path}")

        with open(conversations_file, "r", encoding="utf-8") as f:
            raw_convos = json.load(f)

        self._conversations_cache = raw_convos
        return [self._parse_conversation(c) for c in raw_convos]

    def _parse_conversation(self, raw: Dict[str, Any]) -> Conversation:
        """Parse a single conversation from raw JSON."""
        messages = []

        for msg in raw.get("chat_messages", []):
            # Extract text content
            content_parts = msg.get("content", [])
            text = ""
            for part in content_parts:
                if part.get("type") == "text":
                    text += part.get("text", "")

            if not text.strip():
                continue

            # Determine timestamp
            timestamp = self._parse_timestamp(
                msg.get("created_at") or
                (content_parts[0].get("start_timestamp") if content_parts else None)
            )

            messages.append(ConversationMessage(
                id=msg.get("uuid", ""),
                role=msg.get("sender", "user"),
                content=text,
                timestamp=timestamp,
                metadata={
                    "attachments": msg.get("attachments", []),
                    "files": msg.get("files", []),
                }
            ))

        return Conversation(
            id=raw.get("uuid", ""),
            name=raw.get("name") or None,
            summary=raw.get("summary") or None,
            messages=messages,
            created_at=self._parse_timestamp(raw.get("created_at", "")),
            updated_at=self._parse_timestamp(raw.get("updated_at", "")),
            metadata={
                "account_uuid": raw.get("account", {}).get("uuid"),
            }
        )

    def load_memories(self) -> List[Fact]:
        """
        Load Claude's extracted memories as Facts.

        These serve as ground truth for what SHOULD be extracted.
        """
        memories_file = self.export_path / "memories.json"
        if not memories_file.exists():
            return []

        with open(memories_file, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                return []
            raw_memories = json.loads(content)

        self._memories_cache = raw_memories
        facts = []

        for mem in raw_memories:
            # Claude memories have a "conversations_memory" field with text
            memory_text = mem.get("conversations_memory", "")
            if not memory_text:
                continue

            # Parse the structured memory text
            # Format is typically "**Category**\n\nContent\n\n**Category2**\n\nContent2"
            sections = self._parse_memory_sections(memory_text)

            for category, content in sections.items():
                facts.append(Fact(
                    content=content,
                    fact_type=self._categorize_memory(category),
                    confidence=0.95,  # High confidence - these are Claude's extractions
                    metadata={
                        "source": "claude_memory_export",
                        "original_category": category,
                    }
                ))

        return facts

    def _parse_memory_sections(self, text: str) -> Dict[str, str]:
        """Parse Claude's memory format into sections."""
        sections = {}
        current_category = "general"
        current_content = []

        for line in text.split("\n"):
            line = line.strip()
            if line.startswith("**") and line.endswith("**"):
                # Save previous section
                if current_content:
                    sections[current_category] = "\n".join(current_content).strip()
                # Start new section
                current_category = line.strip("*").strip().lower()
                current_content = []
            elif line:
                current_content.append(line)

        # Save last section
        if current_content:
            sections[current_category] = "\n".join(current_content).strip()

        return sections

    def _categorize_memory(self, category: str) -> FactType:
        """Map Claude's memory categories to our fact types."""
        category = category.lower()
        if "personal" in category or "about" in category:
            return FactType.PERSONAL
        elif "work" in category or "professional" in category:
            return FactType.PROFESSIONAL
        elif "preference" in category or "like" in category:
            return FactType.PREFERENCE
        elif "relationship" in category or "family" in category:
            return FactType.RELATIONSHIP
        elif "project" in category or "technical" in category:
            return FactType.PROJECT
        else:
            return FactType.GENERAL

    def iter_conversations(self) -> Generator[Conversation, None, None]:
        """Iterate over conversations without loading all into memory."""
        conversations_file = self.export_path / "conversations.json"
        if not conversations_file.exists():
            return

        with open(conversations_file, "r", encoding="utf-8") as f:
            raw_convos = json.load(f)

        for raw in raw_convos:
            yield self._parse_conversation(raw)

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the export."""
        convos = self.load_conversations()
        memories = self.load_memories()

        total_messages = sum(len(c.messages) for c in convos)
        named_convos = sum(1 for c in convos if c.name)
        summarized_convos = sum(1 for c in convos if c.summary)

        dates = [c.created_at for c in convos]
        date_range = (min(dates), max(dates)) if dates else (None, None)

        return {
            "conversations": len(convos),
            "total_messages": total_messages,
            "named_conversations": named_convos,
            "summarized_conversations": summarized_convos,
            "extracted_facts": len(memories),
            "date_range": date_range,
            "avg_messages_per_convo": total_messages / len(convos) if convos else 0,
        }


def load_claude_export(path: str) -> tuple[List[Conversation], List[Fact]]:
    """Convenience function to load Claude export."""
    loader = ClaudeExportLoader(path)
    return loader.load_conversations(), loader.load_memories()
