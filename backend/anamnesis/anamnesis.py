"""
Anamnesis - Main Memory System Interface.

Ties together all memory components into a unified API.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from .consolidation.pipeline import ConsolidationPipeline
from .integration import FormatStyle, InjectionMode, MemoryInjector
from .loaders.claude_loader import ClaudeExportLoader
from .models import (
    Conversation,
    ConversationMessage,
    EmotionalValence,
    Episode,
    Fact,
    FactType,
    WorkingMemory,
)
from .stores.episodic import EpisodicStore
from .stores.semantic import SemanticStore
from .stores.sqlite_backend import SQLiteBackend


class Anamnesis:
    """
    Main interface to the Anamnesis memory system.

    Provides unified access to:
    - Episodic memory (experiences)
    - Semantic memory (facts)
    - Working memory (active context)
    - Data import/export
    """

    def __init__(self, db_path: Optional[str] = None):
        """Initialize Anamnesis with database path."""
        self.backend = SQLiteBackend(db_path)
        self.episodic = EpisodicStore(self.backend)
        self.semantic = SemanticStore(self.backend)
        self.working = WorkingMemory()

    # === Import/Export ===

    def import_claude_export(self, export_path: str) -> Dict[str, int]:
        """
        Import a Claude web export into the memory system.

        Returns counts of imported items.
        """
        loader = ClaudeExportLoader(export_path)

        # Load conversations
        conversations = loader.load_conversations()
        convo_count = 0
        for convo in conversations:
            self._import_conversation(convo)
            convo_count += 1

        # Load existing memories as facts
        facts = loader.load_memories()
        fact_count = 0
        for fact in facts:
            self.semantic.save(fact)
            fact_count += 1

        return {
            "conversations": convo_count,
            "facts": fact_count,
        }

    def _import_conversation(self, convo: Conversation):
        """Import a single conversation."""
        # Save raw conversation
        convo_dict = {
            "id": convo.id,
            "name": convo.name,
            "summary": convo.summary,
            "messages": [
                {
                    "role": m.role,
                    "content": m.content,
                    "timestamp": m.timestamp.isoformat() if m.timestamp else None,
                }
                for m in convo.messages
            ],
            "created_at": convo.created_at.isoformat() if convo.created_at else None,
            "updated_at": convo.updated_at.isoformat() if convo.updated_at else None,
            "metadata": convo.metadata,
            "processed": False,
        }
        self.backend.save_conversation(convo_dict)

        # If conversation has a summary, create an episode
        if convo.summary or convo.name:
            episode = Episode(
                summary=convo.summary or convo.name or "",
                messages=[
                    {"role": m.role, "content": m.content}
                    for m in convo.messages
                ],
                topic=convo.name,
                started_at=convo.created_at,
                ended_at=convo.updated_at,
                source_id=convo.id,
                importance=0.5,
            )
            self.episodic.save(episode)

    # === Query Interface ===

    def remember(self, query: str, limit: int = 5) -> Dict[str, List]:
        """
        Retrieve relevant memories for a query.

        Searches both episodic and semantic stores.
        """
        episodes = self.episodic.search(query, limit)
        facts = self.semantic.search(query, limit)

        return {
            "episodes": episodes,
            "facts": facts,
        }

    def recall_timeframe(
        self,
        description: str,
        days_back: int = 30
    ) -> List[Episode]:
        """
        Recall episodes from a described timeframe.

        E.g., "last week", "in October", "recently"
        """
        # For now, just use recent - could add NLP parsing later
        return self.episodic.get_recent(days=days_back)

    def get_context_for_conversation(self, limit: int = 10) -> Dict[str, Any]:
        """
        Get relevant context for starting/continuing a conversation.

        Returns a mix of recent episodes and important facts.
        """
        recent = self.episodic.get_recent(days=7, limit=5)
        important_episodes = self.episodic.get_important(threshold=0.7, limit=3)
        personal_facts = self.semantic.get_personal_facts(limit=5)
        preferences = self.semantic.get_preferences(limit=5)

        return {
            "recent_episodes": recent,
            "important_episodes": important_episodes,
            "personal_facts": personal_facts,
            "preferences": preferences,
            "working_memory": self.working.get_context(),
        }

    def format_context_for_llm(self, context: Dict[str, Any]) -> str:
        """
        Format memory context as text for LLM injection.
        """
        lines = ["## Memory Context\n"]

        # Personal facts
        if context.get("personal_facts"):
            lines.append("### About the User")
            for fact in context["personal_facts"]:
                lines.append(f"- {fact.content}")
            lines.append("")

        # Preferences
        if context.get("preferences"):
            lines.append("### Preferences")
            for fact in context["preferences"]:
                lines.append(f"- {fact.content}")
            lines.append("")

        # Recent episodes
        if context.get("recent_episodes"):
            lines.append("### Recent Conversations")
            for ep in context["recent_episodes"][:3]:
                date_str = ep.started_at.strftime("%Y-%m-%d") if ep.started_at else "Unknown"
                lines.append(f"- [{date_str}] {ep.summary[:200] if ep.summary else ep.topic or 'Untitled'}")
            lines.append("")

        return "\n".join(lines)

    # === Stats ===

    def get_stats(self) -> Dict[str, Any]:
        """Get memory system statistics."""
        db_stats = self.backend.get_stats()
        return {
            "episodes": db_stats["episodes"],
            "facts": db_stats["facts"],
            "raw_conversations": db_stats["conversations"],
            "working_memory_items": len(self.working.items),
        }

    def get_fact_summary(self) -> Dict[str, List[str]]:
        """Get a summary of known facts by category."""
        summary = {}
        for fact_type in [FactType.PERSONAL, FactType.PREFERENCE, FactType.PROFESSIONAL, FactType.PROJECT, FactType.GENERAL]:
            facts = self.semantic.get_by_type(fact_type, limit=20)
            summary[fact_type] = [f.content for f in facts]
        return summary

    # === Real-time Integration API ===

    def get_injection(
        self,
        query: str = "",
        format: str = "markdown",
        budget: str = "standard",
        max_episodes: int = 5,
        max_facts: int = 10,
    ) -> Dict[str, Any]:
        """
        Get memory context injection for an LLM prompt.

        This is the main API for integrating Anamnesis with any LLM system.

        Args:
            query: Optional query to find relevant memories
            format: Output format ("markdown", "xml", "natural", "bullet", "minimal")
            budget: Token budget preset ("minimal", "standard", "extensive", "long_context")
            max_episodes: Maximum episodes to include
            max_facts: Maximum facts to include

        Returns:
            Dict containing:
            - text: Formatted memory context string
            - episodes_used: Number of episodes included
            - facts_used: Number of facts included
            - tokens: Estimated token count

        Example:
            >>> ana = Anamnesis("memory.db")
            >>> result = ana.get_injection("help with Python")
            >>> system_prompt = f"{base_prompt}\\n\\n{result['text']}"
        """
        format_map = {
            "markdown": FormatStyle.MARKDOWN,
            "xml": FormatStyle.XML,
            "natural": FormatStyle.NATURAL,
            "bullet": FormatStyle.BULLET,
            "minimal": FormatStyle.MINIMAL,
        }

        injector = MemoryInjector(
            episodic_store=self.episodic,
            semantic_store=self.semantic,
            budget=budget,
            format_style=format_map.get(format, FormatStyle.MARKDOWN),
        )

        result = injector.inject_for_query(
            query=query,
            mode=InjectionMode.SYSTEM_PROMPT,
            max_episodes=max_episodes,
            max_facts=max_facts,
        )

        return {
            "text": result.injected_text,
            "episodes_used": result.episodes_used,
            "facts_used": result.facts_used,
            "tokens": result.total_tokens,
            "budget_used": result.budget_used,
        }

    def add_fact(
        self,
        content: str,
        fact_type: FactType = FactType.GENERAL,
        confidence: float = 0.8,
        importance: float = 0.5,
    ) -> Fact:
        """
        Add a new fact to semantic memory.

        Args:
            content: The fact content
            fact_type: Category (personal, preference, project, etc.)
            confidence: How confident (0.0-1.0)
            importance: How important (0.0-1.0)

        Returns:
            The created Fact object

        Example:
            >>> ana = Anamnesis("memory.db")
            >>> ana.add_fact("User prefers Python over JavaScript", "preference", 0.9)
        """
        return self.semantic.add_fact(
            content=content,
            fact_type=fact_type,
            confidence=confidence,
            importance=importance,
        )

    def add_episode(
        self,
        summary: str,
        topic: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        valence: EmotionalValence = EmotionalValence.NEUTRAL,
        importance: float = 0.5,
    ) -> Episode:
        """
        Add a new episode to episodic memory.

        Args:
            summary: Summary of the experience
            topic: Topic or title
            messages: Optional conversation messages
            valence: Emotional valence
            importance: How important (0.0-1.0)

        Returns:
            The created Episode object

        Example:
            >>> ana = Anamnesis("memory.db")
            >>> ana.add_episode("Discussed Python best practices", "Python help")
        """
        episode = Episode(
            summary=summary,
            topic=topic,
            messages=messages or [],
            started_at=datetime.now(),
            overall_valence=valence,
            importance=importance,
        )
        self.episodic.save(episode)
        return episode

    def record_conversation(
        self,
        messages: List[Dict[str, str]],
        topic: Optional[str] = None,
        consolidate: bool = True,
    ) -> Optional[Episode]:
        """
        Record a conversation and optionally consolidate it.

        This is the main API for recording conversations in real-time.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            topic: Optional topic/title
            consolidate: Whether to run consolidation (extract facts, etc.)

        Returns:
            Episode if consolidated, None otherwise

        Example:
            >>> ana = Anamnesis("memory.db")
            >>> messages = [
            ...     {"role": "user", "content": "Help me with Python"},
            ...     {"role": "assistant", "content": "Sure! What do you need?"}
            ... ]
            >>> episode = ana.record_conversation(messages, "Python help")
        """
        # Create conversation object
        convo = Conversation(
            name=topic,
            messages=[
                ConversationMessage(
                    role=m.get("role", "user"),
                    content=m.get("content", ""),
                    timestamp=datetime.now(),
                )
                for m in messages
            ],
            created_at=datetime.now(),
        )

        # Save raw conversation
        self._import_conversation(convo)

        if consolidate:
            # Run consolidation
            pipeline = ConsolidationPipeline(
                episodic_store=self.episodic,
                semantic_store=self.semantic,
            )
            result = pipeline.consolidate(convo)
            if result.success and result.episode:
                return result.episode

        return None

    def touch(self, memory_id: str, memory_type: str = "episode") -> bool:
        """
        Mark a memory as accessed, reinforcing its importance.

        Args:
            memory_id: ID of the memory
            memory_type: "episode" or "fact"

        Returns:
            True if memory was found and touched

        Example:
            >>> ana = Anamnesis("memory.db")
            >>> ana.touch("episode-123", "episode")
        """
        if memory_type == "episode":
            return self.episodic.touch(memory_id)
        elif memory_type == "fact":
            return self.semantic.touch(memory_id)
        return False

    def get_emotional_episodes(
        self,
        valence: str,
        limit: int = 10,
    ) -> List[Episode]:
        """
        Get episodes with a specific emotional valence.

        Args:
            valence: "positive", "negative", "neutral", "very_positive", "very_negative"
            limit: Maximum episodes to return

        Returns:
            List of Episode objects
        """
        valence_map = {
            "very_positive": EmotionalValence.VERY_POSITIVE,
            "positive": EmotionalValence.POSITIVE,
            "neutral": EmotionalValence.NEUTRAL,
            "negative": EmotionalValence.NEGATIVE,
            "very_negative": EmotionalValence.VERY_NEGATIVE,
        }
        v = valence_map.get(valence.lower(), EmotionalValence.NEUTRAL)
        return self.episodic.get_emotional(v, limit)

    def get_access_stats(self) -> Dict[str, Any]:
        """Get memory access statistics."""
        return self.backend.get_access_stats()
