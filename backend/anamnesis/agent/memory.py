"""
AgentMemory - High-level memory API for LLM applications.

Provides a simple, intuitive interface for:
- Conversation session management
- Memory recall based on context
- Automatic memory capture during conversations
- User profile and preference tracking

Example usage:
    from src.agent import AgentMemory

    # Initialize
    memory = AgentMemory("memories.db")

    # Start a conversation session
    session = memory.start_session()

    # Get relevant memories for a message
    context = memory.recall_for("User wants to learn Python")

    # Observe conversation exchanges
    memory.observe(
        user_message="How do I start learning Python?",
        assistant_response="Start with the basics..."
    )

    # End session (auto-consolidate to long-term memory)
    memory.end_session(auto_consolidate=True)
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from ..consolidation.pipeline import ConsolidationPipeline
from ..integration import (
    FormatStyle,
    InjectionMode,
    MemoryInjector,
)
from ..models import EmotionalValence, Episode, Fact, FactType
from ..stores.episodic import EpisodicStore
from ..stores.semantic import SemanticStore
from ..stores.sqlite_backend import SQLiteBackend


class SessionState(Enum):
    """State of a conversation session."""
    ACTIVE = "active"
    PAUSED = "paused"
    ENDED = "ended"


@dataclass
class RecallResult:
    """Result of a memory recall operation."""
    episodes: List[Episode] = field(default_factory=list)
    facts: List[Fact] = field(default_factory=list)
    context_text: str = ""
    relevance_scores: Dict[str, float] = field(default_factory=dict)
    token_count: int = 0

    @property
    def has_memories(self) -> bool:
        """Check if any memories were recalled."""
        return bool(self.episodes or self.facts)

    def as_system_prompt(self) -> str:
        """Format as system prompt addition."""
        return self.context_text if self.context_text else ""

    def as_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "episodes": [ep.to_dict() for ep in self.episodes],
            "facts": [f.to_dict() for f in self.facts],
            "context_text": self.context_text,
            "token_count": self.token_count,
        }


@dataclass
class UserProfile:
    """User profile built from memories."""
    preferences: List[Fact] = field(default_factory=list)
    traits: List[Fact] = field(default_factory=list)
    interests: List[str] = field(default_factory=list)
    interaction_style: Optional[str] = None
    emotional_baseline: EmotionalValence = EmotionalValence.NEUTRAL
    last_updated: Optional[datetime] = None

    def as_prompt_context(self) -> str:
        """Format profile as context for LLM."""
        lines = ["## User Profile"]

        if self.preferences:
            lines.append("\n### Preferences")
            for pref in self.preferences[:5]:
                lines.append(f"- {pref.content}")

        if self.traits:
            lines.append("\n### Traits")
            for trait in self.traits[:5]:
                lines.append(f"- {trait.content}")

        if self.interests:
            lines.append("\n### Interests")
            lines.append(", ".join(self.interests[:10]))

        if self.interaction_style:
            lines.append("\n### Interaction Style")
            lines.append(self.interaction_style)

        return "\n".join(lines)


@dataclass
class Session:
    """
    Conversation session tracking.

    Maintains context for the current conversation including:
    - Messages exchanged
    - Topics discussed
    - Emotional tone
    - Recalled memories
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    started_at: datetime = field(default_factory=datetime.now)
    ended_at: Optional[datetime] = None
    state: SessionState = SessionState.ACTIVE
    messages: List[Dict[str, str]] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)
    emotional_valence: EmotionalValence = EmotionalValence.NEUTRAL
    recalled_memory_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float:
        """Get session duration in seconds."""
        end = self.ended_at or datetime.now()
        return (end - self.started_at).total_seconds()

    @property
    def message_count(self) -> int:
        """Get number of messages in session."""
        return len(self.messages)

    def add_message(self, role: str, content: str):
        """Add a message to the session."""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        })


class AgentMemory:
    """
    High-level memory API for LLM applications.

    Provides a simple interface for memory operations without
    requiring knowledge of the underlying storage and processing
    components.
    """

    def __init__(
        self,
        db_path: str = "anamnesis.db",
        budget: str = "standard",
        auto_observe: bool = True,
        consolidation_threshold: int = 10,
    ):
        """
        Initialize AgentMemory.

        Args:
            db_path: Path to SQLite database
            budget: Context budget preset ('minimal', 'standard', 'generous', 'unlimited')
            auto_observe: Whether to auto-observe conversation exchanges
            consolidation_threshold: Messages before auto-consolidation triggers
        """
        self.db_path = db_path
        self.budget = budget
        self.auto_observe = auto_observe
        self.consolidation_threshold = consolidation_threshold

        # Initialize stores
        self.backend = SQLiteBackend(db_path)
        self.episodic_store = EpisodicStore(self.backend)
        self.semantic_store = SemanticStore(self.backend)

        # Initialize injector
        self.injector = MemoryInjector(
            episodic_store=self.episodic_store,
            semantic_store=self.semantic_store,
            budget=budget,
        )

        # Session tracking
        self._current_session: Optional[Session] = None
        self._session_history: List[Session] = []

        # Callbacks
        self._on_memory_recalled: Optional[Callable[[RecallResult], None]] = None
        self._on_memory_stored: Optional[Callable[[str], None]] = None

    # Session Management

    def start_session(self, metadata: Optional[Dict[str, Any]] = None) -> Session:
        """
        Start a new conversation session.

        Args:
            metadata: Optional session metadata

        Returns:
            New Session object
        """
        if self._current_session and self._current_session.state == SessionState.ACTIVE:
            # End previous session
            self.end_session(auto_consolidate=False)

        session = Session(metadata=metadata or {})
        self._current_session = session
        return session

    def pause_session(self) -> Optional[Session]:
        """
        Pause the current session (can be resumed).

        Returns:
            Paused session or None
        """
        if self._current_session:
            self._current_session.state = SessionState.PAUSED
            return self._current_session
        return None

    def resume_session(self) -> Optional[Session]:
        """
        Resume a paused session.

        Returns:
            Resumed session or None
        """
        if self._current_session and self._current_session.state == SessionState.PAUSED:
            self._current_session.state = SessionState.ACTIVE
            return self._current_session
        return None

    def end_session(
        self,
        auto_consolidate: bool = True,
        summary: Optional[str] = None,
    ) -> Optional[Session]:
        """
        End the current session.

        Args:
            auto_consolidate: Whether to consolidate messages into episode
            summary: Optional session summary

        Returns:
            Ended session or None
        """
        if not self._current_session:
            return None

        session = self._current_session
        session.state = SessionState.ENDED
        session.ended_at = datetime.now()

        # Consolidate to long-term memory if requested
        if auto_consolidate and session.messages:
            self._consolidate_session(session, summary)

        self._session_history.append(session)
        self._current_session = None

        return session

    @property
    def current_session(self) -> Optional[Session]:
        """Get current active session."""
        return self._current_session

    # Memory Recall

    def recall_for(
        self,
        query: str,
        max_episodes: int = 5,
        max_facts: int = 10,
        include_profile: bool = True,
        format_style: FormatStyle = FormatStyle.MARKDOWN,
    ) -> RecallResult:
        """
        Recall relevant memories for a query/message.

        Args:
            query: Query or user message to find relevant memories for
            max_episodes: Maximum episodes to include
            max_facts: Maximum facts to include
            include_profile: Whether to include user profile in context
            format_style: How to format the context text

        Returns:
            RecallResult with relevant memories and formatted context
        """
        # Get injection result
        result = self.injector.inject_for_query(
            query=query,
            mode=InjectionMode.SYSTEM_PROMPT,
            max_episodes=max_episodes,
            max_facts=max_facts,
        )

        # Build RecallResult
        recall = RecallResult(
            context_text=result.injected_text,
            token_count=result.total_tokens,
        )

        # Get the actual episodes and facts
        episodes = self.episodic_store.search(query, limit=max_episodes)
        facts = self.semantic_store.search(query, limit=max_facts)

        recall.episodes = episodes
        recall.facts = facts

        # Calculate relevance scores
        for i, ep in enumerate(episodes):
            score = 1.0 - (i * 0.1)  # Simple position-based score
            recall.relevance_scores[ep.id] = score

        # Track recalled memories in session
        if self._current_session:
            for ep in episodes:
                if ep.id not in self._current_session.recalled_memory_ids:
                    self._current_session.recalled_memory_ids.append(ep.id)

        # Trigger callback
        if self._on_memory_recalled:
            self._on_memory_recalled(recall)

        return recall

    def recall_recent(
        self,
        days_back: int = 7,
        limit: int = 10,
    ) -> List[Episode]:
        """
        Recall recent episodes.

        Args:
            days_back: How many days to look back
            limit: Maximum episodes to return

        Returns:
            List of recent episodes
        """
        return self.episodic_store.get_recent(days=days_back, limit=limit)

    def recall_by_topic(
        self,
        topic: str,
        limit: int = 10,
    ) -> List[Episode]:
        """
        Recall episodes about a specific topic.

        Args:
            topic: Topic to search for
            limit: Maximum episodes to return

        Returns:
            List of matching episodes
        """
        return self.episodic_store.search(topic, limit=limit)

    def get_user_profile(self) -> UserProfile:
        """
        Build user profile from stored facts.

        Returns:
            UserProfile with preferences, traits, interests
        """
        profile = UserProfile(last_updated=datetime.now())

        # Get preferences
        pref_facts = self.semantic_store.get_by_type(FactType.PREFERENCE, limit=20)
        profile.preferences = pref_facts

        # Get traits
        trait_facts = self.semantic_store.get_by_type("trait", limit=10)
        profile.traits = trait_facts

        # Extract interests from preferences and traits
        interests = set()
        for fact in pref_facts + trait_facts:
            # Simple keyword extraction
            words = fact.content.lower().split()
            for word in words:
                if len(word) > 5 and word not in {'prefers', 'enjoys', 'likes', 'about'}:
                    interests.add(word.strip('.,!?'))
        profile.interests = list(interests)[:10]

        return profile

    # Observation (Memory Capture)

    def observe(
        self,
        user_message: str,
        assistant_response: str,
        importance: float = 0.5,
        valence: EmotionalValence = EmotionalValence.NEUTRAL,
        topic: Optional[str] = None,
    ):
        """
        Observe a conversation exchange.

        Records the exchange in the current session for later
        consolidation into long-term memory.

        Args:
            user_message: What the user said
            assistant_response: What the assistant responded
            importance: Importance rating (0-1)
            valence: Emotional valence
            topic: Optional topic of the exchange
        """
        if not self._current_session:
            self.start_session()

        session = self._current_session

        # Add messages to session
        session.add_message("user", user_message)
        session.add_message("assistant", assistant_response)

        # Update session emotional valence
        if valence != EmotionalValence.NEUTRAL:
            session.emotional_valence = valence

        # Update topics
        if topic and topic not in session.topics:
            session.topics.append(topic)

        # Auto-consolidate if threshold reached
        if (self.auto_observe and
            session.message_count >= self.consolidation_threshold * 2):
            self._mid_session_consolidate()

    def store_fact(
        self,
        content: str,
        fact_type: FactType = FactType.GENERAL,
        confidence: float = 0.8,
        importance: float = 0.5,
    ) -> str:
        """
        Store a fact directly.

        Args:
            content: Fact content
            fact_type: Type (preference, trait, skill, etc.)
            confidence: Confidence level (0-1)
            importance: Importance level (0-1)

        Returns:
            Fact ID
        """
        fact = Fact(
            id=f"fact-{uuid.uuid4().hex[:8]}",
            content=content,
            fact_type=fact_type,
            confidence=confidence,
            importance=importance,
            first_learned=datetime.now(),
            last_confirmed=datetime.now(),
        )

        self.semantic_store.save(fact)

        if self._on_memory_stored:
            self._on_memory_stored(fact.id)

        return fact.id

    def store_episode(
        self,
        messages: List[Dict[str, str]],
        summary: str,
        topic: Optional[str] = None,
        importance: float = 0.5,
        valence: EmotionalValence = EmotionalValence.NEUTRAL,
    ) -> str:
        """
        Store an episode directly.

        Args:
            messages: List of message dicts with role/content
            summary: Episode summary
            topic: Episode topic
            importance: Importance level (0-1)
            valence: Emotional valence

        Returns:
            Episode ID
        """
        episode = Episode(
            id=f"ep-{uuid.uuid4().hex[:8]}",
            messages=messages,
            summary=summary,
            topic=topic,
            importance=importance,
            overall_valence=valence,
            started_at=datetime.now(),
            ended_at=datetime.now(),
        )

        self.episodic_store.save(episode)

        if self._on_memory_stored:
            self._on_memory_stored(episode.id)

        return episode.id

    # Context Generation

    def get_context_for_prompt(
        self,
        query: str,
        max_tokens: int = 1000,
        include_profile: bool = True,
    ) -> str:
        """
        Get formatted context for LLM prompt injection.

        Args:
            query: Current query/message
            max_tokens: Maximum token budget
            include_profile: Whether to include user profile

        Returns:
            Formatted context string
        """
        parts = []

        # Get user profile
        if include_profile:
            profile = self.get_user_profile()
            if profile.preferences or profile.traits:
                parts.append(profile.as_prompt_context())

        # Get relevant memories
        recall = self.recall_for(query)
        if recall.context_text:
            parts.append(recall.context_text)

        return "\n\n".join(parts)

    # Maintenance

    def consolidate_all(self, limit: int = 50) -> Dict[str, int]:
        """
        Run full consolidation pipeline.

        Args:
            limit: Maximum conversations to process

        Returns:
            Stats dict with counts
        """
        from ..models import Conversation

        pipeline = ConsolidationPipeline(
            episodic_store=self.episodic_store,
            semantic_store=self.semantic_store,
        )

        # Get unprocessed conversations from backend
        raw_convos = self.backend.get_unprocessed_conversations(limit=limit)

        if not raw_convos:
            return {"episodes_created": 0, "facts_extracted": 0, "conversations_processed": 0}

        # Convert to Conversation objects
        conversations = []
        for raw in raw_convos:
            from ..models import ConversationMessage
            messages = [
                ConversationMessage(
                    role=m.get("role", "user"),
                    content=m.get("content", ""),
                )
                for m in raw.get("messages", [])
            ]
            conv = Conversation(
                id=raw["id"],
                name=raw.get("name", ""),
                messages=messages,
                created_at=datetime.fromisoformat(raw["created_at"]) if raw.get("created_at") else None,
            )
            conversations.append(conv)

        results = pipeline.consolidate_batch(conversations)

        # Mark as processed
        for raw in raw_convos:
            self.backend.mark_conversation_processed(raw["id"])

        return {
            "episodes_created": len([r for r in results if r.episode]),
            "facts_extracted": sum(len(r.facts) for r in results),
            "conversations_processed": len(results),
        }

    def get_stats(self) -> Dict[str, Any]:
        """
        Get memory statistics.

        Returns:
            Dict with counts and stats
        """
        return self.backend.get_stats()

    # Callbacks

    def on_memory_recalled(self, callback: Callable[[RecallResult], None]):
        """Set callback for when memories are recalled."""
        self._on_memory_recalled = callback

    def on_memory_stored(self, callback: Callable[[str], None]):
        """Set callback for when a memory is stored."""
        self._on_memory_stored = callback

    # Private Methods

    def _consolidate_session(self, session: Session, summary: Optional[str] = None):
        """Consolidate a session into an episode."""
        if not session.messages:
            return

        # Generate summary if not provided
        if not summary:
            # Simple summary from topics and message count
            topic_str = ", ".join(session.topics) if session.topics else "general conversation"
            summary = f"Conversation about {topic_str} ({session.message_count} messages)"

        # Determine topic
        topic = session.topics[0] if session.topics else None

        # Create episode
        episode = Episode(
            id=f"ep-{session.id[:8]}",
            messages=session.messages,
            summary=summary,
            topic=topic,
            started_at=session.started_at,
            ended_at=session.ended_at,
            overall_valence=session.emotional_valence,
            importance=0.6,  # Default importance for new conversations
            metadata=session.metadata,
        )

        self.episodic_store.save(episode)

        if self._on_memory_stored:
            self._on_memory_stored(episode.id)

    def _mid_session_consolidate(self):
        """Partial consolidation during long sessions."""
        if not self._current_session:
            return

        session = self._current_session

        # Take half the messages and consolidate them
        midpoint = len(session.messages) // 2
        messages_to_consolidate = session.messages[:midpoint]

        if len(messages_to_consolidate) >= 4:  # Minimum meaningful exchange
            # Create episode from partial conversation
            topic = session.topics[0] if session.topics else "ongoing conversation"
            summary = f"Part of conversation about {topic}"

            episode = Episode(
                id=f"ep-{uuid.uuid4().hex[:8]}",
                messages=messages_to_consolidate,
                summary=summary,
                topic=topic,
                started_at=session.started_at,
                overall_valence=session.emotional_valence,
                importance=0.5,
            )

            self.episodic_store.save(episode)

            # Remove consolidated messages from session
            session.messages = session.messages[midpoint:]
