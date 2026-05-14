"""
Core data models for Anamnesis memory system.
"""

import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, TypedDict

# === Named Constants ===

DEFAULT_IMPORTANCE = 0.5
DEFAULT_CONFIDENCE = 0.8
DEFAULT_DECAY_RATE = 0.1
WORKING_MEMORY_CAPACITY = 7


# === Type Definitions ===

class MessageDict(TypedDict):
    """A single message in a conversation."""
    role: str
    content: str


# === Enums ===

class MemoryType(Enum):
    """Types of memories in the system."""
    EPISODIC = "episodic"      # Specific experiences/events
    SEMANTIC = "semantic"       # Facts, preferences, patterns
    WORKING = "working"         # Current active context


class EmotionalValence(Enum):
    """Emotional tone of a memory."""
    VERY_NEGATIVE = -2
    NEGATIVE = -1
    NEUTRAL = 0
    POSITIVE = 1
    VERY_POSITIVE = 2


class FactType(str, Enum):
    """Categories for semantic facts. Inherits str for transparent serialization."""
    PERSONAL = "personal"
    PREFERENCE = "preference"
    PROFESSIONAL = "professional"
    PROJECT = "project"
    RELATIONSHIP = "relationship"
    WORLD = "world"
    GENERAL = "general"


@dataclass
class Memory:
    """Base memory unit."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    memory_type: MemoryType = MemoryType.EPISODIC

    # Temporal
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0

    # Importance & Decay
    importance: float = DEFAULT_IMPORTANCE
    decay_rate: float = DEFAULT_DECAY_RATE

    # Emotional
    valence: EmotionalValence = EmotionalValence.NEUTRAL

    # Metadata
    tags: List[str] = field(default_factory=list)
    source_conversation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Embedding (populated later)
    embedding: Optional[List[float]] = None

    def touch(self):
        """Mark memory as accessed, reinforcing it."""
        self.last_accessed = datetime.now()
        self.access_count += 1
        self.importance = min(1.0, self.importance + 0.05)

    def calculate_strength(self) -> float:
        """
        Calculate current memory strength based on:
        - Base importance
        - Time decay
        - Access frequency
        """
        now = datetime.now()
        age_hours = (now - self.last_accessed).total_seconds() / 3600

        decay_factor = 0.5 ** (age_hours * self.decay_rate / 24)
        access_bonus = math.log1p(self.access_count) * 0.1

        strength = (self.importance * decay_factor) + access_bonus
        return min(1.0, max(0.0, strength))


@dataclass
class Episode:
    """
    An episodic memory - a specific experience or event.

    Episodes are conversation-level memories that capture
    "what happened" with temporal and emotional context.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Content
    summary: str = ""
    messages: List[MessageDict] = field(default_factory=list)

    # Temporal
    started_at: datetime = field(default_factory=datetime.now)
    ended_at: Optional[datetime] = None

    # Context
    topic: Optional[str] = None
    participants: List[str] = field(default_factory=lambda: ["user", "assistant"])

    # Emotional
    overall_valence: EmotionalValence = EmotionalValence.NEUTRAL
    emotional_moments: List[Dict[str, Any]] = field(default_factory=list)

    # Extracted facts (links to semantic memories)
    extracted_facts: List[str] = field(default_factory=list)

    # Importance
    importance: float = DEFAULT_IMPORTANCE

    # Source tracking
    source_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Tags for categorization
    tags: List[str] = field(default_factory=list)

    # Access tracking
    access_count: int = 0
    last_accessed: Optional[datetime] = None


@dataclass
class Fact:
    """
    A semantic memory - a fact, preference, or pattern.

    Facts are extracted from episodes and represent
    context-independent knowledge about the user/world.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Content
    content: str = ""
    fact_type: FactType = FactType.GENERAL

    # Confidence
    confidence: float = DEFAULT_CONFIDENCE

    # Source tracking
    source_episodes: List[str] = field(default_factory=list)
    first_learned: datetime = field(default_factory=datetime.now)
    last_confirmed: datetime = field(default_factory=datetime.now)
    confirmation_count: int = 1

    # Contradictions
    contradicted_by: List[str] = field(default_factory=list)
    is_superseded: bool = False

    # Importance
    importance: float = DEFAULT_IMPORTANCE

    metadata: Dict[str, Any] = field(default_factory=dict)

    # Tags for categorization
    tags: List[str] = field(default_factory=list)

    def confirm(self):
        """Mark fact as confirmed again, increasing confidence."""
        self.last_confirmed = datetime.now()
        self.confirmation_count += 1
        self.confidence = min(1.0, self.confidence + 0.1)


@dataclass
class WorkingMemory:
    """
    Working memory - the active context buffer.

    Limited capacity (like human 7±2 items).
    Contains currently relevant memories for the active conversation.
    """
    capacity: int = WORKING_MEMORY_CAPACITY
    items: List[Memory] = field(default_factory=list)
    current_conversation_id: Optional[str] = None

    def add(self, memory: Memory) -> Optional[Memory]:
        """
        Add item to working memory.
        Returns evicted item if at capacity.
        """
        evicted = None
        if len(self.items) >= self.capacity:
            # Evict lowest strength item
            self.items.sort(key=lambda m: m.calculate_strength())
            evicted = self.items.pop(0)

        memory.touch()
        self.items.append(memory)
        return evicted

    def clear(self):
        """Clear working memory (e.g., session end)."""
        self.items = []
        self.current_conversation_id = None

    def get_context(self) -> List[Memory]:
        """Get current working memory contents, sorted by strength."""
        return sorted(self.items, key=lambda m: m.calculate_strength(), reverse=True)


@dataclass
class ConversationMessage:
    """A single message in a conversation."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    role: str = "user"  # "user" or "assistant"
    content: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Conversation:
    """A full conversation to be processed."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: Optional[str] = None
    summary: Optional[str] = None
    messages: List[ConversationMessage] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
