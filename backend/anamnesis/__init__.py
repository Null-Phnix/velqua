"""
Anamnesis - Memory Architecture for AI Systems

A human-like memory system with:
- Working Memory (active context)
- Episodic Memory (timestamped experiences)
- Semantic Memory (extracted facts/preferences)
- Consolidation (raw -> compressed memories)
- Forgetting (importance decay)
- Integration (LLM context injection)
- Agent API (high-level interface for LLM apps)
- Temporal Recall (time-aware memory retrieval)
"""

__version__ = "0.2.0"

# Agent API (high-level interface)
from .agent import (
    AgentMemory,
    RecallResult,
    Session,
    SessionState,
    UserProfile,
)
from .anamnesis import Anamnesis
from .models import (
    Conversation,
    ConversationMessage,
    EmotionalValence,
    Episode,
    Fact,
    FactType,
    Memory,
    MemoryType,
    WorkingMemory,
)

__all__ = [
    # Core
    "Anamnesis",
    "Memory",
    "Episode",
    "Fact",
    "WorkingMemory",
    "Conversation",
    "ConversationMessage",
    "EmotionalValence",
    "FactType",
    "MemoryType",
    # Agent API
    "AgentMemory",
    "Session",
    "SessionState",
    "RecallResult",
    "UserProfile",
]
