"""
Conversation chunking module.

Intelligently splits long conversations into
topic-based episodes for better retrieval.
"""

from .chunker import (
    ChunkBoundary,
    ChunkResult,
    ConversationChunker,
    chunk_conversation,
)

__all__ = [
    "ConversationChunker",
    "ChunkResult",
    "ChunkBoundary",
    "chunk_conversation",
]
