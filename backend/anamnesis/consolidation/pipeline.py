"""
Main consolidation pipeline.

Orchestrates the conversion of raw conversations into memories:
1. Summarize conversation → Episode summary
2. Extract facts → Semantic memories
3. Detect emotions → Emotional valence
4. Store results → Memory stores
"""

import logging
import uuid
from dataclasses import dataclass
from typing import List, Optional

from ..models import Conversation, EmotionalValence, Episode, Fact
from ..stores.episodic import EpisodicStore
from ..stores.semantic import SemanticStore
from .contradiction import ContradictionDetector
from .extractor import EmotionExtractor, ExtractionResult, FactExtractor, extract_all
from .summarizer import HeuristicSummarizer, Summarizer, SummaryResult

logger = logging.getLogger(__name__)


@dataclass
class ConsolidationResult:
    """Result of consolidating a conversation."""
    episode: Optional[Episode]
    facts: List[Fact]
    summary: SummaryResult
    extraction: ExtractionResult
    success: bool
    error: Optional[str] = None


class ConsolidationPipeline:
    """
    Pipeline for converting raw conversations into memories.

    Steps:
    1. Summarization - Create episode summary
    2. Fact extraction - Identify facts, preferences, patterns
    3. Emotion detection - Assess emotional content
    4. Storage - Save to memory stores
    """

    def __init__(
        self,
        episodic_store: Optional[EpisodicStore] = None,
        semantic_store: Optional[SemanticStore] = None,
        summarizer: Optional[Summarizer] = None,
    ):
        """
        Initialize pipeline.

        Args:
            episodic_store: Store for episodes (optional, won't save if None)
            semantic_store: Store for facts (optional, won't save if None)
            summarizer: Summarizer to use (defaults to HeuristicSummarizer)
        """
        self.episodic_store = episodic_store
        self.semantic_store = semantic_store
        self.summarizer = summarizer or HeuristicSummarizer()

        # Initialize extractors and detectors
        self.fact_extractor = FactExtractor()
        self.emotion_extractor = EmotionExtractor()
        self.contradiction_detector = ContradictionDetector()

    def consolidate(
        self,
        conversation: Conversation,
        min_messages: int = 2,
        importance_threshold: float = 0.3,
    ) -> ConsolidationResult:
        """
        Consolidate a single conversation into memories.

        Args:
            conversation: The conversation to process
            min_messages: Minimum messages required to process
            importance_threshold: Minimum importance to create episode

        Returns:
            ConsolidationResult with episode, facts, and metadata
        """
        try:
            messages = [
                {"role": m.role, "content": m.content}
                for m in conversation.messages
            ]

            # Skip if too few messages
            if len(messages) < min_messages:
                return ConsolidationResult(
                    episode=None,
                    facts=[],
                    summary=SummaryResult("", [], [], 0, 0.0),
                    extraction=ExtractionResult(),
                    success=True,
                    error="Too few messages",
                )

            # Step 1: Summarize
            if conversation.summary:
                # Use existing summary if available
                summary = SummaryResult(
                    summary=conversation.summary,
                    key_points=[],
                    topics=[conversation.name] if conversation.name else [],
                    word_count=len(conversation.summary.split()),
                    compression_ratio=0.0,
                )
            else:
                summary = self.summarizer.summarize(messages)

            # Step 2: Extract facts and emotions
            extraction = extract_all(messages)

            # Step 3: Calculate importance
            importance = self._calculate_importance(
                message_count=len(messages),
                has_facts=len(extraction.facts) > 0,
                has_emotions=len(extraction.emotions) > 0,
                emotional_intensity=max((e.intensity for e in extraction.emotions), default=0),
            )

            # Step 4: Create episode if important enough
            episode = None
            if importance >= importance_threshold:
                valence = EmotionalValence(extraction.overall_valence)

                episode = Episode(
                    id=str(uuid.uuid4()),
                    summary=summary.summary,
                    messages=messages,
                    topic=conversation.name or (summary.topics[0] if summary.topics else None),
                    started_at=conversation.created_at,
                    ended_at=conversation.updated_at,
                    overall_valence=valence,
                    importance=importance,
                    source_id=conversation.id,
                    extracted_facts=[],  # Will populate after saving facts
                    metadata={
                        "topics": summary.topics,
                        "key_points": summary.key_points,
                        "compression_ratio": summary.compression_ratio,
                    }
                )

                # Save episode if store available
                if self.episodic_store:
                    self.episodic_store.save(episode)

            # Step 5: Create and save facts (with contradiction detection)
            facts = []
            existing_facts = (
                self.semantic_store.list_all(limit=500) if self.semantic_store else []
            )

            for extracted in extraction.facts:
                fact = Fact(
                    id=str(uuid.uuid4()),
                    content=extracted.content,
                    fact_type=extracted.fact_type,
                    confidence=extracted.confidence,
                    source_episodes=[episode.id] if episode else [],
                    importance=importance * 0.8,
                    metadata={
                        "source_text": extracted.source_text,
                        "context": extracted.context,
                    }
                )

                # Check for contradictions against existing facts
                if existing_facts:
                    contradictions = self.contradiction_detector.find_contradictions(
                        fact, existing_facts, threshold=0.5,
                    )
                    for contradiction in contradictions:
                        old_fact = contradiction.existing_fact
                        if contradiction.confidence >= 0.8 and self.semantic_store:
                            # High confidence: auto-supersede old fact
                            self.semantic_store.supersede(old_fact.id, fact.id)
                            fact.metadata["superseded"] = old_fact.id
                            logger.info(
                                "Superseded fact %s with %s (%s)",
                                old_fact.id, fact.id, contradiction.explanation,
                            )
                        elif self.semantic_store:
                            # Lower confidence: mark for review
                            fact.metadata["contradicts"] = old_fact.id
                            fact.metadata["contradiction_type"] = contradiction.contradiction_type
                            fact.metadata["contradiction_confidence"] = contradiction.confidence

                facts.append(fact)

                # Save fact if store available
                if self.semantic_store:
                    self.semantic_store.add_fact(
                        content=fact.content,
                        fact_type=fact.fact_type,
                        source_episode_id=episode.id if episode else None,
                        confidence=fact.confidence,
                        importance=fact.importance,
                        metadata=fact.metadata,
                    )

            # Update episode with fact IDs
            if episode and facts:
                episode.extracted_facts = [f.id for f in facts]
                if self.episodic_store:
                    self.episodic_store.save(episode)

            return ConsolidationResult(
                episode=episode,
                facts=facts,
                summary=summary,
                extraction=extraction,
                success=True,
            )

        except Exception as e:
            logger.error("Consolidation failed for conversation %s: %s", conversation.id, e)
            return ConsolidationResult(
                episode=None,
                facts=[],
                summary=SummaryResult("", [], [], 0, 0.0),
                extraction=ExtractionResult(),
                success=False,
                error=str(e),
            )

    def consolidate_batch(
        self,
        conversations: List[Conversation],
        min_messages: int = 2,
        importance_threshold: float = 0.3,
        progress_callback: Optional[callable] = None,
    ) -> List[ConsolidationResult]:
        """
        Consolidate multiple conversations.

        Args:
            conversations: List of conversations to process
            min_messages: Minimum messages required
            importance_threshold: Minimum importance for episodes
            progress_callback: Optional callback(current, total)

        Returns:
            List of ConsolidationResult
        """
        results = []
        total = len(conversations)

        for i, convo in enumerate(conversations):
            result = self.consolidate(
                convo,
                min_messages=min_messages,
                importance_threshold=importance_threshold,
            )
            results.append(result)

            if progress_callback:
                progress_callback(i + 1, total)

        return results

    def _calculate_importance(
        self,
        message_count: int,
        has_facts: bool,
        has_emotions: bool,
        emotional_intensity: float,
    ) -> float:
        """
        Calculate importance score for a conversation.

        Factors:
        - Message count (longer = more important, up to a point)
        - Contains extractable facts
        - Contains emotional content
        - Emotional intensity
        """
        importance = 0.3  # Base importance

        # Message count bonus (logarithmic)
        import math
        message_bonus = min(0.2, math.log1p(message_count) * 0.05)
        importance += message_bonus

        # Fact bonus
        if has_facts:
            importance += 0.2

        # Emotion bonus
        if has_emotions:
            importance += 0.1
            importance += emotional_intensity * 0.2

        return min(1.0, importance)

    def reprocess_conversation(
        self,
        conversation_id: str,
        conversation: Conversation,
    ) -> ConsolidationResult:
        """
        Re-consolidate a conversation (e.g., with better summarizer).

        Removes old memories and creates new ones.
        """
        # Remove old episode if exists
        if self.episodic_store:
            old_episodes = self.episodic_store.search(conversation_id, limit=5)
            for ep in old_episodes:
                if ep.source_id == conversation_id:
                    self.episodic_store.delete(ep.id)

        # Re-consolidate
        return self.consolidate(conversation)


def create_default_pipeline(
    db_path: Optional[str] = None,
) -> ConsolidationPipeline:
    """Create a pipeline with default stores."""
    from ..stores.sqlite_backend import SQLiteBackend

    backend = SQLiteBackend(db_path)
    episodic = EpisodicStore(backend)
    semantic = SemanticStore(backend)

    return ConsolidationPipeline(
        episodic_store=episodic,
        semantic_store=semantic,
    )
