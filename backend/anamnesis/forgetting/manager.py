"""
Forgetting manager for memory maintenance.

Handles:
- Periodic strength recalculation
- Garbage collection of weak memories
- Memory consolidation (merging similar weak memories)
- Statistics and monitoring
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

from ..models import EmotionalValence, Episode, Fact
from ..stores.episodic import EpisodicStore
from ..stores.semantic import SemanticStore
from .decay import AdaptiveDecay, DecayFunction, MemoryStrengthFactors

logger = logging.getLogger(__name__)


@dataclass
class ForgettingStats:
    """Statistics from a forgetting cycle."""
    memories_checked: int
    memories_decayed: int
    memories_garbage_collected: int
    memories_compressed: int
    facts_checked: int
    facts_garbage_collected: int
    duration_seconds: float


@dataclass
class MemoryHealth:
    """Health status of a memory."""
    memory_id: str
    current_strength: float
    days_until_forgotten: Optional[float]
    is_at_risk: bool
    recommendation: str


class ForgettingManager:
    """
    Manages memory forgetting and maintenance.

    Periodically runs to:
    1. Recalculate memory strengths
    2. Garbage collect memories below threshold
    3. Compress fading memories (keep gist, lose details)
    4. Generate health reports
    """

    def __init__(
        self,
        episodic_store: Optional[EpisodicStore] = None,
        semantic_store: Optional[SemanticStore] = None,
        decay_function: Optional[DecayFunction] = None,
        gc_threshold: float = 0.05,  # Garbage collect below this
        compress_threshold: float = 0.2,  # Compress below this
        protect_threshold: float = 0.7,  # Never touch above this
    ):
        self.episodic_store = episodic_store
        self.semantic_store = semantic_store
        self.decay_function = decay_function or AdaptiveDecay()
        self.gc_threshold = gc_threshold
        self.compress_threshold = compress_threshold
        self.protect_threshold = protect_threshold

    def run_forgetting_cycle(
        self,
        dry_run: bool = False,
    ) -> ForgettingStats:
        """
        Run a complete forgetting cycle.

        Args:
            dry_run: If True, don't actually modify anything

        Returns:
            ForgettingStats with cycle results
        """
        start_time = datetime.now()

        stats = {
            "memories_checked": 0,
            "memories_decayed": 0,
            "memories_garbage_collected": 0,
            "memories_compressed": 0,
            "facts_checked": 0,
            "facts_garbage_collected": 0,
        }

        # Process episodes
        if self.episodic_store:
            episodes = self.episodic_store.list_all(limit=1000)
            for episode in episodes:
                stats["memories_checked"] += 1
                result = self._process_episode(episode, dry_run)
                if result == "decayed":
                    stats["memories_decayed"] += 1
                elif result == "gc":
                    stats["memories_garbage_collected"] += 1
                elif result == "compressed":
                    stats["memories_compressed"] += 1

        # Process facts
        if self.semantic_store:
            facts = self.semantic_store.list_all(limit=1000)
            for fact in facts:
                stats["facts_checked"] += 1
                if self._process_fact(fact, dry_run) == "gc":
                    stats["facts_garbage_collected"] += 1

        duration = (datetime.now() - start_time).total_seconds()

        return ForgettingStats(
            memories_checked=stats["memories_checked"],
            memories_decayed=stats["memories_decayed"],
            memories_garbage_collected=stats["memories_garbage_collected"],
            memories_compressed=stats["memories_compressed"],
            facts_checked=stats["facts_checked"],
            facts_garbage_collected=stats["facts_garbage_collected"],
            duration_seconds=duration,
        )

    def _process_episode(self, episode: Episode, dry_run: bool) -> str:
        """Process a single episode for forgetting."""
        # Calculate current strength
        factors = self._get_episode_factors(episode)
        strength = self.decay_function.calculate_strength(factors)

        # Protected - do nothing
        if strength >= self.protect_threshold:
            return "protected"

        # Garbage collect
        if strength < self.gc_threshold:
            if not dry_run and self.episodic_store:
                self.episodic_store.delete(episode.id)
            return "gc"

        # Compress
        if strength < self.compress_threshold:
            if not dry_run and self.episodic_store:
                self._compress_episode(episode)
            return "compressed"

        # Just update importance to reflect decay
        if not dry_run and self.episodic_store:
            episode.importance = strength
            self.episodic_store.save(episode)

        return "decayed"

    def _process_fact(self, fact: Fact, dry_run: bool) -> str:
        """Process a single fact for forgetting."""
        # Facts use simpler decay - based on confirmation
        age_hours = (datetime.now() - fact.last_confirmed).total_seconds() / 3600

        # High confidence facts are protected
        if fact.confidence >= 0.9:
            return "protected"

        # Calculate effective strength
        strength = fact.confidence * (1 - age_hours / (24 * 30))  # Decay over 30 days

        if strength < self.gc_threshold:
            if not dry_run and self.semantic_store:
                self.semantic_store.delete(fact.id)
            return "gc"

        return "kept"

    def _get_episode_factors(self, episode: Episode) -> MemoryStrengthFactors:
        """Extract factors from an episode."""
        now = datetime.now()

        # Calculate age
        if episode.last_accessed:
            age_hours = (now - episode.last_accessed).total_seconds() / 3600
        elif episode.started_at:
            age_hours = (now - episode.started_at).total_seconds() / 3600
        else:
            age_hours = 0

        # Access count
        access_count = episode.access_count

        # Emotional intensity
        valence = episode.overall_valence
        if isinstance(valence, EmotionalValence):
            emotional_intensity = abs(valence.value) / 2  # 0-1 scale
        else:
            emotional_intensity = abs(valence) / 2 if isinstance(valence, int) else 0

        # Reinforcement count
        reinforcement_count = episode.metadata.get("reinforcement_count", 0)

        return MemoryStrengthFactors(
            base_importance=episode.importance,
            age_hours=age_hours,
            access_count=access_count,
            emotional_intensity=emotional_intensity,
            reinforcement_count=reinforcement_count,
        )

    def _compress_episode(self, episode: Episode):
        """
        Compress an episode by removing details, keeping gist.

        - Keep summary
        - Reduce message count
        - Keep key emotional moments
        """
        # Keep only summary and key points
        compressed_messages = []

        # Keep first and last message for context
        if episode.messages:
            if len(episode.messages) > 0:
                compressed_messages.append(episode.messages[0])
            if len(episode.messages) > 1:
                compressed_messages.append(episode.messages[-1])

        episode.messages = compressed_messages
        episode.metadata["compressed"] = True
        episode.metadata["original_message_count"] = len(episode.messages)

        if self.episodic_store:
            self.episodic_store.save(episode)

    def get_memory_health(self, episode: Episode) -> MemoryHealth:
        """Get health status of a specific memory."""
        factors = self._get_episode_factors(episode)
        strength = self.decay_function.calculate_strength(factors)
        time_until = self.decay_function.time_until_threshold(factors, self.gc_threshold)

        days_until = time_until / 24 if time_until else None

        is_at_risk = strength < self.compress_threshold

        if strength >= self.protect_threshold:
            recommendation = "Healthy - no action needed"
        elif strength >= self.compress_threshold:
            recommendation = "Aging - will be compressed if not accessed"
        elif strength >= self.gc_threshold:
            recommendation = "At risk - access to reinforce or will be forgotten"
        else:
            recommendation = "Forgotten - will be garbage collected"

        return MemoryHealth(
            memory_id=episode.id,
            current_strength=strength,
            days_until_forgotten=days_until,
            is_at_risk=is_at_risk,
            recommendation=recommendation,
        )

    def get_system_health(self) -> Dict[str, Any]:
        """Get overall memory system health."""
        health = {
            "total_episodes": 0,
            "healthy": 0,
            "aging": 0,
            "at_risk": 0,
            "forgotten": 0,
            "total_facts": 0,
            "high_confidence_facts": 0,
            "low_confidence_facts": 0,
        }

        if self.episodic_store:
            episodes = self.episodic_store.list_all(limit=1000)
            health["total_episodes"] = len(episodes)

            for ep in episodes:
                status = self.get_memory_health(ep)
                if status.current_strength >= self.protect_threshold:
                    health["healthy"] += 1
                elif status.current_strength >= self.compress_threshold:
                    health["aging"] += 1
                elif status.current_strength >= self.gc_threshold:
                    health["at_risk"] += 1
                else:
                    health["forgotten"] += 1

        if self.semantic_store:
            facts = self.semantic_store.list_all(limit=1000)
            health["total_facts"] = len(facts)

            for fact in facts:
                if fact.confidence >= 0.8:
                    health["high_confidence_facts"] += 1
                else:
                    health["low_confidence_facts"] += 1

        return health

    def manually_protect(self, episode_id: str):
        """Manually protect a memory from forgetting."""
        if not self.episodic_store:
            return

        episode = self.episodic_store.get(episode_id)
        if not episode:
            return

        episode.importance = 1.0
        episode.metadata["protected"] = True
        self.episodic_store.save(episode)
