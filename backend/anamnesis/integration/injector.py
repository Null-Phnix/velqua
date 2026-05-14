"""
Memory injection for LLM prompts.

High-level API for injecting memories into LLM context.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from ..models import EmotionalValence, Episode, Fact, WorkingMemory
from ..quality.scorer import QualityScorer
from ..retrieval.query_expansion import QueryExpander
from ..stores.episodic import EpisodicStore
from ..stores.semantic import SemanticStore
from .context_manager import (
    ContextAllocation,
    ContextBudget,
    ContextManager,
    PriorityStrategy,
    get_preset,
)
from .formatter import FormatStyle, MemoryFormatter

logger = logging.getLogger(__name__)


class InjectionMode(Enum):
    """Where to inject memories in the prompt."""
    SYSTEM_PROMPT = "system"      # Inject into system prompt
    USER_CONTEXT = "user"         # Inject as user context before query
    ASSISTANT_PREFACE = "preface" # Inject as assistant recall
    SEPARATE = "separate"         # Return separate from prompt


@dataclass
class InjectionResult:
    """Result of memory injection."""
    injected_text: str
    mode: InjectionMode
    episodes_used: int
    facts_used: int
    total_tokens: int
    budget_used: float
    items_dropped: int


class MemoryInjector:
    """
    Injects relevant memories into LLM prompts.

    This is the main high-level API for using Anamnesis
    with any LLM system.

    Example usage:
        injector = MemoryInjector(episodic_store, semantic_store)

        # For system prompt injection
        result = injector.inject_for_query(
            query="Help me with my Python project",
            mode=InjectionMode.SYSTEM_PROMPT,
        )
        system_prompt = f"{base_system_prompt}\\n\\n{result.injected_text}"

        # For user context injection
        result = injector.inject_for_query(
            query=user_message,
            mode=InjectionMode.USER_CONTEXT,
        )
        enhanced_message = f"{result.injected_text}\\n\\nUser: {user_message}"
    """

    def __init__(
        self,
        episodic_store: EpisodicStore,
        semantic_store: SemanticStore,
        budget: Optional[Union[ContextBudget, str]] = None,
        format_style: FormatStyle = FormatStyle.MARKDOWN,
        priority_strategy: PriorityStrategy = PriorityStrategy.BALANCED,
        enable_query_expansion: bool = True,
    ):
        """
        Initialize memory injector.

        Args:
            episodic_store: Store for episodic memories
            semantic_store: Store for semantic facts
            budget: Context budget or preset name ("minimal", "standard", "extensive", "long_context")
            format_style: How to format memories
            priority_strategy: How to prioritize memories
            enable_query_expansion: Whether to use query expansion for better retrieval
        """
        self.episodic_store = episodic_store
        self.semantic_store = semantic_store

        # Resolve budget
        if isinstance(budget, str):
            self.budget = get_preset(budget)
        elif budget is None:
            self.budget = get_preset("standard")
        else:
            self.budget = budget

        self.format_style = format_style
        self.priority_strategy = priority_strategy
        self.enable_query_expansion = enable_query_expansion

        # Create formatter and context manager
        self.formatter = MemoryFormatter(style=format_style)
        self.context_manager = ContextManager(
            budget=self.budget,
            formatter=self.formatter,
            priority_strategy=priority_strategy,
        )

        # Create query expander and quality scorer
        if enable_query_expansion:
            self.query_expander = QueryExpander()
        else:
            self.query_expander = None
        self.quality_scorer = QualityScorer()

    def inject_for_query(
        self,
        query: str,
        mode: InjectionMode = InjectionMode.SYSTEM_PROMPT,
        max_episodes: int = 10,
        max_facts: int = 20,
        max_tokens: Optional[int] = None,
        include_recent: bool = True,
        recent_days: int = 7,
        include_relevant: bool = True,
        relevance_threshold: float = 0.3,
        emotion_filter: Optional[EmotionalValence] = None,
        emotion_boost: Optional[EmotionalValence] = None,
        use_query_expansion: bool = True,
    ) -> InjectionResult:
        """
        Inject memories relevant to a query.

        Args:
            query: The user's query/message
            mode: Where to inject memories
            max_episodes: Max episodes to consider
            max_facts: Max facts to consider
            max_tokens: Optional token budget override (uses injector's budget if None)
            include_recent: Include recent memories regardless of relevance
            recent_days: Days to consider for "recent"
            include_relevant: Include topic-relevant memories
            relevance_threshold: Minimum relevance score to include
            emotion_filter: Only include episodes with this valence
            emotion_boost: Boost relevance of episodes with this valence
            use_query_expansion: Whether to expand query for better retrieval

        Returns:
            InjectionResult with formatted memories
        """
        # Apply token budget override if specified
        original_budget = None
        if max_tokens is not None:
            original_budget = self.budget
            self.budget = ContextBudget(
                max_tokens=max_tokens,
                episode_ratio=self.budget.episode_ratio,
                fact_ratio=self.budget.fact_ratio,
                reserved_ratio=self.budget.reserved_ratio,
            )
            self.context_manager.budget = self.budget

        try:
            return self._do_inject_for_query(
                query=query,
                mode=mode,
                max_episodes=max_episodes,
                max_facts=max_facts,
                include_recent=include_recent,
                recent_days=recent_days,
                include_relevant=include_relevant,
                relevance_threshold=relevance_threshold,
                emotion_filter=emotion_filter,
                emotion_boost=emotion_boost,
                use_query_expansion=use_query_expansion,
            )
        finally:
            # Restore original budget
            if original_budget is not None:
                self.budget = original_budget
                self.context_manager.budget = original_budget

    def _do_inject_for_query(
        self,
        query: str,
        mode: InjectionMode,
        max_episodes: int,
        max_facts: int,
        include_recent: bool,
        recent_days: int,
        include_relevant: bool,
        relevance_threshold: float,
        emotion_filter: Optional[EmotionalValence],
        emotion_boost: Optional[EmotionalValence],
        use_query_expansion: bool,
    ) -> InjectionResult:
        """Internal implementation of inject_for_query."""
        # Expand query if enabled
        search_query = query
        expanded_terms = set()
        if use_query_expansion and self.query_expander and query:
            expansion = self.query_expander.expand(query)
            search_query = expansion.expanded_query
            expanded_terms = set(expansion.added_terms)

        # Gather candidate memories
        episodes = []
        facts = []
        query_relevance = {}

        # Get recent episodes
        if include_recent:
            recent = self.episodic_store.get_recent(days=recent_days, limit=max_episodes // 2)
            # If no recent episodes, try with longer window
            if not recent:
                recent = self.episodic_store.get_recent(days=90, limit=max_episodes // 2)
            for ep in recent:
                if ep not in episodes:
                    episodes.append(ep)
                    # Give recent memories a baseline relevance
                    query_relevance[ep.id] = 0.5

        # Get episodes by emotion if filtering
        if emotion_filter is not None:
            emotional = self.episodic_store.get_emotional(emotion_filter, limit=max_episodes)
            for ep in emotional:
                if ep not in episodes:
                    episodes.append(ep)
                    query_relevance[ep.id] = 0.6  # Emotional match gets decent relevance

        # Search for relevant episodes by topic (using expanded query)
        if include_relevant and search_query:
            relevant = self.episodic_store.search(search_query, limit=max_episodes)
            for ep in relevant:
                if ep not in episodes:
                    episodes.append(ep)
                # Search results get higher relevance
                query_relevance[ep.id] = max(query_relevance.get(ep.id, 0), 0.7)

        # If still no episodes, get most important ones
        if not episodes:
            important = self.episodic_store.get_important(threshold=0.3, limit=max_episodes // 2)
            for ep in important:
                if ep not in episodes:
                    episodes.append(ep)
                    query_relevance[ep.id] = ep.importance

        # Apply emotion filter if specified
        if emotion_filter is not None:
            episodes = [ep for ep in episodes if ep.overall_valence == emotion_filter]

        # Apply emotion boost if specified
        if emotion_boost is not None:
            for ep in episodes:
                if ep.overall_valence == emotion_boost:
                    query_relevance[ep.id] = min(1.0, query_relevance.get(ep.id, 0.5) + 0.2)

        # Get all facts (they're typically more compact)
        all_facts = self.semantic_store.list_all(limit=max_facts * 2)

        # Filter and score facts by query relevance (using expanded query)
        search_query_lower = search_query.lower() if search_query else ""
        for fact in all_facts:
            # Keyword relevance using expanded query
            content_lower = fact.content.lower()
            words = search_query_lower.split()
            matches = sum(1 for w in words if w in content_lower)
            relevance = matches / max(len(words), 1)

            # Boost for expanded term matches
            if expanded_terms:
                expanded_matches = sum(1 for t in expanded_terms if t.lower() in content_lower)
                if expanded_matches > 0:
                    relevance = min(1.0, relevance + expanded_matches * 0.1)

            if relevance >= relevance_threshold or not query:
                facts.append(fact)
                query_relevance[fact.id] = relevance

        # Limit to max
        episodes = episodes[:max_episodes]
        facts = facts[:max_facts]

        # Apply quality bonus to relevance scores (20% weight)
        for ep in episodes:
            quality = self.quality_scorer.score_episode(ep)
            base_relevance = query_relevance.get(ep.id, 0.5)
            query_relevance[ep.id] = 0.8 * base_relevance + 0.2 * quality.overall_score

        # Allocate context budget
        allocation = self.context_manager.allocate(
            episodes=episodes,
            facts=facts,
            query_relevance=query_relevance,
        )

        # Build context string based on mode
        text = self._build_injection_text(allocation, mode, query)

        # Record access for used memories (reinforces importance)
        self._record_access(episodes, facts)

        return InjectionResult(
            injected_text=text,
            mode=mode,
            episodes_used=len(allocation.episodes),
            facts_used=len(allocation.facts),
            total_tokens=allocation.total_tokens,
            budget_used=allocation.budget_used,
            items_dropped=allocation.items_dropped,
        )

    def inject_working_memory(
        self,
        working_memory: WorkingMemory,
        mode: InjectionMode = InjectionMode.SYSTEM_PROMPT,
    ) -> InjectionResult:
        """
        Inject from a WorkingMemory object.

        WorkingMemory is the in-session memory that tracks
        the current conversation context.
        """
        # Extract episodes and facts from working memory
        episodes = [item for item in working_memory.recent_context if isinstance(item, Episode)]
        facts = list(working_memory.active_facts)

        # Allocate context budget
        allocation = self.context_manager.allocate(
            episodes=episodes,
            facts=facts,
        )

        # Build context string
        text = self._build_injection_text(allocation, mode)

        return InjectionResult(
            injected_text=text,
            mode=mode,
            episodes_used=len(allocation.episodes),
            facts_used=len(allocation.facts),
            total_tokens=allocation.total_tokens,
            budget_used=allocation.budget_used,
            items_dropped=allocation.items_dropped,
        )

    def inject_specific(
        self,
        episodes: Optional[List[Episode]] = None,
        facts: Optional[List[Fact]] = None,
        mode: InjectionMode = InjectionMode.SYSTEM_PROMPT,
    ) -> InjectionResult:
        """
        Inject specific memories (bypass retrieval).

        Useful when you've already determined which
        memories to include.
        """
        episodes = episodes or []
        facts = facts or []

        allocation = self.context_manager.allocate(
            episodes=episodes,
            facts=facts,
        )

        text = self._build_injection_text(allocation, mode)

        return InjectionResult(
            injected_text=text,
            mode=mode,
            episodes_used=len(allocation.episodes),
            facts_used=len(allocation.facts),
            total_tokens=allocation.total_tokens,
            budget_used=allocation.budget_used,
            items_dropped=allocation.items_dropped,
        )

    def _build_injection_text(
        self,
        allocation: ContextAllocation,
        mode: InjectionMode,
        query: Optional[str] = None,
    ) -> str:
        """Build the injection text based on mode."""

        if mode == InjectionMode.SYSTEM_PROMPT:
            header = "# Memory Context\n\nYou have the following memories about this user:"
            return self.context_manager.build_context(
                allocation,
                header=header,
                include_metadata=False,
            )

        elif mode == InjectionMode.USER_CONTEXT:
            header = "[Relevant context from previous conversations]"
            return self.context_manager.build_context(
                allocation,
                header=header,
                include_metadata=False,
            )

        elif mode == InjectionMode.ASSISTANT_PREFACE:
            # Frame as the assistant recalling information
            lines = []

            if allocation.facts:
                lines.append("I recall the following about you:")
                for ff in allocation.facts:
                    lines.append(f"- {ff.text.lstrip('- ')}")
                lines.append("")

            if allocation.episodes:
                lines.append("From our previous conversations:")
                for fe in allocation.episodes:
                    lines.append(fe.text)
                lines.append("")

            return "\n".join(lines)

        else:  # SEPARATE
            return self.context_manager.build_context(
                allocation,
                include_metadata=True,
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get injector statistics."""
        return {
            "budget_preset": {
                "max_tokens": self.budget.max_tokens,
                "episode_ratio": self.budget.episode_ratio,
                "fact_ratio": self.budget.fact_ratio,
            },
            "format_style": self.format_style.value,
            "priority_strategy": self.priority_strategy.value,
            "episodic_count": self.episodic_store.count(),
            "semantic_count": self.semantic_store.count(),
        }

    def _record_access(self, episodes: List[Episode], facts: List[Fact]):
        """
        Record that memories were accessed during injection.

        This reinforces importance, making frequently-used memories
        more likely to surface in future queries.

        Args:
            episodes: List of Episode objects that were used
            facts: List of Fact objects that were used
        """
        # Touch episodes that were used
        for episode in episodes:
            try:
                self.episodic_store.touch(episode.id, reinforce=True)
            except (KeyError, ValueError, OSError) as e:
                logger.debug("Access tracking failed for episode %s: %s", episode.id, e)

        # Touch facts that were used
        for fact in facts:
            try:
                self.semantic_store.touch(fact.id, reinforce=True)
            except (KeyError, ValueError, OSError) as e:
                logger.debug("Access tracking failed for fact %s: %s", fact.id, e)


def create_injector_for_model(
    episodic_store: EpisodicStore,
    semantic_store: SemanticStore,
    model_context_window: int,
    target_memory_percentage: float = 0.15,
) -> MemoryInjector:
    """
    Create an injector configured for a specific model's context window.

    Args:
        episodic_store: Store for episodes
        semantic_store: Store for facts
        model_context_window: Model's total context window in tokens
        target_memory_percentage: What percentage of context to use for memories

    Returns:
        Configured MemoryInjector
    """
    # Calculate budget based on model context
    max_tokens = int(model_context_window * target_memory_percentage)

    # Clamp to reasonable range
    max_tokens = max(200, min(max_tokens, 8000))

    budget = ContextBudget(
        max_tokens=max_tokens,
        episode_ratio=0.4,
        fact_ratio=0.4,
        reserved_ratio=0.2,
    )

    return MemoryInjector(
        episodic_store=episodic_store,
        semantic_store=semantic_store,
        budget=budget,
    )


# Convenience functions for common use cases

def quick_inject(
    episodic_store: EpisodicStore,
    semantic_store: SemanticStore,
    query: str,
    max_tokens: int = 500,
) -> str:
    """
    Quick one-liner for memory injection.

    Returns formatted memory context string.
    """
    injector = MemoryInjector(
        episodic_store=episodic_store,
        semantic_store=semantic_store,
        budget=ContextBudget(max_tokens=max_tokens),
    )

    result = injector.inject_for_query(query)
    return result.injected_text
