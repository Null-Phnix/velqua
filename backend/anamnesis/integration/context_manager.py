"""
Context budget management for LLM memory injection.

Handles the challenge of fitting relevant memories into
limited context windows.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

from ..models import Episode, Fact
from .formatter import FormatStyle, FormattedMemory, MemoryFormatter

# Import accurate token counting
try:
    from ..context.manager import get_token_counter
    HAS_TOKEN_COUNTER = True
except ImportError:
    HAS_TOKEN_COUNTER = False


class PriorityStrategy(Enum):
    """Strategy for prioritizing memories when budget is limited."""
    IMPORTANCE = "importance"      # By importance score
    RECENCY = "recency"            # Most recent first
    RELEVANCE = "relevance"        # By query relevance
    BALANCED = "balanced"          # Mix of all factors


@dataclass
class ContextBudget:
    """Budget configuration for context injection."""
    max_tokens: int = 1000        # Total token budget
    episode_ratio: float = 0.4    # Fraction for episodes
    fact_ratio: float = 0.4       # Fraction for facts
    reserved_ratio: float = 0.2   # Reserved for headers/formatting
    min_per_item: int = 20        # Minimum tokens per item


@dataclass
class ContextAllocation:
    """Allocation result showing what fits in budget."""
    episodes: List[FormattedMemory]
    facts: List[FormattedMemory]
    total_tokens: int
    budget_used: float  # 0.0 to 1.0
    items_dropped: int  # How many items didn't fit


class ContextManager:
    """
    Manages context budget for memory injection.

    Ensures memories fit within LLM context limits while
    maximizing relevance and usefulness.
    """

    def __init__(
        self,
        budget: Optional[ContextBudget] = None,
        formatter: Optional[MemoryFormatter] = None,
        priority_strategy: PriorityStrategy = PriorityStrategy.BALANCED,
    ):
        self.budget = budget or ContextBudget()
        self.formatter = formatter or MemoryFormatter(style=FormatStyle.MARKDOWN)
        self.priority_strategy = priority_strategy

    def allocate(
        self,
        episodes: List[Episode],
        facts: List[Fact],
        query_relevance: Optional[Dict[str, float]] = None,
    ) -> ContextAllocation:
        """
        Allocate context budget to memories.

        Args:
            episodes: Available episodes
            facts: Available facts
            query_relevance: Optional dict mapping memory IDs to relevance scores

        Returns:
            ContextAllocation with selected memories
        """
        # Format all memories
        formatted_episodes = [self.formatter.format_episode(e) for e in episodes]
        formatted_facts = [self.formatter.format_fact(f) for f in facts]

        # Apply query relevance if provided
        if query_relevance:
            for fe in formatted_episodes:
                if fe.source_id in query_relevance:
                    fe.priority = (fe.priority + query_relevance[fe.source_id]) / 2
            for ff in formatted_facts:
                if ff.source_id in query_relevance:
                    ff.priority = (ff.priority + query_relevance[ff.source_id]) / 2

        # Calculate budgets
        available = int(self.budget.max_tokens * (1 - self.budget.reserved_ratio))
        episode_budget = int(available * self.budget.episode_ratio)
        fact_budget = int(available * self.budget.fact_ratio)

        # Sort by priority
        formatted_episodes = self._sort_by_priority(formatted_episodes, episodes)
        formatted_facts = self._sort_by_priority(formatted_facts, facts)

        # Select within budget
        selected_episodes = self._select_within_budget(formatted_episodes, episode_budget)
        selected_facts = self._select_within_budget(formatted_facts, fact_budget)

        # Calculate stats
        total_tokens = sum(fe.token_estimate for fe in selected_episodes)
        total_tokens += sum(ff.token_estimate for ff in selected_facts)

        items_dropped = (
            len(formatted_episodes) - len(selected_episodes) +
            len(formatted_facts) - len(selected_facts)
        )

        return ContextAllocation(
            episodes=selected_episodes,
            facts=selected_facts,
            total_tokens=total_tokens,
            budget_used=total_tokens / self.budget.max_tokens,
            items_dropped=items_dropped,
        )

    def _sort_by_priority(
        self,
        formatted: List[FormattedMemory],
        originals: List,
    ) -> List[FormattedMemory]:
        """Sort memories by priority based on strategy."""
        if self.priority_strategy == PriorityStrategy.IMPORTANCE:
            return sorted(formatted, key=lambda x: x.priority, reverse=True)

        elif self.priority_strategy == PriorityStrategy.RECENCY:
            # Need to sort by timestamp
            id_to_time = {}
            for orig in originals:
                if hasattr(orig, 'started_at') and orig.started_at:
                    id_to_time[orig.id] = orig.started_at.timestamp()
                elif hasattr(orig, 'first_learned') and orig.first_learned:
                    id_to_time[orig.id] = orig.first_learned.timestamp()
                else:
                    id_to_time[orig.id] = 0

            return sorted(
                formatted,
                key=lambda x: id_to_time.get(x.source_id, 0),
                reverse=True,
            )

        elif self.priority_strategy == PriorityStrategy.RELEVANCE:
            # Already sorted by relevance if query_relevance provided
            return sorted(formatted, key=lambda x: x.priority, reverse=True)

        else:  # BALANCED
            # Combine importance and recency
            id_to_time = {}
            for orig in originals:
                if hasattr(orig, 'started_at') and orig.started_at:
                    id_to_time[orig.id] = orig.started_at.timestamp()
                elif hasattr(orig, 'first_learned') and orig.first_learned:
                    id_to_time[orig.id] = orig.first_learned.timestamp()
                else:
                    id_to_time[orig.id] = 0

            # Normalize timestamps
            times = list(id_to_time.values())
            if times:
                min_time = min(times)
                max_time = max(times)
                time_range = max_time - min_time if max_time > min_time else 1

                for fm in formatted:
                    t = id_to_time.get(fm.source_id, min_time)
                    recency = (t - min_time) / time_range
                    # Combined score: 60% importance, 40% recency
                    fm.priority = 0.6 * fm.priority + 0.4 * recency

            return sorted(formatted, key=lambda x: x.priority, reverse=True)

    def _select_within_budget(
        self,
        formatted: List[FormattedMemory],
        budget: int,
    ) -> List[FormattedMemory]:
        """Select memories that fit within token budget."""
        selected = []
        used = 0

        for fm in formatted:
            if fm.token_estimate < self.budget.min_per_item:
                continue

            if used + fm.token_estimate <= budget:
                selected.append(fm)
                used += fm.token_estimate

        return selected

    def build_context(
        self,
        allocation: ContextAllocation,
        header: Optional[str] = None,
        include_metadata: bool = False,
    ) -> str:
        """
        Build the final context string from allocation.

        Args:
            allocation: The context allocation
            header: Optional header text
            include_metadata: Include debug metadata

        Returns:
            Formatted context string
        """
        lines = []

        if header:
            lines.append(header)
            lines.append("")

        # Facts section
        if allocation.facts:
            if self.formatter.style == FormatStyle.MARKDOWN:
                lines.append("## User Background")
            elif self.formatter.style == FormatStyle.XML:
                lines.append("<user_background>")

            for ff in allocation.facts:
                lines.append(ff.text)

            if self.formatter.style == FormatStyle.XML:
                lines.append("</user_background>")
            lines.append("")

        # Episodes section
        if allocation.episodes:
            if self.formatter.style == FormatStyle.MARKDOWN:
                lines.append("## Recent Context")
            elif self.formatter.style == FormatStyle.XML:
                lines.append("<recent_context>")

            for fe in allocation.episodes:
                lines.append(fe.text)
                lines.append("")

            if self.formatter.style == FormatStyle.XML:
                lines.append("</recent_context>")

        # Metadata footer
        if include_metadata:
            lines.append("")
            lines.append(f"<!-- Tokens: {allocation.total_tokens}, Budget: {allocation.budget_used:.0%} -->")

        return "\n".join(lines)

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.

        Uses tiktoken if available, otherwise ~4 characters per token.
        """
        if HAS_TOKEN_COUNTER and hasattr(self, '_token_counter'):
            return self._token_counter.count(text)
        return len(text) // 4

    def set_model(self, model: str):
        """
        Set the model for accurate token counting.

        Args:
            model: Model name (e.g., 'gpt-4', 'claude-2')
        """
        if HAS_TOKEN_COUNTER:
            self._token_counter = get_token_counter(model)


# Preset configurations
PRESETS = {
    "minimal": ContextBudget(
        max_tokens=500,
        episode_ratio=0.3,
        fact_ratio=0.5,
    ),
    "standard": ContextBudget(
        max_tokens=1000,
        episode_ratio=0.4,
        fact_ratio=0.4,
    ),
    "extensive": ContextBudget(
        max_tokens=2000,
        episode_ratio=0.5,
        fact_ratio=0.3,
    ),
    "long_context": ContextBudget(
        max_tokens=4000,
        episode_ratio=0.5,
        fact_ratio=0.3,
    ),
}


def get_preset(name: str) -> ContextBudget:
    """Get a preset context budget configuration."""
    return PRESETS.get(name, PRESETS["standard"])
