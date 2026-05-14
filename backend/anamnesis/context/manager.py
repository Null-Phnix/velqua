"""
Context window management for LLM integration.

Handles token counting, priority selection, and smart packing
of memories into context windows.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..models import Episode, Fact


@dataclass
class ContextBudget:
    """Token budget configuration."""
    max_tokens: int
    reserved_tokens: int = 500  # Reserve for system prompt, etc.
    episode_budget_ratio: float = 0.6  # 60% for episodes
    fact_budget_ratio: float = 0.4  # 40% for facts

    @property
    def available_tokens(self) -> int:
        return self.max_tokens - self.reserved_tokens

    @property
    def episode_budget(self) -> int:
        return int(self.available_tokens * self.episode_budget_ratio)

    @property
    def fact_budget(self) -> int:
        return int(self.available_tokens * self.fact_budget_ratio)


@dataclass
class PackingResult:
    """Result of context packing."""
    episodes: List[Episode]
    facts: List[Fact]
    total_tokens: int
    episode_tokens: int
    fact_tokens: int
    truncated_episodes: int
    truncated_facts: int
    dropped_episodes: int
    dropped_facts: int


class TokenCounter(ABC):
    """Abstract base for token counters."""

    @abstractmethod
    def count(self, text: str) -> int:
        """Count tokens in text."""
        pass

    @abstractmethod
    def count_messages(self, messages: List[Dict[str, Any]]) -> int:
        """Count tokens in message list."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return model name."""
        pass


class SimpleTokenCounter(TokenCounter):
    """
    Simple token counter based on word/character estimation.

    Fallback when tiktoken is not available.
    Uses ~4 characters per token as a rough estimate.
    """

    CHARS_PER_TOKEN = 4

    def __init__(self, model: str = "default"):
        self._model = model

    def count(self, text: str) -> int:
        """Estimate tokens from character count."""
        return len(text) // self.CHARS_PER_TOKEN + 1

    def count_messages(self, messages: List[Dict[str, Any]]) -> int:
        """Count tokens in messages."""
        total = 0
        for msg in messages:
            # Role overhead
            total += 4  # Rough estimate for message structure

            content = msg.get("content", "")
            if isinstance(content, str):
                total += self.count(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and "text" in item:
                        total += self.count(item["text"])

        return total

    @property
    def model_name(self) -> str:
        return f"simple-{self._model}"


class TiktokenCounter(TokenCounter):
    """
    Token counter using tiktoken (OpenAI's tokenizer).

    Requires: pip install tiktoken
    """

    def __init__(self, model: str = "gpt-4"):
        self._model = model
        self._encoding = None

    def _load_encoding(self):
        if self._encoding is not None:
            return

        try:
            import tiktoken
            try:
                self._encoding = tiktoken.encoding_for_model(self._model)
            except KeyError:
                # Fall back to cl100k_base for unknown models
                self._encoding = tiktoken.get_encoding("cl100k_base")
        except ImportError:
            raise RuntimeError(
                "tiktoken required. Install with: pip install tiktoken"
            )

    def count(self, text: str) -> int:
        self._load_encoding()
        return len(self._encoding.encode(text))

    def count_messages(self, messages: List[Dict[str, Any]]) -> int:
        self._load_encoding()

        # Based on OpenAI's token counting guide
        tokens_per_message = 3  # Every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = 1  # If there's a name, the role is omitted

        total = 0
        for msg in messages:
            total += tokens_per_message
            for key, value in msg.items():
                if isinstance(value, str):
                    total += self.count(value)
                if key == "name":
                    total += tokens_per_name

        total += 3  # Every reply is primed with <|start|>assistant<|message|>
        return total

    @property
    def model_name(self) -> str:
        return self._model


def get_token_counter(model: str = "gpt-4", use_tiktoken: bool = True) -> TokenCounter:
    """
    Get appropriate token counter for model.

    Args:
        model: Model name (gpt-4, gpt-3.5-turbo, claude, llama, etc.)
        use_tiktoken: Use tiktoken if available

    Returns:
        TokenCounter instance
    """
    if use_tiktoken:
        try:
            return TiktokenCounter(model)
        except RuntimeError:
            pass  # Fall through to simple counter

    return SimpleTokenCounter(model)


class ContextManager:
    """
    Manages context window packing for LLM integration.

    Handles:
    - Token counting for different models
    - Priority-based memory selection
    - Smart truncation of content
    - Optimal packing within budget
    """

    def __init__(
        self,
        token_counter: Optional[TokenCounter] = None,
        model: str = "gpt-4",
    ):
        """
        Initialize context manager.

        Args:
            token_counter: Custom token counter (auto-created if None)
            model: Model name for auto-created counter
        """
        self.counter = token_counter or get_token_counter(model)

    def pack_context(
        self,
        episodes: List[Episode],
        facts: List[Fact],
        budget: ContextBudget,
    ) -> PackingResult:
        """
        Pack memories into context budget.

        Episodes and facts are prioritized by importance.
        Content is truncated if necessary to fit budget.

        Args:
            episodes: Available episodes
            facts: Available facts
            budget: Token budget configuration

        Returns:
            PackingResult with selected and truncated content
        """
        # Sort by importance (highest first)
        sorted_episodes = sorted(episodes, key=lambda e: e.importance, reverse=True)
        sorted_facts = sorted(facts, key=lambda f: f.importance, reverse=True)

        # Pack episodes
        selected_episodes = []
        episode_tokens = 0
        truncated_eps = 0
        dropped_eps = 0

        for ep in sorted_episodes:
            ep_tokens = self._count_episode_tokens(ep)

            if episode_tokens + ep_tokens <= budget.episode_budget:
                selected_episodes.append(ep)
                episode_tokens += ep_tokens
            else:
                # Try truncating
                truncated = self._truncate_episode(
                    ep, budget.episode_budget - episode_tokens
                )
                if truncated:
                    selected_episodes.append(truncated)
                    episode_tokens += self._count_episode_tokens(truncated)
                    truncated_eps += 1
                else:
                    dropped_eps += 1

        # Pack facts
        selected_facts = []
        fact_tokens = 0
        truncated_facts_count = 0
        dropped_facts = 0

        for fact in sorted_facts:
            fact_tok = self._count_fact_tokens(fact)

            if fact_tokens + fact_tok <= budget.fact_budget:
                selected_facts.append(fact)
                fact_tokens += fact_tok
            else:
                # Try truncating fact content
                truncated = self._truncate_fact(
                    fact, budget.fact_budget - fact_tokens
                )
                if truncated:
                    selected_facts.append(truncated)
                    fact_tokens += self._count_fact_tokens(truncated)
                    truncated_facts_count += 1
                else:
                    dropped_facts += 1

        return PackingResult(
            episodes=selected_episodes,
            facts=selected_facts,
            total_tokens=episode_tokens + fact_tokens,
            episode_tokens=episode_tokens,
            fact_tokens=fact_tokens,
            truncated_episodes=truncated_eps,
            truncated_facts=truncated_facts_count,
            dropped_episodes=dropped_eps,
            dropped_facts=dropped_facts,
        )

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return self.counter.count(text)

    def count_episode_tokens(self, episode: Episode) -> int:
        """Count tokens for an episode."""
        return self._count_episode_tokens(episode)

    def count_fact_tokens(self, fact: Fact) -> int:
        """Count tokens for a fact."""
        return self._count_fact_tokens(fact)

    def estimate_context_usage(
        self,
        episodes: List[Episode],
        facts: List[Fact],
    ) -> Dict[str, int]:
        """
        Estimate token usage for content.

        Returns:
            Dict with token counts
        """
        episode_tokens = sum(self._count_episode_tokens(e) for e in episodes)
        fact_tokens = sum(self._count_fact_tokens(f) for f in facts)

        return {
            "episodes": episode_tokens,
            "facts": fact_tokens,
            "total": episode_tokens + fact_tokens,
            "episode_count": len(episodes),
            "fact_count": len(facts),
        }

    def _count_episode_tokens(self, episode: Episode) -> int:
        """Count tokens for episode content."""
        tokens = 0

        # Topic and summary
        if episode.topic:
            tokens += self.counter.count(episode.topic)
        if episode.summary:
            tokens += self.counter.count(episode.summary)

        # Messages (if included)
        if episode.messages:
            tokens += self.counter.count_messages(episode.messages)

        return tokens

    def _count_fact_tokens(self, fact: Fact) -> int:
        """Count tokens for fact content."""
        tokens = self.counter.count(fact.content)
        tokens += self.counter.count(fact.fact_type)
        return tokens

    def _truncate_episode(
        self,
        episode: Episode,
        max_tokens: int,
    ) -> Optional[Episode]:
        """
        Truncate episode to fit token budget.

        Strategy: Remove messages, keep summary.
        """
        if max_tokens < 50:  # Minimum viable
            return None

        # Create truncated copy
        from copy import deepcopy
        truncated = deepcopy(episode)

        # Remove messages, keep only summary
        truncated.messages = []

        # Check if it fits now
        if self._count_episode_tokens(truncated) <= max_tokens:
            truncated.metadata["truncated"] = True
            truncated.metadata["original_message_count"] = len(episode.messages)
            return truncated

        # Still too big - truncate summary
        if truncated.summary:
            # Binary search for max summary length
            summary = truncated.summary
            max_len = len(summary)
            min_len = 0

            while max_len - min_len > 10:
                mid = (max_len + min_len) // 2
                truncated.summary = summary[:mid] + "..."

                if self._count_episode_tokens(truncated) <= max_tokens:
                    min_len = mid
                else:
                    max_len = mid

            truncated.summary = summary[:min_len] + "..."

            if self._count_episode_tokens(truncated) <= max_tokens:
                return truncated

        return None

    def _truncate_fact(
        self,
        fact: Fact,
        max_tokens: int,
    ) -> Optional[Fact]:
        """Truncate fact to fit token budget."""
        if max_tokens < 20:
            return None

        from copy import deepcopy
        truncated = deepcopy(fact)

        # Check if it already fits
        if self._count_fact_tokens(truncated) <= max_tokens:
            return truncated

        # Truncate content
        content = truncated.content
        max_len = len(content)
        min_len = 0

        while max_len - min_len > 10:
            mid = (max_len + min_len) // 2
            truncated.content = content[:mid] + "..."

            if self._count_fact_tokens(truncated) <= max_tokens:
                min_len = mid
            else:
                max_len = mid

        truncated.content = content[:min_len] + "..."
        truncated.metadata["truncated"] = True

        if self._count_fact_tokens(truncated) <= max_tokens:
            return truncated

        return None


# Model context limits (approximate)
MODEL_CONTEXT_LIMITS = {
    "gpt-4": 8192,
    "gpt-4-32k": 32768,
    "gpt-4-turbo": 128000,
    "gpt-3.5-turbo": 4096,
    "gpt-3.5-turbo-16k": 16384,
    "claude-2": 100000,
    "claude-3": 200000,
    "llama-2-7b": 4096,
    "llama-2-13b": 4096,
    "llama-2-70b": 4096,
    "mistral-7b": 8192,
}


def get_model_context_limit(model: str) -> int:
    """Get context limit for a model."""
    # Try exact match
    if model in MODEL_CONTEXT_LIMITS:
        return MODEL_CONTEXT_LIMITS[model]

    # Try partial match
    for key, limit in MODEL_CONTEXT_LIMITS.items():
        if key in model.lower():
            return limit

    # Default to conservative estimate
    return 4096
