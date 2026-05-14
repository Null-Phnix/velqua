"""
Memory formatting for LLM context injection.

Converts raw memories into text suitable for LLM prompts.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, Optional

from ..models import Episode, Fact, FactType


class FormatStyle(Enum):
    """Formatting styles for memory injection."""
    MARKDOWN = "markdown"     # Structured markdown
    NATURAL = "natural"       # Natural language paragraphs
    BULLET = "bullet"         # Bullet points
    MINIMAL = "minimal"       # Bare minimum info
    XML = "xml"               # XML-like tags
    JSON = "json"             # JSON format


@dataclass
class FormattedMemory:
    """A formatted memory ready for injection."""
    text: str
    token_estimate: int  # Rough token count (~4 chars per token)
    priority: float      # 0.0 to 1.0
    source_type: str     # "episode", "fact", etc.
    source_id: str


class MemoryFormatter:
    """
    Formats memories for LLM context injection.

    Different models and use cases benefit from different formats:
    - Markdown: Good for models trained on markdown (most modern LLMs)
    - Natural: Good for conversational contexts
    - Bullet: Concise, scannable
    - Minimal: Maximum info density
    - XML: Good for models that respect tags
    """

    def __init__(
        self,
        style: FormatStyle = FormatStyle.MARKDOWN,
        include_timestamps: bool = True,
        include_importance: bool = False,
        max_episode_length: int = 500,
        max_fact_length: int = 200,
    ):
        self.style = style
        self.include_timestamps = include_timestamps
        self.include_importance = include_importance
        self.max_episode_length = max_episode_length
        self.max_fact_length = max_fact_length

    def format_episode(self, episode: Episode) -> FormattedMemory:
        """Format a single episode."""
        if self.style == FormatStyle.MARKDOWN:
            text = self._format_episode_markdown(episode)
        elif self.style == FormatStyle.NATURAL:
            text = self._format_episode_natural(episode)
        elif self.style == FormatStyle.BULLET:
            text = self._format_episode_bullet(episode)
        elif self.style == FormatStyle.MINIMAL:
            text = self._format_episode_minimal(episode)
        elif self.style == FormatStyle.XML:
            text = self._format_episode_xml(episode)
        else:
            text = self._format_episode_markdown(episode)

        # Truncate if needed
        if len(text) > self.max_episode_length:
            text = text[:self.max_episode_length - 3] + "..."

        return FormattedMemory(
            text=text,
            token_estimate=len(text) // 4,
            priority=episode.importance,
            source_type="episode",
            source_id=episode.id,
        )

    def format_fact(self, fact: Fact) -> FormattedMemory:
        """Format a single fact."""
        if self.style == FormatStyle.MARKDOWN:
            text = self._format_fact_markdown(fact)
        elif self.style == FormatStyle.NATURAL:
            text = self._format_fact_natural(fact)
        elif self.style == FormatStyle.BULLET:
            text = self._format_fact_bullet(fact)
        elif self.style == FormatStyle.MINIMAL:
            text = self._format_fact_minimal(fact)
        elif self.style == FormatStyle.XML:
            text = self._format_fact_xml(fact)
        else:
            text = self._format_fact_markdown(fact)

        # Truncate if needed
        if len(text) > self.max_fact_length:
            text = text[:self.max_fact_length - 3] + "..."

        return FormattedMemory(
            text=text,
            token_estimate=len(text) // 4,
            priority=fact.importance * fact.confidence,
            source_type="fact",
            source_id=fact.id,
        )

    def format_context(
        self,
        episodes: List[Episode],
        facts: List[Fact],
        header: Optional[str] = None,
    ) -> str:
        """
        Format a complete context block.

        Combines episodes and facts into a single context string.
        """
        sections = []

        if header:
            sections.append(header)

        # Format facts first (more concise, user background)
        if facts:
            if self.style == FormatStyle.MARKDOWN:
                sections.append("## User Context")
            elif self.style == FormatStyle.XML:
                sections.append("<user_context>")

            fact_texts = []
            for fact in facts:
                formatted = self.format_fact(fact)
                fact_texts.append(formatted.text)

            sections.append("\n".join(fact_texts))

            if self.style == FormatStyle.XML:
                sections.append("</user_context>")

        # Format episodes (recent interactions)
        if episodes:
            if self.style == FormatStyle.MARKDOWN:
                sections.append("\n## Recent Interactions")
            elif self.style == FormatStyle.XML:
                sections.append("<recent_interactions>")

            for ep in episodes[:5]:  # Limit to 5 most recent
                formatted = self.format_episode(ep)
                sections.append(formatted.text)

            if self.style == FormatStyle.XML:
                sections.append("</recent_interactions>")

        return "\n\n".join(sections)

    # === Markdown formatting ===

    def _format_episode_markdown(self, episode: Episode) -> str:
        """Format episode as markdown."""
        lines = []

        # Title line
        title = episode.topic or "Conversation"
        if self.include_timestamps and episode.started_at:
            date_str = episode.started_at.strftime("%Y-%m-%d")
            lines.append(f"### {title} ({date_str})")
        else:
            lines.append(f"### {title}")

        # Summary
        if episode.summary:
            # Take first 2 sentences
            summary = episode.summary
            sentences = summary.split(". ")[:2]
            lines.append(". ".join(sentences) + ("." if not sentences[-1].endswith(".") else ""))

        return "\n".join(lines)

    def _format_fact_markdown(self, fact: Fact) -> str:
        """Format fact as markdown."""
        type_emoji = {
            FactType.PERSONAL: "👤",
            FactType.PREFERENCE: "❤️",
            FactType.PROFESSIONAL: "💼",
            FactType.PROJECT: "🔧",
            FactType.RELATIONSHIP: "👥",
            FactType.GENERAL: "📌",
        }.get(fact.fact_type, "📌")

        return f"- {type_emoji} {fact.content}"

    # === Natural formatting ===

    def _format_episode_natural(self, episode: Episode) -> str:
        """Format episode as natural language."""
        parts = []

        if episode.started_at:
            # Relative time description
            days_ago = (datetime.now() - episode.started_at).days
            if days_ago == 0:
                time_desc = "Today"
            elif days_ago == 1:
                time_desc = "Yesterday"
            elif days_ago < 7:
                time_desc = f"{days_ago} days ago"
            elif days_ago < 30:
                weeks = days_ago // 7
                time_desc = f"{weeks} week{'s' if weeks > 1 else ''} ago"
            else:
                time_desc = episode.started_at.strftime("%B %d")

            parts.append(time_desc)

        if episode.topic:
            parts.append(f"we discussed {episode.topic}")

        if episode.summary:
            # First sentence only
            first_sentence = episode.summary.split(". ")[0]
            parts.append(first_sentence)

        return ". ".join(parts) + "."

    def _format_fact_natural(self, fact: Fact) -> str:
        """Format fact as natural language."""
        # Add contextual prefix based on type
        if fact.fact_type == FactType.PERSONAL:
            return f"The user {fact.content.lower() if not fact.content[0].isupper() else fact.content}"
        elif fact.fact_type == FactType.PREFERENCE:
            return f"The user prefers {fact.content.lower()}"
        elif fact.fact_type == FactType.PROFESSIONAL:
            return f"Professionally, {fact.content.lower()}"
        else:
            return fact.content

    # === Bullet formatting ===

    def _format_episode_bullet(self, episode: Episode) -> str:
        """Format episode as bullet point."""
        date = ""
        if self.include_timestamps and episode.started_at:
            date = f"[{episode.started_at.strftime('%m/%d')}] "

        topic = episode.topic or "Conversation"
        summary = ""
        if episode.summary:
            summary = f": {episode.summary.split('.')[0]}"

        return f"• {date}{topic}{summary}"

    def _format_fact_bullet(self, fact: Fact) -> str:
        """Format fact as bullet point."""
        type_tag = f"[{fact.fact_type}] " if self.include_importance else ""
        return f"• {type_tag}{fact.content}"

    # === Minimal formatting ===

    def _format_episode_minimal(self, episode: Episode) -> str:
        """Format episode minimally."""
        return episode.summary[:200] if episode.summary else (episode.topic or "")

    def _format_fact_minimal(self, fact: Fact) -> str:
        """Format fact minimally."""
        return fact.content

    # === XML formatting ===

    def _format_episode_xml(self, episode: Episode) -> str:
        """Format episode as XML."""
        lines = ["<episode>"]

        if episode.topic:
            lines.append(f"  <topic>{episode.topic}</topic>")

        if self.include_timestamps and episode.started_at:
            lines.append(f"  <date>{episode.started_at.strftime('%Y-%m-%d')}</date>")

        if episode.summary:
            summary = episode.summary.split(".")[0] + "."
            lines.append(f"  <summary>{summary}</summary>")

        lines.append("</episode>")
        return "\n".join(lines)

    def _format_fact_xml(self, fact: Fact) -> str:
        """Format fact as XML."""
        return f'<fact type="{fact.fact_type}">{fact.content}</fact>'


def create_system_prompt_context(
    episodes: List[Episode],
    facts: List[Fact],
    max_tokens: int = 500,
) -> str:
    """
    Create a context block suitable for system prompts.

    Returns formatted text optimized for system prompt injection.
    """
    formatter = MemoryFormatter(style=FormatStyle.MARKDOWN)

    # Prioritize facts (user background)
    formatted_facts = [formatter.format_fact(f) for f in facts]
    formatted_facts.sort(key=lambda x: x.priority, reverse=True)

    # Then episodes
    formatted_episodes = [formatter.format_episode(e) for e in episodes]
    formatted_episodes.sort(key=lambda x: x.priority, reverse=True)

    # Build context within budget
    lines = ["## Memory Context\n"]
    token_count = 5  # Header

    # Add facts first
    if formatted_facts:
        lines.append("### About the User")
        token_count += 5

        for ff in formatted_facts:
            if token_count + ff.token_estimate > max_tokens:
                break
            lines.append(ff.text)
            token_count += ff.token_estimate

    # Add episodes
    if formatted_episodes and token_count < max_tokens - 50:
        lines.append("\n### Recent Context")
        token_count += 5

        for fe in formatted_episodes[:3]:
            if token_count + fe.token_estimate > max_tokens:
                break
            lines.append(fe.text)
            token_count += fe.token_estimate

    return "\n".join(lines)
