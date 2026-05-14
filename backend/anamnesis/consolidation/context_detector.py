"""
Context detector for distinguishing fiction vs reality in conversations.

Problem: Regex fact extractors pull fictional character facts from novel-writing conversations.
Example: "Elite is a Firehawk from Riven" → extracts as user facts

Solution: Detect conversation context (fiction vs reality) and filter appropriately.
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Literal, Tuple

from .extractor import ExtractedFact
from ..models import FactType


# Fiction context markers - phrases that indicate discussing fictional content
FICTION_MARKERS = [
    # Explicit creative writing references
    "my character", "the protagonist", "the antagonist", "the villain",
    "in the story", "in my novel", "in the book", "in my draft",
    "working on my novel", "writing about", "writing a novel",
    "worldbuilding", "world-building", "magic system",
    "in chapter", "the plot", "my mc", "main character",
    "character development", "narrative", "story arc",
    "this scene", "the scene where", "character named",

    # Creative speculation
    "imagine a", "what if there was", "picture this",
    "let me describe", "here's the world",

    # Project names (Josii's known novels)
    "the talker inside", "alderwick", "riku", "levi baxter",
    "prestige npc", "red rising", "malice"
]

# Fantasy/fiction keywords - terms unlikely in reality facts
FANTASY_KEYWORDS = {
    # Magic/Powers
    'magic', 'mana', 'spell', 'wizard', 'sorcerer', 'necromancer',
    'summoner', 'anima', 'aura', 'celestial', 'flux', 'void',
    'enchantment', 'rune', 'ward', 'portal', 'teleport',

    # Creatures
    'dragon', 'phoenix', 'werewolf', 'vampire', 'demon', 'angel',
    'unicorn', 'griffin', 'basilisk', 'chimera', 'kraken',

    # Fantasy roles/titles
    'firehawk', 'tuner', 'hand', 'collector', 'seeker',
    'paladin', 'rogue', 'ranger', 'barbarian',

    # Places (Josii's worldbuilding)
    'riven', 'alderwick', 'codexia', 'ashenfell', 'bellhouse',

    # Fantasy concepts
    'quest', 'dungeon', 'realm', 'kingdom', 'guild',
    'artifact', 'prophecy', 'curse', 'blessing'
}

# Meta-fact patterns - extract facts about the user's creative work
META_PATTERNS = [
    # Novel/story metadata
    r"(?:i'm (?:working on|writing|drafting))\s+(?:a (?:novel|story|book))\s+(?:called|named|titled)\s+(['\"]?[A-Z][^'\",.]{2,40}['\"]?)",
    r"(?:my (?:novel|story|book))\s+(?:is called|titled|named)\s+(['\"]?[A-Z][^'\",.]{2,40}['\"]?)",

    # Character names
    r"(?:character (?:named|called))\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
    r"(?:protagonist (?:is|named))\s+([A-Z][a-z]+)",

    # Project details
    r"(?:setting is)\s+([^.!?,]{5,50})",
    r"(?:writing (?:a|an))\s+(fantasy|sci-fi|romance|thriller|mystery|horror)\s+(?:novel|book)",
]


@dataclass
class ContextResult:
    """Result of context detection."""
    context: Literal["fiction", "reality", "uncertain"]
    confidence: float  # 0.0 to 1.0
    fiction_markers: List[str]  # Which markers triggered fiction detection
    fantasy_keywords: List[str]  # Which keywords found in content


class ContextDetector:
    """
    Detect whether conversation is about fiction or reality.

    Uses:
    - Fiction markers in conversation
    - Fantasy keywords in extracted facts
    - Conversation title analysis
    """

    def detect_context(
        self,
        messages: List[Dict[str, str]],
        conversation_title: str = ""
    ) -> ContextResult:
        """
        Detect if conversation is about fiction or reality.

        Returns ContextResult with classification and evidence.
        """
        fiction_markers_found = []
        total_words = 0

        # Check messages for fiction markers
        for msg in messages:
            content = msg.get("content", "").lower()
            total_words += len(content.split())

            for marker in FICTION_MARKERS:
                if marker in content:
                    fiction_markers_found.append(marker)

        # Check conversation title
        if conversation_title:
            title_lower = conversation_title.lower()
            for marker in FICTION_MARKERS:
                if marker in title_lower:
                    fiction_markers_found.append(f"title:{marker}")

        # Calculate confidence
        if len(fiction_markers_found) >= 3:
            return ContextResult(
                context="fiction",
                confidence=0.9,
                fiction_markers=fiction_markers_found,
                fantasy_keywords=[]
            )
        elif len(fiction_markers_found) >= 1:
            return ContextResult(
                context="fiction",
                confidence=0.7,
                fiction_markers=fiction_markers_found,
                fantasy_keywords=[]
            )
        else:
            return ContextResult(
                context="reality",
                confidence=0.8,
                fiction_markers=[],
                fantasy_keywords=[]
            )

    def filter_fiction_facts(
        self,
        facts: List[ExtractedFact],
        messages: List[Dict[str, str]],
        conversation_title: str = ""
    ) -> Tuple[List[ExtractedFact], List[ExtractedFact]]:
        """
        Split facts into reality vs fiction based on context.

        Returns:
            (reality_facts, fiction_facts)
        """
        # Detect overall conversation context
        context_result = self.detect_context(messages, conversation_title)

        reality_facts = []
        fiction_facts = []

        # If conversation is clearly fiction, filter all facts
        if context_result.context == "fiction" and context_result.confidence > 0.8:
            # Check each fact for fantasy keywords
            for fact in facts:
                fantasy_keywords = self._detect_fantasy_keywords(fact.content)

                if fantasy_keywords:
                    # Fact contains fantasy terms → definitely fiction
                    fiction_facts.append(fact)
                else:
                    # No fantasy keywords, but in fiction context
                    # Could be meta-fact about the user's work
                    if self._is_meta_fact(fact.content):
                        reality_facts.append(fact)
                    else:
                        fiction_facts.append(fact)
        else:
            # Reality or uncertain context - check individual facts
            for fact in facts:
                fantasy_keywords = self._detect_fantasy_keywords(fact.content)

                if fantasy_keywords:
                    # Contains fantasy keywords → likely fiction
                    fiction_facts.append(fact)
                else:
                    reality_facts.append(fact)

        return reality_facts, fiction_facts

    def extract_meta_facts(
        self,
        messages: List[Dict[str, str]],
        fiction_facts: List[ExtractedFact]
    ) -> List[ExtractedFact]:
        """
        Extract meta-facts about user's creative work from fiction conversations.

        Examples:
        - "I'm writing a novel called The Talker" → project fact
        - "My protagonist is named Elite" → project detail fact
        """
        meta_facts = []

        for msg in messages:
            content = msg.get("content", "")
            role = msg.get("role", "")

            # Only extract from user messages
            if role not in ("user", "human"):
                continue

            # Try meta-patterns
            for pattern in META_PATTERNS:
                matches = re.finditer(pattern, content, re.IGNORECASE)

                for match in matches:
                    # Extract the matched detail
                    detail = match.group(1).strip()

                    # Determine fact type based on pattern
                    if any(word in pattern for word in ['novel', 'story', 'book']):
                        fact_type = FactType.PROJECT
                        fact_content = f"Working on project: {detail}"
                    elif 'character' in pattern or 'protagonist' in pattern:
                        fact_type = FactType.PROJECT
                        fact_content = f"Created character: {detail}"
                    elif 'setting' in pattern:
                        fact_type = FactType.PROJECT
                        fact_content = f"Story setting: {detail}"
                    elif any(genre in content.lower() for genre in ['fantasy', 'sci-fi', 'romance']):
                        genre = match.group(1)
                        fact_type = FactType.PROJECT
                        fact_content = f"Writing {genre} novel"
                    else:
                        continue

                    meta_fact = ExtractedFact(
                        content=fact_content,
                        fact_type=fact_type.value,
                        confidence=0.7,  # Meta-extraction is less reliable
                        source_text=content,
                        context="meta-fact from fiction context"
                    )
                    meta_facts.append(meta_fact)

        return meta_facts

    def _detect_fantasy_keywords(self, text: str) -> List[str]:
        """Find fantasy keywords in text."""
        text_lower = text.lower()
        found = []

        for keyword in FANTASY_KEYWORDS:
            # Match whole words only (avoid "hand" matching "handling")
            if re.search(rf'\b{keyword}\b', text_lower):
                found.append(keyword)

        return found

    def _is_meta_fact(self, text: str) -> bool:
        """
        Check if fact is meta-fact about creative work (vs fictional content).

        Meta-facts:
        - "Writing a novel called X"
        - "Working on project Y"
        - "Character development for Z"

        Not meta-facts:
        - "Elite is a Firehawk"
        - "Lives in Riven"
        - "Can cast fire spells"
        """
        text_lower = text.lower()

        # Meta-fact indicators
        meta_indicators = [
            'writing', 'working on', 'project', 'novel', 'story',
            'developing', 'creating', 'planning', 'drafting',
            'character named', 'protagonist is', 'book called'
        ]

        return any(indicator in text_lower for indicator in meta_indicators)
