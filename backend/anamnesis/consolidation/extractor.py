"""
Extractors for facts, emotions, and patterns from conversations.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from ..models import FactType


class EmotionCategory(Enum):
    """Categories of emotions."""
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    TRUST = "trust"
    ANTICIPATION = "anticipation"
    NEUTRAL = "neutral"


@dataclass
class ExtractedFact:
    """A fact extracted from conversation."""
    content: str
    fact_type: str  # "personal", "preference", "project", "relationship", "general"
    confidence: float
    source_text: str
    context: str = ""


@dataclass
class ExtractedEmotion:
    """An emotion detected in conversation."""
    category: EmotionCategory
    intensity: float  # 0.0 to 1.0
    valence: int  # -2 to +2
    source_text: str
    context: str = ""


@dataclass
class ExtractionResult:
    """Result of extraction process."""
    facts: List[ExtractedFact] = field(default_factory=list)
    emotions: List[ExtractedEmotion] = field(default_factory=list)
    overall_valence: int = 0  # -2 to +2
    dominant_emotion: Optional[EmotionCategory] = None


class FactExtractor:
    """
    Extract facts from conversations.

    Uses pattern matching and keyword detection to identify:
    - Personal information (name, age, location, occupation)
    - Preferences (likes, dislikes, favorites)
    - Projects and work
    - Relationships
    - General knowledge
    """

    # Patterns for different fact types
    # Note: Some patterns use (?-i:) to disable case-insensitivity for proper name matching
    PERSONAL_PATTERNS = [
        # Identity - names need case-sensitive matching for proper nouns
        (r"(?:my name is|i am|i'm)\s+(?-i:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", "name", True),
        (r"(?:i am|i'm)\s+(\d{1,3})\s*(?:years? old|yo)", "age", False),
        (r"(?:i live in|i'm from|i'm in|based in)\s+(?-i:[A-Z][a-zA-Z]+(?:[\s,]+[A-Z]?[a-zA-Z]+)*)", "location", True),
        (r"(?:i work (?:as|at)|i'm a|my job is)\s+(?:a\s+)?([^.!?,]+?)(?:\.|,|!|\?|$)", "occupation", False),
        # Background
        (r"(?:i have|i've got)\s+(\d+)\s+(kids?|children|sons?|daughters?|pets?|cats?|dogs?)", "family", False),
        (r"(?:i've been|i have been)\s+([^.]+?)\s+for\s+(\d+\s*(?:years?|months?|days?))", "experience", False),
    ]

    PREFERENCE_PATTERNS = [
        # Likes - limit to reasonable length to avoid grabbing full sentences
        (r"(?:i (?:really )?(?:like|love|enjoy|prefer))\s+([^.!?,]{3,50})(?:\.|,|!|\?|$)", "like", False),
        (r"(?:my favorite[s]? (?:is|are))\s+([^.!?,]{3,50})(?:\.|,|!|\?|$)", "favorite", False),
        (r"(?:i'm (?:a fan of|into|passionate about))\s+([^.!?,]{3,50})(?:\.|,|!|\?|$)", "favorite", False),
        # Dislikes
        (r"(?:i (?:don't|do not|never) (?:like|enjoy))\s+([^.!?,]{3,50})(?:\.|,|!|\?|$)", "dislike", False),
        (r"(?:i hate|i dislike|i can't stand)\s+([^.!?,]{3,50})(?:\.|,|!|\?|$)", "dislike", False),
    ]

    PROJECT_PATTERNS = [
        (r"(?:i'm (?:working on|building|creating|developing))\s+([^.!?,]{3,60})(?:\.|,|!|\?|$)", "current_project", False),
        (r"(?:my (?:project|app|site|book|novel|story) (?:is called|is))\s+([^.!?,]{3,40})(?:\.|,|!|\?|$)", "project_name", False),
        (r"(?:i'm trying to|i want to|my goal is to)\s+([^.!?,]{3,60})(?:\.|,|!|\?|$)", "goal", False),
    ]

    RELATIONSHIP_PATTERNS = [
        # Partner - capture name only (proper noun)
        (r"(?:my (?:wife|husband|partner|girlfriend|boyfriend|spouse)(?:'s name is|,?\s+))(?-i:[A-Z][a-z]+)", "partner", True),
        (r"(?:my (?:wife|husband|partner|girlfriend|boyfriend|spouse))\s+(?-i:[A-Z][a-z]+)", "partner", True),
        (r"(?:my (?:friend|best friend|colleague))\s+(?-i:[A-Z][a-z]+)", "friend", True),
        (r"(?:my (?:mom|dad|mother|father|sister|brother|parent)(?:'s name is|,?\s+))(?-i:[A-Z][a-z]+)", "family_member", True),
    ]

    def extract(self, messages: List[Dict[str, str]]) -> List[ExtractedFact]:
        """Extract facts from messages."""
        facts = []

        for msg in messages:
            content = msg.get("content", "")
            role = msg.get("role", "")

            # Only extract from user messages (contains personal info)
            if role not in ("user", "human"):
                continue

            # Try each pattern category
            facts.extend(self._extract_pattern_facts(content, self.PERSONAL_PATTERNS, FactType.PERSONAL))
            facts.extend(self._extract_pattern_facts(content, self.PREFERENCE_PATTERNS, FactType.PREFERENCE))
            facts.extend(self._extract_pattern_facts(content, self.PROJECT_PATTERNS, FactType.PROJECT))
            facts.extend(self._extract_pattern_facts(content, self.RELATIONSHIP_PATTERNS, FactType.RELATIONSHIP))

        # Deduplicate and filter low-quality extractions
        seen = set()
        unique_facts = []
        for fact in facts:
            # Skip very short or generic content
            if len(fact.content) < 8:
                continue
            # Skip if just the subtype with no real content
            if ':' in fact.content and len(fact.content.split(':', 1)[1].strip()) < 3:
                continue

            key = (fact.content.lower(), fact.fact_type)
            if key not in seen:
                seen.add(key)
                unique_facts.append(fact)

        return unique_facts

    def _extract_pattern_facts(
        self,
        text: str,
        patterns: List[Tuple],
        fact_type: str,
    ) -> List[ExtractedFact]:
        """Extract facts using regex patterns."""
        facts = []

        for pattern_tuple in patterns:
            # Handle both old (pattern, subtype) and new (pattern, subtype, is_proper_noun) formats
            if len(pattern_tuple) == 3:
                pattern, subtype, is_proper_noun = pattern_tuple
            else:
                pattern, subtype = pattern_tuple
                is_proper_noun = False

            # Use case-insensitive matching except for proper noun patterns
            # Note: (?-i:...) in pattern handles case sensitivity for specific parts
            flags = re.IGNORECASE

            try:
                matches = re.finditer(pattern, text, flags)
            except re.error:
                # If pattern has inline flags that conflict, try without global flag
                try:
                    matches = re.finditer(pattern, text)
                except re.error:
                    continue

            for match in matches:
                # Get the captured group(s) or full match for non-capturing patterns
                groups = match.groups()

                if groups and any(g for g in groups):
                    # Use captured groups
                    captured = ' '.join(g.strip() for g in groups if g)
                else:
                    # For patterns using (?-i:...) without explicit groups, extract the name part
                    full_match = match.group(0)
                    # Find the proper noun part (capitalized words at the end)
                    if is_proper_noun:
                        captured = self._extract_proper_noun(full_match)
                    else:
                        captured = full_match

                if not captured:
                    continue

                # Clean up the captured content
                captured = self._clean_extracted_content(captured, subtype)

                if not captured or len(captured) < 2:
                    continue

                # Build fact content
                content = f"{subtype}: {captured}"

                # Determine confidence based on pattern quality
                confidence = 0.8 if is_proper_noun else 0.7

                facts.append(ExtractedFact(
                    content=content,
                    fact_type=fact_type,
                    confidence=confidence,
                    source_text=match.group(0)[:100],
                    context=text[:200],
                ))

        return facts

    def _extract_proper_noun(self, text: str) -> str:
        """Extract proper noun (capitalized name) from text."""
        # Find capitalized words at the end of the text
        words = text.split()
        proper_nouns = []

        for word in reversed(words):
            # Check if word starts with capital (and isn't just "I")
            clean_word = word.strip('.,!?\'\"')
            if clean_word and clean_word[0].isupper() and clean_word != 'I' and len(clean_word) > 1:
                proper_nouns.insert(0, clean_word)
            elif proper_nouns:
                # Stop when we hit a non-capitalized word after finding names
                break

        return ' '.join(proper_nouns)

    def _clean_extracted_content(self, content: str, subtype: str) -> str:
        """Clean up extracted content."""
        # Remove leading articles
        content = re.sub(r'^(?:a|an|the)\s+', '', content, flags=re.IGNORECASE)

        # Remove trailing punctuation and whitespace
        content = content.strip().rstrip('.,!?;:')

        # Remove common filler words at boundaries
        content = re.sub(r'\s+(?:and|or|but|so|then)\s*$', '', content, flags=re.IGNORECASE)
        content = re.sub(r'^\s*(?:and|or|but|so|then)\s+', '', content, flags=re.IGNORECASE)

        # Normalize whitespace
        content = re.sub(r'\s+', ' ', content).strip()

        # For names, only keep proper nouns
        if subtype in ('name', 'partner', 'friend', 'family_member'):
            # Keep only capitalized words
            words = [w for w in content.split() if w[0].isupper()]
            content = ' '.join(words)

        return content


class EmotionExtractor:
    """
    Extract emotions from conversations.

    Uses keyword matching and patterns to detect emotional content.
    """

    # Emotion keywords with intensity
    EMOTION_KEYWORDS = {
        EmotionCategory.JOY: {
            "high": ["ecstatic", "thrilled", "overjoyed", "elated", "amazing", "wonderful"],
            "medium": ["happy", "glad", "pleased", "excited", "great", "good", "love"],
            "low": ["nice", "okay", "fine", "alright", "content"],
        },
        EmotionCategory.SADNESS: {
            "high": ["devastated", "heartbroken", "miserable", "depressed", "terrible"],
            "medium": ["sad", "unhappy", "disappointed", "upset", "down", "hurt"],
            "low": ["melancholy", "blue", "low", "not great"],
        },
        EmotionCategory.ANGER: {
            "high": ["furious", "enraged", "livid", "outraged", "hate"],
            "medium": ["angry", "mad", "frustrated", "annoyed", "irritated"],
            "low": ["bothered", "miffed", "displeased"],
        },
        EmotionCategory.FEAR: {
            "high": ["terrified", "petrified", "horrified", "panic"],
            "medium": ["scared", "afraid", "worried", "anxious", "nervous"],
            "low": ["concerned", "uneasy", "apprehensive"],
        },
        EmotionCategory.SURPRISE: {
            "high": ["shocked", "astonished", "stunned", "amazed"],
            "medium": ["surprised", "unexpected", "wow"],
            "low": ["interesting", "huh", "oh"],
        },
        EmotionCategory.TRUST: {
            "high": ["completely trust", "faith in", "believe in"],
            "medium": ["trust", "rely on", "confident"],
            "low": ["think", "hope", "expect"],
        },
        EmotionCategory.ANTICIPATION: {
            "high": ["can't wait", "so excited for", "counting down"],
            "medium": ["looking forward", "excited about", "eager"],
            "low": ["interested in", "curious about"],
        },
    }

    # Valence mapping
    VALENCE_MAP = {
        EmotionCategory.JOY: 2,
        EmotionCategory.TRUST: 1,
        EmotionCategory.ANTICIPATION: 1,
        EmotionCategory.SURPRISE: 0,
        EmotionCategory.SADNESS: -1,
        EmotionCategory.FEAR: -1,
        EmotionCategory.ANGER: -2,
        EmotionCategory.DISGUST: -2,
        EmotionCategory.NEUTRAL: 0,
    }

    def extract(self, messages: List[Dict[str, str]]) -> List[ExtractedEmotion]:
        """Extract emotions from messages."""
        emotions = []

        for msg in messages:
            content = msg.get("content", "").lower()
            role = msg.get("role", "")

            # Focus on user messages for emotional content
            if role not in ("user", "human"):
                continue

            # Check each emotion category
            for category, intensity_keywords in self.EMOTION_KEYWORDS.items():
                for intensity_level, keywords in intensity_keywords.items():
                    for keyword in keywords:
                        if keyword in content:
                            # Find the sentence containing the keyword
                            sentences = re.split(r'[.!?]+', content)
                            source = next(
                                (s.strip() for s in sentences if keyword in s.lower()),
                                content[:100]
                            )

                            intensity = {"high": 0.9, "medium": 0.6, "low": 0.3}[intensity_level]

                            emotions.append(ExtractedEmotion(
                                category=category,
                                intensity=intensity,
                                valence=self.VALENCE_MAP.get(category, 0),
                                source_text=source,
                                context=content[:200],
                            ))

        return emotions

    def get_overall_valence(self, emotions: List[ExtractedEmotion]) -> int:
        """Calculate overall emotional valence from extracted emotions."""
        if not emotions:
            return 0

        # Weight by intensity
        weighted_sum = sum(e.valence * e.intensity for e in emotions)
        total_intensity = sum(e.intensity for e in emotions)

        if total_intensity == 0:
            return 0

        avg_valence = weighted_sum / total_intensity

        # Map to -2 to +2 scale
        if avg_valence >= 1.5:
            return 2
        elif avg_valence >= 0.5:
            return 1
        elif avg_valence <= -1.5:
            return -2
        elif avg_valence <= -0.5:
            return -1
        else:
            return 0

    def get_dominant_emotion(self, emotions: List[ExtractedEmotion]) -> Optional[EmotionCategory]:
        """Find the dominant emotion."""
        if not emotions:
            return None

        # Count and weight by intensity
        emotion_scores = {}
        for e in emotions:
            if e.category not in emotion_scores:
                emotion_scores[e.category] = 0
            emotion_scores[e.category] += e.intensity

        if not emotion_scores:
            return None

        return max(emotion_scores.items(), key=lambda x: x[1])[0]


def extract_all(messages: List[Dict[str, str]]) -> ExtractionResult:
    """Extract all facts and emotions from messages."""
    fact_extractor = FactExtractor()
    emotion_extractor = EmotionExtractor()

    facts = fact_extractor.extract(messages)
    emotions = emotion_extractor.extract(messages)

    return ExtractionResult(
        facts=facts,
        emotions=emotions,
        overall_valence=emotion_extractor.get_overall_valence(emotions),
        dominant_emotion=emotion_extractor.get_dominant_emotion(emotions),
    )
