"""
Sentiment and emotion analysis.

Provides rule-based and keyword-based sentiment analysis
without requiring external ML dependencies.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set

from ..models import EmotionalValence


class EmotionCategory(Enum):
    """Specific emotion categories beyond positive/negative."""
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    TRUST = "trust"
    ANTICIPATION = "anticipation"
    NEUTRAL = "neutral"


class EmotionalIntensity(Enum):
    """Intensity levels for emotions."""
    VERY_LOW = 1
    LOW = 2
    MODERATE = 3
    HIGH = 4
    VERY_HIGH = 5


@dataclass
class SentimentResult:
    """Result of sentiment analysis."""
    text: str
    valence: EmotionalValence
    primary_emotion: EmotionCategory
    secondary_emotions: List[EmotionCategory] = field(default_factory=list)
    intensity: EmotionalIntensity = EmotionalIntensity.MODERATE
    confidence: float = 0.8
    sentiment_score: float = 0.0  # -1 to 1
    keywords_found: List[str] = field(default_factory=list)

    @property
    def is_positive(self) -> bool:
        return self.valence == EmotionalValence.POSITIVE

    @property
    def is_negative(self) -> bool:
        return self.valence == EmotionalValence.NEGATIVE

    @property
    def is_neutral(self) -> bool:
        return self.valence == EmotionalValence.NEUTRAL


class SentimentAnalyzer:
    """
    Rule-based sentiment and emotion analyzer.

    Uses keyword matching and patterns to detect emotional
    content in text without requiring ML models.
    """

    # Emotion keyword dictionaries
    JOY_KEYWORDS = {
        'happy', 'glad', 'joyful', 'excited', 'thrilled', 'delighted',
        'pleased', 'cheerful', 'elated', 'wonderful', 'fantastic',
        'amazing', 'awesome', 'great', 'love', 'loving', 'loved',
        'enjoy', 'enjoying', 'enjoyed', 'fun', 'laugh', 'laughing',
        'smile', 'smiling', 'yay', 'woohoo', 'hooray', 'brilliant',
        'excellent', 'perfect', 'beautiful', 'grateful', 'thankful',
    }

    SADNESS_KEYWORDS = {
        'sad', 'unhappy', 'depressed', 'down', 'blue', 'miserable',
        'heartbroken', 'devastated', 'upset', 'disappointed', 'sorry',
        'regret', 'miss', 'missing', 'lonely', 'alone', 'grief',
        'mourning', 'cry', 'crying', 'tears', 'weep', 'melancholy',
        'gloomy', 'hopeless', 'despair', 'hurt', 'hurting', 'pain',
    }

    ANGER_KEYWORDS = {
        'angry', 'mad', 'furious', 'enraged', 'irritated', 'annoyed',
        'frustrated', 'pissed', 'hate', 'hatred', 'resent', 'resentful',
        'outraged', 'livid', 'hostile', 'bitter', 'rage', 'fury',
        'damn', 'hell', 'stupid', 'idiot', 'ridiculous', 'unfair',
    }

    FEAR_KEYWORDS = {
        'afraid', 'scared', 'frightened', 'terrified', 'anxious',
        'worried', 'nervous', 'panic', 'dread', 'horror', 'alarmed',
        'uneasy', 'apprehensive', 'fearful', 'phobia', 'terror',
        'scary', 'creepy', 'threatening', 'danger', 'dangerous',
    }

    SURPRISE_KEYWORDS = {
        'surprised', 'shocked', 'amazed', 'astonished', 'stunned',
        'unexpected', 'wow', 'omg', 'whoa', 'incredible', 'unbelievable',
        'remarkable', 'extraordinary', 'speechless', 'astounding',
    }

    TRUST_KEYWORDS = {
        'trust', 'believe', 'faith', 'confident', 'sure', 'certain',
        'reliable', 'dependable', 'honest', 'sincere', 'loyal',
        'faithful', 'trustworthy', 'genuine', 'authentic',
    }

    ANTICIPATION_KEYWORDS = {
        'excited', 'eager', 'looking forward', 'can\'t wait', 'anticipate',
        'expect', 'hope', 'hoping', 'waiting', 'upcoming', 'soon',
        'planning', 'plan', 'future', 'tomorrow', 'next',
    }

    # Intensity modifiers
    INTENSIFIERS = {
        'very', 'really', 'extremely', 'incredibly', 'absolutely',
        'totally', 'completely', 'utterly', 'so', 'super', 'truly',
    }

    DIMINISHERS = {
        'slightly', 'somewhat', 'a bit', 'a little', 'kind of',
        'sort of', 'mildly', 'barely', 'hardly',
    }

    # Negation words
    NEGATIONS = {
        'not', 'no', 'never', 'none', 'nobody', 'nothing', 'neither',
        'nowhere', 'hardly', 'barely', 'don\'t', 'doesn\'t', 'didn\'t',
        'won\'t', 'wouldn\'t', 'couldn\'t', 'shouldn\'t', 'isn\'t',
        'aren\'t', 'wasn\'t', 'weren\'t', 'haven\'t', 'hasn\'t',
    }

    def __init__(self, custom_keywords: Optional[Dict[str, Set[str]]] = None):
        """
        Initialize analyzer.

        Args:
            custom_keywords: Optional dict mapping emotion names to keyword sets
        """
        self.custom_keywords = custom_keywords or {}

        # Build combined keyword maps
        self.emotion_keywords = {
            EmotionCategory.JOY: self.JOY_KEYWORDS,
            EmotionCategory.SADNESS: self.SADNESS_KEYWORDS,
            EmotionCategory.ANGER: self.ANGER_KEYWORDS,
            EmotionCategory.FEAR: self.FEAR_KEYWORDS,
            EmotionCategory.SURPRISE: self.SURPRISE_KEYWORDS,
            EmotionCategory.TRUST: self.TRUST_KEYWORDS,
            EmotionCategory.ANTICIPATION: self.ANTICIPATION_KEYWORDS,
        }

        # Map emotions to valence
        self.emotion_valence = {
            EmotionCategory.JOY: EmotionalValence.POSITIVE,
            EmotionCategory.SADNESS: EmotionalValence.NEGATIVE,
            EmotionCategory.ANGER: EmotionalValence.NEGATIVE,
            EmotionCategory.FEAR: EmotionalValence.NEGATIVE,
            EmotionCategory.SURPRISE: EmotionalValence.NEUTRAL,
            EmotionCategory.TRUST: EmotionalValence.POSITIVE,
            EmotionCategory.ANTICIPATION: EmotionalValence.POSITIVE,
            EmotionCategory.NEUTRAL: EmotionalValence.NEUTRAL,
            EmotionCategory.DISGUST: EmotionalValence.NEGATIVE,
        }

    def analyze(self, text: str) -> SentimentResult:
        """
        Analyze sentiment and emotion in text.

        Args:
            text: Text to analyze

        Returns:
            SentimentResult with analysis
        """
        if not text:
            return SentimentResult(
                text="",
                valence=EmotionalValence.NEUTRAL,
                primary_emotion=EmotionCategory.NEUTRAL,
            )

        text_lower = text.lower()
        words = set(re.findall(r'\b\w+\b', text_lower))

        # Detect negation context
        has_negation = bool(words & self.NEGATIONS)

        # Score each emotion
        emotion_scores: Dict[EmotionCategory, float] = {}
        keywords_found = []

        for emotion, keywords in self.emotion_keywords.items():
            matches = words & keywords
            if matches:
                score = len(matches)

                # Check for intensifiers/diminishers near matches
                for match in matches:
                    pattern = rf'\b({"|".join(self.INTENSIFIERS)})\s+\w*{match}'
                    if re.search(pattern, text_lower):
                        score *= 1.5
                    pattern = rf'\b({"|".join(self.DIMINISHERS)})\s+\w*{match}'
                    if re.search(pattern, text_lower):
                        score *= 0.6

                emotion_scores[emotion] = score
                keywords_found.extend(matches)

        # Determine primary emotion
        if emotion_scores:
            primary = max(emotion_scores, key=emotion_scores.get)
            score = emotion_scores[primary]

            # Get secondary emotions (any with score > 0)
            secondary = [e for e, s in emotion_scores.items()
                        if e != primary and s > 0]

            # Handle negation - flips positive/negative
            if has_negation:
                if primary in [EmotionCategory.JOY, EmotionCategory.TRUST]:
                    primary = EmotionCategory.SADNESS
                elif primary in [EmotionCategory.SADNESS, EmotionCategory.ANGER]:
                    primary = EmotionCategory.NEUTRAL

            # Calculate sentiment score (-1 to 1)
            valence = self.emotion_valence.get(primary, EmotionalValence.NEUTRAL)
            if valence == EmotionalValence.POSITIVE:
                sentiment_score = min(score / 5, 1.0)
            elif valence == EmotionalValence.NEGATIVE:
                sentiment_score = -min(score / 5, 1.0)
            else:
                sentiment_score = 0.0

            if has_negation:
                sentiment_score *= -0.7  # Partial flip

            # Determine intensity
            if score >= 5:
                intensity = EmotionalIntensity.VERY_HIGH
            elif score >= 3:
                intensity = EmotionalIntensity.HIGH
            elif score >= 2:
                intensity = EmotionalIntensity.MODERATE
            elif score >= 1:
                intensity = EmotionalIntensity.LOW
            else:
                intensity = EmotionalIntensity.VERY_LOW

            return SentimentResult(
                text=text,
                valence=valence,
                primary_emotion=primary,
                secondary_emotions=secondary,
                intensity=intensity,
                confidence=min(0.5 + score * 0.1, 0.95),
                sentiment_score=sentiment_score,
                keywords_found=keywords_found,
            )

        # No emotions detected - neutral
        return SentimentResult(
            text=text,
            valence=EmotionalValence.NEUTRAL,
            primary_emotion=EmotionCategory.NEUTRAL,
            intensity=EmotionalIntensity.VERY_LOW,
            confidence=0.6,
            sentiment_score=0.0,
        )

    def analyze_conversation(
        self,
        messages: List[Dict[str, str]],
    ) -> SentimentResult:
        """
        Analyze overall sentiment of a conversation.

        Args:
            messages: List of message dicts with 'content'

        Returns:
            Aggregated sentiment result
        """
        if not messages:
            return SentimentResult(
                text="",
                valence=EmotionalValence.NEUTRAL,
                primary_emotion=EmotionCategory.NEUTRAL,
            )

        # Analyze each message
        results = [
            self.analyze(msg.get("content", ""))
            for msg in messages
            if msg.get("content")
        ]

        if not results:
            return SentimentResult(
                text="",
                valence=EmotionalValence.NEUTRAL,
                primary_emotion=EmotionCategory.NEUTRAL,
            )

        # Aggregate scores
        total_score = sum(r.sentiment_score for r in results)
        avg_score = total_score / len(results)

        # Count emotion occurrences
        emotion_counts: Dict[EmotionCategory, int] = {}
        all_keywords = []
        for r in results:
            emotion_counts[r.primary_emotion] = emotion_counts.get(r.primary_emotion, 0) + 1
            all_keywords.extend(r.keywords_found)

        # Most common emotion
        primary = max(emotion_counts, key=emotion_counts.get)
        secondary = [e for e, c in emotion_counts.items() if e != primary and c > 0]

        # Determine overall valence
        if avg_score > 0.2:
            valence = EmotionalValence.POSITIVE
        elif avg_score < -0.2:
            valence = EmotionalValence.NEGATIVE
        elif abs(avg_score) < 0.1:
            valence = EmotionalValence.NEUTRAL
        else:
            valence = EmotionalValence.NEUTRAL

        # Aggregate intensity
        intensities = [r.intensity.value for r in results]
        avg_intensity = sum(intensities) / len(intensities)
        if avg_intensity >= 4:
            intensity = EmotionalIntensity.VERY_HIGH
        elif avg_intensity >= 3:
            intensity = EmotionalIntensity.HIGH
        elif avg_intensity >= 2:
            intensity = EmotionalIntensity.MODERATE
        else:
            intensity = EmotionalIntensity.LOW

        # Combine text
        combined_text = " ".join(msg.get("content", "") for msg in messages)

        return SentimentResult(
            text=combined_text[:500],
            valence=valence,
            primary_emotion=primary,
            secondary_emotions=secondary,
            intensity=intensity,
            confidence=sum(r.confidence for r in results) / len(results),
            sentiment_score=avg_score,
            keywords_found=list(set(all_keywords)),
        )

    def get_valence(self, text: str) -> EmotionalValence:
        """Quick valence check."""
        return self.analyze(text).valence

    def is_positive(self, text: str) -> bool:
        """Check if text is positive."""
        return self.analyze(text).is_positive

    def is_negative(self, text: str) -> bool:
        """Check if text is negative."""
        return self.analyze(text).is_negative
