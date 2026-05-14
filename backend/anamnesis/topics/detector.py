"""
Topic detection for conversations and episodes.

Extracts key topics and categorizes content.
"""

import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional

from ..models import Conversation, Episode


@dataclass
class TopicResult:
    """Result of topic detection."""
    main_topic: str
    subtopics: List[str]
    keywords: List[str]
    category: str  # "technical", "creative", "personal", "general"
    confidence: float


class TopicDetector:
    """
    Detects topics from text content.

    Uses keyword extraction, pattern matching, and
    category classification to identify topics.
    """

    # Topic categories with keywords
    CATEGORIES = {
        "technical": [
            "code", "python", "javascript", "api", "database", "bug", "error",
            "function", "class", "programming", "debug", "deploy", "server",
            "algorithm", "data", "software", "development", "testing", "git",
            "docker", "kubernetes", "linux", "web", "app", "framework",
        ],
        "creative": [
            "story", "novel", "character", "plot", "chapter", "write", "writing",
            "fiction", "fantasy", "narrative", "scene", "dialogue", "draft",
            "manuscript", "outline", "worldbuilding", "lore", "magic", "hero",
            "villain", "romance", "adventure", "poetry", "creative",
        ],
        "personal": [
            "feel", "feeling", "stressed", "happy", "sad", "worried", "excited",
            "relationship", "friend", "family", "work", "career", "life",
            "health", "hobby", "goal", "plan", "decision", "advice", "help",
        ],
        "general": [
            "question", "explain", "how", "what", "why", "learn", "understand",
            "help", "information", "know", "think", "idea", "opinion",
        ],
    }

    # Stop words to ignore
    STOPWORDS = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "must", "can", "to", "of", "in", "for",
        "on", "with", "at", "by", "from", "as", "into", "through", "during",
        "before", "after", "above", "below", "between", "under", "again",
        "further", "then", "once", "here", "there", "when", "where", "why",
        "how", "all", "each", "few", "more", "most", "other", "some", "such",
        "no", "nor", "not", "only", "own", "same", "so", "than", "too",
        "very", "just", "also", "now", "and", "but", "or", "if", "because",
        "until", "while", "about", "against", "i", "me", "my", "we", "our",
        "you", "your", "he", "she", "it", "they", "them", "this", "that",
        "these", "those", "am", "up", "down", "out", "off", "over", "any",
    }

    def __init__(self):
        # Build reverse lookup for category keywords
        self._keyword_to_category = {}
        for category, keywords in self.CATEGORIES.items():
            for kw in keywords:
                self._keyword_to_category[kw] = category

    def detect(
        self,
        text: str,
        existing_topic: Optional[str] = None,
    ) -> TopicResult:
        """
        Detect topics from text.

        Args:
            text: The text to analyze
            existing_topic: Optional existing topic hint

        Returns:
            TopicResult with detected topics
        """
        # Normalize and tokenize
        words = self._tokenize(text)

        # Extract keywords (most frequent meaningful words)
        keywords = self._extract_keywords(words, limit=10)

        # Determine category
        category, category_confidence = self._detect_category(words)

        # Generate main topic
        if existing_topic:
            main_topic = existing_topic
        else:
            main_topic = self._generate_topic(keywords)

        # Find subtopics
        subtopics = self._find_subtopics(words, main_topic)

        return TopicResult(
            main_topic=main_topic,
            subtopics=subtopics[:5],
            keywords=keywords,
            category=category,
            confidence=category_confidence,
        )

    def detect_from_episode(self, episode: Episode) -> TopicResult:
        """Detect topics from an episode."""
        # Combine summary and messages
        text_parts = [episode.summary or ""]

        if episode.topic:
            text_parts.append(episode.topic)

        for msg in episode.messages[:20]:  # Limit messages
            content = msg.get("content", "")
            if content:
                text_parts.append(content[:500])  # Truncate long messages

        full_text = " ".join(text_parts)
        return self.detect(full_text, episode.topic)

    def detect_from_conversation(self, conversation: Conversation) -> TopicResult:
        """Detect topics from a conversation."""
        text_parts = [conversation.name or "", conversation.summary or ""]

        for msg in conversation.messages[:20]:
            text_parts.append(msg.content[:500])

        full_text = " ".join(text_parts)
        return self.detect(full_text, conversation.name)

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        # Convert to lowercase and extract words
        text = text.lower()
        words = re.findall(r'\b[a-z]+\b', text)
        return words

    def _extract_keywords(self, words: List[str], limit: int = 10) -> List[str]:
        """Extract top keywords from words."""
        # Filter stopwords and short words
        meaningful = [
            w for w in words
            if w not in self.STOPWORDS and len(w) > 2
        ]

        # Count frequencies
        counts = Counter(meaningful)

        # Return top keywords
        return [w for w, _ in counts.most_common(limit)]

    def _detect_category(self, words: List[str]) -> tuple:
        """Detect category based on keyword matches."""
        word_set = set(words)
        scores = {}

        for category, keywords in self.CATEGORIES.items():
            matches = word_set & set(keywords)
            scores[category] = len(matches)

        if not scores or max(scores.values()) == 0:
            return "general", 0.3

        best_category = max(scores, key=scores.get)
        max_score = scores[best_category]

        # Calculate confidence
        total_matches = sum(scores.values())
        confidence = max_score / max(total_matches, 1)

        return best_category, min(1.0, confidence + 0.3)

    def _generate_topic(self, keywords: List[str]) -> str:
        """Generate a topic string from keywords."""
        if not keywords:
            return "General Discussion"

        # Use top 2-3 keywords
        topic_words = keywords[:3]
        return " ".join(w.title() for w in topic_words)

    def _find_subtopics(self, words: List[str], main_topic: str) -> List[str]:
        """Find subtopics related to the main topic."""
        # Filter out main topic words
        main_words = set(main_topic.lower().split())

        # Find other meaningful words not in main topic
        other_words = [
            w for w in words
            if w not in self.STOPWORDS and w not in main_words and len(w) > 3
        ]

        counts = Counter(other_words)
        return [w.title() for w, _ in counts.most_common(5)]

    def categorize_multiple(self, texts: List[str]) -> Dict[str, List[int]]:
        """
        Categorize multiple texts and group by category.

        Args:
            texts: List of texts to categorize

        Returns:
            Dict mapping category to list of text indices
        """
        categorized = {cat: [] for cat in self.CATEGORIES}
        categorized["general"] = []

        for i, text in enumerate(texts):
            result = self.detect(text)
            categorized[result.category].append(i)

        return categorized


def detect_topics(text: str) -> TopicResult:
    """Convenience function to detect topics."""
    detector = TopicDetector()
    return detector.detect(text)
