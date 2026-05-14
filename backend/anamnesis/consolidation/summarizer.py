"""
Summarization for memory consolidation.

Provides:
- HeuristicSummarizer: Fast rule-based summarization (no LLM needed)
- EnhancedSummarizer: Extractive + abstractive hybrid approach
"""

import math
import re
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


@dataclass
class SummaryResult:
    """Result of summarization."""
    summary: str
    key_points: List[str]
    topics: List[str]
    word_count: int
    compression_ratio: float
    entities: Dict[str, List[str]] = field(default_factory=dict)  # Entity type -> values


class Summarizer(ABC):
    """Abstract base class for summarizers."""

    @abstractmethod
    def summarize(
        self,
        messages: List[Dict[str, str]],
        max_length: int = 200,
    ) -> SummaryResult:
        """Summarize a list of messages."""
        pass

    @abstractmethod
    def extract_topics(self, messages: List[Dict[str, str]]) -> List[str]:
        """Extract main topics from messages."""
        pass


class HeuristicSummarizer(Summarizer):
    """
    Rule-based summarizer that doesn't require an LLM.

    Uses extraction and compression heuristics to create summaries.
    Fast and deterministic, but lower quality than LLM-based.
    """

    # Common filler words to skip
    FILLER_WORDS = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "must", "shall",
        "can", "to", "of", "in", "for", "on", "with", "at", "by",
        "from", "as", "into", "through", "during", "before", "after",
        "above", "below", "between", "under", "again", "further",
        "then", "once", "here", "there", "when", "where", "why",
        "how", "all", "each", "few", "more", "most", "other", "some",
        "such", "no", "nor", "not", "only", "own", "same", "so",
        "than", "too", "very", "just", "also", "now", "i", "you",
        "he", "she", "it", "we", "they", "me", "him", "her", "us",
        "them", "my", "your", "his", "its", "our", "their", "this",
        "that", "these", "those", "am", "and", "but", "or", "if",
        "because", "while", "although", "though", "even", "what",
    }

    # Topic indicator patterns
    TOPIC_PATTERNS = [
        r"(?:about|regarding|concerning|discussing)\s+(.+?)(?:\.|,|$)",
        r"(?:working on|building|creating|developing)\s+(.+?)(?:\.|,|$)",
        r"(?:question about|asked about|wondering about)\s+(.+?)(?:\.|,|$)",
        r"(?:help with|assist with|support for)\s+(.+?)(?:\.|,|$)",
    ]

    def summarize(
        self,
        messages: List[Dict[str, str]],
        max_length: int = 200,
    ) -> SummaryResult:
        """Create a heuristic summary of messages."""
        if not messages:
            return SummaryResult(
                summary="Empty conversation",
                key_points=[],
                topics=[],
                word_count=0,
                compression_ratio=0.0,
            )

        # Calculate original length
        original_text = " ".join(m.get("content", "") for m in messages)
        original_words = len(original_text.split())

        # Extract key sentences
        key_sentences = self._extract_key_sentences(messages)

        # Extract topics
        topics = self.extract_topics(messages)

        # Build summary
        summary_parts = []

        # Add topic context
        if topics:
            topic_str = ", ".join(topics[:3])
            summary_parts.append(f"Discussion about {topic_str}.")

        # Add key points
        for sentence in key_sentences[:3]:
            if len(" ".join(summary_parts)) + len(sentence) < max_length:
                summary_parts.append(sentence)

        summary = " ".join(summary_parts)

        # Truncate if needed
        if len(summary) > max_length:
            summary = summary[:max_length-3] + "..."

        summary_words = len(summary.split())

        return SummaryResult(
            summary=summary,
            key_points=key_sentences[:5],
            topics=topics,
            word_count=summary_words,
            compression_ratio=summary_words / original_words if original_words > 0 else 0,
        )

    def extract_topics(self, messages: List[Dict[str, str]]) -> List[str]:
        """Extract topics using keyword frequency and patterns."""
        all_text = " ".join(m.get("content", "") for m in messages).lower()

        topics = []

        # Pattern-based extraction
        for pattern in self.TOPIC_PATTERNS:
            matches = re.findall(pattern, all_text, re.IGNORECASE)
            for match in matches:
                topic = match.strip()[:50]  # Limit length
                if topic and topic not in topics:
                    topics.append(topic)

        # Keyword frequency (nouns tend to be topics)
        words = re.findall(r'\b[a-z]{4,}\b', all_text)
        word_freq = {}
        for word in words:
            if word not in self.FILLER_WORDS:
                word_freq[word] = word_freq.get(word, 0) + 1

        # Add top frequent words as potential topics
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        for word, count in sorted_words[:5]:
            if count >= 2 and word not in topics:
                topics.append(word)

        return topics[:10]

    def _extract_key_sentences(self, messages: List[Dict[str, str]]) -> List[str]:
        """Extract key sentences from messages."""
        key_sentences = []

        for msg in messages:
            content = msg.get("content", "")
            role = msg.get("role", "")

            # Split into sentences
            sentences = re.split(r'[.!?]+', content)

            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence or len(sentence) < 20:
                    continue

                # Score sentence importance
                score = self._score_sentence(sentence, role)

                if score > 0.5:
                    key_sentences.append((sentence, score))

        # Sort by score and return top sentences
        key_sentences.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in key_sentences[:10]]

    def _score_sentence(self, sentence: str, role: str) -> float:
        """Score a sentence's importance."""
        score = 0.5  # Base score

        lower = sentence.lower()

        # Boost for certain patterns
        importance_patterns = [
            (r'\b(important|key|main|crucial|essential)\b', 0.2),
            (r'\b(conclusion|summary|result|finding)\b', 0.2),
            (r'\b(decided|concluded|determined|found)\b', 0.15),
            (r'\b(should|must|need to|have to)\b', 0.1),
            (r'\b(because|therefore|thus|hence)\b', 0.1),
            (r'\?$', 0.1),  # Questions often important
        ]

        for pattern, boost in importance_patterns:
            if re.search(pattern, lower):
                score += boost

        # User messages slightly more important (contain intent)
        if role == "user" or role == "human":
            score += 0.1

        # Penalize very short or very long
        word_count = len(sentence.split())
        if word_count < 5:
            score -= 0.2
        elif word_count > 50:
            score -= 0.1

        return min(1.0, max(0.0, score))


class EnhancedSummarizer(Summarizer):
    """
    Enhanced extractive + abstractive summarizer.

    Combines:
    - TF-IDF-like sentence scoring for extraction
    - Position-based weighting (lead/tail bias)
    - Entity density scoring
    - Template-based abstractive sentence fusion
    - Compression patterns for conciseness
    """

    FILLER_WORDS = HeuristicSummarizer.FILLER_WORDS

    # Abstractive templates for different summary types
    SUMMARY_TEMPLATES = {
        "discussion": "Discussed {topics}. {key_point}",
        "question": "User asked about {topics}. {key_point}",
        "task": "Worked on {topics}. {key_point}",
        "learning": "Explored {topics}. {key_point}",
        "problem": "Addressed issue with {topics}. {key_point}",
        "default": "Conversation about {topics}. {key_point}",
    }

    # Sentence compression patterns (regex, replacement)
    COMPRESSION_PATTERNS = [
        (r'\bI think (that )?(.*)', r'\2'),  # Remove "I think"
        (r'\bbasically,?\s*', ''),  # Remove "basically"
        (r'\bso,?\s*', ''),  # Remove filler "so"
        (r'\bactually,?\s*', ''),  # Remove "actually"
        (r'\bjust\s+', ''),  # Remove filler "just"
        (r'\breally\s+', ''),  # Remove filler "really"
        (r'\bkind of\s+', ''),  # Remove "kind of"
        (r'\bsort of\s+', ''),  # Remove "sort of"
        (r'\byou know,?\s*', ''),  # Remove "you know"
        (r'\bI mean,?\s*', ''),  # Remove "I mean"
        (r'\blike,?\s+', ''),  # Remove filler "like"
    ]

    # Intent indicators
    INTENT_PATTERNS = {
        "question": [r'\?$', r'^(?:how|what|why|when|where|who|can|could|would|is|are|do|does)\b'],
        "task": [r'\b(?:build|create|implement|write|add|fix|update|change|modify)\b'],
        "learning": [r'\b(?:learn|understand|explain|study|explore|research)\b'],
        "problem": [r'\b(?:error|bug|issue|problem|fail|broken|wrong)\b'],
    }

    def __init__(self, use_entities: bool = True):
        """
        Initialize enhanced summarizer.

        Args:
            use_entities: Whether to use entity extraction for better summaries
        """
        self.use_entities = use_entities
        self._entity_extractor = None

    def _get_entity_extractor(self):
        """Lazy load entity extractor."""
        if self._entity_extractor is None and self.use_entities:
            try:
                from .entities import EntityExtractor
                self._entity_extractor = EntityExtractor()
            except ImportError:
                self.use_entities = False
        return self._entity_extractor

    def summarize(
        self,
        messages: List[Dict[str, str]],
        max_length: int = 200,
    ) -> SummaryResult:
        """
        Create an enhanced extractive + abstractive summary.

        Process:
        1. Extract all sentences with TF-IDF-like scoring
        2. Identify conversation intent/type
        3. Extract named entities
        4. Select top sentences (extractive)
        5. Apply compression patterns
        6. Use template-based fusion (abstractive)
        """
        if not messages:
            return SummaryResult(
                summary="Empty conversation",
                key_points=[],
                topics=[],
                word_count=0,
                compression_ratio=0.0,
                entities={},
            )

        # Collect all text
        all_text = " ".join(m.get("content", "") for m in messages)
        original_words = len(all_text.split())

        # Extract entities
        entities_by_type = {}
        if self.use_entities:
            extractor = self._get_entity_extractor()
            if extractor:
                entities_by_type = extractor.extract_from_messages(messages)
                # Convert enum keys to strings
                entities_by_type = {k.value: v for k, v in entities_by_type.items()}

        # Build term frequencies for TF-IDF-like scoring
        term_freq = self._build_term_frequencies(all_text)

        # Score and extract sentences
        scored_sentences = self._score_sentences_tfidf(messages, term_freq, entities_by_type)

        # Detect conversation intent
        intent = self._detect_intent(messages)

        # Extract topics
        topics = self._extract_enhanced_topics(messages, entities_by_type)

        # Select top sentences (extractive phase)
        top_sentences = [s for s, _ in sorted(scored_sentences, key=lambda x: x[1], reverse=True)[:5]]

        # Compress selected sentences
        compressed = [self._compress_sentence(s) for s in top_sentences]
        compressed = [s for s in compressed if s]  # Remove empty

        # Generate abstractive summary using templates
        summary = self._generate_abstractive_summary(
            intent,
            topics,
            compressed,
            entities_by_type,
            max_length,
        )

        # Extract key points (clean versions of top sentences)
        key_points = compressed[:5]

        summary_words = len(summary.split())

        return SummaryResult(
            summary=summary,
            key_points=key_points,
            topics=topics,
            word_count=summary_words,
            compression_ratio=summary_words / original_words if original_words > 0 else 0,
            entities=entities_by_type,
        )

    def extract_topics(self, messages: List[Dict[str, str]]) -> List[str]:
        """Extract topics using entity-aware method."""
        entities_by_type = {}
        if self.use_entities:
            extractor = self._get_entity_extractor()
            if extractor:
                entities_by_type = extractor.extract_from_messages(messages)
                entities_by_type = {k.value: v for k, v in entities_by_type.items()}

        return self._extract_enhanced_topics(messages, entities_by_type)

    def _build_term_frequencies(self, text: str) -> Dict[str, float]:
        """Build term frequencies with IDF-like weighting."""
        words = re.findall(r'\b[a-z]{3,}\b', text.lower())

        # Count frequencies
        freq = Counter(words)
        total = len(words)

        # Calculate TF-IDF-like scores
        # Higher score for less frequent (more unique) terms
        scores = {}
        for word, count in freq.items():
            if word not in self.FILLER_WORDS:
                tf = count / total
                # Inverse frequency boost for rare terms
                idf = math.log(total / (count + 1)) + 1
                scores[word] = tf * idf

        return scores

    def _score_sentences_tfidf(
        self,
        messages: List[Dict[str, str]],
        term_freq: Dict[str, float],
        entities: Dict[str, List[str]],
    ) -> List[Tuple[str, float]]:
        """Score sentences using TF-IDF-like weighting + position + entities."""
        scored = []
        all_sentences = []

        # Collect all sentences with positions
        for msg_idx, msg in enumerate(messages):
            content = msg.get("content", "")
            role = msg.get("role", "")
            sentences = re.split(r'(?<=[.!?])\s+', content)

            for sent_idx, sentence in enumerate(sentences):
                sentence = sentence.strip()
                if len(sentence) < 15:  # Skip very short
                    continue

                all_sentences.append({
                    "text": sentence,
                    "msg_idx": msg_idx,
                    "sent_idx": sent_idx,
                    "role": role,
                    "total_msgs": len(messages),
                    "total_sents": len(sentences),
                })

        # Score each sentence
        for sent_data in all_sentences:
            score = self._compute_sentence_score(sent_data, term_freq, entities)
            scored.append((sent_data["text"], score))

        return scored

    def _compute_sentence_score(
        self,
        sent_data: Dict[str, Any],
        term_freq: Dict[str, float],
        entities: Dict[str, List[str]],
    ) -> float:
        """Compute comprehensive sentence score."""
        sentence = sent_data["text"]
        lower = sentence.lower()
        words = re.findall(r'\b[a-z]{3,}\b', lower)

        # 1. TF-IDF score (sum of term scores)
        tfidf_score = sum(term_freq.get(w, 0) for w in words) / (len(words) + 1)
        tfidf_score = min(1.0, tfidf_score * 10)  # Normalize

        # 2. Position score (first/last sentences more important)
        msg_idx = sent_data["msg_idx"]
        sent_idx = sent_data["sent_idx"]
        total_msgs = sent_data["total_msgs"]
        total_sents = sent_data["total_sents"]

        # First and last messages weighted higher
        msg_position_score = 0.0
        if msg_idx == 0:  # First message
            msg_position_score = 0.3
        elif msg_idx == total_msgs - 1:  # Last message
            msg_position_score = 0.2

        # First sentence of any message weighted higher
        sent_position_score = 0.0
        if sent_idx == 0:
            sent_position_score = 0.15
        elif sent_idx == total_sents - 1:
            sent_position_score = 0.1

        position_score = msg_position_score + sent_position_score

        # 3. Entity density score
        entity_score = 0.0
        all_entities = []
        for ent_list in entities.values():
            all_entities.extend(ent_list)

        for entity in all_entities:
            if entity.lower() in lower:
                entity_score += 0.1

        entity_score = min(0.3, entity_score)

        # 4. Importance indicators
        importance_score = 0.0
        importance_patterns = [
            (r'\b(important|key|main|crucial|essential|critical)\b', 0.15),
            (r'\b(conclusion|summary|result|finding|outcome)\b', 0.15),
            (r'\b(decided|concluded|determined|found|discovered)\b', 0.1),
            (r'\b(should|must|need to|have to|requires)\b', 0.08),
            (r'\b(because|therefore|thus|hence|consequently)\b', 0.08),
            (r'^[A-Z][^.!?]*[.!?]$', 0.05),  # Complete sentence
        ]

        for pattern, boost in importance_patterns:
            if re.search(pattern, lower):
                importance_score += boost

        importance_score = min(0.3, importance_score)

        # 5. Role bonus (user intent more important)
        role_score = 0.1 if sent_data["role"] in ("user", "human") else 0.0

        # 6. Length penalty (optimal 15-40 words)
        word_count = len(sentence.split())
        length_score = 0.0
        if 15 <= word_count <= 40:
            length_score = 0.1
        elif word_count < 10:
            length_score = -0.1
        elif word_count > 60:
            length_score = -0.15

        # Combine scores
        total = (
            tfidf_score * 0.25 +
            position_score * 0.2 +
            entity_score * 0.2 +
            importance_score * 0.2 +
            role_score * 0.1 +
            length_score * 0.05
        )

        return min(1.0, max(0.0, total + 0.3))  # Base of 0.3

    def _detect_intent(self, messages: List[Dict[str, str]]) -> str:
        """Detect the primary intent of the conversation."""
        all_text = " ".join(m.get("content", "") for m in messages).lower()

        intent_scores = {}
        for intent, patterns in self.INTENT_PATTERNS.items():
            score = 0
            for pattern in patterns:
                matches = re.findall(pattern, all_text, re.IGNORECASE | re.MULTILINE)
                score += len(matches)
            intent_scores[intent] = score

        if not intent_scores or max(intent_scores.values()) == 0:
            return "discussion"

        return max(intent_scores, key=intent_scores.get)

    def _extract_enhanced_topics(
        self,
        messages: List[Dict[str, str]],
        entities: Dict[str, List[str]],
    ) -> List[str]:
        """Extract topics using entities + keyword frequency."""
        topics = []

        # Add technology entities as topics
        if "technology" in entities:
            topics.extend(entities["technology"][:3])

        # Add project names
        if "project" in entities:
            topics.extend(entities["project"][:2])

        # Keyword frequency for remaining topics
        all_text = " ".join(m.get("content", "") for m in messages).lower()
        words = re.findall(r'\b[a-z]{4,}\b', all_text)
        word_freq = Counter(words)

        # Remove filler words and already-added topics
        existing_lower = {t.lower() for t in topics}
        for word, count in word_freq.most_common(20):
            if word not in self.FILLER_WORDS and word not in existing_lower:
                if count >= 2:
                    topics.append(word)
                    existing_lower.add(word)

        return topics[:8]

    def _compress_sentence(self, sentence: str) -> str:
        """Apply compression patterns to reduce sentence verbosity."""
        result = sentence

        for pattern, replacement in self.COMPRESSION_PATTERNS:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

        # Clean up extra spaces
        result = re.sub(r'\s+', ' ', result).strip()

        # Capitalize first letter
        if result:
            result = result[0].upper() + result[1:] if len(result) > 1 else result.upper()

        return result

    def _generate_abstractive_summary(
        self,
        intent: str,
        topics: List[str],
        sentences: List[str],
        entities: Dict[str, List[str]],
        max_length: int,
    ) -> str:
        """Generate abstractive summary using templates and fusion."""
        # Get template for intent
        template = self.SUMMARY_TEMPLATES.get(intent, self.SUMMARY_TEMPLATES["default"])

        # Format topics
        if topics:
            topic_str = ", ".join(topics[:3])
        else:
            topic_str = "various topics"

        # Get best key point (first compressed sentence or entity-based)
        if sentences:
            key_point = sentences[0]
            # Ensure it ends with period
            if key_point and not key_point.endswith(('.', '!', '?')):
                key_point += "."
        else:
            # Generate from entities
            if "person" in entities and entities["person"]:
                key_point = f"Mentioned {entities['person'][0]}."
            elif "technology" in entities and entities["technology"]:
                key_point = f"Worked with {', '.join(entities['technology'][:2])}."
            else:
                key_point = "Key details discussed."

        # Apply template
        summary = template.format(topics=topic_str, key_point=key_point)

        # Add additional context if room
        if len(summary) < max_length - 50 and len(sentences) > 1:
            additional = sentences[1]
            if len(summary) + len(additional) + 1 < max_length:
                if not additional.endswith(('.', '!', '?')):
                    additional += "."
                summary += " " + additional

        # Truncate if needed
        if len(summary) > max_length:
            summary = summary[:max_length-3].rsplit(' ', 1)[0] + "..."

        return summary



# NOTE: LLMSummarizer was removed in v0.2.0 (required torch/transformers,
# not viable on Phase 0 hardware). Will be rebuilt when hardware supports it.
