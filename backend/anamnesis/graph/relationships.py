"""
Fact relationship detection engine.

Detects three types of relationships between facts:
  - Contradiction: same topic, conflicting claims
  - Elaboration: one fact expands on another
  - Temporal sequence: events in chronological order

Uses heuristic NLP — no ML models required.
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class RelationshipType(Enum):
    """Types of fact-to-fact relationships."""
    CONTRADICTION = "contradiction"
    ELABORATION = "elaboration"
    TEMPORAL_SEQUENCE = "temporal_sequence"


@dataclass
class FactRelationship:
    """An edge between two facts."""
    source_id: str
    target_id: str
    relationship_type: RelationshipType
    confidence: float  # 0.0-1.0, how confident the detection is
    evidence: str       # Human-readable reason
    metadata: Dict[str, Any] = field(default_factory=dict)


# --- Keyword sets for heuristic detection ---

_NEGATION_PREFIXES = {
    "not", "no", "never", "don't", "doesn't", "didn't", "isn't",
    "aren't", "wasn't", "weren't", "won't", "can't", "cannot",
    "hardly", "rarely", "seldom",
}

_ANTONYM_PAIRS = [
    ("like", "dislike"), ("likes", "dislikes"),
    ("love", "hate"), ("loves", "hates"),
    ("enjoy", "avoid"), ("enjoys", "avoids"),
    ("prefer", "reject"), ("prefers", "rejects"),
    ("support", "oppose"), ("supports", "opposes"),
    ("agree", "disagree"), ("agrees", "disagrees"),
    ("happy", "unhappy"), ("happy", "sad"),
    ("good", "bad"), ("best", "worst"),
    ("always", "never"),
    ("true", "false"),
    ("yes", "no"),
    ("start", "stop"), ("started", "stopped"),
    ("begin", "end"), ("began", "ended"),
    ("buy", "sell"), ("bought", "sold"),
    ("join", "leave"), ("joined", "left"),
    ("accept", "decline"), ("accepted", "declined"),
    ("increase", "decrease"), ("increased", "decreased"),
]

# Build a fast lookup: word -> set of antonyms
_ANTONYM_MAP: Dict[str, Set[str]] = {}
for a, b in _ANTONYM_PAIRS:
    _ANTONYM_MAP.setdefault(a, set()).add(b)
    _ANTONYM_MAP.setdefault(b, set()).add(a)

_TEMPORAL_KEYWORDS = {
    "before", "after", "then", "later", "earlier", "previously",
    "recently", "now", "currently", "formerly", "originally",
    "used to", "started", "stopped", "began", "ended",
    "first", "next", "finally", "eventually", "since", "until",
    "yesterday", "today", "tomorrow", "last year", "this year",
    "in 2020", "in 2021", "in 2022", "in 2023", "in 2024", "in 2025", "in 2026",
}

# Patterns that indicate past vs present state
_PAST_PATTERNS = re.compile(
    r"\b(used to|formerly|previously|originally|was|were|had been|"
    r"stopped|quit|gave up|moved from|left)\b", re.IGNORECASE
)
_PRESENT_PATTERNS = re.compile(
    r"\b(currently|now|is|are|has been|recently started|"
    r"lives in|works at|works as)\b", re.IGNORECASE
)

# Year extraction
_YEAR_PATTERN = re.compile(r"\b((?:19|20)\d{2})\b")


def _tokenize(text: str) -> Set[str]:
    """Extract lowercased word tokens from text."""
    return set(re.findall(r"[a-z]+", text.lower()))


def _extract_topic(fact: Dict[str, Any]) -> Optional[str]:
    """Extract topic from fact metadata or infer from content."""
    meta = fact.get("metadata", {})
    if isinstance(meta, dict):
        topic = meta.get("topic") or meta.get("category")
        if topic:
            return topic.lower().strip()
    return None


def _word_overlap_ratio(tokens_a: Set[str], tokens_b: Set[str]) -> float:
    """Jaccard-like overlap between two token sets, ignoring stopwords."""
    stopwords = {
        "i", "me", "my", "we", "our", "you", "your", "he", "she", "it",
        "they", "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "shall", "can", "to", "of", "in",
        "for", "on", "with", "at", "by", "from", "as", "into", "about",
        "that", "this", "which", "who", "whom", "what", "and", "but", "or",
        "if", "than", "very", "just", "also", "so", "not", "no",
    }
    a = tokens_a - stopwords
    b = tokens_b - stopwords
    if not a or not b:
        return 0.0
    intersection = a & b
    union = a | b
    return len(intersection) / len(union)


def detect_contradiction(
    fact_a: Dict[str, Any], fact_b: Dict[str, Any]
) -> Optional[FactRelationship]:
    """
    Detect if two facts contradict each other.

    Looks for: same topic + opposing claims (negation, antonyms, sentiment flip).
    """
    content_a = fact_a.get("content", "")
    content_b = fact_b.get("content", "")
    tokens_a = _tokenize(content_a)
    tokens_b = _tokenize(content_b)

    overlap = _word_overlap_ratio(tokens_a, tokens_b)

    # Need some topical overlap to be a contradiction (not random unrelated facts)
    topic_a = _extract_topic(fact_a)
    topic_b = _extract_topic(fact_b)
    same_topic = topic_a and topic_b and topic_a == topic_b

    if overlap < 0.15 and not same_topic:
        return None

    confidence = 0.0
    evidence_parts = []

    # Check 1: Same topic with different sentiment scores
    meta_a = fact_a.get("metadata", {}) or {}
    meta_b = fact_b.get("metadata", {}) or {}
    sent_a = meta_a.get("sentiment_score")
    sent_b = meta_b.get("sentiment_score")
    if sent_a is not None and sent_b is not None:
        if isinstance(sent_a, (int, float)) and isinstance(sent_b, (int, float)):
            # Opposing sentiments on same topic
            if sent_a * sent_b < 0 and abs(sent_a - sent_b) > 0.8:
                confidence += 0.3
                evidence_parts.append(
                    f"opposing sentiment ({sent_a:.1f} vs {sent_b:.1f})"
                )

    # Check 2: Negation — one fact negates the other's key claim
    words_a = content_a.lower().split()
    words_b = content_b.lower().split()
    neg_in_a = bool(_NEGATION_PREFIXES & set(words_a))
    neg_in_b = bool(_NEGATION_PREFIXES & set(words_b))
    if neg_in_a != neg_in_b and overlap >= 0.25:
        confidence += 0.35
        evidence_parts.append("negation pattern detected")

    # Check 3: Antonym pairs
    for word in tokens_a:
        if word in _ANTONYM_MAP:
            if _ANTONYM_MAP[word] & tokens_b:
                antonyms_found = _ANTONYM_MAP[word] & tokens_b
                confidence += 0.35
                evidence_parts.append(
                    f"antonym pair: {word} vs {', '.join(antonyms_found)}"
                )
                break  # One pair is enough

    # Check 4: Same fact_type + high overlap but different claims
    if fact_a.get("fact_type") == fact_b.get("fact_type") and same_topic:
        confidence += 0.15
        evidence_parts.append("same type and topic")

    # Check 5: One fact supersedes the other
    if fact_a.get("is_superseded") or fact_b.get("is_superseded"):
        if overlap >= 0.2:
            confidence += 0.2
            evidence_parts.append("one fact is superseded")

    if confidence < 0.3:
        return None

    confidence = min(1.0, confidence)
    return FactRelationship(
        source_id=fact_a["id"],
        target_id=fact_b["id"],
        relationship_type=RelationshipType.CONTRADICTION,
        confidence=confidence,
        evidence="; ".join(evidence_parts),
    )


def detect_elaboration(
    fact_a: Dict[str, Any], fact_b: Dict[str, Any]
) -> Optional[FactRelationship]:
    """
    Detect if one fact elaborates on (expands) another.

    The shorter/more general fact is the source, the longer/more specific is the target.
    """
    content_a = fact_a.get("content", "")
    content_b = fact_b.get("content", "")
    tokens_a = _tokenize(content_a)
    tokens_b = _tokenize(content_b)

    overlap = _word_overlap_ratio(tokens_a, tokens_b)

    # Need meaningful overlap — elaboration means same subject
    if overlap < 0.2:
        return None

    # Same topic boosts confidence
    topic_a = _extract_topic(fact_a)
    topic_b = _extract_topic(fact_b)
    same_topic = topic_a and topic_b and topic_a == topic_b

    confidence = 0.0
    evidence_parts = []

    # Check 1: One fact's content tokens are mostly a subset of the other's
    stopwords = {
        "i", "me", "my", "the", "a", "an", "is", "are", "was", "were",
        "to", "of", "in", "for", "on", "with", "and", "but", "or", "that",
        "this", "have", "has", "had",
    }
    meaningful_a = tokens_a - stopwords
    meaningful_b = tokens_b - stopwords

    if meaningful_a and meaningful_b:
        # How much of the shorter fact appears in the longer one?
        if len(meaningful_a) <= len(meaningful_b):
            shorter, longer = meaningful_a, meaningful_b
            source, target = fact_a, fact_b
        else:
            shorter, longer = meaningful_b, meaningful_a
            source, target = fact_b, fact_a

        containment = len(shorter & longer) / len(shorter) if shorter else 0
        if containment >= 0.6:
            length_ratio = len(longer) / len(shorter) if shorter else 1.0
            if length_ratio >= 1.3:
                confidence += 0.4
                evidence_parts.append(
                    f"containment {containment:.0%}, "
                    f"target has {len(longer - shorter)} additional terms"
                )

    # Check 2: Same topic/category
    if same_topic:
        confidence += 0.2
        evidence_parts.append(f"same topic: {topic_a}")

    # Check 3: Same fact_type
    if fact_a.get("fact_type") == fact_b.get("fact_type"):
        confidence += 0.1
        evidence_parts.append("same fact type")

    # Check 4: Significant length difference (elaboration = more detail)
    len_a = len(content_a)
    len_b = len(content_b)
    if max(len_a, len_b) > 0:
        length_diff = abs(len_a - len_b) / max(len_a, len_b)
        if length_diff > 0.3:
            confidence += 0.15
            evidence_parts.append(f"length difference {length_diff:.0%}")

    if confidence < 0.3:
        return None

    # Determine direction: general -> specific
    if len(content_a) <= len(content_b):
        src_id, tgt_id = fact_a["id"], fact_b["id"]
    else:
        src_id, tgt_id = fact_b["id"], fact_a["id"]

    confidence = min(1.0, confidence)
    return FactRelationship(
        source_id=src_id,
        target_id=tgt_id,
        relationship_type=RelationshipType.ELABORATION,
        confidence=confidence,
        evidence="; ".join(evidence_parts),
    )


def detect_temporal_sequence(
    fact_a: Dict[str, Any], fact_b: Dict[str, Any]
) -> Optional[FactRelationship]:
    """
    Detect if two facts form a temporal sequence (A happened before B).

    Uses temporal keywords, tense patterns, year references, and first_learned timestamps.
    """
    content_a = fact_a.get("content", "")
    content_b = fact_b.get("content", "")
    tokens_a = _tokenize(content_a)
    tokens_b = _tokenize(content_b)

    overlap = _word_overlap_ratio(tokens_a, tokens_b)

    # Need some topical connection
    topic_a = _extract_topic(fact_a)
    topic_b = _extract_topic(fact_b)
    same_topic = topic_a and topic_b and topic_a == topic_b

    if overlap < 0.1 and not same_topic:
        return None

    confidence = 0.0
    evidence_parts = []
    # Positive = A before B, negative = B before A
    direction_score = 0.0

    # Check 1: Past vs present patterns
    past_a = bool(_PAST_PATTERNS.search(content_a))
    present_a = bool(_PRESENT_PATTERNS.search(content_a))
    past_b = bool(_PAST_PATTERNS.search(content_b))
    present_b = bool(_PRESENT_PATTERNS.search(content_b))

    if past_a and present_b and not past_b:
        confidence += 0.35
        direction_score += 1.0
        evidence_parts.append("A uses past tense, B uses present")
    elif past_b and present_a and not past_a:
        confidence += 0.35
        direction_score -= 1.0
        evidence_parts.append("B uses past tense, A uses present")

    # Check 2: Year references
    years_a = [int(y) for y in _YEAR_PATTERN.findall(content_a)]
    years_b = [int(y) for y in _YEAR_PATTERN.findall(content_b)]
    if years_a and years_b:
        min_a, min_b = min(years_a), min(years_b)
        if min_a != min_b:
            confidence += 0.4
            if min_a < min_b:
                direction_score += 1.0
                evidence_parts.append(f"A references {min_a}, B references {min_b}")
            else:
                direction_score -= 1.0
                evidence_parts.append(f"B references {min_b}, A references {min_a}")

    # Check 3: first_learned timestamps
    learned_a = fact_a.get("first_learned")
    learned_b = fact_b.get("first_learned")
    if learned_a and learned_b:
        try:
            ts_a = (
                datetime.fromisoformat(learned_a)
                if isinstance(learned_a, str)
                else learned_a
            )
            ts_b = (
                datetime.fromisoformat(learned_b)
                if isinstance(learned_b, str)
                else learned_b
            )
            delta = abs((ts_a - ts_b).total_seconds())
            # Only use timestamp if there's meaningful time gap (>1 hour)
            if delta > 3600:
                confidence += 0.15
                if ts_a < ts_b:
                    direction_score += 0.5
                else:
                    direction_score -= 0.5
                evidence_parts.append("first_learned timestamps differ")
        except (ValueError, TypeError):
            pass

    # Check 4: Temporal keywords in content
    content_lower_a = content_a.lower()
    content_lower_b = content_b.lower()
    temporal_a = any(kw in content_lower_a for kw in _TEMPORAL_KEYWORDS)
    temporal_b = any(kw in content_lower_b for kw in _TEMPORAL_KEYWORDS)
    if temporal_a or temporal_b:
        confidence += 0.1
        evidence_parts.append("temporal keywords present")

    # Check 5: Same topic strengthens temporal link
    if same_topic:
        confidence += 0.15
        evidence_parts.append(f"same topic: {topic_a}")

    if confidence < 0.3:
        return None

    # Determine direction
    if direction_score >= 0:
        src_id, tgt_id = fact_a["id"], fact_b["id"]
    else:
        src_id, tgt_id = fact_b["id"], fact_a["id"]

    confidence = min(1.0, confidence)
    return FactRelationship(
        source_id=src_id,
        target_id=tgt_id,
        relationship_type=RelationshipType.TEMPORAL_SEQUENCE,
        confidence=confidence,
        evidence="; ".join(evidence_parts),
    )


def detect_relationships(
    facts: List[Dict[str, Any]],
    types: Optional[List[RelationshipType]] = None,
) -> List[FactRelationship]:
    """
    Analyze a list of facts and detect all pairwise relationships.

    Args:
        facts: List of fact dicts (as returned by SQLiteBackend)
        types: Optional filter — only detect these relationship types

    Returns:
        List of detected FactRelationship edges
    """
    if types is None:
        types = list(RelationshipType)

    detectors = []
    if RelationshipType.CONTRADICTION in types:
        detectors.append(detect_contradiction)
    if RelationshipType.ELABORATION in types:
        detectors.append(detect_elaboration)
    if RelationshipType.TEMPORAL_SEQUENCE in types:
        detectors.append(detect_temporal_sequence)

    relationships: List[FactRelationship] = []
    seen_pairs: Set[Tuple[str, str, str]] = set()

    for i, fact_a in enumerate(facts):
        for fact_b in facts[i + 1:]:
            for detector in detectors:
                rel = detector(fact_a, fact_b)
                if rel is None:
                    continue
                # Deduplicate (same pair + same type)
                key = (rel.source_id, rel.target_id, rel.relationship_type.value)
                reverse_key = (rel.target_id, rel.source_id, rel.relationship_type.value)
                if key not in seen_pairs and reverse_key not in seen_pairs:
                    seen_pairs.add(key)
                    relationships.append(rel)

    return relationships
