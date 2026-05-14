"""
Contradiction detection for facts.

Identifies when new facts contradict existing knowledge.
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from ..models import Fact


@dataclass
class ContradictionResult:
    """Result of contradiction check."""
    is_contradiction: bool
    existing_fact: Optional[Fact]
    confidence: float  # 0.0 to 1.0
    contradiction_type: str  # "direct", "temporal", "negation", "value_change"
    explanation: str


class ContradictionDetector:
    """
    Detects contradictions between facts.

    Uses pattern matching and keyword analysis to identify:
    - Direct contradictions (X is Y vs X is not Y)
    - Negations (likes X vs dislikes X)
    - Value changes (age is 25 vs age is 30)
    - Temporal conflicts (works at A vs works at B)
    """

    # Patterns that indicate opposites
    NEGATION_PAIRS = [
        ("like", "dislike"),
        ("love", "hate"),
        ("prefer", "avoid"),
        ("enjoy", "dislike"),
        ("can", "cannot"),
        ("is", "is not"),
        ("has", "has no"),
        ("want", "don't want"),
        ("happy", "unhappy"),
        ("works at", "left"),
        ("lives in", "moved from"),
    ]

    # Fact types that are typically singular (only one value)
    SINGULAR_FACT_TYPES = [
        "name", "age", "location", "occupation", "partner",
    ]

    def __init__(self):
        pass

    def find_contradictions(
        self,
        new_fact: Fact,
        existing_facts: List[Fact],
        threshold: float = 0.5,
    ) -> List[ContradictionResult]:
        """
        Find facts that contradict the new fact.

        Args:
            new_fact: The new fact to check
            existing_facts: List of existing facts to check against
            threshold: Minimum confidence to report contradiction

        Returns:
            List of ContradictionResult for each contradiction found
        """
        contradictions = []

        for existing in existing_facts:
            # Skip if same fact or superseded
            if existing.id == new_fact.id or existing.is_superseded:
                continue

            # Check for contradiction
            result = self._check_contradiction(new_fact, existing)
            if result.is_contradiction and result.confidence >= threshold:
                contradictions.append(result)

        return contradictions

    def _check_contradiction(
        self,
        new_fact: Fact,
        existing: Fact,
    ) -> ContradictionResult:
        """Check if two facts contradict each other."""
        new_content = new_fact.content.lower()
        existing_content = existing.content.lower()

        # Extract key elements
        new_subject = self._extract_subject(new_content)
        existing_subject = self._extract_subject(existing_content)

        # If subjects don't overlap, no contradiction
        if not self._subjects_overlap(new_subject, existing_subject):
            return ContradictionResult(
                is_contradiction=False,
                existing_fact=None,
                confidence=0.0,
                contradiction_type="",
                explanation="Different subjects",
            )

        # Check for direct negation
        neg_result = self._check_negation(new_content, existing_content)
        if neg_result[0]:
            return ContradictionResult(
                is_contradiction=True,
                existing_fact=existing,
                confidence=neg_result[1],
                contradiction_type="negation",
                explanation=f"Negation detected: {neg_result[2]}",
            )

        # Check for value change (same attribute, different value)
        value_result = self._check_value_change(new_fact, existing)
        if value_result[0]:
            return ContradictionResult(
                is_contradiction=True,
                existing_fact=existing,
                confidence=value_result[1],
                contradiction_type="value_change",
                explanation=f"Value changed: {value_result[2]}",
            )

        # Check for singular fact conflict
        singular_result = self._check_singular_conflict(new_fact, existing)
        if singular_result[0]:
            return ContradictionResult(
                is_contradiction=True,
                existing_fact=existing,
                confidence=singular_result[1],
                contradiction_type="singular",
                explanation=f"Conflicting {singular_result[2]}",
            )

        return ContradictionResult(
            is_contradiction=False,
            existing_fact=None,
            confidence=0.0,
            contradiction_type="",
            explanation="No contradiction detected",
        )

    def _extract_subject(self, content: str) -> str:
        """Extract the subject of a fact."""
        # Try to extract from "X: Y" format
        if ':' in content:
            return content.split(':')[0].strip()

        # Try to extract first few words
        words = content.split()[:3]
        return ' '.join(words)

    def _subjects_overlap(self, subject1: str, subject2: str) -> bool:
        """Check if two subjects refer to the same thing."""
        # Simple word overlap check
        words1 = set(subject1.lower().split())
        words2 = set(subject2.lower().split())

        # Remove common words
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'user', 'my', 'i'}
        words1 -= stopwords
        words2 -= stopwords

        if not words1 or not words2:
            return False

        overlap = words1 & words2
        return len(overlap) > 0

    def _check_negation(
        self,
        new_content: str,
        existing_content: str,
    ) -> Tuple[bool, float, str]:
        """Check for negation patterns."""
        for pos, neg in self.NEGATION_PAIRS:
            # Check if one has positive and other has negative
            if pos in new_content and neg in existing_content:
                return (True, 0.8, f"'{pos}' vs '{neg}'")
            if neg in new_content and pos in existing_content:
                return (True, 0.8, f"'{neg}' vs '{pos}'")

            # Check for "not" negation
            if pos in new_content and f"not {pos}" in existing_content:
                return (True, 0.9, f"'{pos}' vs 'not {pos}'")
            if f"not {pos}" in new_content and pos in existing_content:
                return (True, 0.9, f"'not {pos}' vs '{pos}'")

        return (False, 0.0, "")

    # Attributes that can have multiple values (not singular)
    MULTI_VALUE_ATTRIBUTES = [
        "like", "dislike", "prefer", "enjoy", "want", "need",
        "favorite", "interest", "hobby", "skill", "goal",
        "project", "topic", "friend", "relationship",
    ]

    def _check_value_change(
        self,
        new_fact: Fact,
        existing: Fact,
    ) -> Tuple[bool, float, str]:
        """Check if the same attribute has different values."""
        new_content = new_fact.content.lower()
        existing_content = existing.content.lower()

        # Check for "X: Y" pattern with same X but different Y
        if ':' in new_content and ':' in existing_content:
            new_key, new_val = new_content.split(':', 1)
            exist_key, exist_val = existing_content.split(':', 1)

            new_key = new_key.strip()
            exist_key = exist_key.strip()
            new_val = new_val.strip()
            exist_val = exist_val.strip()

            # Skip multi-value attributes - different likes/preferences aren't contradictions
            if new_key in self.MULTI_VALUE_ATTRIBUTES:
                return (False, 0.0, "")

            # Skip if values are too short or look like noise
            if len(new_val) < 3 or len(exist_val) < 3:
                return (False, 0.0, "")

            # Skip if values look like sentence fragments (contain common words)
            noise_words = ['the', 'is', 'are', 'not', 'just', 'still', 'being', 'doing']
            if any(new_val.startswith(w) for w in noise_words):
                return (False, 0.0, "")
            if any(exist_val.startswith(w) for w in noise_words):
                return (False, 0.0, "")

            # For name facts, require capitalized proper nouns
            if new_key == 'name':
                # Both values should look like proper names (start with capital)
                if not (new_val[0].isupper() or exist_val[0].isupper()):
                    return (False, 0.0, "")
                # Skip very short names
                if len(new_val) < 2 or len(exist_val) < 2:
                    return (False, 0.0, "")

            # Same key, different value (only for singular attributes)
            if new_key == exist_key and new_val != exist_val:
                return (True, 0.85, f"{new_key}: '{new_val[:40]}' vs '{exist_val[:40]}'")

        # Check for number changes (age, count, etc.)
        new_nums = re.findall(r'\d+', new_content)
        exist_nums = re.findall(r'\d+', existing_content)

        if new_nums and exist_nums:
            # Check if context is similar but numbers differ
            new_no_nums = re.sub(r'\d+', '', new_content)
            exist_no_nums = re.sub(r'\d+', '', existing_content)

            if self._text_similarity(new_no_nums, exist_no_nums) > 0.7:
                if new_nums[0] != exist_nums[0]:
                    return (True, 0.7, f"Number changed: {exist_nums[0]} → {new_nums[0]}")

        return (False, 0.0, "")

    def _check_singular_conflict(
        self,
        new_fact: Fact,
        existing: Fact,
    ) -> Tuple[bool, float, str]:
        """Check for conflicts in singular facts (only one value allowed)."""
        new_content = new_fact.content.lower()
        existing_content = existing.content.lower()

        # Only check "X: Y" format facts for singular conflicts
        if ':' not in new_content or ':' not in existing_content:
            return (False, 0.0, "")

        new_key = new_content.split(':')[0].strip()
        exist_key = existing_content.split(':')[0].strip()

        # Keys must match exactly for singular conflict
        if new_key != exist_key:
            return (False, 0.0, "")

        # Only check for singular fact types
        if new_key not in self.SINGULAR_FACT_TYPES:
            return (False, 0.0, "")

        new_val = new_content.split(':', 1)[1].strip()
        exist_val = existing_content.split(':', 1)[1].strip()

        # Skip noise values
        noise_indicators = ['not', 'still', 'doing', 'being', 'just', 'the']
        if any(new_val.startswith(n) for n in noise_indicators):
            return (False, 0.0, "")
        if any(exist_val.startswith(n) for n in noise_indicators):
            return (False, 0.0, "")

        # For names, both should look like proper names
        if new_key == 'name':
            # At least 2 chars, starts with letter
            if len(new_val) < 2 or len(exist_val) < 2:
                return (False, 0.0, "")
            # Skip if either looks like a sentence fragment
            if ' ' in new_val and len(new_val.split()) > 2:
                return (False, 0.0, "")
            if ' ' in exist_val and len(exist_val.split()) > 2:
                return (False, 0.0, "")

        if new_val != exist_val and len(new_val) > 2 and len(exist_val) > 2:
            return (True, 0.75, new_key)

        return (False, 0.0, "")

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union)


def detect_contradictions(
    new_fact: Fact,
    existing_facts: List[Fact],
    threshold: float = 0.5,
) -> List[ContradictionResult]:
    """
    Convenience function to detect contradictions.

    Args:
        new_fact: The new fact to check
        existing_facts: List of existing facts
        threshold: Minimum confidence

    Returns:
        List of contradictions found
    """
    detector = ContradictionDetector()
    return detector.find_contradictions(new_fact, existing_facts, threshold)
