"""
Entity extraction from text.

Simple rule-based NER that doesn't require external libraries.
Extracts named entities like people, locations, and organizations.
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List


class EntityType(Enum):
    """Types of entities to extract."""
    PERSON = "person"           # People's names
    LOCATION = "location"       # Places
    ORGANIZATION = "organization"  # Companies, groups
    PROJECT = "project"         # Projects, products
    TECHNOLOGY = "technology"   # Programming languages, tools
    DATE = "date"               # Dates and times
    NUMBER = "number"           # Numbers and quantities


@dataclass
class Entity:
    """An extracted entity."""
    text: str
    entity_type: EntityType
    confidence: float
    start: int  # Character position
    end: int


class EntityExtractor:
    """
    Simple rule-based entity extraction.

    Uses patterns and keyword lists to identify entities.
    For better accuracy, consider using spaCy:
        pip install spacy
        python -m spacy download en_core_web_sm
    """

    # Known technology terms
    TECHNOLOGIES = {
        "python", "javascript", "typescript", "java", "rust", "go", "golang",
        "c++", "c#", "ruby", "php", "swift", "kotlin", "scala", "react",
        "vue", "angular", "django", "flask", "fastapi", "express", "nodejs",
        "docker", "kubernetes", "k8s", "aws", "azure", "gcp", "linux", "windows",
        "macos", "postgresql", "mysql", "mongodb", "redis", "sqlite", "git",
        "github", "gitlab", "vscode", "vim", "emacs", "pytorch", "tensorflow",
        "numpy", "pandas", "chromadb", "openai", "claude", "anthropic", "llm",
        "gpt", "transformer", "embedding", "api", "rest", "graphql",
    }

    # Location indicators
    LOCATION_PATTERNS = [
        r"\bin\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b",  # in California
        r"\bat\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b",  # at Paris
        r"\bfrom\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b",  # from Tokyo
        r"\blive[ds]?\s+in\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b",
        r"\bbased\s+in\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b",
    ]

    # Project/product indicators
    PROJECT_PATTERNS = [
        r"\bproject\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b",
        r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+project\b",
        r"\bapp\s+called\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b",
        r"\bbuilding\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b",
    ]

    # Person name indicators
    PERSON_PATTERNS = [
        r"\bmy\s+(?:friend|wife|husband|partner|colleague)\s+([A-Z][a-z]+)\b",
        r"\bname\s+is\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b",
        r"\bi'?m\s+([A-Z][a-z]+)\b",
        r"\bcall\s+me\s+([A-Z][a-z]+)\b",
    ]

    # Date patterns
    DATE_PATTERNS = [
        r"\b(\d{4}-\d{2}-\d{2})\b",  # 2024-01-15
        r"\b(\d{1,2}/\d{1,2}/\d{2,4})\b",  # 1/15/2024
        r"\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?)\b",
        r"\b(last\s+(?:week|month|year))\b",
        r"\b(yesterday|today|tomorrow)\b",
    ]

    # Common non-entity words that look like names
    FALSE_POSITIVES = {
        "The", "This", "That", "These", "Those", "What", "Which", "Where",
        "When", "Why", "How", "Hello", "Hi", "Thanks", "Please", "Sure",
        "Yes", "No", "Maybe", "But", "And", "Or", "If", "So", "Just",
        "Actually", "Really", "Very", "Also", "Then", "Now", "Here",
        "There", "Always", "Never", "Sometimes", "Often", "Usually",
    }

    def __init__(self, use_spacy: bool = False):
        """
        Initialize entity extractor.

        Args:
            use_spacy: If True, try to use spaCy for better extraction
        """
        self.nlp = None
        if use_spacy:
            self._init_spacy()

    def _init_spacy(self):
        """Try to initialize spaCy."""
        try:
            import spacy
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                # Model not installed
                pass
        except ImportError:
            pass

    def extract(self, text: str) -> List[Entity]:
        """
        Extract entities from text.

        Args:
            text: Text to extract from

        Returns:
            List of Entity objects
        """
        if self.nlp:
            return self._extract_with_spacy(text)
        else:
            return self._extract_with_patterns(text)

    def _extract_with_spacy(self, text: str) -> List[Entity]:
        """Extract using spaCy NER."""
        doc = self.nlp(text)
        entities = []

        type_map = {
            "PERSON": EntityType.PERSON,
            "GPE": EntityType.LOCATION,
            "LOC": EntityType.LOCATION,
            "ORG": EntityType.ORGANIZATION,
            "PRODUCT": EntityType.PROJECT,
            "DATE": EntityType.DATE,
            "TIME": EntityType.DATE,
            "CARDINAL": EntityType.NUMBER,
            "QUANTITY": EntityType.NUMBER,
        }

        for ent in doc.ents:
            if ent.label_ in type_map:
                entities.append(Entity(
                    text=ent.text,
                    entity_type=type_map[ent.label_],
                    confidence=0.9,  # spaCy is fairly confident
                    start=ent.start_char,
                    end=ent.end_char,
                ))

        return entities

    def _extract_with_patterns(self, text: str) -> List[Entity]:
        """Extract using rule-based patterns."""
        entities = []

        # Extract technologies (case-insensitive)
        text_lower = text.lower()
        for tech in self.TECHNOLOGIES:
            if tech in text_lower:
                # Find all occurrences
                for match in re.finditer(r'\b' + re.escape(tech) + r'\b', text_lower):
                    entities.append(Entity(
                        text=tech,
                        entity_type=EntityType.TECHNOLOGY,
                        confidence=0.95,
                        start=match.start(),
                        end=match.end(),
                    ))

        # Extract locations
        for pattern in self.LOCATION_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                name = match.group(1) if match.groups() else match.group(0)
                if name and name not in self.FALSE_POSITIVES:
                    entities.append(Entity(
                        text=name,
                        entity_type=EntityType.LOCATION,
                        confidence=0.7,
                        start=match.start(),
                        end=match.end(),
                    ))

        # Extract person names
        for pattern in self.PERSON_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                name = match.group(1) if match.groups() else match.group(0)
                if name and name not in self.FALSE_POSITIVES:
                    entities.append(Entity(
                        text=name,
                        entity_type=EntityType.PERSON,
                        confidence=0.75,
                        start=match.start(),
                        end=match.end(),
                    ))

        # Extract project names
        for pattern in self.PROJECT_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                name = match.group(1) if match.groups() else match.group(0)
                if name and name not in self.FALSE_POSITIVES:
                    entities.append(Entity(
                        text=name,
                        entity_type=EntityType.PROJECT,
                        confidence=0.7,
                        start=match.start(),
                        end=match.end(),
                    ))

        # Extract dates
        for pattern in self.DATE_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                date_text = match.group(1) if match.groups() else match.group(0)
                entities.append(Entity(
                    text=date_text,
                    entity_type=EntityType.DATE,
                    confidence=0.9,
                    start=match.start(),
                    end=match.end(),
                ))

        # Deduplicate by position
        seen = set()
        unique = []
        for e in entities:
            key = (e.start, e.end, e.entity_type)
            if key not in seen:
                seen.add(key)
                unique.append(e)

        return sorted(unique, key=lambda e: e.start)

    def extract_from_messages(
        self,
        messages: List[Dict[str, str]],
    ) -> Dict[EntityType, List[str]]:
        """
        Extract entities from a list of messages.

        Args:
            messages: List of message dicts with 'content' key

        Returns:
            Dict mapping entity type to list of unique entities
        """
        all_entities = []

        for msg in messages:
            content = msg.get("content", "")
            if content:
                entities = self.extract(content)
                all_entities.extend(entities)

        # Group by type and deduplicate
        by_type = {t: set() for t in EntityType}
        for e in all_entities:
            by_type[e.entity_type].add(e.text)

        return {t: list(v) for t, v in by_type.items() if v}


def extract_entities(text: str) -> List[Entity]:
    """Convenience function to extract entities."""
    extractor = EntityExtractor()
    return extractor.extract(text)
