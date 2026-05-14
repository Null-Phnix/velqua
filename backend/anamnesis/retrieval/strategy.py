"""
Adaptive retrieval strategy selection.

Analyses a query to detect its type (named-entity, thematic, cross-cultural,
follow-up) and returns tuned retrieval parameters so that
:class:`HybridRetriever` can adjust weights, MMR, and scoring on the fly.

Detection is intentionally keyword-heuristic — no ML model required, fast
enough to run on every query with zero overhead.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, FrozenSet, List, Optional, Set

from .synonyms import SYNONYM_MAP

# ── Query type taxonomy ───────────────────────────────────────────────

class QueryType(Enum):
    """Detected query intent category."""
    NAMED_ENTITY = auto()     # Specific god/figure/place name
    THEMATIC = auto()         # Theme, emotion, abstract concept
    CROSS_CULTURAL = auto()   # Comparison across mythological traditions
    FOLLOW_UP = auto()        # Refers back to earlier conversation
    GENERAL = auto()          # No strong signal — use default weights


# ── Retrieval parameter overrides ─────────────────────────────────────

@dataclass
class RetrievalStrategy:
    """
    Parameter overrides for :class:`HybridRetriever`.

    Only non-None fields should be applied — ``None`` means "keep default".
    """
    query_type: QueryType
    text_weight: Optional[float] = None
    vector_weight: Optional[float] = None
    mmr_lambda: Optional[float] = None
    mmr_diversity_threshold: Optional[float] = None
    boost_confirmed: bool = False
    confidence: float = 0.0  # How confident the classifier is (0-1)
    reasons: List[str] = field(default_factory=list)

    def describe(self) -> str:
        """Human-readable summary of the strategy."""
        parts = [f"type={self.query_type.name}"]
        if self.text_weight is not None:
            parts.append(f"fts={self.text_weight:.0%}")
        if self.vector_weight is not None:
            parts.append(f"vec={self.vector_weight:.0%}")
        if self.mmr_lambda is not None:
            parts.append(f"mmr_λ={self.mmr_lambda:.2f}")
        if self.boost_confirmed:
            parts.append("boost_confirmed")
        return " | ".join(parts)


# ── Keyword inventories ──────────────────────────────────────────────

# Culture / tradition markers — used to detect cross-cultural queries.
_CULTURE_TAGS: Dict[str, str] = {}
_CULTURE_GROUPS: Dict[str, List[str]] = {
    "norse":        ["odin", "thor", "freya", "loki", "tyr", "frigg", "baldur",
                     "heimdall", "fenrir", "ragnarok", "yggdrasil", "valhalla",
                     "asgard", "norse", "viking"],
    "greek":        ["zeus", "hera", "poseidon", "athena", "ares", "aphrodite",
                     "hermes", "artemis", "hephaestus", "demeter", "dionysus",
                     "hades", "apollo", "olympus", "greek"],
    "roman":        ["jupiter", "juno", "neptune", "minerva", "mars", "venus",
                     "mercury", "diana", "vulcan", "ceres", "bacchus", "pluto",
                     "roman"],
    "hindu":        ["indra", "shiva", "vishnu", "brahma", "lakshmi", "saraswati",
                     "ganesha", "hanuman", "kali", "durga", "hindu", "vedic"],
    "egyptian":     ["ra", "osiris", "isis", "anubis", "thoth", "horus", "set",
                     "bastet", "sekhmet", "egyptian", "pharaoh"],
    "mesopotamian": ["marduk", "ishtar", "inanna", "enlil", "enki", "tiamat",
                     "sumerian", "babylonian", "mesopotamian"],
    "celtic":       ["dagda", "brigid", "morrigan", "lugh", "cernunnos",
                     "celtic", "druid"],
    "japanese":     ["amaterasu", "susanoo", "tsukuyomi", "izanagi", "izanami",
                     "shinto", "kami", "japanese"],
    "chinese":      ["jade emperor", "guanyin", "sun wukong", "monkey king",
                     "chinese", "taoist"],
}

# Flatten to term → culture tag
for _culture, _terms in _CULTURE_GROUPS.items():
    for _t in _terms:
        _CULTURE_TAGS[_t] = _culture

# Named-entity set: all keys from SYNONYM_MAP that represent specific names
# (not thematic clusters).  The thematic cluster terms overlap, so we build
# a separate set of *names* by including only terms whose synonym group is
# ≤ 4 members and at least one member is a capitalizable proper noun.
_ENTITY_NAMES: Set[str] = set()
for _term in SYNONYM_MAP:
    # If the term appears in any culture group, it's a named entity
    if _term in _CULTURE_TAGS:
        _ENTITY_NAMES.add(_term)
    # Also include any term that is a synonym of a culture-tagged term
    for _syn in SYNONYM_MAP[_term]:
        if _syn in _CULTURE_TAGS:
            _ENTITY_NAMES.add(_term)
            break

# Thematic / emotional keywords
_THEMATIC_KEYWORDS: FrozenSet[str] = frozenset({
    # Abstract themes
    "death", "mortality", "afterlife", "underworld", "creation", "origin",
    "genesis", "cosmogony", "flood", "deluge", "cataclysm", "trickster",
    "shapeshifter", "hero", "champion", "prophecy", "oracle", "divination",
    "sacrifice", "offering", "fate", "destiny", "rebirth", "resurrection",
    "reincarnation", "chaos", "void", "abyss", "sacred", "divine", "quest",
    "journey", "pilgrimage", "magic", "sorcery", "spirit", "ghost", "demon",
    "giant", "titan",
    # Emotional registers
    "love", "grief", "rage", "fear", "jealousy", "pride", "shame", "guilt",
    "hope", "despair", "longing", "betrayal", "revenge", "forgiveness",
    "sorrow", "joy", "wrath", "mercy", "suffering", "redemption",
    # Abstract concepts
    "morality", "justice", "power", "wisdom", "transformation", "duality",
    "balance", "order", "fertility", "war", "peace", "eternal", "immortal",
    "transcendence",
})

# Cross-cultural comparison signals
_COMPARISON_PATTERNS: List[re.Pattern] = [
    re.compile(r"\bcompar(?:e|ed|ing|ison)\b", re.IGNORECASE),
    re.compile(r"\bvs\.?\b", re.IGNORECASE),
    re.compile(r"\bversus\b", re.IGNORECASE),
    re.compile(r"\bdifferen(?:ce|t|ces)\b", re.IGNORECASE),
    re.compile(r"\bsimilar(?:ity|ities|ly)?\b", re.IGNORECASE),
    re.compile(r"\bacross\b.*\b(?:cultur|tradition|mytholog)", re.IGNORECASE),
    re.compile(r"\bequivalent\b", re.IGNORECASE),
    re.compile(r"\bcounterpart\b", re.IGNORECASE),
    re.compile(r"\banalog(?:ue|ous)?\b", re.IGNORECASE),
    re.compile(r"\bparallel\b", re.IGNORECASE),
    re.compile(r"\bboth\b.*\band\b", re.IGNORECASE),
]

# Follow-up / conversational reference patterns
_FOLLOWUP_PATTERNS: List[re.Pattern] = [
    re.compile(r"\byou (?:said|mentioned|told)\b", re.IGNORECASE),
    re.compile(r"\bearlier\b", re.IGNORECASE),
    re.compile(r"\bbefore\b", re.IGNORECASE),
    re.compile(r"\blast time\b", re.IGNORECASE),
    re.compile(r"\bremember when\b", re.IGNORECASE),
    re.compile(r"\bwe (?:talked|discussed|covered)\b", re.IGNORECASE),
    re.compile(r"\bconfirm\b", re.IGNORECASE),
    re.compile(r"\bfollow.?up\b", re.IGNORECASE),
    re.compile(r"\bwhat (?:was|were)\b.*\bagain\b", re.IGNORECASE),
    re.compile(r"\byou know about\b", re.IGNORECASE),
    re.compile(r"\bdo you (?:recall|remember)\b", re.IGNORECASE),
    re.compile(r"\bpreviously\b", re.IGNORECASE),
]


# ── Core classifier ──────────────────────────────────────────────────

def _tokenize(query: str) -> List[str]:
    """Lowercase word tokens from a query string, stripping possessives."""
    raw = re.findall(r"[a-z][a-z'-]*", query.lower())
    tokens = []
    for w in raw:
        # Strip trailing possessive 's or lone apostrophe
        w = re.sub(r"'s$", "", w)
        w = w.rstrip("'")
        if len(w) > 1:
            tokens.append(w)
    return tokens


def _count_matches(tokens: List[str], term_set: Set[str] | FrozenSet[str]) -> int:
    """Count how many tokens appear in a term set."""
    return sum(1 for t in tokens if t in term_set)


def _cultures_present(tokens: List[str], query_lower: str) -> Set[str]:
    """Return the set of distinct culture tags detected in the query."""
    cultures: Set[str] = set()
    for token in tokens:
        if token in _CULTURE_TAGS:
            cultures.add(_CULTURE_TAGS[token])
    # Also check multi-word culture terms in the raw query
    for term, culture in _CULTURE_TAGS.items():
        if " " in term and term in query_lower:
            cultures.add(culture)
    return cultures


def _any_pattern_matches(patterns: List[re.Pattern], text: str) -> int:
    """Count how many patterns match against text."""
    return sum(1 for p in patterns if p.search(text))


def classify_query(query: str) -> RetrievalStrategy:
    """
    Classify a query and return the optimal retrieval strategy.

    The classifier runs all heuristics and picks the strongest signal.
    When signals are ambiguous, it falls back to GENERAL (default weights).

    Returns:
        RetrievalStrategy with overrides for HybridRetriever parameters.
    """
    if not query or not query.strip():
        return RetrievalStrategy(query_type=QueryType.GENERAL, confidence=0.0)

    tokens = _tokenize(query)
    query_lower = query.lower()

    # ── Score each query type ─────────────────────────────────────

    # 1. Follow-up detection (highest priority — conversational context)
    followup_hits = _any_pattern_matches(_FOLLOWUP_PATTERNS, query)
    followup_score = min(1.0, followup_hits * 0.4)

    # 2. Cross-cultural detection
    cultures = _cultures_present(tokens, query_lower)
    comparison_hits = _any_pattern_matches(_COMPARISON_PATTERNS, query)
    cross_cultural_score = 0.0
    if len(cultures) >= 2:
        # Two or more cultures is a very strong signal — score high enough
        # to beat entity detection (which also fires on the same names).
        cross_cultural_score = 0.75 + min(0.2, comparison_hits * 0.1)
    elif comparison_hits > 0 and len(cultures) >= 1:
        cross_cultural_score = 0.3 + min(0.2, comparison_hits * 0.1)

    # 3. Named entity detection
    entity_hits = _count_matches(tokens, _ENTITY_NAMES)
    entity_score = min(1.0, entity_hits * 0.35)

    # 4. Thematic / emotional detection
    thematic_hits = _count_matches(tokens, _THEMATIC_KEYWORDS)
    thematic_score = min(1.0, thematic_hits * 0.3)

    # ── Pick the winner ───────────────────────────────────────────

    scores = {
        QueryType.FOLLOW_UP: followup_score,
        QueryType.CROSS_CULTURAL: cross_cultural_score,
        QueryType.NAMED_ENTITY: entity_score,
        QueryType.THEMATIC: thematic_score,
    }

    best_type = max(scores, key=scores.get)
    best_score = scores[best_type]

    # Minimum confidence threshold — below this, fall back to GENERAL
    if best_score < 0.25:
        return RetrievalStrategy(
            query_type=QueryType.GENERAL,
            confidence=best_score,
            reasons=["no strong signal detected"],
        )

    # ── Build strategy for the winning type ───────────────────────

    if best_type == QueryType.NAMED_ENTITY:
        return RetrievalStrategy(
            query_type=QueryType.NAMED_ENTITY,
            text_weight=0.4,
            vector_weight=0.6,
            confidence=best_score,
            reasons=[f"entity keywords: {entity_hits}"],
        )

    if best_type == QueryType.THEMATIC:
        return RetrievalStrategy(
            query_type=QueryType.THEMATIC,
            text_weight=0.1,
            vector_weight=0.9,
            confidence=best_score,
            reasons=[f"thematic keywords: {thematic_hits}"],
        )

    if best_type == QueryType.CROSS_CULTURAL:
        return RetrievalStrategy(
            query_type=QueryType.CROSS_CULTURAL,
            mmr_lambda=0.3,
            mmr_diversity_threshold=0.7,
            confidence=best_score,
            reasons=[
                f"cultures: {sorted(cultures)}",
                f"comparison signals: {comparison_hits}",
            ],
        )

    if best_type == QueryType.FOLLOW_UP:
        return RetrievalStrategy(
            query_type=QueryType.FOLLOW_UP,
            boost_confirmed=True,
            confidence=best_score,
            reasons=[f"follow-up patterns: {followup_hits}"],
        )

    # Unreachable, but satisfy the type checker
    return RetrievalStrategy(query_type=QueryType.GENERAL, confidence=0.0)


def apply_strategy(
    retriever,
    strategy: RetrievalStrategy,
) -> Dict[str, float | bool]:
    """
    Apply a :class:`RetrievalStrategy` to a :class:`HybridRetriever`,
    returning a dict of the original values so callers can restore them.

    Usage::

        original = apply_strategy(retriever, strategy)
        results = retriever.search(query)
        restore_strategy(retriever, original)
    """
    original: Dict[str, float | bool] = {}

    if strategy.text_weight is not None:
        original["text_weight"] = retriever.text_weight
        retriever.text_weight = strategy.text_weight

    if strategy.vector_weight is not None:
        original["vector_weight"] = retriever.vector_weight
        retriever.vector_weight = strategy.vector_weight

    if strategy.mmr_lambda is not None:
        original["mmr_lambda"] = retriever.mmr_lambda
        retriever.mmr_lambda = strategy.mmr_lambda

    if strategy.mmr_diversity_threshold is not None:
        original["mmr_diversity_threshold"] = retriever.mmr_diversity_threshold
        retriever.mmr_diversity_threshold = strategy.mmr_diversity_threshold

    return original


def restore_strategy(
    retriever,
    original: Dict[str, float | bool],
) -> None:
    """Restore retriever parameters from the dict returned by :func:`apply_strategy`."""
    for attr, value in original.items():
        setattr(retriever, attr, value)
