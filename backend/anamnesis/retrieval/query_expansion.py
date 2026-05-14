"""
Query expansion for improved memory retrieval.

Expands search queries with:
- Synonyms and related terms
- Technical terminology mappings
- Entity-based expansion
- Graph-based related concepts
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set


@dataclass
class ExpansionResult:
    """Result of query expansion."""
    original_query: str
    expanded_query: str
    added_terms: List[str]
    expansion_sources: Dict[str, List[str]]  # source -> terms

    @property
    def all_terms(self) -> Set[str]:
        """Get all unique terms in expanded query."""
        return set(re.findall(r'\b\w+\b', self.expanded_query.lower()))


class QueryExpander:
    """
    Expands search queries for better retrieval coverage.

    Uses multiple expansion strategies:
    1. Synonym expansion (built-in mappings)
    2. Technical term expansion (domain-specific)
    3. Entity expansion (from known entities)
    4. Graph expansion (from memory graph relationships)
    """

    # Common synonyms for query expansion
    SYNONYMS = {
        # Emotions
        "happy": ["glad", "pleased", "joyful", "content"],
        "sad": ["unhappy", "upset", "disappointed", "down"],
        "angry": ["mad", "frustrated", "annoyed", "irritated"],
        "excited": ["enthusiastic", "thrilled", "eager"],
        "worried": ["concerned", "anxious", "nervous"],

        # Actions
        "build": ["create", "make", "develop", "construct"],
        "fix": ["repair", "solve", "resolve", "debug"],
        "learn": ["study", "understand", "explore"],
        "help": ["assist", "support", "aid"],
        "want": ["need", "desire", "wish", "looking for"],

        # Tech terms
        "code": ["program", "script", "software"],
        "bug": ["error", "issue", "problem", "defect"],
        "test": ["check", "verify", "validate"],
        "deploy": ["release", "ship", "launch"],
        "database": ["db", "data store", "storage"],

        # Common substitutions
        "fast": ["quick", "rapid", "speedy"],
        "slow": ["sluggish", "laggy", "delayed"],
        "good": ["great", "excellent", "nice"],
        "bad": ["poor", "terrible", "awful"],

        # Time
        "recent": ["latest", "new", "current"],
        "old": ["previous", "past", "earlier", "former"],
    }

    # Technical term expansions
    TECH_EXPANSIONS = {
        "python": ["py", "python3", "cpython"],
        "javascript": ["js", "ecmascript", "node"],
        "typescript": ["ts"],
        "api": ["endpoint", "rest", "graphql", "interface"],
        "frontend": ["front-end", "ui", "client-side"],
        "backend": ["back-end", "server-side", "server"],
        "ml": ["machine learning", "ai", "artificial intelligence"],
        "ai": ["artificial intelligence", "machine learning", "ml"],
        "db": ["database", "data store"],
        "k8s": ["kubernetes"],
        "kubernetes": ["k8s", "container orchestration"],
        "docker": ["container", "containerization"],
        "git": ["version control", "source control"],
        "ci": ["continuous integration", "cicd", "ci/cd"],
        "cd": ["continuous deployment", "cicd", "ci/cd"],
        "auth": ["authentication", "authorization", "login"],
        "oauth": ["authentication", "oauth2", "auth"],
    }

    # Query intent patterns -> expansion hints
    INTENT_EXPANSIONS = {
        r'\bhow to\b': ["guide", "tutorial", "steps"],
        r'\bwhat is\b': ["definition", "explanation", "meaning"],
        r'\bwhy\b': ["reason", "cause", "because"],
        r'\bproblem\b': ["issue", "error", "bug", "trouble"],
        r'\bworking on\b': ["building", "developing", "creating"],
    }

    def __init__(
        self,
        enable_synonyms: bool = True,
        enable_tech: bool = True,
        enable_intent: bool = True,
        max_expansions_per_term: int = 3,
        custom_synonyms: Optional[Dict[str, List[str]]] = None,
    ):
        """
        Initialize query expander.

        Args:
            enable_synonyms: Use synonym expansion
            enable_tech: Use technical term expansion
            enable_intent: Use intent-based expansion
            max_expansions_per_term: Max synonyms per term
            custom_synonyms: Additional custom synonym mappings
        """
        self.enable_synonyms = enable_synonyms
        self.enable_tech = enable_tech
        self.enable_intent = enable_intent
        self.max_expansions = max_expansions_per_term

        # Merge custom synonyms
        self.synonyms = self.SYNONYMS.copy()
        if custom_synonyms:
            for term, syns in custom_synonyms.items():
                if term in self.synonyms:
                    self.synonyms[term] = list(set(self.synonyms[term] + syns))
                else:
                    self.synonyms[term] = syns

    def expand(
        self,
        query: str,
        entities: Optional[Dict[str, List[str]]] = None,
        graph_related: Optional[List[str]] = None,
    ) -> ExpansionResult:
        """
        Expand a query with related terms.

        Args:
            query: Original search query
            entities: Known entities to consider {type: [values]}
            graph_related: Related terms from memory graph

        Returns:
            ExpansionResult with expanded query
        """
        original_words = set(re.findall(r'\b\w+\b', query.lower()))
        added_terms = []
        sources = {
            "synonyms": [],
            "tech": [],
            "intent": [],
            "entities": [],
            "graph": [],
        }

        # 1. Synonym expansion
        if self.enable_synonyms:
            syn_terms = self._expand_synonyms(original_words)
            sources["synonyms"] = syn_terms
            added_terms.extend(syn_terms)

        # 2. Technical term expansion
        if self.enable_tech:
            tech_terms = self._expand_tech_terms(original_words)
            sources["tech"] = tech_terms
            added_terms.extend(tech_terms)

        # 3. Intent-based expansion
        if self.enable_intent:
            intent_terms = self._expand_by_intent(query)
            sources["intent"] = intent_terms
            added_terms.extend(intent_terms)

        # 4. Entity expansion
        if entities:
            entity_terms = self._expand_from_entities(original_words, entities)
            sources["entities"] = entity_terms
            added_terms.extend(entity_terms)

        # 5. Graph expansion
        if graph_related:
            # Filter to relevant terms
            graph_terms = [t for t in graph_related if t.lower() not in original_words][:5]
            sources["graph"] = graph_terms
            added_terms.extend(graph_terms)

        # Build expanded query
        # Remove duplicates while preserving order
        unique_added = []
        seen = set(original_words)
        for term in added_terms:
            if term.lower() not in seen:
                unique_added.append(term)
                seen.add(term.lower())

        # Limit total expansions
        unique_added = unique_added[:10]

        if unique_added:
            expanded = f"{query} {' '.join(unique_added)}"
        else:
            expanded = query

        return ExpansionResult(
            original_query=query,
            expanded_query=expanded,
            added_terms=unique_added,
            expansion_sources={k: v for k, v in sources.items() if v},
        )

    def _expand_synonyms(self, words: Set[str]) -> List[str]:
        """Expand words with synonyms."""
        expanded = []
        for word in words:
            if word in self.synonyms:
                syns = self.synonyms[word][:self.max_expansions]
                expanded.extend(syns)
        return expanded

    def _expand_tech_terms(self, words: Set[str]) -> List[str]:
        """Expand technical terms."""
        expanded = []
        for word in words:
            if word in self.TECH_EXPANSIONS:
                terms = self.TECH_EXPANSIONS[word][:self.max_expansions]
                expanded.extend(terms)
        return expanded

    def _expand_by_intent(self, query: str) -> List[str]:
        """Add terms based on query intent."""
        expanded = []
        query_lower = query.lower()

        for pattern, terms in self.INTENT_EXPANSIONS.items():
            if re.search(pattern, query_lower):
                expanded.extend(terms[:2])

        return expanded

    def _expand_from_entities(
        self,
        words: Set[str],
        entities: Dict[str, List[str]],
    ) -> List[str]:
        """Expand using known entities."""
        expanded = []

        # If query mentions a technology, add related entities
        if "technology" in entities:
            for tech in entities["technology"]:
                if tech.lower() in words:
                    # Add co-occurring technologies
                    related = [t for t in entities["technology"] if t != tech]
                    expanded.extend(related[:2])

        # If query mentions a person, add their context
        if "person" in entities:
            for person in entities["person"]:
                if person.lower() in words:
                    expanded.append(f"about {person}")

        return expanded

    def get_expansion_stats(self, result: ExpansionResult) -> Dict[str, Any]:
        """Get statistics about an expansion."""
        return {
            "original_terms": len(result.original_query.split()),
            "added_terms": len(result.added_terms),
            "total_terms": len(result.all_terms),
            "expansion_ratio": len(result.all_terms) / max(1, len(result.original_query.split())),
            "sources_used": list(result.expansion_sources.keys()),
        }


def expand_query(
    query: str,
    entities: Optional[Dict[str, List[str]]] = None,
) -> ExpansionResult:
    """Convenience function to expand a query."""
    expander = QueryExpander()
    return expander.expand(query, entities=entities)
