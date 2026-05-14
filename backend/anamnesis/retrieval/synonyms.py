"""
FTS5 synonym expansion for mythology and thematic retrieval.

Provides a bidirectional synonym map loaded once at import time.
Query terms are expanded with OR-joined alternatives before FTS5 MATCH.
"""

from typing import Dict, FrozenSet, List, Set


# ------------------------------------------------------------------
# Raw synonym groups.  Each inner tuple is a cluster of equivalent
# terms — every term maps to every other term in its cluster.
# ------------------------------------------------------------------

_SYNONYM_GROUPS: List[tuple] = [
    # ── Mythology names ──────────────────────────────────────────
    # Norse
    ("odin", "wodan", "woden", "allfather"),
    ("thor", "donar"),
    ("freya", "freyja", "vanadis"),
    ("loki", "loptr"),
    ("tyr", "tiw", "tiwaz"),
    ("frigg", "frigga"),
    ("baldur", "baldr", "balder"),
    ("heimdall", "heimdallr"),
    ("fenrir", "fenris"),
    ("ragnarok", "ragnarök"),

    # Greek / Roman
    ("zeus", "jupiter", "jove"),
    ("hera", "juno"),
    ("poseidon", "neptune"),
    ("athena", "minerva"),
    ("ares", "mars"),
    ("aphrodite", "venus"),
    ("hermes", "mercury"),
    ("artemis", "diana"),
    ("hephaestus", "vulcan"),
    ("demeter", "ceres"),
    ("dionysus", "bacchus"),
    ("hades", "pluto"),
    ("persephone", "proserpina"),
    ("apollo", "phoebus"),
    ("eros", "cupid"),
    ("kronos", "cronus", "saturn"),

    # Hindu
    ("indra", "sakra"),
    ("shiva", "mahadeva", "rudra"),
    ("vishnu", "narayana", "hari"),
    ("brahma", "prajapati"),
    ("lakshmi", "sri"),
    ("saraswati", "sarasvati"),
    ("ganesha", "ganesh", "ganapati"),
    ("hanuman", "anjaneya"),
    ("kali", "kalika"),
    ("durga", "parvati", "uma"),

    # Egyptian
    ("ra", "amun-ra", "amen-ra"),
    ("osiris", "wesir"),
    ("isis", "aset"),
    ("anubis", "anpu"),
    ("thoth", "djehuty"),
    ("horus", "hor"),
    ("set", "seth", "sutekh"),
    ("bastet", "bast"),
    ("sekhmet", "sachmis"),

    # Mesopotamian
    ("marduk", "bel"),
    ("ishtar", "inanna"),
    ("enlil", "ellil"),
    ("enki", "ea"),
    ("tiamat", "tamtu"),

    # Celtic
    ("dagda", "eochaid", "ollathair"),
    ("brigid", "brigit", "bride"),
    ("morrigan", "morrigu"),
    ("lugh", "lug"),
    ("cernunnos", "horned god"),

    # Japanese
    ("amaterasu", "ohirume"),
    ("susanoo", "susano-o"),
    ("tsukuyomi", "tsukiyomi"),
    ("izanagi", "izanaki"),
    ("izanami", "izanami-no-mikoto"),

    # Chinese
    ("jade emperor", "yu huang"),
    ("guanyin", "kuan yin", "avalokitesvara"),
    ("sun wukong", "monkey king"),

    # ── Thematic clusters ────────────────────────────────────────
    ("death", "mortality", "afterlife", "underworld"),
    ("creation", "origin", "genesis", "cosmogony"),
    ("flood", "deluge", "cataclysm"),
    ("trickster", "shapeshifter", "deceiver"),
    ("hero", "champion", "warrior"),
    ("prophecy", "oracle", "divination", "seer"),
    ("sacrifice", "offering", "immolation"),
    ("fate", "destiny", "wyrd", "moira"),
    ("rebirth", "resurrection", "reincarnation", "renewal"),
    ("chaos", "void", "abyss", "primordial"),
    ("sacred", "holy", "divine", "hallowed"),
    ("quest", "journey", "pilgrimage", "odyssey"),
    ("dragon", "serpent", "wyrm"),
    ("world tree", "yggdrasil", "axis mundi"),
    ("heaven", "paradise", "elysium", "valhalla", "nirvana"),
    ("hell", "tartarus", "naraka", "hel", "duat"),
    ("magic", "sorcery", "witchcraft", "enchantment"),
    ("spirit", "ghost", "phantom", "shade"),
    ("demon", "devil", "fiend", "rakshasa", "oni"),
    ("giant", "titan", "jotun", "asura"),
]


def _build_synonym_map(
    groups: List[tuple],
) -> Dict[str, FrozenSet[str]]:
    """
    Build a bidirectional synonym lookup from synonym groups.

    Each term maps to a frozenset of *all other* terms in its cluster.
    Multi-word terms (e.g. "monkey king") are stored as-is (lowered).
    """
    mapping: Dict[str, Set[str]] = {}
    for group in groups:
        normed = [t.lower() for t in group]
        for term in normed:
            if term not in mapping:
                mapping[term] = set()
            mapping[term].update(t for t in normed if t != term)
    return {k: frozenset(v) for k, v in mapping.items()}


# Module-level singleton — built once at import time.
SYNONYM_MAP: Dict[str, FrozenSet[str]] = _build_synonym_map(_SYNONYM_GROUPS)


def get_synonyms(term: str) -> FrozenSet[str]:
    """
    Return synonyms for *term* (case-insensitive).

    Returns an empty frozenset if no synonyms are known.
    """
    return SYNONYM_MAP.get(term.lower(), frozenset())


def expand_terms(words: List[str]) -> List[str]:
    """
    Given a list of query words, return the original words plus any
    synonym expansions.  Duplicates are removed; order is originals first.
    """
    seen: Set[str] = set()
    result: List[str] = []
    for w in words:
        low = w.lower()
        if low not in seen:
            seen.add(low)
            result.append(low)
    # Second pass: add synonyms
    for w in words:
        for syn in get_synonyms(w):
            if syn not in seen:
                seen.add(syn)
                result.append(syn)
    return result
