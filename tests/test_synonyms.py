"""Tests for FTS5 synonym expansion."""

import pytest

from backend.anamnesis.retrieval.synonyms import (
    SYNONYM_MAP,
    _SYNONYM_GROUPS,
    expand_terms,
    get_synonyms,
)


# ── Synonym map construction ────────────────────────────────────


class TestSynonymMap:
    """Verify the module-level synonym map is built correctly."""

    def test_map_is_not_empty(self):
        assert len(SYNONYM_MAP) > 0

    def test_all_group_terms_present(self):
        """Every term in every group should appear as a key."""
        for group in _SYNONYM_GROUPS:
            for term in group:
                assert term.lower() in SYNONYM_MAP, f"{term!r} missing from map"

    def test_bidirectional(self):
        """If A maps to B, then B maps to A."""
        for term, syns in SYNONYM_MAP.items():
            for syn in syns:
                assert term in SYNONYM_MAP[syn], (
                    f"{term!r} -> {syn!r} but {syn!r} does not map back"
                )

    def test_no_self_reference(self):
        """No term should list itself as a synonym."""
        for term, syns in SYNONYM_MAP.items():
            assert term not in syns, f"{term!r} maps to itself"


# ── Mythology name lookups ───────────────────────────────────────


class TestMythologyNames:
    """Spot-check required mythology name clusters."""

    @pytest.mark.parametrize(
        "term, expected_synonym",
        [
            ("odin", "wodan"),
            ("wodan", "odin"),
            ("woden", "odin"),
            ("odin", "allfather"),
            ("zeus", "jupiter"),
            ("jupiter", "zeus"),
            ("jove", "zeus"),
            ("indra", "sakra"),
            ("sakra", "indra"),
            ("hera", "juno"),
            ("poseidon", "neptune"),
            ("athena", "minerva"),
        ],
    )
    def test_name_synonym(self, term, expected_synonym):
        syns = get_synonyms(term)
        assert expected_synonym in syns

    def test_case_insensitive(self):
        assert get_synonyms("Odin") == get_synonyms("odin")
        assert get_synonyms("ZEUS") == get_synonyms("zeus")

    def test_unknown_term_returns_empty(self):
        result = get_synonyms("xyznonexistent")
        assert result == frozenset()
        assert isinstance(result, frozenset)


# ── Thematic cluster lookups ─────────────────────────────────────


class TestThemeClusters:
    """Spot-check required thematic clusters."""

    @pytest.mark.parametrize(
        "term, expected_synonym",
        [
            ("death", "mortality"),
            ("death", "afterlife"),
            ("mortality", "death"),
            ("creation", "origin"),
            ("creation", "genesis"),
            ("origin", "cosmogony"),
            ("fate", "destiny"),
            ("fate", "wyrd"),
            ("rebirth", "resurrection"),
            ("chaos", "void"),
            ("quest", "odyssey"),
        ],
    )
    def test_theme_synonym(self, term, expected_synonym):
        syns = get_synonyms(term)
        assert expected_synonym in syns


# ── expand_terms ─────────────────────────────────────────────────


class TestExpandTerms:
    """Verify query-level expansion produces correct FTS-ready word lists."""

    def test_no_expansion_for_unknown(self):
        result = expand_terms(["hello", "world"])
        assert result == ["hello", "world"]

    def test_expands_known_term(self):
        result = expand_terms(["odin"])
        assert "odin" in result
        assert "wodan" in result
        assert "woden" in result
        assert "allfather" in result

    def test_originals_come_first(self):
        result = expand_terms(["zeus", "and", "heroes"])
        # Original words should be the first entries
        assert result[0] == "zeus"
        assert result[1] == "and"

    def test_no_duplicates(self):
        result = expand_terms(["odin", "wodan"])
        assert len(result) == len(set(result))

    def test_mixed_known_and_unknown(self):
        result = expand_terms(["tell", "me", "about", "death"])
        # 'death' should expand; others should pass through
        assert "mortality" in result
        assert "afterlife" in result
        assert "tell" in result

    def test_preserves_lowercase(self):
        result = expand_terms(["Zeus"])
        assert all(t == t.lower() for t in result)

    def test_empty_input(self):
        assert expand_terms([]) == []


# ── Integration: synonym expansion in FTS query path ─────────────


class TestFTSIntegration:
    """
    Verify that HybridRetriever uses synonym expansion on the FTS path.

    Uses a real SQLite backend with FTS5 tables to confirm that synonym
    terms actually surface matching rows.
    """

    @pytest.fixture
    def backend(self, tmp_path):
        from backend.anamnesis.stores.sqlite_backend import SQLiteBackend

        return SQLiteBackend(str(tmp_path / "test.db"))

    @pytest.fixture
    def retriever(self, backend):
        from backend.anamnesis.retrieval.hybrid import HybridRetriever, SearchMode

        return HybridRetriever(sqlite_backend=backend)

    def _store_fact(self, backend, fact_id, content):
        """Helper to store a fact with FTS indexing."""
        backend.save_fact(
            {
                "id": fact_id,
                "content": content,
                "fact_type": "mythology",
                "confidence": 1.0,
                "importance": 0.8,
                "confirmation_count": 1,
                "first_learned": "2025-01-01T00:00:00",
            }
        )

    def test_synonym_finds_wodan_when_searching_odin(self, backend, retriever):
        """Searching 'odin' should find a fact that only contains 'wodan'."""
        from backend.anamnesis.retrieval.hybrid import SearchMode

        self._store_fact(backend, "f1", "Wodan is a Norse deity")
        results = retriever.search("odin", mode=SearchMode.TEXT_ONLY)
        assert any("Wodan" in r.content for r in results)

    def test_synonym_finds_jupiter_when_searching_zeus(self, backend, retriever):
        """Searching 'zeus' should find a fact that only contains 'jupiter'."""
        from backend.anamnesis.retrieval.hybrid import SearchMode

        self._store_fact(backend, "f2", "Jupiter rules the Roman pantheon")
        results = retriever.search("zeus", mode=SearchMode.TEXT_ONLY)
        assert any("Jupiter" in r.content for r in results)

    def test_synonym_finds_mortality_when_searching_death(self, backend, retriever):
        """Thematic expansion: 'death' should find 'mortality'."""
        from backend.anamnesis.retrieval.hybrid import SearchMode

        self._store_fact(backend, "f3", "Mortality is central to the Gilgamesh epic")
        results = retriever.search("death", mode=SearchMode.TEXT_ONLY)
        assert any("Mortality" in r.content for r in results)

    def test_synonym_finds_genesis_when_searching_creation(self, backend, retriever):
        from backend.anamnesis.retrieval.hybrid import SearchMode

        self._store_fact(backend, "f4", "The genesis of the world from chaos")
        results = retriever.search("creation", mode=SearchMode.TEXT_ONLY)
        assert any("genesis" in r.content for r in results)

    def test_original_term_still_matches(self, backend, retriever):
        """Expansion should not break exact matches."""
        from backend.anamnesis.retrieval.hybrid import SearchMode

        self._store_fact(backend, "f5", "Odin hung from Yggdrasil for nine nights")
        results = retriever.search("odin", mode=SearchMode.TEXT_ONLY)
        assert any("Odin" in r.content for r in results)

    def test_no_expansion_for_plain_query(self, backend, retriever):
        """A query with no synonym hits should still work normally."""
        from backend.anamnesis.retrieval.hybrid import SearchMode

        self._store_fact(backend, "f6", "The weather today is sunny")
        results = retriever.search("weather sunny", mode=SearchMode.TEXT_ONLY)
        assert any("sunny" in r.content for r in results)
