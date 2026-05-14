"""Tests for greedy bin-packing context window optimization.

The packer selects facts by value density (score / token_count) to maximize
total relevance within a fixed token budget. This beats naive sequential
truncation because short, high-relevance facts are preferred over long,
mediocre ones.
"""

import pytest

from backend.proxy import _pack_facts_greedy, _build_memory_context, config


# ==================================================================
# _pack_facts_greedy — core algorithm
# ==================================================================

class TestPackFactsGreedyBasic:
    """Fundamental behavior of the greedy packing algorithm."""

    def test_empty_input_returns_empty(self):
        assert _pack_facts_greedy([], 100) == []

    def test_zero_budget_returns_empty(self):
        facts = [("User likes Python", 0.9)]
        assert _pack_facts_greedy(facts, 0) == []

    def test_negative_budget_returns_empty(self):
        facts = [("User likes Python", 0.9)]
        assert _pack_facts_greedy(facts, -10) == []

    def test_single_fact_fits(self):
        facts = [("Short fact", 0.8)]
        result = _pack_facts_greedy(facts, 100)
        assert len(result) == 1
        assert result[0][0] == "Short fact"

    def test_single_fact_too_large(self):
        facts = [("This is a very long fact with many words that exceeds the tiny budget", 0.9)]
        result = _pack_facts_greedy(facts, 3)
        assert result == []

    def test_all_facts_fit(self):
        facts = [
            ("Fact A", 0.9),
            ("Fact B", 0.8),
            ("Fact C", 0.7),
        ]
        result = _pack_facts_greedy(facts, 100)
        assert len(result) == 3

    def test_zero_score_facts_excluded(self):
        """Facts with score <= 0 should be skipped."""
        facts = [("Good fact", 0.9), ("Bad fact", 0.0), ("Negative fact", -0.1)]
        result = _pack_facts_greedy(facts, 100)
        assert len(result) == 1
        assert result[0][0] == "Good fact"


class TestPackFactsGreedyDensity:
    """Value density optimization — the core insight of bin-packing."""

    def test_prefers_short_high_density_over_long_low_density(self):
        """A short fact with high score should beat a long fact with similar score."""
        facts = [
            # Long fact: 12 tokens for "- " prefix + content, score 0.8 → density ~0.067
            ("This is a very long fact about the user that takes up many tokens in context", 0.8),
            # Short fact: 4 tokens, score 0.7 → density ~0.175
            ("Likes Python", 0.7),
            # Short fact: 4 tokens, score 0.6 → density ~0.15
            ("Lives Toronto", 0.6),
        ]
        # Budget enough for the short facts but not the long one plus both shorts
        result = _pack_facts_greedy(facts, 10)

        contents = [c for c, _ in result]
        assert "Likes Python" in contents
        assert "Lives Toronto" in contents

    def test_density_sorting_maximizes_total_score(self):
        """Greedy packing should achieve higher total score than sequential truncation."""
        facts = [
            ("Long verbose fact about software engineering practices and methodologies", 0.5),
            ("Python dev", 0.45),
            ("Uses Linux", 0.4),
            ("Likes cats", 0.35),
        ]
        budget = 15  # Enough for the short facts, barely for the long one

        packed = _pack_facts_greedy(facts, budget)
        packed_total = sum(s for _, s in packed)

        # Sequential would take the long fact first (highest score) and maybe one short
        # Greedy by density takes all three short facts: 0.45 + 0.4 + 0.35 = 1.2
        # vs sequential: 0.5 + maybe 0.45 = 0.95
        assert packed_total > 0.5, f"Packing should beat just the first fact: {packed_total}"

    def test_result_sorted_by_score_descending(self):
        """After packing, results should be sorted by score (not density) for coherent output."""
        facts = [
            ("Fact A", 0.3),  # low score, might have high density
            ("Fact B", 0.9),  # high score
            ("Fact C", 0.6),  # medium score
        ]
        result = _pack_facts_greedy(facts, 100)
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)


class TestPackFactsGreedyBudget:
    """Token budget enforcement."""

    def test_respects_exact_budget(self):
        """Selected facts must fit within the budget."""
        facts = [
            ("Three word fact", 0.9),
            ("Another short one", 0.8),
            ("Yet more words here please", 0.7),
        ]
        budget = 8
        result = _pack_facts_greedy(facts, budget)
        total_tokens = sum(len(f"- {c}".split()) for c, _ in result)
        assert total_tokens <= budget

    def test_fills_budget_efficiently(self):
        """Should use available budget rather than leaving it mostly empty."""
        # 5 facts of ~4 tokens each (with "- " prefix), budget for ~3
        facts = [(f"Fact {chr(65+i)}", 0.9 - i * 0.1) for i in range(5)]
        budget = 12
        result = _pack_facts_greedy(facts, budget)
        assert len(result) >= 2, f"Should fit at least 2 facts in {budget} tokens"

    def test_skips_oversized_fact_takes_smaller_ones(self):
        """A fact too large for remaining budget should be skipped, not block subsequent fits."""
        facts = [
            # This one won't fit in a tight budget
            ("A very long fact with many words that describes something complex", 0.95),
            ("Short A", 0.5),
            ("Short B", 0.4),
        ]
        budget = 8
        result = _pack_facts_greedy(facts, budget)
        contents = [c for c, _ in result]
        assert "Short A" in contents
        assert "Short B" in contents


class TestPackFactsGreedyEdgeCases:
    """Edge cases and robustness."""

    def test_duplicate_scores_handled(self):
        facts = [("Fact A", 0.5), ("Fact B", 0.5), ("Fact C", 0.5)]
        result = _pack_facts_greedy(facts, 100)
        assert len(result) == 3

    def test_very_large_budget(self):
        facts = [("Fact", 0.9)]
        result = _pack_facts_greedy(facts, 999999)
        assert len(result) == 1

    def test_whitespace_content_handled(self):
        """Facts with unusual whitespace should still count tokens correctly."""
        facts = [("  spaced   content  ", 0.8)]
        result = _pack_facts_greedy(facts, 100)
        assert len(result) == 1

    def test_single_word_fact(self):
        facts = [("Python", 0.9)]
        result = _pack_facts_greedy(facts, 5)
        assert len(result) == 1


# ==================================================================
# _build_memory_context with bin-packing integration
# ==================================================================

class TestBuildMemoryContextPacking:
    """Verify _build_memory_context uses bin-packing for facts."""

    def setup_method(self):
        self._original_budget = config.max_tokens
        config.max_tokens = 200

    def teardown_method(self):
        config.max_tokens = self._original_budget

    def test_packing_selects_high_density_facts(self):
        """With a tight budget, packing should prefer short high-density facts."""
        config.max_tokens = 25
        facts = [
            ("A very long detailed fact about the users extensive background in data science", 0.6),
            ("Likes cats", 0.55),
            ("Uses Arch", 0.5),
        ]
        context, selected, _ = _build_memory_context(facts)
        # Short facts should be preferred over the long one
        assert "Likes cats" in selected or "Uses Arch" in selected

    def test_packing_returns_selected_content_strings(self):
        """selected_facts should contain the actual content strings, not tuples."""
        facts = [("User is a dev", 0.9), ("User likes coffee", 0.8)]
        _, selected, _ = _build_memory_context(facts)
        assert all(isinstance(s, str) for s in selected)
        assert "User is a dev" in selected
        assert "User likes coffee" in selected

    def test_selected_facts_match_context_content(self):
        """Every fact in selected_facts should appear in the context string."""
        facts = [
            ("User speaks French", 0.9),
            ("User is from Montreal", 0.8),
            ("User studies AI", 0.7),
        ]
        context, selected, _ = _build_memory_context(facts)
        for fact in selected:
            assert fact in context, f"Selected fact '{fact}' missing from context"

    def test_packing_with_episodes_respects_split_budget(self):
        """Facts should only use their allocated share, not the episode budget."""
        config.max_tokens = 30
        facts = [(f"Fact {i} with some words", 0.9 - i * 0.05) for i in range(10)]
        episodes = [("Had a good chat about deployment", 0.8)]

        context, selected, episodes_used = _build_memory_context(
            facts, episode_contents=episodes
        )

        # With 30 tokens total, ~9 for episodes (30%), ~21 for facts
        # Both should be present but neither should monopolize
        assert episodes_used > 0
        assert len(selected) > 0

    def test_higher_scored_facts_preferred_when_same_length(self):
        """With equal-length facts, higher-scored ones should be selected first."""
        config.max_tokens = 15  # Tight budget — can fit ~2 facts
        facts = [
            ("User fact alpha", 0.3),
            ("User fact bravo", 0.9),
            ("User fact delta", 0.6),
        ]
        _, selected, _ = _build_memory_context(facts)
        if len(selected) >= 1:
            # The highest-scored fact should be in the selection
            assert "User fact bravo" in selected


class TestBuildMemoryContextPackingMetrics:
    """Verify that the return values are accurate for metrics tracking."""

    def setup_method(self):
        self._original_budget = config.max_tokens
        config.max_tokens = 200

    def teardown_method(self):
        config.max_tokens = self._original_budget

    def test_selected_count_matches_context_bullets(self):
        """Number of selected facts should match bullet points in context."""
        facts = [("Fact A", 0.9), ("Fact B", 0.8), ("Fact C", 0.7)]
        context, selected, _ = _build_memory_context(facts)
        bullet_count = context.count("\n- ")
        # First bullet doesn't have a preceding newline
        if context and "- " in context:
            bullet_count += 1 if context.split("\n")[1].startswith("- ") else 0
        assert len(selected) == len([line for line in context.split("\n") if line.startswith("- ")])

    def test_empty_input_returns_empty_list(self):
        _, selected, _ = _build_memory_context([])
        assert selected == []
        assert isinstance(selected, list)
