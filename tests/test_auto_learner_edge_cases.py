"""Edge-case coverage for AutoLearner behavior."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from anamnesis import Anamnesis
from backend.auto_learner import AutoLearner


@pytest.fixture
def learner_with_memory(tmp_path: Path):
    os.environ["VELQUA_DB_PATH"] = str(tmp_path / "learner_edge.db")
    memory = Anamnesis(str(tmp_path / "learner_edge.db"))
    learner = AutoLearner(memory)
    return learner, memory


@pytest.mark.asyncio
async def test_learn_from_message_noop_when_disabled(learner_with_memory):
    learner, _memory = learner_with_memory
    learner.enabled = False
    before_learned = learner.facts_learned
    result = await learner.learn_from_message("My favorite editor is Neovim", source="test")
    # learn_from_message returns None (no explicit return) when disabled
    assert result is None
    assert learner.facts_learned == before_learned


@pytest.mark.asyncio
async def test_learn_from_message_empty_input_returns_empty(learner_with_memory):
    learner, _memory = learner_with_memory
    result = await learner.learn_from_message("", source="test")
    # Empty text produces no facts; function returns None implicitly
    assert result is None


@pytest.mark.asyncio
async def test_learn_from_assistant_message_empty_returns_empty(learner_with_memory):
    learner, _memory = learner_with_memory
    result = await learner.learn_from_assistant_message("", source="assistant")
    assert result is None


@pytest.mark.asyncio
async def test_learn_from_assistant_message_handles_extractor_failure(monkeypatch, learner_with_memory):
    learner, _memory = learner_with_memory

    def _boom(text):
        raise RuntimeError("extractor failed")

    # Monkeypatch the module-level extraction function (AutoLearner has no .extractor attribute)
    monkeypatch.setattr("backend.auto_learner.extract_facts_from_assistant", _boom)
    result = await learner.learn_from_assistant_message(
        "You told me you use Python daily.", source="assistant"
    )
    # Exception is caught internally; function returns None
    assert result is None


@pytest.mark.asyncio
async def test_learn_from_message_handles_extractor_failure(monkeypatch, learner_with_memory):
    learner, _memory = learner_with_memory

    def _boom(text):
        raise RuntimeError("extractor failed")

    monkeypatch.setattr("backend.auto_learner.extract_facts_from_text", _boom)
    result = await learner.learn_from_message("I work with FastAPI every day.", source="user")
    assert result is None


def test_get_stats_shape(learner_with_memory):
    learner, _memory = learner_with_memory
    stats = learner.get_stats()
    # Actual stat keys from AutoLearner.get_stats()
    assert "enabled" in stats
    assert "facts_learned" in stats
    assert "duplicates_seen" in stats
    assert "facts_rejected" in stats
    assert "contradictions_found" in stats
