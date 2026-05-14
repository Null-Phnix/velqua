from types import SimpleNamespace

import pytest
from fastapi.responses import StreamingResponse

from backend import proxy


class DummyProvider:
    name = "dummy"

    def __init__(self):
        self.captured_messages = None

    def inject_memory(self, messages, context):
        self.captured_messages = [{"role": "system", "content": context}, *messages]
        return self.captured_messages

    @property
    def config(self):
        return SimpleNamespace(base_url="http://dummy")

    def get_auth_headers(self):
        return {}


@pytest.fixture(autouse=True)
def restore_config_tokens():
    original = proxy.config.max_tokens
    yield
    proxy.config.max_tokens = original


def test_facts_are_injected_into_system_prompt_correctly(monkeypatch):
    provider = DummyProvider()

    monkeypatch.setattr(
        proxy,
        "_retrieve_relevant_facts",
        lambda query: (["I prefer concise answers.", "I live in Berlin."], "fts"),
    )
    monkeypatch.setattr(proxy, "_retrieve_relevant_episodes", lambda query: [])
    monkeypatch.setattr(proxy.learner, "learn_from_message", lambda *args, **kwargs: None)
    monkeypatch.setattr(proxy.learner, "learn_from_assistant_message", lambda *args, **kwargs: None)

    body = {
        "messages": [{"role": "user", "content": "Where should I travel next?"}],
        "stream": False,
        "model": "test-model",
    }

    async def fake_forward(body, stream, metadata, provider):
        return {"ok": True, "body": body, "metadata": metadata}

    monkeypatch.setattr(proxy, "_forward_openai_compat", fake_forward)

    result = pytest.run(asyncio=False) if False else None  # no-op to satisfy static tools

    import asyncio
    response = asyncio.run(
        proxy._handle_chat_request(body, source="openai", provider=provider, request=None)
    )

    system_message = response["body"]["messages"][0]
    assert system_message["role"] == "system"
    assert "Context about the user:" in system_message["content"]
    assert "- I prefer concise answers." in system_message["content"]
    assert "- I live in Berlin." in system_message["content"]
    assert response["metadata"]["facts_injected"] == 2
    assert response["metadata"]["episodes_injected"] == 0


def test_token_budget_is_respected_never_exceeded():
    proxy.config.max_tokens = 10
    facts = [
        "one two three four five six",
        "seven eight nine ten eleven",
        "tiny fact",
    ]

    context, facts_used, episodes_used = proxy._build_memory_context(facts, episode_contents=[])

    assert episodes_used == 0
    assert facts_used >= 1
    assert len(context.split()) <= proxy.config.max_tokens


def test_empty_fact_store_returns_unmodified_prompt(monkeypatch):
    monkeypatch.setattr(proxy, "_retrieve_relevant_facts", lambda query: ([], "fts"))
    monkeypatch.setattr(proxy, "_retrieve_relevant_episodes", lambda query: [])

    prompt = "Tell me about Rust lifetimes."
    enhanced, metadata = proxy.inject_memory(prompt, max_tokens=25)

    assert enhanced == prompt
    assert metadata["facts_injected"] == 0
    assert metadata["episodes_injected"] == 0


@pytest.mark.asyncio
async def test_streaming_responses_work_with_injected_context(monkeypatch):
    provider = DummyProvider()

    monkeypatch.setattr(proxy, "_retrieve_relevant_facts", lambda query: (["I use Neovim."], "fts"))
    monkeypatch.setattr(proxy, "_retrieve_relevant_episodes", lambda query: [])
    monkeypatch.setattr(proxy.learner, "learn_from_message", lambda *args, **kwargs: None)
    monkeypatch.setattr(proxy.learner, "learn_from_assistant_message", lambda *args, **kwargs: None)

    async def fake_forward(body, stream, metadata, provider):
        async def gen():
            yield b'{"delta":"hello"}\n'
            yield b'{"delta":"world"}\n'
        return StreamingResponse(gen(), media_type="text/event-stream")

    monkeypatch.setattr(proxy, "_forward_openai_compat", fake_forward)

    body = {
        "messages": [{"role": "user", "content": "Continue"}],
        "stream": True,
        "model": "test-model",
    }

    response = await proxy._handle_chat_request(body, source="openai", provider=provider, request=None)

    assert isinstance(response, StreamingResponse)
    assert provider.captured_messages is not None
    assert provider.captured_messages[0]["role"] == "system"
    assert "I use Neovim." in provider.captured_messages[0]["content"]


def test_episode_injection_differs_from_fact_injection():
    proxy.config.max_tokens = 40

    context, facts_used, episodes_used = proxy._build_memory_context(
        ["I prefer dark mode."],
        episode_contents=[("Yesterday we debugged a websocket timeout together.", 0.92)],
    )

    assert "[Recent experiences:]" in context
    assert "[Known facts:]" in context
    assert "- Yesterday we debugged a websocket timeout together." in context
    assert "- I prefer dark mode." in context
    assert episodes_used == 1
    assert facts_used == 1
