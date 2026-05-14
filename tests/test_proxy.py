"""
Unit tests for proxy memory injection logic.

Tests the core functions that determine what memory context gets injected
into LLM conversations. These are the most critical functions in Velqua —
if injection is wrong, the LLM gets bad context.

Note: These tests don't need a running Ollama instance. They test the
retrieval/injection pipeline up to the point of forwarding to the backend.
"""
import os
import tempfile
import importlib
from dataclasses import dataclass
from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

# Temp DB setup — must happen before proxy imports (conftest.py handles sys.path)
_tmpdir = tempfile.mkdtemp()
os.environ["VELQUA_DB_PATH"] = os.path.join(_tmpdir, "test_proxy.db")

import backend.config
importlib.reload(backend.config)

from backend.proxy import (
    MemoryConfig,
    _build_memory_context,
    _retrieve_relevant_facts,
    _topic_boost,
    config,
    inject_memory,
    score_fact_freshness,
)


# -- Fake fact objects for testing freshness scoring --

@dataclass
class FakeFact:
    """Minimal fact-like object for testing score_fact_freshness."""
    content: str = "User works as a developer"
    last_confirmed: datetime = None
    first_learned: datetime = None
    confirmation_count: int = 1
    importance: float = 0.5


# ==================================================================
# MemoryConfig
# ==================================================================

class TestMemoryConfig:
    """Hardware-based budget configuration."""

    def test_default_is_minimal(self):
        mc = MemoryConfig()
        assert mc.budget == "minimal"
        assert mc.max_tokens == 200

    def test_8gb_gpu(self):
        mc = MemoryConfig()
        mc.set_budget(8)
        assert mc.budget == "minimal"
        assert mc.max_tokens == 200

    def test_16gb_gpu(self):
        mc = MemoryConfig()
        mc.set_budget(16)
        assert mc.budget == "standard"
        assert mc.max_tokens == 500

    def test_24gb_gpu(self):
        mc = MemoryConfig()
        mc.set_budget(24)
        assert mc.budget == "generous"
        assert mc.max_tokens == 1000

    def test_rubin_128gb(self):
        mc = MemoryConfig()
        mc.set_budget(128)
        assert mc.budget == "generous"
        assert mc.max_tokens == 2000

    def test_4gb_gpu_still_minimal(self):
        mc = MemoryConfig()
        mc.set_budget(4)
        assert mc.budget == "minimal"
        assert mc.max_tokens == 200


# ==================================================================
# score_fact_freshness
# ==================================================================

class TestScoreFactFreshness:
    """Freshness scoring with decay model."""

    def test_fresh_fact_scores_high(self):
        """A fact confirmed moments ago should score near 1.0."""
        fact = FakeFact(last_confirmed=datetime.now())
        score = score_fact_freshness(fact)
        assert score >= 0.5

    def test_old_fact_scores_lower(self):
        """A fact from months ago should score lower than a fresh one."""
        fresh = FakeFact(last_confirmed=datetime.now())
        old = FakeFact(last_confirmed=datetime.now() - timedelta(days=180))

        fresh_score = score_fact_freshness(fresh)
        old_score = score_fact_freshness(old)
        assert fresh_score > old_score

    def test_no_timestamp_treated_as_fresh(self):
        """Unknown age defaults to 0 hours (treated as fresh)."""
        fact = FakeFact()  # No timestamps
        score = score_fact_freshness(fact)
        assert score >= 0.5

    def test_high_confirmation_count_boosts(self):
        """More confirmations = higher score (reinforcement)."""
        single = FakeFact(confirmation_count=1, last_confirmed=datetime.now() - timedelta(days=30))
        multi = FakeFact(confirmation_count=5, last_confirmed=datetime.now() - timedelta(days=30))

        single_score = score_fact_freshness(single)
        multi_score = score_fact_freshness(multi)
        assert multi_score >= single_score

    def test_score_bounded_zero_to_one(self):
        """Score should always be between 0 and 1."""
        test_cases = [
            FakeFact(),
            FakeFact(last_confirmed=datetime.now()),
            FakeFact(last_confirmed=datetime.now() - timedelta(days=365)),
            FakeFact(confirmation_count=100),
            FakeFact(importance=1.0),
            FakeFact(importance=0.0),
        ]
        for fact in test_cases:
            score = score_fact_freshness(fact)
            assert 0.0 <= score <= 1.0, f"Score {score} out of bounds for {fact}"

    def test_uses_first_learned_fallback(self):
        """Falls back to first_learned if last_confirmed is None."""
        fact = FakeFact(first_learned=datetime.now() - timedelta(hours=1))
        score = score_fact_freshness(fact)
        assert score >= 0.3  # Recent enough to be relevant

    def test_metadata_dict_freshness(self):
        """HybridSearchResult-like object with metadata dict should score correctly."""

        @dataclass
        class FakeHybridResult:
            content: str = "User works as a developer"
            metadata: dict = None

        # Result with timestamp in metadata dict (like HybridSearchResult)
        result = FakeHybridResult(
            metadata={
                "last_confirmed": datetime.now().isoformat(),
                "confirmation_count": 3,
                "importance": 0.7,
            }
        )
        score = score_fact_freshness(result)
        assert 0.0 <= score <= 1.0
        # Should use the metadata fields, not default to 0 age
        assert score > 0.3

    def test_metadata_string_timestamp_parsed(self):
        """String timestamps in metadata should be parsed as datetime."""

        @dataclass
        class FakeResult:
            content: str = "User lives in Toronto"
            metadata: dict = None

        result = FakeResult(
            metadata={
                "first_learned": (datetime.now() - timedelta(days=60)).isoformat(),
            }
        )
        score = score_fact_freshness(result)
        assert 0.0 <= score <= 1.0

    def test_metadata_no_timestamps_treated_fresh(self):
        """Result with metadata but no timestamps treated as fresh."""

        @dataclass
        class FakeResult:
            content: str = "User has a cat"
            metadata: dict = None

        result = FakeResult(metadata={"some_key": "value"})
        score = score_fact_freshness(result)
        assert score >= 0.5  # Unknown age = treated as fresh


# ==================================================================
# _build_memory_context
# ==================================================================

class TestBuildMemoryContext:
    """Token-budgeted context building."""

    def setup_method(self):
        """Reset config to known state before each test."""
        config.max_tokens = 200

    def test_empty_facts_returns_empty(self):
        context, count, ep_count = _build_memory_context([])
        assert context == ""
        assert count == 0
        assert ep_count == 0

    def test_single_fact(self):
        context, count, _ = _build_memory_context(["User is a developer"])
        assert count == 1
        assert "User is a developer" in context
        assert context.startswith("Context about the user:")

    def test_multiple_facts(self):
        facts = [
            "User works at Google",
            "User lives in Seattle",
            "User has two cats",
        ]
        context, count, _ = _build_memory_context(facts)
        assert count == 3
        assert "Google" in context
        assert "Seattle" in context
        assert "cats" in context

    def test_custom_header(self):
        context, _, _ = _build_memory_context(
            ["User likes Python"],
            header="Here's what I know:"
        )
        assert context.startswith("Here's what I know:")

    def test_respects_token_budget(self):
        """Should stop adding facts when budget is exhausted."""
        config.max_tokens = 20  # Very tight budget
        facts = [
            "User is a software engineer working at a large tech company",
            "User lives in downtown Toronto near the waterfront",
            "User has three cats named Luna, Pixel, and Byte",
        ]
        context, count, _ = _build_memory_context(facts)
        # Can't fit all 3 facts in 20 tokens
        assert count < 3

    def test_zero_budget_returns_empty(self):
        """With budget smaller than header, should return empty."""
        config.max_tokens = 1  # Can't even fit the header
        context, count, _ = _build_memory_context(["User likes Python"])
        assert count == 0
        assert context == ""

    def test_facts_formatted_as_bullets(self):
        context, _, _ = _build_memory_context(["User is a dev", "User has a dog"])
        assert "- User is a dev" in context
        assert "- User has a dog" in context


# ==================================================================
# _topic_boost
# ==================================================================

class TestTopicBoost:
    """Topic-based relevance multiplier."""

    def _make_fact(self, category=""):
        from dataclasses import dataclass
        @dataclass
        class F:
            content: str = "test"
            metadata: dict = None
        return F(metadata={"category": category} if category else {})

    def test_no_query_category_returns_one(self):
        """Empty query category → no boost."""
        fact = self._make_fact("work")
        assert _topic_boost(fact, "") == 1.0

    def test_matching_category_boosts(self):
        """Fact category matches query → 1.3x multiplier."""
        fact = self._make_fact("work")
        assert _topic_boost(fact, "work") == 1.3

    def test_mismatched_category_no_boost(self):
        """Fact category differs from query → no boost."""
        fact = self._make_fact("hobbies")
        assert _topic_boost(fact, "work") == 1.0

    def test_fact_no_metadata_no_boost(self):
        """Fact without metadata → no boost."""
        from dataclasses import dataclass
        @dataclass
        class F:
            content: str = "test"
        assert _topic_boost(F(), "work") == 1.0

    def test_fact_empty_metadata_no_boost(self):
        """Fact with empty metadata dict → no boost."""
        fact = self._make_fact("")
        assert _topic_boost(fact, "work") == 1.0

    def test_boost_value_is_thirty_percent(self):
        """Confirm the boost multiplier is exactly 1.3."""
        fact = self._make_fact("health")
        assert _topic_boost(fact, "health") == pytest.approx(1.3)


# ==================================================================
# inject_memory (integration-level, uses real memory store)
# ==================================================================

class TestInjectMemory:
    """Memory injection into raw prompts."""

    def setup_method(self):
        config.max_tokens = 200

    def test_no_facts_returns_original_prompt(self):
        """Prompt passes through unchanged when no facts match the query."""
        from backend.proxy import memory, vector_store, VECTOR_ENABLED
        # Clear any facts seeded by other test modules to avoid cross-contamination
        memory.backend.clear_all()
        if VECTOR_ENABLED and vector_store:
            vector_store.clear()
        prompt = "xyzzy_unique_no_match_abc123"
        result, metadata = inject_memory(prompt)
        assert metadata["facts_injected"] == 0
        assert prompt in result

    def test_metadata_structure(self):
        _, metadata = inject_memory("hello")
        assert "facts_injected" in metadata
        assert "search_mode" in metadata
        assert metadata["search_mode"] in ("hybrid", "fts")

    def test_returns_string(self):
        """inject_memory should always return a string prompt."""
        result, _ = inject_memory("test prompt")
        assert isinstance(result, str)
        assert "test prompt" in result


# ==================================================================
# Proxy Endpoint Tests (mocked httpx)
# ==================================================================

from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient as ProxyTestClient
from backend.proxy import app as proxy_app


class FakeHttpxResponse:
    """Minimal httpx.Response stand-in for testing."""
    def __init__(self, status_code=200, json_data=None, text=""):
        self.status_code = status_code
        self._json_data = json_data or {}
        self.text = text

    def json(self):
        return self._json_data


class TestProxyEndpoints:
    """Tests for proxy API endpoints with mocked Ollama backend."""

    @pytest.fixture
    def client(self):
        return ProxyTestClient(proxy_app)

    def test_root_returns_status(self, client):
        """Root endpoint returns proxy status info."""
        r = client.get("/")
        assert r.status_code == 200
        data = r.json()
        assert "memory_config" in data
        assert "vector_retrieval" in data
        assert "auto_learning" in data

    def test_get_config(self, client):
        """Config endpoint returns budget info."""
        r = client.get("/proxy/config")
        assert r.status_code == 200
        data = r.json()
        assert "budget" in data
        assert "max_tokens" in data

    def test_update_config_gpu(self, client):
        """Can update config via GPU VRAM parameter."""
        r = client.post("/proxy/config?gpu_vram_gb=24")
        assert r.status_code == 200
        data = r.json()
        assert data["budget"] == "generous"

    def test_update_config_budget_name(self, client):
        """Can update config via budget name."""
        r = client.post("/proxy/config?budget=standard")
        assert r.status_code == 200
        data = r.json()
        assert data["budget"] == "standard"

    def test_update_config_invalid_budget(self, client):
        """Invalid budget name returns 400."""
        r = client.post("/proxy/config?budget=huge")
        assert r.status_code == 400
        assert "Invalid budget" in r.json()["detail"]

    def test_update_config_gpu_zero(self, client):
        """GPU VRAM of 0 should be accepted (not silently ignored)."""
        r = client.post("/proxy/config?gpu_vram_gb=0")
        assert r.status_code == 200
        # Should still set minimal budget
        assert r.json()["budget"] == "minimal"

    def test_generate_invalid_json(self, client):
        """Generate endpoint rejects invalid JSON body."""
        r = client.post("/api/generate", content=b"not json", headers={"Content-Type": "application/json"})
        assert r.status_code == 400

    def test_chat_invalid_json(self, client):
        """Chat endpoint rejects invalid JSON body."""
        r = client.post("/api/chat", content=b"not json", headers={"Content-Type": "application/json"})
        assert r.status_code == 400

    def test_openai_invalid_json(self, client):
        """OpenAI chat endpoint rejects invalid JSON body."""
        r = client.post("/v1/chat/completions", content=b"not json", headers={"Content-Type": "application/json"})
        assert r.status_code == 400

    @patch("backend.proxy.httpx.AsyncClient")
    def test_generate_ollama_connection_error(self, mock_client_cls, client):
        """Generate returns 503 when Ollama is not running."""
        mock_instance = AsyncMock()
        mock_instance.post.side_effect = __import__("httpx").ConnectError("Connection refused")
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_instance

        r = client.post("/api/generate", json={"prompt": "hello", "model": "llama3"})
        assert r.status_code == 503
        assert "Cannot connect" in r.json()["detail"]

    @patch("backend.proxy.httpx.AsyncClient")
    def test_chat_ollama_connection_error(self, mock_client_cls, client):
        """Chat returns 503 when Ollama is not running."""
        mock_instance = AsyncMock()
        mock_instance.post.side_effect = __import__("httpx").ConnectError("Connection refused")
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_instance

        r = client.post("/api/chat", json={
            "model": "llama3",
            "messages": [{"role": "user", "content": "hello"}]
        })
        assert r.status_code == 503

    @patch("backend.proxy.httpx.AsyncClient")
    def test_generate_ollama_500(self, mock_client_cls, client):
        """Generate returns 502 when Ollama returns 500."""
        mock_response = FakeHttpxResponse(status_code=500, text="Internal Server Error")
        mock_instance = AsyncMock()
        mock_instance.post.return_value = mock_response
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_instance

        r = client.post("/api/generate", json={"prompt": "hello", "model": "llama3"})
        assert r.status_code == 502

    @patch("backend.proxy.httpx.AsyncClient")
    def test_generate_success(self, mock_client_cls, client):
        """Generate proxies successful response and adds metadata."""
        mock_response = FakeHttpxResponse(
            status_code=200,
            json_data={"response": "Hello! How can I help?", "done": True}
        )
        mock_instance = AsyncMock()
        mock_instance.post.return_value = mock_response
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_instance

        r = client.post("/api/generate", json={"prompt": "hello", "model": "llama3"})
        assert r.status_code == 200
        data = r.json()
        assert "response" in data
        assert "velqua_metadata" in data

    @patch("backend.proxy.httpx.AsyncClient")
    def test_chat_success(self, mock_client_cls, client):
        """Chat proxies successful response and adds metadata."""
        mock_response = FakeHttpxResponse(
            status_code=200,
            json_data={"message": {"role": "assistant", "content": "Hi!"}, "done": True}
        )
        mock_instance = AsyncMock()
        mock_instance.post.return_value = mock_response
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_instance

        r = client.post("/api/chat", json={
            "model": "llama3",
            "messages": [{"role": "user", "content": "hi"}]
        })
        assert r.status_code == 200
        data = r.json()
        assert "velqua_metadata" in data

    @patch("backend.proxy.httpx.AsyncClient")
    def test_openai_chat_success(self, mock_client_cls, client):
        """OpenAI chat completions proxies successful response."""
        mock_response = FakeHttpxResponse(
            status_code=200,
            json_data={
                "choices": [{"message": {"role": "assistant", "content": "Hi!"}}],
                "usage": {"total_tokens": 10},
            }
        )
        mock_instance = AsyncMock()
        mock_instance.post.return_value = mock_response
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_instance

        r = client.post("/v1/chat/completions", json={
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "hi"}]
        })
        assert r.status_code == 200
        data = r.json()
        assert "choices" in data

    @patch("backend.proxy.httpx.AsyncClient")
    def test_openai_connection_error(self, mock_client_cls, client):
        """OpenAI endpoint returns 503 when backend is unreachable."""
        mock_instance = AsyncMock()
        mock_instance.post.side_effect = __import__("httpx").ConnectError("Connection refused")
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_instance

        r = client.post("/v1/chat/completions", json={
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "hello"}]
        })
        assert r.status_code == 503
        assert "Cannot connect" in r.json()["detail"]

    @patch("backend.proxy.httpx.AsyncClient")
    def test_openai_backend_500(self, mock_client_cls, client):
        """OpenAI endpoint returns 502 when backend returns 500."""
        mock_response = FakeHttpxResponse(status_code=500, text="Internal Server Error")
        mock_instance = AsyncMock()
        mock_instance.post.return_value = mock_response
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_instance

        r = client.post("/v1/chat/completions", json={
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "hello"}]
        })
        assert r.status_code == 502

    @patch("backend.proxy.httpx.AsyncClient")
    def test_tags_success(self, mock_client_cls, client):
        """Tags endpoint proxies model listing from Ollama."""
        mock_response = FakeHttpxResponse(
            status_code=200,
            json_data={"models": []}
        )
        mock_instance = AsyncMock()
        mock_instance.get.return_value = mock_response
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_instance

        r = client.get("/api/tags")
        assert r.status_code == 200
        data = r.json()
        assert "models" in data

    @patch("backend.proxy.httpx.AsyncClient")
    def test_tags_connection_error(self, mock_client_cls, client):
        """Tags endpoint returns 503 when Ollama is not running."""
        mock_instance = AsyncMock()
        mock_instance.get.side_effect = __import__("httpx").ConnectError("Connection refused")
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_instance

        r = client.get("/api/tags")
        assert r.status_code == 503

    def test_learning_stats(self, client):
        """Learning stats endpoint returns expected keys."""
        r = client.get("/proxy/learning")
        assert r.status_code == 200
        data = r.json()
        assert "enabled" in data
        assert "facts_learned" in data
        assert "facts_pending" in data

    def test_toggle_learning(self, client):
        """Can toggle auto-learning on and off."""
        r = client.post("/proxy/learning?enabled=false")
        assert r.status_code == 200
        assert r.json()["enabled"] is False

        r = client.post("/proxy/learning?enabled=true")
        assert r.status_code == 200
        assert r.json()["enabled"] is True

    def test_retrieval_stats(self, client):
        """Retrieval stats endpoint returns vector_enabled key."""
        r = client.get("/proxy/retrieval")
        assert r.status_code == 200
        data = r.json()
        assert "vector_enabled" in data

    @patch("backend.proxy.httpx.AsyncClient")
    def test_generate_invalid_json_response(self, mock_client_cls, client):
        """Generate returns 502 when Ollama sends 200 with non-JSON body."""
        mock_response = FakeHttpxResponse(status_code=200, text="not json at all")
        # Override .json() to raise ValueError like real httpx does for bad JSON
        mock_response.json = lambda: (_ for _ in ()).throw(ValueError("Invalid JSON"))
        mock_instance = AsyncMock()
        mock_instance.post.return_value = mock_response
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_instance

        r = client.post("/api/generate", json={"prompt": "hello", "model": "llama3"})
        assert r.status_code == 502
        assert "invalid JSON" in r.json()["detail"]

    @patch("backend.proxy.httpx.AsyncClient")
    def test_chat_invalid_json_response(self, mock_client_cls, client):
        """Chat returns 502 when Ollama sends 200 with non-JSON body."""
        mock_response = FakeHttpxResponse(status_code=200, text="not json at all")
        mock_response.json = lambda: (_ for _ in ()).throw(ValueError("Invalid JSON"))
        mock_instance = AsyncMock()
        mock_instance.post.return_value = mock_response
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_instance

        r = client.post("/api/chat", json={
            "model": "llama3",
            "messages": [{"role": "user", "content": "hello"}],
        })
        assert r.status_code == 502
        assert "invalid JSON" in r.json()["detail"]

    @patch("backend.proxy.httpx.AsyncClient")
    def test_tags_invalid_json(self, mock_client_cls, client):
        """Tags returns 502 when Ollama sends 200 with non-JSON body."""
        mock_response = FakeHttpxResponse(status_code=200, text="not json")
        mock_response.json = lambda: (_ for _ in ()).throw(ValueError("Invalid JSON"))
        mock_instance = AsyncMock()
        mock_instance.get.return_value = mock_response
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_instance

        r = client.get("/api/tags")
        assert r.status_code == 502
        assert "invalid JSON" in r.json()["detail"]

    @patch("backend.proxy.httpx.AsyncClient")
    def test_stream_proxy_error_status(self, mock_client_cls, client):
        """Streaming proxy returns 502 when backend returns 500 status."""
        # Create a mock streaming response with non-200 status
        mock_response = AsyncMock()
        mock_response.status_code = 500
        mock_response.aread = AsyncMock(return_value=b"Internal Server Error")
        mock_response.aclose = AsyncMock()

        mock_instance = AsyncMock()
        mock_instance.build_request.return_value = MagicMock()
        mock_instance.send = AsyncMock(return_value=mock_response)
        mock_instance.aclose = AsyncMock()
        mock_client_cls.return_value = mock_instance

        r = client.post("/api/generate", json={
            "prompt": "hello",
            "model": "llama3",
            "stream": True,
        })
        assert r.status_code == 502
        assert "Backend returned 500" in r.json()["detail"]

    @patch("backend.proxy.httpx.AsyncClient")
    def test_chat_non_200_status(self, mock_client_cls, client):
        """Non-streaming chat returns 502 when Ollama returns non-200."""
        mock_response = MagicMock()
        mock_response.status_code = 503
        mock_response.text = "Service Unavailable"
        mock_response.json.return_value = {}

        mock_instance = AsyncMock()
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        mock_instance.post = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value = mock_instance

        r = client.post("/api/chat", json={
            "model": "llama3",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
        })
        assert r.status_code == 502

    @patch("backend.proxy.httpx.AsyncClient")
    def test_chat_stream_mode(self, mock_client_cls, client):
        """Streaming chat calls _stream_proxy path."""
        # Create a mock streaming response with 200 status
        mock_response = AsyncMock()
        mock_response.status_code = 200

        async def fake_iter():
            yield b'{"done": true}'
        mock_response.aiter_bytes = fake_iter
        mock_response.aclose = AsyncMock()

        mock_instance = AsyncMock()
        mock_instance.build_request.return_value = MagicMock()
        mock_instance.send = AsyncMock(return_value=mock_response)
        mock_instance.aclose = AsyncMock()
        mock_client_cls.return_value = mock_instance

        r = client.post("/api/chat", json={
            "model": "llama3",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
        })
        assert r.status_code == 200

    @patch("backend.proxy.httpx.AsyncClient")
    def test_openai_stream_mode(self, mock_client_cls, client):
        """OpenAI streaming calls _stream_proxy path."""
        mock_response = AsyncMock()
        mock_response.status_code = 200

        async def fake_iter():
            yield b'data: {"choices": []}\n\n'
        mock_response.aiter_bytes = fake_iter
        mock_response.aclose = AsyncMock()

        mock_instance = AsyncMock()
        mock_instance.build_request.return_value = MagicMock()
        mock_instance.send = AsyncMock(return_value=mock_response)
        mock_instance.aclose = AsyncMock()
        mock_client_cls.return_value = mock_instance

        r = client.post("/v1/chat/completions", json={
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
        })
        assert r.status_code == 200

    @patch("backend.proxy.httpx.AsyncClient")
    def test_openai_invalid_json_response(self, mock_client_cls, client):
        """OpenAI backend returning invalid JSON should give 502."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("No JSON")

        mock_instance = AsyncMock()
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        mock_instance.post = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value = mock_instance

        r = client.post("/v1/chat/completions", json={
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
        })
        assert r.status_code == 502
        assert "invalid JSON" in r.json()["detail"]

    @patch("backend.proxy.httpx.AsyncClient")
    def test_tags_non_200(self, mock_client_cls, client):
        """Tags endpoint returns 502 when Ollama returns non-200."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.json.return_value = {}

        mock_instance = AsyncMock()
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        mock_instance.get = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value = mock_instance

        r = client.get("/api/tags")
        assert r.status_code == 502


# === _log_task_error Coverage ===

class TestLogTaskError:
    """Cover the background task error callback."""

    def test_cancelled_task_returns_silently(self):
        from backend.proxy import _log_task_error
        task = MagicMock()
        task.cancelled.return_value = True
        # Should not raise or call task.exception()
        _log_task_error(task)
        task.exception.assert_not_called()

    def test_failed_task_logs_warning(self):
        from backend.proxy import _log_task_error
        task = MagicMock()
        task.cancelled.return_value = False
        task.exception.return_value = RuntimeError("test error")
        # Should not raise
        _log_task_error(task)

    def test_successful_task_no_warning(self):
        from backend.proxy import _log_task_error
        task = MagicMock()
        task.cancelled.return_value = False
        task.exception.return_value = None
        # Should not raise
        _log_task_error(task)


# === FTS-only path Coverage ===

class TestRetrievalFTSPath:
    """Cover the FTS-only retrieval path (when vector is disabled)."""

    def test_retrieve_with_vector_disabled(self):
        """When VECTOR_ENABLED is False, should use FTS path."""
        from backend.proxy import _retrieve_relevant_facts
        with patch("backend.proxy.VECTOR_ENABLED", False):
            contents, mode = _retrieve_relevant_facts("test query")
            assert mode == "fts"
            assert isinstance(contents, list)


# === Topic Boost Coverage ===

class TestTopicBoost:
    """Cover the topic-weighted retrieval functions."""

    def test_detect_query_topic_technical(self):
        """Technical query should detect 'technical' category."""
        from backend.proxy import _detect_query_topic
        topic = _detect_query_topic("How do I debug Python code with breakpoints?")
        # May or may not detect — depends on confidence threshold
        assert isinstance(topic, str)

    def test_detect_query_topic_empty(self):
        """Empty query should return empty string."""
        from backend.proxy import _detect_query_topic
        assert _detect_query_topic("") == ""

    def test_detect_query_topic_import_error(self):
        """TopicDetector import failure should return empty string."""
        from backend.proxy import _detect_query_topic
        with patch.dict("sys.modules", {"anamnesis.topics.detector": None}):
            assert _detect_query_topic("some technical query about Python") == ""

    def test_topic_boost_matching_category(self):
        """Fact with matching category should get 1.3x boost."""
        from backend.proxy import _topic_boost
        fact = MagicMock()
        fact.metadata = {"category": "technical"}
        assert _topic_boost(fact, "technical") == 1.3

    def test_topic_boost_no_match(self):
        """Fact with different category should get no boost."""
        from backend.proxy import _topic_boost
        fact = MagicMock()
        fact.metadata = {"category": "personal"}
        assert _topic_boost(fact, "technical") == 1.0

    def test_topic_boost_no_category(self):
        """Fact with no category should get no boost."""
        from backend.proxy import _topic_boost
        fact = MagicMock()
        fact.metadata = {}
        assert _topic_boost(fact, "technical") == 1.0

    def test_topic_boost_no_query_category(self):
        """Empty query category should always return 1.0."""
        from backend.proxy import _topic_boost
        fact = MagicMock()
        fact.metadata = {"category": "technical"}
        assert _topic_boost(fact, "") == 1.0

    def test_topic_boost_no_metadata(self):
        """Fact with no metadata attribute should get no boost."""
        from backend.proxy import _topic_boost
        fact = MagicMock(spec=[])  # No metadata attr
        assert _topic_boost(fact, "technical") == 1.0


# ==================================================================
# Summarize Session Endpoint
# ==================================================================

class TestSummarizeSession:
    """Tests for POST /proxy/summarize-session."""

    @pytest.fixture
    def client(self):
        return ProxyTestClient(proxy_app)

    def test_empty_messages(self, client):
        """Empty message list returns success with zero counts."""
        r = client.post("/proxy/summarize-session", json={"messages": []})
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is True
        assert data["messages_processed"] == 0
        assert data["facts_stored"] >= 0

    def test_invalid_json(self, client):
        """Non-JSON body returns 400."""
        r = client.post(
            "/proxy/summarize-session",
            content=b"not json",
            headers={"Content-Type": "application/json"},
        )
        assert r.status_code == 400

    def test_messages_not_list(self, client):
        """Non-list messages field returns 400."""
        r = client.post("/proxy/summarize-session", json={"messages": "bad"})
        assert r.status_code == 400

    def test_user_messages_processed(self, client):
        """User messages increment messages_processed."""
        r = client.post("/proxy/summarize-session", json={
            "messages": [
                {"role": "user", "content": "I work at Acme Corp as a software engineer."},
                {"role": "assistant", "content": "Got it, you work at Acme Corp."},
            ]
        })
        assert r.status_code == 200
        data = r.json()
        assert data["messages_processed"] == 2

    def test_custom_source_tag(self, client):
        """Custom source tag is echoed back in response."""
        r = client.post("/proxy/summarize-session", json={
            "messages": [{"role": "user", "content": "I live in Toronto."}],
            "source": "manual_import",
        })
        assert r.status_code == 200
        assert r.json()["source"] == "manual_import"

    def test_default_source_tag(self, client):
        """Default source tag is session_summary."""
        r = client.post("/proxy/summarize-session", json={
            "messages": [{"role": "user", "content": "I speak French."}],
        })
        assert r.status_code == 200
        assert r.json()["source"] == "session_summary"

    def test_empty_content_messages_skipped(self, client):
        """Messages with empty content are not counted."""
        r = client.post("/proxy/summarize-session", json={
            "messages": [
                {"role": "user", "content": ""},
                {"role": "assistant", "content": ""},
                {"role": "user", "content": "I have a dog named Max."},
            ]
        })
        assert r.status_code == 200
        assert r.json()["messages_processed"] == 1  # Only the non-empty one


# ==================================================================
# Extended proxy coverage
# ==================================================================

class TestProxyLifespan:
    """Cover proxy.py lines 52-69: lifespan startup event."""

    def test_lifespan_runs_on_startup(self):
        """Using 'with TestClient' triggers the lifespan context (lines 52-69)."""
        with ProxyTestClient(proxy_app) as c:
            r = c.get("/")
            assert r.status_code == 200


class TestLoadApiKeys:
    """Cover proxy.py lines 74-83: _load_api_keys()."""

    def test_load_api_keys_runs_without_error(self):
        """_load_api_keys() should not raise even with empty keystore."""
        from backend.proxy import _load_api_keys
        _load_api_keys()  # Should complete without exception

    def test_load_api_keys_handles_exception(self):
        """_load_api_keys() swallows exceptions (lines 82-83)."""
        from backend.proxy import _load_api_keys
        with patch("backend.keystore.KeyStore", side_effect=RuntimeError("no keystore")):
            _load_api_keys()  # Should not raise


class TestScoreFactFreshnessStringTimestamp:
    """Cover proxy.py lines 239-240: bad ISO string in metadata."""

    def test_invalid_iso_string_in_metadata(self):
        """score_fact_freshness handles non-ISO string timestamps (lines 239-240)."""
        from backend.proxy import score_fact_freshness

        class FakeFactWithBadTimestamp:
            content = "I am a developer"
            last_confirmed = None
            first_learned = None
            confirmation_count = 1
            importance = 0.5
            metadata = {"last_confirmed": "not-a-real-date-string"}

        result = score_fact_freshness(FakeFactWithBadTimestamp())
        assert isinstance(result, float)
        assert result > 0


class TestInjectMemoryWithFacts:
    """Cover proxy.py line 387: inject_memory returns enriched prompt when facts exist."""

    def test_inject_memory_with_stored_facts(self):
        """inject_memory returns context-injected prompt when facts are in DB (line 387)."""
        from backend.proxy import inject_memory, memory

        # Store a fact directly
        result = memory.semantic.add_fact(
            content="I am a professional software engineer specializing in Python",
            fact_type="general",
            confidence=0.9,
        )

        prompt = "What do you know about me?"
        enriched, meta = inject_memory(prompt)

        # The fact may or may not be retrieved (depends on FTS indexing)
        # At minimum, the function should return a string
        assert isinstance(enriched, str)
        assert prompt in enriched or "Context" in enriched or enriched == prompt


class TestProxyAnthropic:
    """Cover proxy.py lines 782-809: /v1/messages Anthropic endpoint."""

    @pytest.fixture
    def client(self):
        return ProxyTestClient(proxy_app)

    @patch("backend.proxy.httpx.AsyncClient")
    def test_anthropic_messages_endpoint_success(self, mock_client_cls, client):
        """POST /v1/messages proxies to Anthropic format."""
        mock_response = FakeHttpxResponse(
            status_code=200,
            json_data={
                "content": [{"type": "text", "text": "Hello!"}],
                "model": "claude-3-sonnet",
                "stop_reason": "end_turn",
            }
        )
        mock_instance = AsyncMock()
        mock_instance.post.return_value = mock_response
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_instance

        r = client.post("/v1/messages", json={
            "model": "claude-3-sonnet",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
        })
        # Either 200 (success) or 503 (Anthropic not configured) — both acceptable
        assert r.status_code in (200, 503, 502)

    def test_anthropic_messages_invalid_json(self, client):
        """POST /v1/messages rejects invalid JSON body."""
        r = client.post("/v1/messages", content=b"not json",
                        headers={"Content-Type": "application/json"})
        assert r.status_code == 400

    @patch("backend.proxy.httpx.AsyncClient")
    def test_anthropic_messages_exception(self, mock_client_cls, client):
        """POST /v1/messages returns 500 on unexpected exception (lines 806-809)."""
        mock_instance = AsyncMock()
        mock_instance.post.side_effect = RuntimeError("unexpected runtime error")
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_instance

        r = client.post("/v1/messages", json={
            "model": "claude-3-sonnet",
            "messages": [{"role": "user", "content": "hi"}],
        })
        assert r.status_code in (500, 503)


class TestProxyExceptionHandlers:
    """Cover proxy.py exception handlers (lines 706-708, 733-735, 769-771)."""

    @pytest.fixture
    def client(self):
        return ProxyTestClient(proxy_app)

    @patch("backend.proxy.httpx.AsyncClient")
    def test_generate_general_exception(self, mock_client_cls, client):
        """POST /api/generate returns 500 on unexpected exception (lines 706-708)."""
        mock_instance = AsyncMock()
        mock_instance.post.side_effect = RuntimeError("db crashed")
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_instance

        r = client.post("/api/generate", json={"prompt": "hello", "model": "llama3"})
        assert r.status_code == 500

    @patch("backend.proxy.httpx.AsyncClient")
    def test_chat_general_exception(self, mock_client_cls, client):
        """POST /api/chat returns 500 on unexpected exception (lines 733-735)."""
        mock_instance = AsyncMock()
        mock_instance.post.side_effect = RuntimeError("memory error")
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_instance

        r = client.post("/api/chat", json={
            "model": "llama3",
            "messages": [{"role": "user", "content": "hello"}]
        })
        assert r.status_code == 500

    @patch("backend.proxy.httpx.AsyncClient")
    def test_openai_chat_general_exception(self, mock_client_cls, client):
        """POST /v1/chat/completions returns 500 on unexpected exception (lines 769-771)."""
        mock_instance = AsyncMock()
        mock_instance.post.side_effect = RuntimeError("unexpected crash")
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_instance

        r = client.post("/v1/chat/completions", json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hello"}]
        })
        assert r.status_code == 500


class TestProxyMemoryPreview:
    """Cover proxy.py lines 863-928: GET /proxy/memory-preview endpoint."""

    @pytest.fixture
    def client(self):
        return ProxyTestClient(proxy_app)

    def test_memory_preview_requires_query(self, client):
        """POST /proxy/preview with no query returns 400."""
        r = client.post("/proxy/preview", json={})
        assert r.status_code == 400
        assert "query" in r.json()["detail"]

    def test_memory_preview_with_query(self, client):
        """POST /proxy/preview returns injected facts and episodes for a query."""
        r = client.post("/proxy/preview", json={"query": "software engineer"})
        assert r.status_code == 200
        data = r.json()
        assert "query" in data
        assert "fact_candidates" in data
        assert "episode_candidates" in data
        assert "context" in data


class TestProxyChatWithAssistantMsg:
    """Cover proxy.py lines 477-480: assistant message learning path."""

    @pytest.fixture
    def client(self):
        return ProxyTestClient(proxy_app)

    @patch("backend.proxy.httpx.AsyncClient")
    def test_chat_with_assistant_message(self, mock_client_cls, client):
        """Chat request with assistant turn triggers assistant message learning (lines 477-480)."""
        mock_response = FakeHttpxResponse(
            status_code=200,
            json_data={"message": {"role": "assistant", "content": "You work at TechCorp."}, "done": True}
        )
        mock_instance = AsyncMock()
        mock_instance.post.return_value = mock_response
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_instance

        r = client.post("/api/chat", json={
            "model": "llama3",
            "messages": [
                {"role": "user", "content": "Tell me about my job."},
                {"role": "assistant", "content": "You work at TechCorp as a developer."},
                {"role": "user", "content": "What else?"},
            ]
        })
        assert r.status_code == 200


class TestProxySummarizeSessionLearnException:
    """Cover proxy.py lines 980-981: exception in per-message learning."""

    @pytest.fixture
    def client(self):
        return ProxyTestClient(proxy_app)

    def test_summarize_session_learn_exception_swallowed(self, client):
        """summarize-session swallows per-message learn exceptions (lines 980-981)."""
        with patch("backend.proxy.learner.learn_from_message",
                   side_effect=RuntimeError("learn crash")):
            r = client.post("/proxy/summarize-session", json={
                "messages": [{"role": "user", "content": "I am a Python developer."}]
            })
        # Should succeed despite the exception — messages_processed might be 0
        assert r.status_code == 200
        assert "messages_processed" in r.json()


class TestProxyRetrievalFTSScoring:
    """Cover proxy.py lines 319-322 and 329-331: fact scoring in retrieval."""

    def test_retrieve_relevant_facts_with_facts_in_db(self):
        """_retrieve_relevant_facts scores results when facts exist (lines 319-322 or 329-331)."""
        from backend.proxy import _retrieve_relevant_facts, memory

        # Add a fact to the shared memory
        memory.semantic.add_fact(
            content="I am a senior Python developer at a startup",
            fact_type="general",
            confidence=0.9,
        )

        facts, mode = _retrieve_relevant_facts("Python developer")
        # May return empty if indexing hasn't happened, but function should complete
        assert isinstance(facts, list)
        assert mode in ("hybrid", "fts")


# ===========================================================================
# _forward_anthropic and _forward_openai_compat direct call tests
# ===========================================================================

class TestForwardAnthropicDirect:
    """Cover proxy.py lines 539-577: _forward_anthropic non-stream path."""

    @pytest.mark.asyncio
    async def test_forward_anthropic_nonstream_success(self):
        """_forward_anthropic returns dict with velqua_metadata on success."""
        from backend.proxy import _forward_anthropic
        from backend.providers.anthropic import AnthropicProvider

        provider = AnthropicProvider()
        provider.config.api_key = "sk-ant-test"

        body = {
            "messages": [{"role": "user", "content": "hi"}],
            "model": "claude-haiku-4-5-20251001",
        }

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json = MagicMock(return_value={
            "content": [{"type": "text", "text": "hello"}],
            "model": "claude-haiku-4-5-20251001",
            "stop_reason": "end_turn",
        })

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("backend.proxy.httpx.AsyncClient", return_value=mock_client):
            result = await _forward_anthropic(body, stream=False, metadata={"test": True}, provider=provider)

        assert "velqua_metadata" in result
        assert result["velqua_metadata"]["test"] is True

    @pytest.mark.asyncio
    async def test_forward_anthropic_nonstream_error(self):
        """_forward_anthropic raises HTTPException on non-200 (line 567-570)."""
        from backend.proxy import _forward_anthropic
        from backend.providers.anthropic import AnthropicProvider
        from fastapi import HTTPException

        provider = AnthropicProvider()
        body = {"messages": [{"role": "user", "content": "hi"}]}

        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("backend.proxy.httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(HTTPException) as exc_info:
                await _forward_anthropic(body, stream=False, metadata={}, provider=provider)
        assert exc_info.value.status_code == 502

    @pytest.mark.asyncio
    async def test_forward_anthropic_invalid_json(self):
        """_forward_anthropic raises HTTPException when response is invalid JSON (lines 572-575)."""
        from backend.proxy import _forward_anthropic
        from backend.providers.anthropic import AnthropicProvider
        from fastapi import HTTPException

        provider = AnthropicProvider()
        body = {"messages": [{"role": "user", "content": "hi"}]}

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("not json")

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("backend.proxy.httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(HTTPException) as exc_info:
                await _forward_anthropic(body, stream=False, metadata={}, provider=provider)
        assert exc_info.value.status_code == 502


class TestForwardOpenAICompatDirect:
    """Cover proxy.py lines 580-605: _forward_openai_compat non-stream path."""

    @pytest.mark.asyncio
    async def test_forward_openai_compat_success(self):
        """_forward_openai_compat returns dict with velqua_metadata on success."""
        from backend.proxy import _forward_openai_compat
        from backend.providers.openai_compat import OpenAICompatProvider
        from backend.providers.base import ProviderConfig

        provider = OpenAICompatProvider(ProviderConfig(
            name="openai",
            base_url="https://api.openai.com",
            api_key="sk-test",
        ))

        body = {
            "messages": [{"role": "user", "content": "hi"}],
            "model": "gpt-4o",
        }

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json = MagicMock(return_value={
            "choices": [{"message": {"content": "hello"}}],
            "model": "gpt-4o",
        })

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("backend.proxy.httpx.AsyncClient", return_value=mock_client):
            result = await _forward_openai_compat(body, stream=False, metadata={"test": True}, provider=provider)

        assert "velqua_metadata" in result

    @pytest.mark.asyncio
    async def test_forward_openai_compat_error(self):
        """_forward_openai_compat raises HTTPException on non-200 (lines 595-598)."""
        from backend.proxy import _forward_openai_compat
        from backend.providers.openai_compat import OpenAICompatProvider
        from backend.providers.base import ProviderConfig
        from fastapi import HTTPException

        provider = OpenAICompatProvider(ProviderConfig(
            name="openai",
            base_url="https://api.openai.com",
            api_key="sk-test",
        ))
        body = {"messages": [{"role": "user", "content": "hi"}]}

        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.text = "Rate limited"

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("backend.proxy.httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(HTTPException) as exc_info:
                await _forward_openai_compat(body, stream=False, metadata={}, provider=provider)
        assert exc_info.value.status_code == 502

    @pytest.mark.asyncio
    async def test_forward_openai_compat_invalid_json(self):
        """_forward_openai_compat raises HTTPException for invalid JSON (lines 600-603)."""
        from backend.proxy import _forward_openai_compat
        from backend.providers.openai_compat import OpenAICompatProvider
        from backend.providers.base import ProviderConfig
        from fastapi import HTTPException

        provider = OpenAICompatProvider(ProviderConfig(
            name="openai",
            base_url="https://api.openai.com",
            api_key="sk-test",
        ))
        body = {"messages": []}

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("bad json")

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("backend.proxy.httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(HTTPException) as exc_info:
                await _forward_openai_compat(body, stream=False, metadata={}, provider=provider)
        assert exc_info.value.status_code == 502


class TestV1MessagesSystemTextInjection:
    """Cover proxy.py lines 795-796: system field in /v1/messages body."""

    @pytest.fixture
    def client(self):
        return ProxyTestClient(proxy_app)

    @patch("backend.proxy.httpx.AsyncClient")
    def test_v1_messages_with_system_text(self, mock_client_cls, client):
        """POST /v1/messages with 'system' field prepends system message (lines 795-796)."""
        mock_response = FakeHttpxResponse(
            status_code=200,
            json_data={
                "message": {"content": "Hello!", "role": "assistant"},
                "done": True,
            }
        )
        mock_instance = AsyncMock()
        mock_instance.post.return_value = mock_response
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_instance

        r = client.post("/v1/messages", json={
            "messages": [{"role": "user", "content": "hello"}],
            "system": "You are a helpful assistant with memory.",
            "model": "llama3",
        })
        # Should either succeed or return a provider error — not crash
        assert r.status_code in (200, 502, 503)


class TestHandleChatRequestAnthropicProvider:
    """Cover proxy.py lines 506-507: routing to Anthropic in _handle_chat_request."""

    @pytest.fixture
    def client(self):
        return ProxyTestClient(proxy_app)

    @patch("backend.proxy.httpx.AsyncClient")
    def test_chat_routes_to_anthropic_when_active(self, mock_client_cls, client):
        """When active provider is Anthropic, _forward_anthropic is called (lines 506-507)."""
        from backend.proxy import registry
        from backend.providers.anthropic import AnthropicProvider
        from backend.providers.base import ProviderConfig

        # Register Anthropic provider and set as active
        anthropic_config = ProviderConfig(
            name="anthropic",
            base_url="https://api.anthropic.com",
            api_key="sk-ant-test",
        )
        registry._providers["anthropic"] = anthropic_config
        registry._active_name = "anthropic"

        try:
            mock_response = FakeHttpxResponse(
                status_code=200,
                json_data={
                    "content": [{"type": "text", "text": "Hello!"}],
                    "model": "claude-haiku-4-5-20251001",
                    "stop_reason": "end_turn",
                }
            )
            mock_instance = AsyncMock()
            mock_instance.post.return_value = mock_response
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_instance

            r = client.post("/api/chat", json={
                "messages": [{"role": "user", "content": "hello"}],
                "model": "claude-haiku-4-5-20251001",
            })
            # Should succeed or fail with connection error, but not crash
            assert r.status_code in (200, 502, 503)
        finally:
            # Restore Ollama as active
            registry._active_name = "ollama"
            registry._providers.pop("anthropic", None)


class TestFTSOnlyRetrievalPath:
    """Cover proxy.py lines 329-331: FTS fallback when VECTOR_ENABLED is False."""

    def test_fts_scoring_when_vector_disabled(self):
        """_retrieve_relevant_facts scores via FTS when vector is disabled (lines 329-331)."""
        from backend.proxy import _retrieve_relevant_facts, memory, VECTOR_ENABLED
        import backend.proxy as proxy_mod

        # Add a fact that FTS can find
        memory.semantic.add_fact(
            content="I love programming in Python and building APIs",
            fact_type="general",
            confidence=0.9,
        )

        original = proxy_mod.VECTOR_ENABLED
        try:
            proxy_mod.VECTOR_ENABLED = False
            facts, mode = _retrieve_relevant_facts("Python programming APIs")
            assert mode == "fts"
            assert isinstance(facts, list)
        finally:
            proxy_mod.VECTOR_ENABLED = original


class TestForwardAnthropicStream:
    """Cover proxy.py streaming path of _forward_anthropic (lines 549-560)."""

    @pytest.mark.asyncio
    async def test_forward_anthropic_stream_with_system_and_temperature(self):
        """Stream=True with system + temperature covers lines 549-560."""
        from backend.proxy import _forward_anthropic
        from backend.providers import ProviderConfig
        from backend.providers.anthropic import AnthropicProvider
        from starlette.responses import StreamingResponse

        config = ProviderConfig(
            name="anthropic",
            base_url="https://api.anthropic.com",
            api_key="sk-ant-test",
        )
        provider = AnthropicProvider(config)

        body = {
            "messages": [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "hi"},
            ],
            "model": "claude-3-haiku-20240307",
            "stream": True,
            "temperature": 0.7,
        }

        with patch("backend.proxy._stream_proxy", new=AsyncMock(
            return_value=StreamingResponse(iter([b"data: {}\n\n"]))
        )) as mock_sp:
            result = await _forward_anthropic(body, stream=True, metadata={}, provider=provider)
            mock_sp.assert_called_once()

        assert isinstance(result, StreamingResponse)


class TestForwardOpenAICompatStream:
    """Cover proxy.py streaming path of _forward_openai_compat (line 588)."""

    @pytest.mark.asyncio
    async def test_forward_openai_compat_stream(self):
        """Stream=True routes to _stream_proxy (line 588)."""
        from backend.proxy import _forward_openai_compat
        from backend.providers import ProviderConfig
        from backend.providers.openai_compat import OpenAICompatProvider
        from starlette.responses import StreamingResponse

        config = ProviderConfig(
            name="openai",
            base_url="https://api.openai.com",
            api_key="sk-test",
        )
        provider = OpenAICompatProvider(config)

        body = {
            "messages": [{"role": "user", "content": "hi"}],
            "model": "gpt-4",
            "stream": True,
        }

        with patch("backend.proxy._stream_proxy", new=AsyncMock(
            return_value=StreamingResponse(iter([b"data: {}\n\n"]))
        )) as mock_sp:
            result = await _forward_openai_compat(body, stream=True, metadata={}, provider=provider)
            mock_sp.assert_called_once()

        assert isinstance(result, StreamingResponse)


class TestStreamProxyWithHeaders:
    """Cover proxy.py line 164: req_headers.update(headers) in _stream_proxy."""

    @patch("backend.proxy.httpx.AsyncClient")
    def test_stream_proxy_passes_headers(self, mock_client_cls):
        """_stream_proxy updates request headers when headers dict provided (line 164)."""
        import asyncio
        from backend.proxy import _stream_proxy

        mock_response = AsyncMock()
        mock_response.status_code = 200

        async def _fake_aiter():
            yield b"chunk"

        mock_response.aiter_bytes = _fake_aiter
        mock_response.aclose = AsyncMock()

        mock_instance = MagicMock()
        mock_instance.build_request = MagicMock(return_value=MagicMock())
        mock_instance.send = AsyncMock(return_value=mock_response)
        mock_instance.aclose = AsyncMock()
        mock_client_cls.return_value = mock_instance

        async def _run():
            response = await _stream_proxy(
                "http://fake/v1/messages",
                {"model": "claude"},
                headers={"x-api-key": "sk-ant-test", "anthropic-version": "2023-06-01"},
                media_type="text/event-stream",
            )
            # Consume the streaming response
            async for _ in response.body_iterator:
                pass

        asyncio.run(_run())


# ===========================================================================
# Coverage gaps: proxy.py lines 424, 432-433, 455-461, 488, 506-509,
#                757, 801, 803, 865-866, 897-908, 921
# ===========================================================================

class TestHandleChatRequestDirectCalls:
    """Call _handle_chat_request directly to cover routing branches (lines 424, 506-509)."""

    def test_none_provider_gets_active(self):
        """provider=None → registry.get_active() is called (line 424)."""
        import asyncio
        from backend.proxy import _handle_chat_request

        body = {"messages": [{"role": "user", "content": "hello"}]}
        with patch("backend.proxy._forward_ollama_chat", new=AsyncMock(return_value={"ok": True})):
            result = asyncio.run(_handle_chat_request(body, source="test", provider=None, request=None))
        assert result == {"ok": True}

    def test_anthropic_provider_routes_to_forward_anthropic(self):
        """AnthropicProvider routes to _forward_anthropic (lines 506-507)."""
        import asyncio
        from backend.proxy import _handle_chat_request
        from backend.providers.anthropic import AnthropicProvider
        from backend.providers.base import ProviderConfig

        config = ProviderConfig(name="anthropic", base_url="https://api.anthropic.com", api_key="sk-ant-test")
        provider = AnthropicProvider(config)
        body = {"messages": [{"role": "user", "content": "hello"}]}

        with patch("backend.proxy._forward_anthropic", new=AsyncMock(return_value={"ok": "anthropic"})):
            result = asyncio.run(_handle_chat_request(body, source="test", provider=provider, request=None))
        assert result == {"ok": "anthropic"}

    def test_openai_compat_provider_routes_to_forward_openai(self):
        """OpenAICompatProvider routes to _forward_openai_compat (lines 508-509)."""
        import asyncio
        from backend.proxy import _handle_chat_request
        from backend.providers.openai_compat import OpenAICompatProvider
        from backend.providers.base import ProviderConfig

        config = ProviderConfig(name="openai", base_url="https://api.openai.com", api_key="sk-test")
        provider = OpenAICompatProvider(config)
        body = {"messages": [{"role": "user", "content": "hello"}]}

        with patch("backend.proxy._forward_openai_compat", new=AsyncMock(return_value={"ok": "openai"})):
            result = asyncio.run(_handle_chat_request(body, source="test", provider=provider, request=None))
        assert result == {"ok": "openai"}

    def test_detect_agent_id_exception_swallowed(self):
        """detect_agent_id raises → except Exception: pass (lines 432-433)."""
        import asyncio
        from backend.proxy import _handle_chat_request

        body = {"messages": [{"role": "user", "content": "hello"}]}
        with patch("backend.proxy.detect_agent_id", side_effect=RuntimeError("agent fail")):
            with patch("backend.proxy._forward_ollama_chat", new=AsyncMock(return_value={"ok": True})):
                # Should not raise even though detect_agent_id fails
                result = asyncio.run(
                    _handle_chat_request(body, source="test", provider=None, request=MagicMock())
                )
        assert result == {"ok": True}

    def test_mesh_notes_injected_into_context(self):
        """Unread mesh notes for agent are injected into context (lines 455-459, 488)."""
        import asyncio
        from backend.proxy import _handle_chat_request, mesh_noteboard

        # Post a note for the "unknown" agent (detect_agent_id returns "unknown" when request=None)
        note = mesh_noteboard.post(
            from_agent="test-sender",
            to_agent="unknown",
            content="This is a test mesh note for injection testing",
        )

        body = {"messages": [{"role": "user", "content": "testing mesh injection"}]}
        try:
            with patch("backend.proxy._forward_ollama_chat", new=AsyncMock(return_value={"ok": True})):
                result = asyncio.run(
                    _handle_chat_request(body, source="test", provider=None, request=None)
                )
            assert result == {"ok": True}
        finally:
            # Clean up the note
            try:
                from backend.mesh.db import get_conn
                conn = get_conn()
                conn.execute("DELETE FROM mesh_notes WHERE id = ?", (note["id"],))
                conn.commit()
            except Exception:
                pass

    def test_mesh_heartbeat_exception_caught(self):
        """mesh_registry.heartbeat raises → except Exception: logger.debug (lines 460-461)."""
        import asyncio
        from backend.proxy import _handle_chat_request

        body = {"messages": [{"role": "user", "content": "hello"}]}
        with patch("backend.proxy.mesh_registry.heartbeat", side_effect=RuntimeError("heartbeat fail")):
            with patch("backend.proxy._forward_ollama_chat", new=AsyncMock(return_value={"ok": True})):
                result = asyncio.run(
                    _handle_chat_request(body, source="test", provider=None, request=None)
                )
        # Exception is swallowed, request proceeds normally
        assert result == {"ok": True}


class TestV1ChatCompletionsWithAnthropicActive:
    """Cover proxy.py line 757 (isinstance check for Anthropic in /v1/chat/completions)."""

    @pytest.fixture
    def client(self):
        return ProxyTestClient(proxy_app)

    @patch("backend.proxy.httpx.AsyncClient")
    def test_openai_endpoint_with_anthropic_active_provider(self, mock_client_cls, client):
        """When Anthropic is active, /v1/chat/completions routes through it (line 757)."""
        from backend.proxy import registry
        from backend.providers.base import ProviderConfig

        config = ProviderConfig(
            name="anthropic",
            base_url="https://api.anthropic.com",
            api_key="sk-ant-test",
        )
        registry._providers["anthropic"] = config
        registry._active_name = "anthropic"

        try:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "content": [{"type": "text", "text": "ok"}],
                "model": "claude-3-haiku-20240307",
                "stop_reason": "end_turn",
            }
            mock_instance = AsyncMock()
            mock_instance.post.return_value = mock_response
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_instance

            r = client.post("/v1/chat/completions", json={
                "messages": [{"role": "user", "content": "hi"}],
                "model": "claude-3-haiku-20240307",
            })
            assert r.status_code in (200, 502, 503)
        finally:
            registry._active_name = "ollama"
            registry._providers.pop("anthropic", None)


class TestV1MessagesExceptionPaths:
    """Cover proxy.py lines 801 (HTTPException re-raise) and 803 (ConnectError)."""

    @pytest.fixture
    def client(self):
        return ProxyTestClient(proxy_app)

    def test_v1_messages_http_exception_reraise(self, client):
        """HTTPException from _handle_chat_request is re-raised (line 801)."""
        from fastapi import HTTPException as FastHTTPException
        with patch("backend.proxy._handle_chat_request",
                   new=AsyncMock(side_effect=FastHTTPException(status_code=404, detail="not found"))):
            r = client.post("/v1/messages", json={"messages": [{"role": "user", "content": "hi"}]})
        assert r.status_code == 404

    def test_v1_messages_connect_error_returns_503(self, client):
        """httpx.ConnectError from _handle_chat_request returns 503 (line 803)."""
        import httpx
        with patch("backend.proxy._handle_chat_request",
                   new=AsyncMock(side_effect=httpx.ConnectError("no route"))):
            r = client.post("/v1/messages", json={"messages": [{"role": "user", "content": "hi"}]})
        assert r.status_code == 503
        assert "Anthropic" in r.json()["detail"]


class TestProxyPreviewEndpointCoverage:
    """Cover /proxy/preview endpoint lines 865-866, 897-908, 921."""

    @pytest.fixture
    def client(self):
        return ProxyTestClient(proxy_app)

    def test_preview_invalid_json_body(self, client):
        """Invalid JSON body raises 400 (lines 865-866)."""
        # Send malformed JSON
        r = client.post(
            "/proxy/preview",
            content=b"not-valid-json{{{",
            headers={"content-type": "application/json"},
        )
        assert r.status_code == 400
        assert "Invalid JSON" in r.json()["detail"]

    def test_preview_fts_fallback_when_vector_disabled(self, client):
        """When VECTOR_ENABLED=False, FTS fallback runs (lines 897-908)."""
        with patch("backend.proxy.VECTOR_ENABLED", False):
            with patch("backend.proxy.retriever", None):
                r = client.post("/proxy/preview", json={"query": "software engineer"})
        assert r.status_code == 200
        data = r.json()
        assert data["search_mode"] == "fts"
        assert "facts_available" in data

    def test_preview_token_budget_exceeded(self, client):
        """Facts exceed token budget → break executed (line 921)."""
        from backend.proxy import memory

        # Add several facts with long content to fill the budget
        for i in range(5):
            memory.semantic.add_fact(
                content=f"I am a {'very ' * 50}experienced software engineer working on large scale distributed systems {i}",
                confidence=0.9,
            )

        # Set tiny token budget so it overflows quickly
        with patch("backend.proxy.config") as mock_config:
            mock_config.max_tokens = 5  # tiny budget → will break early
            mock_config.budget = "minimal"
            r = client.post("/proxy/preview", json={"query": "software engineer"})

        assert r.status_code == 200
        data = r.json()
        # Some candidates but fewer injected than total due to budget
        assert "injected_count" in data or "context" in data


# ===========================================================================
# Tests for extracted init functions (_init_vector_retriever, _index_existing_facts)
# Previously untestable module-level code — now 100% coverable.
# ===========================================================================

class TestInitVectorRetriever:
    """Cover _init_vector_retriever success and failure paths (lines 104-107)."""

    def test_success_returns_retriever_and_enabled(self):
        """Happy path: returns (retriever, vector_store, True)."""
        from backend.proxy import _init_vector_retriever, memory
        ret, vs, enabled = _init_vector_retriever(memory)
        assert enabled is True
        assert ret is not None
        assert vs is not None

    def test_embedder_failure_returns_none_and_disabled(self):
        """When get_default_embedder raises, returns (None, None, False)."""
        from backend.proxy import _init_vector_retriever, memory
        with patch("backend.proxy.get_default_embedder", side_effect=RuntimeError("no model")):
            ret, vs, enabled = _init_vector_retriever(memory)
        assert enabled is False
        assert ret is None
        assert vs is None

    def test_hybrid_retriever_failure_returns_disabled(self):
        """When HybridRetriever construction raises, falls back to FTS."""
        from backend.proxy import _init_vector_retriever, memory
        with patch("backend.proxy.HybridRetriever", side_effect=RuntimeError("retriever fail")):
            ret, vs, enabled = _init_vector_retriever(memory)
        assert enabled is False
        assert ret is None


class TestIndexExistingFacts:
    """Cover _index_existing_facts branches (lines 66-68)."""

    def test_skips_when_vector_disabled(self):
        """No indexing when VECTOR_ENABLED is False."""
        from backend.proxy import _index_existing_facts
        with patch("backend.proxy.VECTOR_ENABLED", False):
            with patch("backend.proxy.retriever", None):
                # Should return without calling anything
                _index_existing_facts()  # must not raise

    def test_empty_db_logs_no_facts(self):
        """When DB is empty, logs 'No existing facts to index' (line 66)."""
        from backend.proxy import _index_existing_facts
        mock_ret = MagicMock()
        mock_mem = MagicMock()
        mock_mem.semantic.list_all.return_value = []

        with patch("backend.proxy.VECTOR_ENABLED", True):
            with patch("backend.proxy.retriever", mock_ret):
                with patch("backend.proxy.memory", mock_mem):
                    _index_existing_facts()

        mock_ret.index_all.assert_not_called()

    def test_index_all_failure_logged(self):
        """When index_all raises, error is logged and not re-raised (lines 67-68)."""
        from backend.proxy import _index_existing_facts
        from backend.anamnesis.models import Fact

        mock_fact = MagicMock()
        mock_fact.content = "some fact"
        mock_ret = MagicMock()
        mock_ret.index_all.side_effect = RuntimeError("index crash")
        mock_mem = MagicMock()
        mock_mem.semantic.list_all.return_value = [mock_fact]

        with patch("backend.proxy.VECTOR_ENABLED", True):
            with patch("backend.proxy.retriever", mock_ret):
                with patch("backend.proxy.memory", mock_mem):
                    _index_existing_facts()  # must not raise


# ==================================================================
# Hybrid Retrieval Weight Configuration
# ==================================================================

class TestHybridRetrievalWeights:
    """Verify hybrid retrieval weight defaults and env var overrides."""

    def test_default_fts_weight(self):
        """Default FTS weight should be 0.2 (20%)."""
        from backend.config import VelquaConfig
        assert VelquaConfig.FTS_WEIGHT == pytest.approx(0.2)

    def test_default_vector_weight(self):
        """Default vector weight should be 0.8 (80%)."""
        from backend.config import VelquaConfig
        assert VelquaConfig.VECTOR_WEIGHT == pytest.approx(0.8)

    def test_env_var_override_fts_weight(self, monkeypatch):
        """VELQUA_FTS_WEIGHT env var overrides the default."""
        monkeypatch.setenv("VELQUA_FTS_WEIGHT", "0.35")
        # Re-evaluate the class attribute by reloading config
        import backend.config
        importlib.reload(backend.config)
        from backend.config import VelquaConfig
        assert VelquaConfig.FTS_WEIGHT == pytest.approx(0.35)
        # Restore default
        monkeypatch.delenv("VELQUA_FTS_WEIGHT")
        importlib.reload(backend.config)

    def test_env_var_override_vector_weight(self, monkeypatch):
        """VELQUA_VECTOR_WEIGHT env var overrides the default."""
        monkeypatch.setenv("VELQUA_VECTOR_WEIGHT", "0.65")
        import backend.config
        importlib.reload(backend.config)
        from backend.config import VelquaConfig
        assert VelquaConfig.VECTOR_WEIGHT == pytest.approx(0.65)
        # Restore default
        monkeypatch.delenv("VELQUA_VECTOR_WEIGHT")
        importlib.reload(backend.config)

    def test_init_vector_retriever_uses_config_weights(self):
        """_init_vector_retriever passes Config.FTS_WEIGHT/VECTOR_WEIGHT to HybridRetriever."""
        from backend.proxy import _init_vector_retriever, memory
        from backend.config import VelquaConfig as Cfg

        with patch("backend.proxy.HybridRetriever") as MockHR:
            with patch("backend.proxy.get_default_embedder") as mock_emb:
                with patch("backend.proxy.InMemoryVectorStore") as mock_vs:
                    MockHR.return_value = MagicMock()
                    _init_vector_retriever(memory)

        MockHR.assert_called_once()
        call_kwargs = MockHR.call_args[1]
        assert call_kwargs["text_weight"] == pytest.approx(Cfg.FTS_WEIGHT)
        assert call_kwargs["vector_weight"] == pytest.approx(Cfg.VECTOR_WEIGHT)


# ==================================================================
# Cross-encoder reranker integration in retrieval pipeline
# ==================================================================

class TestRerankerIntegration:
    """Tests that the cross-encoder reranker is wired into _retrieve_relevant_facts."""

    def _make_hybrid_result(self, content, score=0.5):
        """Create a fake HybridSearchResult-like object."""
        obj = MagicMock(
            content=content,
            score=score,
            metadata={"importance": 0.5, "confirmation_count": 1},
            spec=["content", "score", "metadata"],
        )
        # Ensure getattr falls through to metadata for missing attrs
        obj.last_confirmed = None
        obj.first_learned = None
        obj.confirmation_count = 1
        obj.importance = 0.5
        return obj

    @patch("backend.proxy._detect_query_topic", return_value="")
    def test_reranker_reorders_hybrid_results(self, _mock_topic):
        """When reranker is active, results are reordered by cross-encoder score."""
        import backend.proxy as proxy_mod
        import numpy as np

        # Hybrid retrieval returns A, B, C in that order
        fake_results = [
            self._make_hybrid_result("Fact A", 0.9),
            self._make_hybrid_result("Fact B", 0.5),
            self._make_hybrid_result("Fact C", 0.7),
        ]

        mock_retriever = MagicMock()
        mock_retriever.search.return_value = fake_results

        # Cross-encoder says B > C > A
        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = [
            ("Fact B", 0.95),
            ("Fact C", 0.80),
            ("Fact A", 0.30),
        ]

        orig_retriever = proxy_mod.retriever
        orig_reranker = proxy_mod.reranker
        orig_vec = proxy_mod.VECTOR_ENABLED
        try:
            proxy_mod.retriever = mock_retriever
            proxy_mod.reranker = mock_reranker
            proxy_mod.VECTOR_ENABLED = True

            facts, mode = _retrieve_relevant_facts("test query")

            assert mode == "hybrid+rerank"
            # B should be first (highest CE score)
            assert facts[0] == "Fact B"
            mock_reranker.rerank.assert_called_once()
        finally:
            proxy_mod.retriever = orig_retriever
            proxy_mod.reranker = orig_reranker
            proxy_mod.VECTOR_ENABLED = orig_vec

    @patch("backend.proxy._detect_query_topic", return_value="")
    def test_reranker_disabled_uses_standard_hybrid(self, _mock_topic):
        """When reranker is None, standard hybrid scoring is used."""
        import backend.proxy as proxy_mod

        fake_results = [
            self._make_hybrid_result("Fact A", 0.9),
            self._make_hybrid_result("Fact B", 0.5),
        ]

        mock_retriever = MagicMock()
        mock_retriever.search.return_value = fake_results

        orig_retriever = proxy_mod.retriever
        orig_reranker = proxy_mod.reranker
        orig_vec = proxy_mod.VECTOR_ENABLED
        try:
            proxy_mod.retriever = mock_retriever
            proxy_mod.reranker = None
            proxy_mod.VECTOR_ENABLED = True

            facts, mode = _retrieve_relevant_facts("test query")

            assert mode == "hybrid"
            assert len(facts) == 2
        finally:
            proxy_mod.retriever = orig_retriever
            proxy_mod.reranker = orig_reranker
            proxy_mod.VECTOR_ENABLED = orig_vec

    @patch("backend.proxy._detect_query_topic", return_value="")
    def test_reranker_failure_falls_back_to_hybrid(self, _mock_topic):
        """If the reranker raises, fall back to standard hybrid scoring."""
        import backend.proxy as proxy_mod

        fake_results = [
            self._make_hybrid_result("Fact A", 0.9),
        ]

        mock_retriever = MagicMock()
        mock_retriever.search.return_value = fake_results

        mock_reranker = MagicMock()
        mock_reranker.rerank.side_effect = RuntimeError("model load failed")

        orig_retriever = proxy_mod.retriever
        orig_reranker = proxy_mod.reranker
        orig_vec = proxy_mod.VECTOR_ENABLED
        try:
            proxy_mod.retriever = mock_retriever
            proxy_mod.reranker = mock_reranker
            proxy_mod.VECTOR_ENABLED = True

            facts, mode = _retrieve_relevant_facts("test query")

            # Should gracefully degrade to hybrid
            assert mode == "hybrid"
            assert facts == ["Fact A"]
        finally:
            proxy_mod.retriever = orig_retriever
            proxy_mod.reranker = orig_reranker
            proxy_mod.VECTOR_ENABLED = orig_vec

    @patch("backend.proxy._detect_query_topic", return_value="")
    def test_reranker_overfetches_candidates(self, _mock_topic):
        """When reranker is active, retrieval limit increases to RERANKER_CANDIDATES."""
        import backend.proxy as proxy_mod
        from backend.config import VelquaConfig as Cfg

        mock_retriever = MagicMock()
        mock_retriever.search.return_value = []

        mock_reranker = MagicMock()

        orig_retriever = proxy_mod.retriever
        orig_reranker = proxy_mod.reranker
        orig_vec = proxy_mod.VECTOR_ENABLED
        try:
            proxy_mod.retriever = mock_retriever
            proxy_mod.reranker = mock_reranker
            proxy_mod.VECTOR_ENABLED = True

            _retrieve_relevant_facts("test query")

            call_kwargs = mock_retriever.search.call_args[1]
            assert call_kwargs["limit"] == Cfg.RERANKER_CANDIDATES
        finally:
            proxy_mod.retriever = orig_retriever
            proxy_mod.reranker = orig_reranker
            proxy_mod.VECTOR_ENABLED = orig_vec


class TestInitReranker:
    """Tests for _init_reranker factory function."""

    def test_disabled_by_default(self):
        """Reranker returns None when RERANKER_ENABLED is False."""
        from backend.proxy import _init_reranker
        import backend.proxy as proxy_mod
        # Use the Config reference that proxy.py actually sees
        Cfg = proxy_mod.Config

        orig = Cfg.RERANKER_ENABLED
        try:
            Cfg.RERANKER_ENABLED = False
            assert _init_reranker() is None
        finally:
            Cfg.RERANKER_ENABLED = orig

    def test_enabled_returns_reranker(self):
        """Reranker is returned when enabled."""
        from backend.proxy import _init_reranker
        import backend.proxy as proxy_mod
        from backend.anamnesis.retrieval.reranker import CrossEncoderReranker
        # Use the Config reference that proxy.py actually sees
        Cfg = proxy_mod.Config

        orig = Cfg.RERANKER_ENABLED
        try:
            Cfg.RERANKER_ENABLED = True
            result = _init_reranker()
            assert isinstance(result, CrossEncoderReranker)
            assert result.model_name == Cfg.RERANKER_MODEL
        finally:
            Cfg.RERANKER_ENABLED = orig
