"""Coverage-focused tests for proxy error handling and edge paths."""
from __future__ import annotations

from pathlib import Path
import importlib
import os
import sys
import types

import httpx
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient


def _reload_proxy_with_temp_db(tmp_path: Path):
    """Reload backend.proxy after pointing config at a temp DB."""
    os.environ["VELQUA_DB_PATH"] = str(tmp_path / "proxy_test.db")
    sys.modules.pop("backend.proxy", None)
    import backend.proxy as proxy
    return importlib.reload(proxy)


@pytest.fixture
def proxy_module(tmp_path):
    return _reload_proxy_with_temp_db(tmp_path)


@pytest.fixture
def client(proxy_module):
    return TestClient(proxy_module.app)


class _DummyResponse:
    def __init__(self, status_code=200, json_data=None, text="", bytes_data=b""):
        self.status_code = status_code
        self._json = json_data if json_data is not None else {}
        self.text = text
        self._bytes = bytes_data

    def json(self):
        if self._json == "__invalid__":
            raise ValueError("invalid json")
        return self._json

    async def aread(self):
        return self._bytes

    async def aclose(self):
        return None

    async def aiter_bytes(self):
        if self._bytes:
            yield self._bytes


class _DummyAsyncClient:
    def __init__(self, response=None, exc=None):
        self._response = response
        self._exc = exc

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, *args, **kwargs):
        if self._exc:
            raise self._exc
        return self._response

    async def get(self, *args, **kwargs):
        if self._exc:
            raise self._exc
        return self._response

    def build_request(self, method, url, json=None, headers=None):
        return {"method": method, "url": url, "json": json, "headers": headers}

    async def send(self, req, stream=False):
        if self._exc:
            raise self._exc
        return self._response

    async def aclose(self):
        return None


def test_stream_proxy_raises_http_502_on_non_200(monkeypatch, proxy_module):
    response = _DummyResponse(status_code=500, bytes_data=b"backend broke")
    monkeypatch.setattr(proxy_module.httpx, "AsyncClient", lambda *a, **k: _DummyAsyncClient(response=response))

    with pytest.raises(HTTPException) as exc:
        import asyncio
        asyncio.run(proxy_module._stream_proxy("http://backend/test", {"x": 1}))

    assert exc.value.status_code == 502
    assert "Backend returned 500" in exc.value.detail


def test_forward_ollama_chat_invalid_json(monkeypatch, proxy_module):
    response = _DummyResponse(status_code=200, json_data="__invalid__")
    monkeypatch.setattr(proxy_module.httpx, "AsyncClient", lambda *a, **k: _DummyAsyncClient(response=response))

    with pytest.raises(HTTPException) as exc:
        import asyncio
        asyncio.run(proxy_module._forward_ollama_chat({"messages": []}, False, {}))

    assert exc.value.status_code == 502
    assert "invalid JSON" in exc.value.detail


def test_forward_ollama_chat_non_200(monkeypatch, proxy_module):
    response = _DummyResponse(status_code=404, text="missing")
    monkeypatch.setattr(proxy_module.httpx, "AsyncClient", lambda *a, **k: _DummyAsyncClient(response=response))

    with pytest.raises(HTTPException) as exc:
        import asyncio
        asyncio.run(proxy_module._forward_ollama_chat({"messages": []}, False, {}))

    assert exc.value.status_code == 502
    assert "Ollama returned 404" in exc.value.detail


def test_forward_openai_compat_invalid_json(monkeypatch, proxy_module):
    provider = types.SimpleNamespace(
        config=types.SimpleNamespace(base_url="http://llm.local"),
        get_auth_headers=lambda: {"Authorization": "Bearer x"},
    )
    response = _DummyResponse(status_code=200, json_data="__invalid__")
    monkeypatch.setattr(proxy_module.httpx, "AsyncClient", lambda *a, **k: _DummyAsyncClient(response=response))

    with pytest.raises(HTTPException) as exc:
        import asyncio
        asyncio.run(proxy_module._forward_openai_compat({"messages": []}, False, {}, provider))

    assert exc.value.status_code == 502
    assert "invalid JSON" in exc.value.detail


def test_forward_openai_compat_non_200(monkeypatch, proxy_module):
    provider = types.SimpleNamespace(
        config=types.SimpleNamespace(base_url="http://llm.local"),
        get_auth_headers=lambda: {"Authorization": "Bearer x"},
    )
    response = _DummyResponse(status_code=429, text="rate limited")
    monkeypatch.setattr(proxy_module.httpx, "AsyncClient", lambda *a, **k: _DummyAsyncClient(response=response))

    with pytest.raises(HTTPException) as exc:
        import asyncio
        asyncio.run(proxy_module._forward_openai_compat({"messages": []}, False, {}, provider))

    assert exc.value.status_code == 502
    assert "Backend returned 429" in exc.value.detail


def test_forward_anthropic_invalid_json(monkeypatch, proxy_module):
    provider = types.SimpleNamespace(
        config=types.SimpleNamespace(base_url="http://anthropic.local"),
        get_auth_headers=lambda: {"x-api-key": "k"},
        _extract_system=lambda messages: ("sys", [{"role": "user", "content": "hi"}]),
        _resolve_model=lambda model: "claude-test",
    )
    response = _DummyResponse(status_code=200, json_data="__invalid__")
    monkeypatch.setattr(proxy_module.httpx, "AsyncClient", lambda *a, **k: _DummyAsyncClient(response=response))

    with pytest.raises(HTTPException) as exc:
        import asyncio
        asyncio.run(proxy_module._forward_anthropic({"messages": []}, False, {}, provider))

    assert exc.value.status_code == 502
    assert "Anthropic returned invalid JSON" in exc.value.detail


def test_forward_anthropic_non_200(monkeypatch, proxy_module):
    provider = types.SimpleNamespace(
        config=types.SimpleNamespace(base_url="http://anthropic.local"),
        get_auth_headers=lambda: {"x-api-key": "k"},
        _extract_system=lambda messages: ("sys", [{"role": "user", "content": "hi"}]),
        _resolve_model=lambda model: "claude-test",
    )
    response = _DummyResponse(status_code=400, text="bad request")
    monkeypatch.setattr(proxy_module.httpx, "AsyncClient", lambda *a, **k: _DummyAsyncClient(response=response))

    with pytest.raises(HTTPException) as exc:
        import asyncio
        asyncio.run(proxy_module._forward_anthropic({"messages": []}, False, {}, provider))

    assert exc.value.status_code == 502
    assert "Anthropic returned 400" in exc.value.detail


def test_proxy_generate_invalid_json_returns_400(client):
    response = client.post("/api/generate", content="{broken", headers={"Content-Type": "application/json"})
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid JSON in request body"


def test_proxy_generate_connect_error_returns_503(monkeypatch, client, proxy_module):
    monkeypatch.setattr(proxy_module, "inject_memory", lambda prompt, max_tokens=200: (prompt, {"facts_injected": 0}))
    monkeypatch.setattr(
        proxy_module.httpx,
        "AsyncClient",
        lambda *a, **k: _DummyAsyncClient(exc=httpx.ConnectError("offline")),
    )

    response = client.post("/api/generate", json={"prompt": "hello", "stream": False})
    assert response.status_code == 503
    assert "Cannot connect to Ollama" in response.json()["detail"]


def test_proxy_chat_connect_error_returns_503(monkeypatch, client, proxy_module):
    async def _boom(*args, **kwargs):
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(proxy_module, "_handle_chat_request", _boom)
    response = client.post("/api/chat", json={"messages": [{"role": "user", "content": "hi"}]})
    assert response.status_code == 503


def test_proxy_openai_chat_connect_error_returns_503(monkeypatch, client, proxy_module):
    async def _boom(*args, **kwargs):
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(proxy_module, "_handle_chat_request", _boom)
    response = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    assert response.status_code == 503
    assert "Cannot connect to" in response.json()["detail"]


def test_proxy_anthropic_invalid_json_returns_400(client):
    response = client.post("/v1/messages", content="{bad", headers={"Content-Type": "application/json"})
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid JSON in request body"


def test_proxy_preview_requires_query(client):
    response = client.post("/proxy/preview", json={"query": ""})
    assert response.status_code == 400
    assert response.json()["detail"] == "query is required"


def test_proxy_preview_invalid_json(client):
    response = client.post("/proxy/preview", content="{bad", headers={"Content-Type": "application/json"})
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid JSON"


def test_proxy_summarize_session_invalid_payloads(client):
    r1 = client.post("/proxy/summarize-session", content="{bad", headers={"Content-Type": "application/json"})
    assert r1.status_code == 400
    assert r1.json()["detail"] == "Invalid JSON"

    r2 = client.post("/proxy/summarize-session", json={"messages": "not-a-list"})
    assert r2.status_code == 400
    assert r2.json()["detail"] == "messages must be an array"


def test_proxy_tags_connect_error(monkeypatch, client, proxy_module):
    monkeypatch.setattr(
        proxy_module.httpx,
        "AsyncClient",
        lambda *a, **k: _DummyAsyncClient(exc=httpx.ConnectError("offline")),
    )
    response = client.get("/api/tags")
    assert response.status_code == 503
    assert response.json()["detail"] == "Ollama not running"


def test_proxy_tags_invalid_json(monkeypatch, client, proxy_module):
    response_obj = _DummyResponse(status_code=200, json_data="__invalid__")
    monkeypatch.setattr(proxy_module.httpx, "AsyncClient", lambda *a, **k: _DummyAsyncClient(response=response_obj))
    response = client.get("/api/tags")
    assert response.status_code == 502
    assert response.json()["detail"] == "Ollama returned invalid JSON"


def test_update_proxy_config_rejects_invalid_budget(client):
    response = client.post("/proxy/config", params={"budget": "ultra"})
    assert response.status_code == 400
    assert "Invalid budget" in response.json()["detail"]


def test_log_task_error_handles_cancelled_and_exception(proxy_module):
    class CancelledTask:
        def cancelled(self):
            return True

    class FailedTask:
        def cancelled(self):
            return False

        def exception(self):
            return RuntimeError("boom")

    proxy_module._log_task_error(CancelledTask())
    proxy_module._log_task_error(FailedTask())
