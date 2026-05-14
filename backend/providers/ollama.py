"""
Ollama provider — wraps Ollama's native /api/chat and /api/generate endpoints.

No API key needed. Base URL defaults to http://localhost:11434.
"""
from typing import AsyncIterator

import httpx

from backend.config import VelquaConfig as Config
from backend.providers.base import BaseProvider, ChatResponse, ProviderConfig


class OllamaProvider(BaseProvider):
    """Ollama local inference server."""

    name = "ollama"

    def __init__(self, config: ProviderConfig | None = None):
        if config is None:
            config = ProviderConfig(
                name="ollama",
                base_url=Config.OLLAMA_BASE_URL,
            )
        super().__init__(config)

    async def chat(
        self,
        messages: list[dict],
        model: str = "",
        stream: bool = False,
        **kwargs,
    ) -> ChatResponse | AsyncIterator[bytes]:
        model = self._resolve_model(model)
        body = {"model": model, "messages": messages, "stream": stream, **kwargs}

        if stream:
            return self._stream_chat(body)

        async with httpx.AsyncClient(timeout=Config.PROXY_TIMEOUT) as client:
            response = await client.post(
                f"{self.config.base_url}/api/chat",
                json=body,
            )
            if response.status_code != 200:
                raise ConnectionError(
                    f"Ollama returned {response.status_code}: {response.text[:200]}"
                )
            data = response.json()
            msg = data.get("message", {})
            return ChatResponse(
                content=msg.get("content", ""),
                model=data.get("model", model),
                finish_reason=data.get("done_reason", ""),
                usage={
                    "prompt_tokens": data.get("prompt_eval_count", 0),
                    "completion_tokens": data.get("eval_count", 0),
                },
                raw=data,
            )

    async def _stream_chat(self, body: dict) -> AsyncIterator[bytes]:
        """Stream Ollama NDJSON responses."""
        client = httpx.AsyncClient(timeout=Config.PROXY_TIMEOUT)
        req = client.build_request(
            "POST",
            f"{self.config.base_url}/api/chat",
            json=body,
            headers={"Content-Type": "application/json"},
        )
        response = await client.send(req, stream=True)

        if response.status_code != 200:
            error_text = (await response.aread()).decode(errors="replace")[:200]
            await response.aclose()
            await client.aclose()
            raise ConnectionError(
                f"Ollama returned {response.status_code}: {error_text}"
            )

        async def _yield():
            try:
                async for chunk in response.aiter_bytes():
                    yield chunk
            finally:
                await response.aclose()
                await client.aclose()

        return _yield()

    async def generate(
        self,
        prompt: str,
        model: str = "",
        stream: bool = False,
        **kwargs,
    ) -> dict | AsyncIterator[bytes]:
        """Ollama-specific /api/generate endpoint (prompt-based, not chat)."""
        model = self._resolve_model(model)
        body = {"model": model, "prompt": prompt, "stream": stream, **kwargs}

        if stream:
            return await self._stream_generate(body)

        async with httpx.AsyncClient(timeout=Config.PROXY_TIMEOUT) as client:
            response = await client.post(
                f"{self.config.base_url}/api/generate",
                json=body,
            )
            if response.status_code != 200:
                raise ConnectionError(
                    f"Ollama returned {response.status_code}: {response.text[:200]}"
                )
            return response.json()

    async def _stream_generate(self, body: dict) -> AsyncIterator[bytes]:
        """Stream Ollama /api/generate NDJSON responses."""
        client = httpx.AsyncClient(timeout=Config.PROXY_TIMEOUT)
        req = client.build_request(
            "POST",
            f"{self.config.base_url}/api/generate",
            json=body,
            headers={"Content-Type": "application/json"},
        )
        response = await client.send(req, stream=True)

        if response.status_code != 200:
            error_text = (await response.aread()).decode(errors="replace")[:200]
            await response.aclose()
            await client.aclose()
            raise ConnectionError(
                f"Ollama returned {response.status_code}: {error_text}"
            )

        async def _yield():
            try:
                async for chunk in response.aiter_bytes():
                    yield chunk
            finally:
                await response.aclose()
                await client.aclose()

        return _yield()

    async def list_models(self) -> list[str]:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(f"{self.config.base_url}/api/tags")
            if response.status_code != 200:
                raise ConnectionError(f"Ollama returned {response.status_code}")
            data = response.json()
            return [m["name"] for m in data.get("models", [])]

    def inject_memory(self, messages: list[dict], context: str) -> list[dict]:
        """Prepend a system message with memory context."""
        if not context:
            return messages
        memory_msg = {"role": "system", "content": context}
        return [memory_msg] + messages

    def get_auth_headers(self) -> dict:
        return {}  # Ollama needs no auth
