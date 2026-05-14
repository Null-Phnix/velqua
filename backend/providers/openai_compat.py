"""
OpenAI-compatible provider — handles OpenAI, Groq, and local backends.

All use the /v1/chat/completions format. Differences are:
- Auth header: Authorization: Bearer {api_key}
- Base URL: api.openai.com, api.groq.com, or localhost
- Model names vary
"""
from typing import AsyncIterator

import httpx

from backend.config import VelquaConfig as Config
from backend.providers.base import BaseProvider, ChatResponse, ProviderConfig


# Well-known provider defaults
KNOWN_PROVIDERS = {
    "openai": {
        "base_url": "https://api.openai.com",
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
    },
    "groq": {
        "base_url": "https://api.groq.com/openai",
        "models": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
    },
}


class OpenAICompatProvider(BaseProvider):
    """
    OpenAI-compatible chat completions provider.

    Works with: OpenAI, Groq, llama.cpp, vLLM, LM Studio, LocalAI,
    text-generation-webui, and any other OpenAI API-compatible backend.
    """

    name = "openai_compat"

    async def chat(
        self,
        messages: list[dict],
        model: str = "",
        stream: bool = False,
        **kwargs,
    ) -> ChatResponse | AsyncIterator[bytes]:
        model = self._resolve_model(model)
        body = {"model": model, "messages": messages, "stream": stream}
        # Pass through supported kwargs
        for key in ("temperature", "max_tokens", "top_p", "frequency_penalty",
                     "presence_penalty", "stop"):
            if key in kwargs:
                body[key] = kwargs[key]

        url = f"{self.config.base_url}/v1/chat/completions"
        headers = {"Content-Type": "application/json", **self.get_auth_headers()}

        if stream:
            return self._stream_chat(url, body, headers)

        async with httpx.AsyncClient(timeout=Config.PROXY_TIMEOUT) as client:
            response = await client.post(url, json=body, headers=headers)
            if response.status_code != 200:
                raise ConnectionError(
                    f"Provider returned {response.status_code}: {response.text[:200]}"
                )
            data = response.json()
            choice = data.get("choices", [{}])[0]
            msg = choice.get("message", {})
            return ChatResponse(
                content=msg.get("content", ""),
                model=data.get("model", model),
                finish_reason=choice.get("finish_reason", ""),
                usage=data.get("usage", {}),
                raw=data,
            )

    async def _stream_chat(
        self, url: str, body: dict, headers: dict
    ) -> AsyncIterator[bytes]:
        """Stream OpenAI SSE responses."""
        client = httpx.AsyncClient(timeout=Config.PROXY_TIMEOUT)
        req = client.build_request("POST", url, json=body, headers=headers)
        response = await client.send(req, stream=True)

        if response.status_code != 200:
            error_text = (await response.aread()).decode(errors="replace")[:200]
            await response.aclose()
            await client.aclose()
            raise ConnectionError(
                f"Provider returned {response.status_code}: {error_text}"
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
        url = f"{self.config.base_url}/v1/models"
        headers = self.get_auth_headers()

        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(url, headers=headers)
            if response.status_code != 200:
                raise ConnectionError(
                    f"Provider returned {response.status_code}: {response.text[:200]}"
                )
            data = response.json()
            return sorted([m["id"] for m in data.get("data", [])])

    def inject_memory(self, messages: list[dict], context: str) -> list[dict]:
        """Prepend a system message with memory context."""
        if not context:
            return messages
        memory_msg = {"role": "system", "content": context}
        return [memory_msg] + messages

    def get_auth_headers(self) -> dict:
        if self.config.api_key:
            return {"Authorization": f"Bearer {self.config.api_key}"}
        return {}


def create_openai_provider(api_key: str = "", base_url: str = "") -> OpenAICompatProvider:
    """Create an OpenAI provider with sensible defaults."""
    return OpenAICompatProvider(ProviderConfig(
        name="openai",
        base_url=base_url or KNOWN_PROVIDERS["openai"]["base_url"],
        api_key=api_key,
        models=KNOWN_PROVIDERS["openai"]["models"],
        default_model="gpt-4o-mini",
    ))


def create_groq_provider(api_key: str = "", base_url: str = "") -> OpenAICompatProvider:
    """Create a Groq provider with sensible defaults."""
    return OpenAICompatProvider(ProviderConfig(
        name="groq",
        base_url=base_url or KNOWN_PROVIDERS["groq"]["base_url"],
        api_key=api_key,
        models=KNOWN_PROVIDERS["groq"]["models"],
        default_model="llama-3.3-70b-versatile",
    ))


def create_local_openai_provider(base_url: str = "") -> OpenAICompatProvider:
    """Create a provider for local OpenAI-compatible backends (llama.cpp, vLLM, etc.)."""
    return OpenAICompatProvider(ProviderConfig(
        name="local_openai",
        base_url=base_url or Config.OPENAI_BASE_URL,
    ))
