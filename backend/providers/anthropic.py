"""
Anthropic provider — handles Claude models via the Messages API.

Key differences from OpenAI format:
- System message is a top-level 'system' parameter, NOT in the messages array
- Auth uses 'x-api-key' header, NOT 'Authorization: Bearer'
- Streaming uses content_block_delta events (not choices[0].delta)
- Response format: content[0].text (not choices[0].message.content)
"""
import json
from typing import AsyncIterator

import httpx

from backend.config import VelquaConfig as Config
from backend.providers.base import BaseProvider, ChatResponse, ProviderConfig

ANTHROPIC_BASE_URL = "https://api.anthropic.com"
ANTHROPIC_API_VERSION = "2023-06-01"

KNOWN_MODELS = [
    "claude-opus-4-0",
    "claude-sonnet-4-5-20250514",
    "claude-haiku-4-5-20251001",
    "claude-sonnet-4-0-20250514",
]


class AnthropicProvider(BaseProvider):
    """Anthropic Messages API provider for Claude models."""

    name = "anthropic"

    def __init__(self, config: ProviderConfig | None = None):
        if config is None:
            config = ProviderConfig(
                name="anthropic",
                base_url=ANTHROPIC_BASE_URL,
                models=KNOWN_MODELS,
                default_model="claude-sonnet-4-5-20250514",
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

        # Extract system message from messages array (Anthropic wants it as a param)
        system_text, user_messages = self._extract_system(messages)

        body = {
            "model": model,
            "messages": user_messages,
            "max_tokens": kwargs.get("max_tokens", 4096),
        }
        if system_text:
            body["system"] = system_text
        if "temperature" in kwargs:
            body["temperature"] = kwargs["temperature"]
        if "top_p" in kwargs:
            body["top_p"] = kwargs["top_p"]
        if "stop" in kwargs:
            body["stop_sequences"] = kwargs["stop"]
        if stream:
            body["stream"] = True

        url = f"{self.config.base_url}/v1/messages"
        headers = {"Content-Type": "application/json", **self.get_auth_headers()}

        if stream:
            return self._stream_chat(url, body, headers)

        async with httpx.AsyncClient(timeout=Config.PROXY_TIMEOUT) as client:
            response = await client.post(url, json=body, headers=headers)
            if response.status_code != 200:
                raise ConnectionError(
                    f"Anthropic returned {response.status_code}: {response.text[:200]}"
                )
            data = response.json()
            content_blocks = data.get("content", [])
            text = "".join(
                block["text"] for block in content_blocks if block.get("type") == "text"
            )
            return ChatResponse(
                content=text,
                model=data.get("model", model),
                finish_reason=data.get("stop_reason", ""),
                usage=data.get("usage", {}),
                raw=data,
            )

    async def _stream_chat(
        self, url: str, body: dict, headers: dict
    ) -> AsyncIterator[bytes]:
        """Stream Anthropic SSE responses."""
        client = httpx.AsyncClient(timeout=Config.PROXY_TIMEOUT)
        req = client.build_request("POST", url, json=body, headers=headers)
        response = await client.send(req, stream=True)

        if response.status_code != 200:
            error_text = (await response.aread()).decode(errors="replace")[:200]
            await response.aclose()
            await client.aclose()
            raise ConnectionError(
                f"Anthropic returned {response.status_code}: {error_text}"
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
        """Return known Claude models (Anthropic has no /models endpoint)."""
        # Anthropic doesn't have a public model listing API.
        # Return known models, optionally validating the API key.
        if self.config.api_key:
            # Validate the key with a minimal request
            url = f"{self.config.base_url}/v1/messages"
            headers = self.get_auth_headers()
            body = {
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 1,
                "messages": [{"role": "user", "content": "hi"}],
            }
            async with httpx.AsyncClient(timeout=15) as client:
                response = await client.post(url, json=body, headers=headers)
                if response.status_code == 401:
                    raise ConnectionError("Invalid API key")
                # Any non-401 response means the key is valid
                # (we might get 200 or a rate limit, either confirms the key works)
        return list(KNOWN_MODELS)

    def inject_memory(self, messages: list[dict], context: str) -> list[dict]:
        """
        For Anthropic, memory goes into the system parameter, not messages.

        This method prepends memory to any existing system message content.
        The caller should extract the system text from the returned messages
        and pass it as the 'system' body parameter.

        We use a convention: if the first message is role=system, it contains
        the memory context + any original system prompt combined.
        """
        if not context:
            return messages

        # Check if there's already a system message
        if messages and messages[0].get("role") == "system":
            existing = messages[0]["content"]
            combined = f"{context}\n\n{existing}"
            return [{"role": "system", "content": combined}] + messages[1:]

        return [{"role": "system", "content": context}] + messages

    def get_auth_headers(self) -> dict:
        headers = {
            "anthropic-version": ANTHROPIC_API_VERSION,
        }
        if self.config.api_key:
            headers["x-api-key"] = self.config.api_key
        return headers

    @staticmethod
    def _extract_system(messages: list[dict]) -> tuple[str, list[dict]]:
        """
        Extract system messages from the messages array.

        Returns (system_text, remaining_messages) where system_text is all
        system message contents joined, and remaining_messages has no system role.
        """
        system_parts = []
        user_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                system_parts.append(msg["content"])
            else:
                user_messages.append(msg)
        return "\n\n".join(system_parts), user_messages

    def format_for_proxy(self, body: dict) -> dict:
        """
        Convert an Anthropic Messages API request into internal format.

        This is used when Velqua receives a request on /v1/messages from
        an app using the Anthropic SDK. We need to normalize it before
        routing through the memory injection pipeline.
        """
        messages = body.get("messages", [])
        system = body.get("system", "")
        if system:
            messages = [{"role": "system", "content": system}] + messages
        return {
            "messages": messages,
            "model": body.get("model", ""),
            "stream": body.get("stream", False),
            "max_tokens": body.get("max_tokens", 4096),
            "temperature": body.get("temperature"),
        }

    def format_response_for_anthropic(self, response: ChatResponse, model: str) -> dict:
        """
        Convert a ChatResponse back into Anthropic Messages API format.

        Used when proxying to non-Anthropic backends but the client
        expects Anthropic response format (because they called /v1/messages).
        """
        return {
            "id": "msg_velqua",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": response.content}],
            "model": model,
            "stop_reason": response.finish_reason or "end_turn",
            "usage": response.usage or {"input_tokens": 0, "output_tokens": 0},
        }
