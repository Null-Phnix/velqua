"""
Abstract base class for LLM providers.

Every provider implements chat() for message-based completion and
list_models() for model discovery. Memory injection is handled per-provider
since different APIs have different conventions (system message vs system param).
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator


@dataclass
class ProviderConfig:
    """Configuration for a single LLM provider."""
    name: str
    base_url: str
    api_key: str = ""
    enabled: bool = True
    models: list[str] = field(default_factory=list)
    default_model: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to dict (without api_key — that goes in keystore)."""
        return {
            "name": self.name,
            "base_url": self.base_url,
            "enabled": self.enabled,
            "models": self.models,
            "default_model": self.default_model,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ProviderConfig":
        return cls(
            name=data["name"],
            base_url=data.get("base_url", ""),
            api_key=data.get("api_key", ""),
            enabled=data.get("enabled", True),
            models=data.get("models", []),
            default_model=data.get("default_model", ""),
            extra=data.get("extra", {}),
        )


@dataclass
class ChatMessage:
    """A single message in a conversation."""
    role: str  # "system", "user", "assistant"
    content: str


@dataclass
class ChatResponse:
    """Normalized response from any provider."""
    content: str
    model: str = ""
    finish_reason: str = ""
    usage: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)


class BaseProvider(ABC):
    """Abstract base for all LLM providers."""

    name: str = ""

    def __init__(self, config: ProviderConfig):
        self.config = config

    @abstractmethod
    async def chat(
        self,
        messages: list[dict],
        model: str = "",
        stream: bool = False,
        **kwargs,
    ) -> ChatResponse | AsyncIterator[bytes]:
        """
        Send a chat completion request.

        When stream=False, returns a ChatResponse.
        When stream=True, returns an AsyncIterator of raw bytes (SSE or NDJSON).
        """

    @abstractmethod
    async def list_models(self) -> list[str]:
        """Return available model names from this provider."""

    @abstractmethod
    def inject_memory(self, messages: list[dict], context: str) -> list[dict]:
        """
        Inject memory context into the message list.

        Different APIs handle system messages differently:
        - OpenAI/Ollama: prepend {"role": "system", "content": context}
        - Anthropic: system goes as a top-level param, not in messages
        """

    async def test_connection(self) -> dict:
        """
        Test that the provider is reachable and the API key is valid.
        Returns {"ok": True, "models": [...]} on success,
        {"ok": False, "error": "..."} on failure.
        """
        try:
            models = await self.list_models()
            return {"ok": True, "models": models}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def get_auth_headers(self) -> dict:
        """Return provider-specific auth headers. Override in subclasses."""
        return {}

    def _resolve_model(self, model: str) -> str:
        """Use provided model, fall back to default, or first available."""
        if model:
            return model
        if self.config.default_model:
            return self.config.default_model
        if self.config.models:
            return self.config.models[0]
        return ""
