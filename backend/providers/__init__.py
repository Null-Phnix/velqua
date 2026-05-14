"""
Provider registry — manages available LLM providers and the active selection.

Usage:
    from backend.providers import registry
    registry.add(provider_config)
    provider = registry.get_active()
    response = await provider.chat(messages, model="gpt-4o")
"""
import json
from pathlib import Path

from backend.providers.base import BaseProvider, ProviderConfig, ChatResponse
from backend.providers.ollama import OllamaProvider
from backend.providers.openai_compat import OpenAICompatProvider
from backend.providers.anthropic import AnthropicProvider


# Maps provider name -> provider class
PROVIDER_CLASSES: dict[str, type[BaseProvider]] = {
    "ollama": OllamaProvider,
    "openai": OpenAICompatProvider,
    "groq": OpenAICompatProvider,
    "anthropic": AnthropicProvider,
    "local_openai": OpenAICompatProvider,
    "custom": OpenAICompatProvider,
}

# Default base URLs for well-known providers
DEFAULT_URLS: dict[str, str] = {
    "ollama": "http://localhost:11434",
    "openai": "https://api.openai.com",
    "groq": "https://api.groq.com/openai",
    "anthropic": "https://api.anthropic.com",
    "local_openai": "http://localhost:8080",
}


class ProviderRegistry:
    """
    Manages configured LLM providers and the currently active one.

    Provider configs are persisted in a JSON file. API keys are stored
    separately in the encrypted keystore (not in this file).
    """

    def __init__(self, config_path: Path | None = None):
        self._providers: dict[str, ProviderConfig] = {}
        self._instances: dict[str, BaseProvider] = {}
        self._active_name: str = "ollama"
        self._config_path = config_path

        # Always register Ollama as the default (no key needed)
        self._ensure_ollama()

    def _ensure_ollama(self):
        """Ensure Ollama is always available as a provider."""
        if "ollama" not in self._providers:
            self._providers["ollama"] = ProviderConfig(
                name="ollama",
                base_url=DEFAULT_URLS["ollama"],
                enabled=True,
            )

    def add(self, config: ProviderConfig) -> None:
        """Add or update a provider configuration."""
        self._providers[config.name] = config
        # Invalidate cached instance
        self._instances.pop(config.name, None)

    def remove(self, name: str) -> bool:
        """Remove a provider. Cannot remove Ollama or the active provider."""
        if name == "ollama":
            return False
        if name == self._active_name:
            return False
        if name not in self._providers:
            return False
        self._providers.pop(name, None)
        self._instances.pop(name, None)
        return True

    def get(self, name: str) -> BaseProvider | None:
        """Get a provider instance by name."""
        config = self._providers.get(name)
        if not config:
            return None

        # Cache instances to reuse connections
        if name not in self._instances:
            cls = PROVIDER_CLASSES.get(name, OpenAICompatProvider)
            self._instances[name] = cls(config)

        return self._instances[name]

    def get_active(self) -> BaseProvider:
        """Get the currently active provider."""
        provider = self.get(self._active_name)
        if provider is None:
            # Fallback to Ollama
            self._active_name = "ollama"
            provider = self.get("ollama")
        return provider

    def set_active(self, name: str) -> bool:
        """Switch the active provider. Returns False if provider doesn't exist."""
        if name not in self._providers:
            return False
        self._active_name = name
        return True

    @property
    def active_name(self) -> str:
        return self._active_name

    def list_providers(self) -> list[dict]:
        """Return all configured providers (without API keys)."""
        result = []
        for name, config in self._providers.items():
            info = config.to_dict()
            info["active"] = (name == self._active_name)
            info["has_api_key"] = bool(config.api_key)
            result.append(info)
        return result

    def get_config(self, name: str) -> ProviderConfig | None:
        """Get raw provider config by name."""
        return self._providers.get(name)

    def update_api_key(self, name: str, api_key: str) -> bool:
        """Update the API key for a provider (also refreshes the instance)."""
        config = self._providers.get(name)
        if not config:
            return False
        config.api_key = api_key
        self._instances.pop(name, None)  # Force re-instantiation
        return True

    def save(self, path: Path | None = None) -> None:
        """Persist provider configs to JSON (API keys excluded)."""
        path = path or self._config_path
        if not path:
            return
        data = {
            "active": self._active_name,
            "providers": {
                name: config.to_dict()
                for name, config in self._providers.items()
            },
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))

    def load(self, path: Path | None = None) -> None:
        """Load provider configs from JSON."""
        path = path or self._config_path
        if not path or not path.exists():
            return
        try:
            data = json.loads(path.read_text())
            for name, pdata in data.get("providers", {}).items():
                self._providers[name] = ProviderConfig.from_dict(pdata)
            active = data.get("active", "ollama")
            if active in self._providers:
                self._active_name = active
            # Clear cached instances after reload
            self._instances.clear()
            self._ensure_ollama()
        except (json.JSONDecodeError, KeyError, TypeError):
            pass  # Corrupt file — keep defaults


# Singleton registry
registry = ProviderRegistry()
