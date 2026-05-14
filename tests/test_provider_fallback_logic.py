"""Tests for provider registry fallback and active-provider behavior."""
from __future__ import annotations

from pathlib import Path

import json

from backend.providers import ProviderRegistry, ProviderConfig


def test_get_active_falls_back_to_first_enabled_provider():
    registry = ProviderRegistry()
    registry._providers["openai"] = ProviderConfig(
        name="openai", enabled=True, api_key="k",
        base_url="https://api.openai.com", default_model="gpt-4o-mini",
    )
    # Point active_name at a nonexistent provider
    registry._active_name = "missing-provider"

    provider = registry.get_active()
    assert provider is not None
    # Should fall back to ollama (always registered by default)
    assert provider.config.name == "ollama"


def test_get_active_returns_named_provider_when_present():
    registry = ProviderRegistry()
    registry._providers["openai"] = ProviderConfig(
        name="openai", enabled=True, api_key="k",
        base_url="https://api.openai.com", default_model="gpt-4o-mini",
    )
    registry._active_name = "openai"

    provider = registry.get_active()
    assert provider.config.name == "openai"


def test_set_active_provider_ignores_unknown_name():
    registry = ProviderRegistry()
    registry._active_name = "ollama"

    result = registry.set_active("does-not-exist")
    assert result is False
    assert registry.active_name == "ollama"


def test_load_handles_missing_file(tmp_path: Path):
    registry = ProviderRegistry()
    missing = tmp_path / "missing.json"
    registry.load(missing)
    # Registry should still be valid with ollama as default
    assert registry.get(registry.active_name) is not None


def test_save_and_reload_round_trip(tmp_path: Path):
    registry = ProviderRegistry()
    registry._providers["openai"] = ProviderConfig(
        name="openai",
        enabled=True,
        api_key="secret",
        base_url="https://api.openai.com",
        default_model="gpt-4o-mini",
    )
    registry._active_name = "openai"

    config_path = tmp_path / "providers.json"
    registry.save(config_path)

    assert config_path.exists()
    raw = json.loads(config_path.read_text())
    # Save format uses "active" key, not "active_provider"
    assert raw["active"] == "openai"

    registry2 = ProviderRegistry()
    registry2.load(config_path)
    assert registry2.active_name == "openai"
    assert registry2.get_config("openai").base_url == "https://api.openai.com"
    assert registry2.get_config("openai").enabled is True
