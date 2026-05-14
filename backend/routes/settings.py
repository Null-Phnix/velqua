"""
Settings routes: provider configuration, API key management, app settings.
"""
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.config import VelquaConfig as Config
from backend.logging_config import get_logger
from backend.providers import registry, ProviderConfig, DEFAULT_URLS, PROVIDER_CLASSES

logger = get_logger("routes.settings")

router = APIRouter(prefix="/settings", tags=["settings"])


# ============================================================
# Request/Response Models
# ============================================================

class ProviderRequest(BaseModel):
    name: str
    api_key: str = ""
    base_url: str = ""
    enabled: bool = True
    default_model: str = ""


class ActiveProviderRequest(BaseModel):
    name: str


class SettingsResponse(BaseModel):
    active_provider: str
    budget: str
    max_tokens: int
    auto_learning: bool = True


class SettingsUpdateRequest(BaseModel):
    budget: Optional[str] = None
    max_tokens: Optional[int] = None
    auto_learning: Optional[bool] = None


# ============================================================
# General Settings
# ============================================================

@router.get("")
async def get_settings():
    """Get all application settings."""
    # Import proxy config at runtime to avoid circular imports
    from backend.proxy import config as proxy_config, learner

    return {
        "active_provider": registry.active_name,
        "budget": proxy_config.budget,
        "max_tokens": proxy_config.max_tokens,
        "format": proxy_config.format,
        "auto_learning": learner.enabled,
        "database": str(Config.DB_PATH),
    }


@router.put("")
async def update_settings(req: SettingsUpdateRequest):
    """Update application settings."""
    from backend.proxy import config as proxy_config, learner

    if req.budget is not None:
        valid = ["minimal", "standard", "generous"]
        if req.budget not in valid:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid budget '{req.budget}'. Must be one of: {', '.join(valid)}",
            )
        proxy_config.budget = req.budget

    if req.max_tokens is not None:
        if req.max_tokens < 50 or req.max_tokens > 5000:
            raise HTTPException(
                status_code=400,
                detail="max_tokens must be between 50 and 5000",
            )
        proxy_config.max_tokens = req.max_tokens

    if req.auto_learning is not None:
        learner.enabled = req.auto_learning

    return await get_settings()


# ============================================================
# Provider Management
# ============================================================

@router.get("/providers")
async def list_providers():
    """List all configured providers (API keys masked)."""
    providers = registry.list_providers()
    # Mask API keys in response
    for p in providers:
        if p.get("has_api_key"):
            p["api_key_masked"] = "****"
    return {"providers": providers}


@router.post("/providers")
async def add_or_update_provider(req: ProviderRequest):
    """Add or update a provider configuration."""
    if req.name not in PROVIDER_CLASSES and req.name != "custom":
        valid = list(PROVIDER_CLASSES.keys()) + ["custom"]
        raise HTTPException(
            status_code=400,
            detail=f"Unknown provider '{req.name}'. Valid: {', '.join(valid)}",
        )

    base_url = req.base_url or DEFAULT_URLS.get(req.name, "")
    if not base_url:
        raise HTTPException(
            status_code=400,
            detail=f"base_url is required for provider '{req.name}'",
        )

    config = ProviderConfig(
        name=req.name,
        base_url=base_url,
        api_key=req.api_key,
        enabled=req.enabled,
        default_model=req.default_model,
    )
    registry.add(config)

    # Store API key in encrypted keystore
    if req.api_key:
        try:
            from backend.keystore import KeyStore
            ks = KeyStore(Config.DATA_DIR)
            ks.store(req.name, req.api_key)
        except Exception as e:
            logger.warning("Failed to store API key in keystore: %s", e)

    # Persist provider config
    registry.save()
    logger.info("Provider configured: %s (%s)", req.name, base_url)

    return {"status": "ok", "provider": config.to_dict()}


@router.delete("/providers/{name}")
async def remove_provider(name: str):
    """Remove a provider configuration."""
    if not registry.remove(name):
        if name == "ollama":
            raise HTTPException(status_code=400, detail="Cannot remove Ollama provider")
        if name == registry.active_name:
            raise HTTPException(
                status_code=400,
                detail="Cannot remove the active provider. Switch to another first.",
            )
        raise HTTPException(status_code=404, detail=f"Provider '{name}' not found")

    # Remove API key from keystore
    try:
        from backend.keystore import KeyStore
        ks = KeyStore(Config.DATA_DIR)
        ks.delete(name)
    except Exception:
        pass

    registry.save()
    return {"status": "ok", "removed": name}


@router.post("/providers/{name}/test")
async def test_provider(name: str):
    """Test connection to a provider (validates API key, returns models)."""
    provider = registry.get(name)
    if not provider:
        raise HTTPException(status_code=404, detail=f"Provider '{name}' not found")

    result = await provider.test_connection()
    if result["ok"]:
        # Update cached model list
        config = registry.get_config(name)
        if config:
            config.models = result["models"]
            registry.save()

    return result


@router.get("/providers/{name}/models")
async def list_provider_models(name: str):
    """List available models for a provider."""
    provider = registry.get(name)
    if not provider:
        raise HTTPException(status_code=404, detail=f"Provider '{name}' not found")

    try:
        models = await provider.list_models()
        return {"provider": name, "models": models}
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@router.post("/active-provider")
async def set_active_provider(req: ActiveProviderRequest):
    """Switch the active LLM provider."""
    if not registry.set_active(req.name):
        raise HTTPException(
            status_code=404,
            detail=f"Provider '{req.name}' not configured",
        )
    registry.save()
    logger.info("Active provider switched to: %s", req.name)
    return {"status": "ok", "active_provider": req.name}
