"""
License activation and validation via LemonSqueezy.

Flow:
1. User enters license key → POST /license/activate
2. Key validated against LemonSqueezy API → cached in data/license.enc
3. Subsequent launches check cache first, re-validate online weekly
4. Offline → cached activation valid for 30 days
5. Failed validation after grace → "Please re-activate" (data NOT deleted)

The license system is deliberately lenient:
- 30-day offline grace period (people travel, networks go down)
- Never deletes user data (that's hostile)
- Graceful degradation: if license check fails, log and continue
- Trial mode works indefinitely with a nag banner (not a hard lock)
"""
import json
import os
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import httpx

from backend.logging_config import get_logger

logger = get_logger("license")

LEMONSQUEEZY_API = "https://api.lemonsqueezy.com/v1"
VALIDATION_INTERVAL_SECONDS = 7 * 24 * 3600  # Re-validate weekly
OFFLINE_GRACE_SECONDS = 30 * 24 * 3600  # 30-day offline grace

DEV_MODE = os.getenv("VELQUA_DEV", "") in ("1", "true", "yes")


class LicenseStatus(str, Enum):
    ACTIVE = "active"
    TRIAL = "trial"
    EXPIRED = "expired"
    INVALID = "invalid"


class LicenseTier(str, Enum):
    FREE = "free"
    PRO = "pro"
    TEAMS = "teams"


# Feature flags gated by tier
TIER_FEATURES = {
    LicenseTier.FREE: {
        "max_facts": 100,
        "max_agents": 1,
        "import_formats": ["claude", "chatgpt"],
        "mesh_enabled": False,
        "shared_memory": False,
        "agent_dashboard": False,
        "custom_providers": True,
        "desktop_app": True,
        "api_access": True,
        "nag_banner": True,
    },
    LicenseTier.PRO: {
        "max_facts": 10000,
        "max_agents": 5,
        "import_formats": ["claude", "chatgpt", "obsidian", "notion"],
        "mesh_enabled": True,
        "shared_memory": False,
        "agent_dashboard": True,
        "custom_providers": True,
        "desktop_app": True,
        "api_access": True,
        "nag_banner": False,
    },
    LicenseTier.TEAMS: {
        "max_facts": 100000,
        "max_agents": 50,
        "import_formats": ["claude", "chatgpt", "obsidian", "notion"],
        "mesh_enabled": True,
        "shared_memory": True,
        "agent_dashboard": True,
        "custom_providers": True,
        "desktop_app": True,
        "api_access": True,
        "nag_banner": False,
    },
}


TIER_PRODUCT_NAMES = {
    "Velqua Free": LicenseTier.FREE,
    "Velqua Pro": LicenseTier.PRO,
    "Velqua Teams": LicenseTier.TEAMS,
}


class TierManager:
    """Resolves tier from license activation, returns feature flags."""

    def __init__(self, data_dir: Path):
        self._data_dir = data_dir

    @staticmethod
    def features_for_tier(tier: LicenseTier) -> dict:
        return dict(TIER_FEATURES.get(tier, TIER_FEATURES[LicenseTier.FREE]))

    def current_tier(self) -> LicenseTier:
        """Read current tier from cached license or default to free."""
        if DEV_MODE:
            return LicenseTier.PRO
        try:
            from backend.keystore import KeyStore
            ks = KeyStore(self._data_dir)
            raw = ks.get("_velqua_license")
            if raw:
                import json
                cache = json.loads(raw)
                product = cache.get("product_name", "")
                tier = TIER_PRODUCT_NAMES.get(product)
                if tier:
                    return tier
        except Exception:
            pass
        return LicenseTier.FREE

    def current_features(self) -> dict:
        return self.features_for_tier(self.current_tier())


@dataclass
class ActivationResult:
    success: bool
    status: LicenseStatus
    message: str
    license_key: str = ""
    customer_email: str = ""
    product_name: str = ""
    valid_until: float = 0  # Unix timestamp


class LicenseManager:
    """
    Manages license activation, caching, and validation.

    License data is stored encrypted using the existing KeyStore.
    The 'license' key in the keystore holds a JSON blob with
    activation details and timestamps.
    """

    LICENSE_STORE_KEY = "_velqua_license"

    def __init__(self, data_dir: Path):
        self._data_dir = data_dir
        self._cached: dict | None = None

    def _get_keystore(self):
        """Lazy import to avoid circular deps."""
        from backend.keystore import KeyStore
        return KeyStore(self._data_dir)

    def _load_cache(self) -> dict:
        """Load cached license from keystore."""
        if self._cached is not None:
            return self._cached
        try:
            ks = self._get_keystore()
            raw = ks.get(self.LICENSE_STORE_KEY)
            if raw:
                self._cached = json.loads(raw)
                return self._cached
        except Exception as e:
            logger.warning("Failed to load license cache: %s", e)
        return {}

    def _save_cache(self, data: dict) -> None:
        """Save license data to encrypted keystore."""
        try:
            ks = self._get_keystore()
            ks.store(self.LICENSE_STORE_KEY, json.dumps(data))
            self._cached = data
        except Exception as e:
            logger.warning("Failed to save license cache: %s", e)

    async def activate(self, key: str) -> ActivationResult:
        """
        Validate a license key against LemonSqueezy and cache the result.
        """
        if DEV_MODE and key == "dev":
            return ActivationResult(
                success=True, status=LicenseStatus.ACTIVE,
                message="Dev mode — license bypassed",
                license_key="dev", product_name="Velqua Pro",
            )
        if not key or not key.strip():
            return ActivationResult(
                success=False,
                status=LicenseStatus.INVALID,
                message="License key cannot be empty",
            )

        key = key.strip()

        try:
            result = await self._validate_with_api(key)
            if result.success:
                # Cache the activation
                cache_data = {
                    "key": key,
                    "status": result.status.value,
                    "activated_at": time.time(),
                    "last_validated": time.time(),
                    "customer_email": result.customer_email,
                    "product_name": result.product_name,
                }
                self._save_cache(cache_data)
                logger.info("License activated successfully")
            return result

        except httpx.ConnectError:
            return ActivationResult(
                success=False,
                status=LicenseStatus.INVALID,
                message="Cannot reach license server. Check your internet connection.",
            )
        except Exception as e:
            logger.error("License activation failed: %s", e)
            return ActivationResult(
                success=False,
                status=LicenseStatus.INVALID,
                message=f"Activation failed: {str(e)}",
            )

    async def _validate_with_api(self, key: str) -> ActivationResult:
        """
        Call LemonSqueezy validation API.

        API response shape (200):
          {
            "valid": true|false,
            "error": null|"<reason>",
            "license_key": {
              "status": "active"|"inactive"|"expired"|"disabled",
              "activation_limit": 5,
              "activation_usage": 1,
              "expires_at": null | "<ISO8601>",
              ...
            },
            "meta": {
              "product_name": "Velqua",
              "customer_email": "user@example.com",
              "customer_name": "...",
              ...
            }
          }

        Non-200 codes:
          400 — malformed request body
          404 — license key not found
          422 — validation error (key format wrong)
          429 — rate limited
        """
        url = f"{LEMONSQUEEZY_API}/licenses/validate"

        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.post(
                url,
                json={"license_key": key},
                headers={"Accept": "application/json"},
            )

            if response.status_code == 429:
                return ActivationResult(
                    success=False,
                    status=LicenseStatus.INVALID,
                    message="License server is rate-limiting requests. Try again in a moment.",
                )

            if response.status_code == 404:
                return ActivationResult(
                    success=False,
                    status=LicenseStatus.INVALID,
                    message="License key not found.",
                )

            if response.status_code in (400, 422):
                try:
                    detail = response.json().get("message", "Invalid request")
                except Exception:
                    detail = f"Bad request ({response.status_code})"
                return ActivationResult(
                    success=False,
                    status=LicenseStatus.INVALID,
                    message=detail,
                )

            if response.status_code != 200:
                return ActivationResult(
                    success=False,
                    status=LicenseStatus.INVALID,
                    message=f"License server error ({response.status_code})",
                )

            data = response.json()
            valid = data.get("valid", False)
            error_msg = data.get("error")  # human-readable reason when valid=false
            meta = data.get("meta", {})
            license_data = data.get("license_key", {})

            if not valid:
                reason = error_msg or "License is not valid"
                # Map known LemonSqueezy error messages to cleaner status
                status = LicenseStatus.INVALID
                if license_data.get("status") == "expired":
                    status = LicenseStatus.EXPIRED
                return ActivationResult(
                    success=False,
                    status=status,
                    message=reason,
                    license_key=key,
                )

            # Check expiry even when valid=true (time-limited licenses)
            expires_at_str = license_data.get("expires_at")
            if expires_at_str:
                try:
                    from datetime import datetime, timezone
                    expires_at = datetime.fromisoformat(
                        expires_at_str.replace("Z", "+00:00")
                    )
                    if expires_at < datetime.now(timezone.utc):
                        return ActivationResult(
                            success=False,
                            status=LicenseStatus.EXPIRED,
                            message="License has expired",
                            license_key=key,
                        )
                except Exception:
                    pass  # Unparseable date — treat as non-expiring

            return ActivationResult(
                success=True,
                status=LicenseStatus.ACTIVE,
                message="License activated",
                license_key=key,
                customer_email=meta.get("customer_email", ""),  # correct field: meta, not license_key
                product_name=meta.get("product_name", "Velqua"),
            )

    def check(self) -> ActivationResult:
        """
        Check current license status from cache.
        """
        if DEV_MODE:
            return ActivationResult(
                success=True, status=LicenseStatus.ACTIVE,
                message="Dev mode — license bypassed",
                license_key="dev", product_name="Velqua Pro",
            )
        cache = self._load_cache()

        if not cache or not cache.get("key"):
            return ActivationResult(
                success=True,
                status=LicenseStatus.TRIAL,
                message="No license activated — running in trial mode",
            )

        last_validated = cache.get("last_validated", 0)
        now = time.time()
        age = now - last_validated

        if age < VALIDATION_INTERVAL_SECONDS:
            # Recently validated
            return ActivationResult(
                success=True,
                status=LicenseStatus.ACTIVE,
                message="License active",
                license_key=cache.get("key", ""),
                customer_email=cache.get("customer_email", ""),
                product_name=cache.get("product_name", "Velqua"),
            )

        if age < OFFLINE_GRACE_SECONDS:
            # Stale but within grace period
            days_left = int((OFFLINE_GRACE_SECONDS - age) / 86400)
            return ActivationResult(
                success=True,
                status=LicenseStatus.ACTIVE,
                message=f"License active (offline, re-validate within {days_left} days)",
                license_key=cache.get("key", ""),
                customer_email=cache.get("customer_email", ""),
            )

        # Beyond grace period
        return ActivationResult(
            success=False,
            status=LicenseStatus.EXPIRED,
            message="License needs re-activation. Please connect to the internet.",
            license_key=cache.get("key", ""),
        )

    async def revalidate(self) -> ActivationResult:
        """
        Re-validate the cached license key online.
        Called periodically (weekly) to keep the cache fresh.
        """
        cache = self._load_cache()
        key = cache.get("key")
        if not key:
            return self.check()

        try:
            result = await self._validate_with_api(key)
            if result.success:
                cache["last_validated"] = time.time()
                cache["status"] = result.status.value
                self._save_cache(cache)
            return result
        except Exception:
            # Network error — use cached status
            return self.check()

    def deactivate(self) -> bool:
        """
        Remove the license activation. Returns to trial mode.
        Does NOT delete user data — just the license cache.
        """
        try:
            ks = self._get_keystore()
            ks.delete(self.LICENSE_STORE_KEY)
            self._cached = None
            logger.info("License deactivated")
            return True
        except Exception as e:
            logger.error("Failed to deactivate license: %s", e)
            return False

    @property
    def is_active(self) -> bool:
        """Quick check: is the license active or in trial?"""
        result = self.check()
        return result.status in (LicenseStatus.ACTIVE, LicenseStatus.TRIAL)

    @property
    def is_trial(self) -> bool:
        """Is this running in trial mode (no license)?"""
        return self.check().status == LicenseStatus.TRIAL
