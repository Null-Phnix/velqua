"""
Tests for the license activation system.

Tests cover: LicenseManager (activate, check, deactivate, revalidate),
license status logic (trial/active/expired), offline grace period,
and the license API endpoints.
"""
import json
import os
import tempfile
import time
import importlib
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Temp DB setup
_tmpdir = tempfile.mkdtemp()
os.environ["VELQUA_DB_PATH"] = os.path.join(_tmpdir, "test_license.db")

import backend.config
importlib.reload(backend.config)

from backend.license import (
    LicenseManager,
    LicenseStatus,
    ActivationResult,
    VALIDATION_INTERVAL_SECONDS,
    OFFLINE_GRACE_SECONDS,
)


@pytest.fixture
def manager(tmp_path):
    return LicenseManager(tmp_path)


# ==================================================================
# LicenseManager.check() — status logic
# ==================================================================

class TestLicenseCheck:
    def test_no_cache_returns_trial(self, manager):
        result = manager.check()
        assert result.status == LicenseStatus.TRIAL
        assert result.success is True

    def test_recently_validated_returns_active(self, manager):
        # Manually inject a cached license
        manager._save_cache({
            "key": "test-key-123",
            "status": "active",
            "activated_at": time.time(),
            "last_validated": time.time(),
            "customer_email": "test@example.com",
        })
        result = manager.check()
        assert result.status == LicenseStatus.ACTIVE
        assert result.success is True
        assert result.customer_email == "test@example.com"

    def test_stale_within_grace_returns_active(self, manager):
        # Validated 10 days ago (within 30-day grace)
        manager._save_cache({
            "key": "test-key-123",
            "status": "active",
            "activated_at": time.time() - 86400 * 10,
            "last_validated": time.time() - 86400 * 10,
        })
        result = manager.check()
        assert result.status == LicenseStatus.ACTIVE
        assert result.success is True
        assert "offline" in result.message.lower()

    def test_beyond_grace_returns_expired(self, manager):
        # Validated 31 days ago (beyond 30-day grace)
        manager._save_cache({
            "key": "test-key-123",
            "status": "active",
            "activated_at": time.time() - 86400 * 31,
            "last_validated": time.time() - 86400 * 31,
        })
        result = manager.check()
        assert result.status == LicenseStatus.EXPIRED
        assert result.success is False

    def test_empty_key_in_cache_returns_trial(self, manager):
        manager._save_cache({"key": "", "status": "active"})
        result = manager.check()
        assert result.status == LicenseStatus.TRIAL


# ==================================================================
# LicenseManager.activate()
# ==================================================================

class TestLicenseActivate:
    @pytest.mark.asyncio
    async def test_empty_key_rejected(self, manager):
        result = await manager.activate("")
        assert result.success is False
        assert result.status == LicenseStatus.INVALID

    @pytest.mark.asyncio
    async def test_whitespace_key_rejected(self, manager):
        result = await manager.activate("   ")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_successful_activation(self, manager):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "valid": True,
            "error": None,
            "meta": {"product_name": "Velqua", "customer_email": "user@test.com"},
            "license_key": {"status": "active", "expires_at": None},
        }

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await manager.activate("valid-key-123")

        assert result.success is True
        assert result.status == LicenseStatus.ACTIVE
        # customer_email comes from meta, not license_key
        assert result.customer_email == "user@test.com"
        assert result.product_name == "Velqua"

    @pytest.mark.asyncio
    async def test_invalid_key_activation(self, manager):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "valid": False,
            "error": None,
            "license_key": {"status": "inactive"},
            "meta": {},
        }

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await manager.activate("bad-key")

        assert result.success is False
        assert result.status == LicenseStatus.INVALID

    @pytest.mark.asyncio
    async def test_valid_false_with_error_field(self, manager):
        """LemonSqueezy returns error string on 200 when valid=false."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "valid": False,
            "error": "This license key has reached its activation limit.",
            "license_key": {"status": "inactive"},
            "meta": {},
        }

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await manager.activate("overused-key")

        assert result.success is False
        assert result.status == LicenseStatus.INVALID
        assert "activation limit" in result.message.lower()

    @pytest.mark.asyncio
    async def test_valid_false_expired_status_maps_to_expired(self, manager):
        """license_key.status == 'expired' maps to LicenseStatus.EXPIRED."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "valid": False,
            "error": "This license key has expired.",
            "license_key": {"status": "expired"},
            "meta": {},
        }

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await manager.activate("expired-key")

        assert result.success is False
        assert result.status == LicenseStatus.EXPIRED

    @pytest.mark.asyncio
    async def test_expires_at_in_past_returns_expired(self, manager):
        """valid=true but expires_at already passed → EXPIRED."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "valid": True,
            "error": None,
            "license_key": {
                "status": "active",
                "expires_at": "2020-01-01T00:00:00Z",  # clearly in the past
            },
            "meta": {"customer_email": "user@test.com", "product_name": "Velqua"},
        }

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await manager.activate("time-limited-expired-key")

        assert result.success is False
        assert result.status == LicenseStatus.EXPIRED
        assert "expired" in result.message.lower()

    @pytest.mark.asyncio
    async def test_expires_at_in_future_is_active(self, manager):
        """valid=true with expires_at in the future → ACTIVE."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "valid": True,
            "error": None,
            "license_key": {
                "status": "active",
                "expires_at": "2099-12-31T23:59:59Z",  # far future
            },
            "meta": {"customer_email": "user@test.com", "product_name": "Velqua"},
        }

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await manager.activate("time-limited-valid-key")

        assert result.success is True
        assert result.status == LicenseStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_expires_at_unparseable_treated_as_non_expiring(self, manager):
        """Unparseable expires_at should not crash — treat as no expiry."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "valid": True,
            "error": None,
            "license_key": {"status": "active", "expires_at": "not-a-date"},
            "meta": {"customer_email": "user@test.com", "product_name": "Velqua"},
        }

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await manager.activate("weird-key")

        assert result.success is True
        assert result.status == LicenseStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_429_rate_limited(self, manager):
        mock_response = MagicMock()
        mock_response.status_code = 429

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await manager.activate("any-key")

        assert result.success is False
        assert result.status == LicenseStatus.INVALID
        assert "rate" in result.message.lower() or "moment" in result.message.lower()

    @pytest.mark.asyncio
    async def test_400_bad_request_with_message(self, manager):
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {"message": "The license_key field is required."}

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await manager.activate("bad-key")

        assert result.success is False
        assert "required" in result.message.lower()

    @pytest.mark.asyncio
    async def test_422_validation_error(self, manager):
        mock_response = MagicMock()
        mock_response.status_code = 422
        mock_response.json.return_value = {"message": "The license_key format is invalid."}

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await manager.activate("bad-format")

        assert result.success is False
        assert "format" in result.message.lower() or result.status == LicenseStatus.INVALID

    @pytest.mark.asyncio
    async def test_400_json_decode_error_fallback(self, manager):
        """400 with non-JSON body falls back to generic message."""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.side_effect = ValueError("not json")

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await manager.activate("any-key")

        assert result.success is False
        assert "400" in result.message  # fallback includes status code

    @pytest.mark.asyncio
    async def test_404_key_not_found(self, manager):
        mock_response = MagicMock()
        mock_response.status_code = 404

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await manager.activate("nonexistent-key")

        assert result.success is False
        assert "not found" in result.message.lower()

    @pytest.mark.asyncio
    async def test_server_error(self, manager):
        mock_response = MagicMock()
        mock_response.status_code = 500

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await manager.activate("some-key")

        assert result.success is False

    @pytest.mark.asyncio
    async def test_network_error(self, manager):
        import httpx as httpx_mod
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.side_effect = httpx_mod.ConnectError("No internet")
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await manager.activate("key-123")

        assert result.success is False
        assert "internet" in result.message.lower() or "reach" in result.message.lower()


# ==================================================================
# LicenseManager.deactivate()
# ==================================================================

class TestLicenseDeactivate:
    def test_deactivate_returns_to_trial(self, manager):
        manager._save_cache({
            "key": "test-key",
            "status": "active",
            "last_validated": time.time(),
        })
        assert manager.check().status == LicenseStatus.ACTIVE
        assert manager.deactivate() is True
        assert manager.check().status == LicenseStatus.TRIAL

    def test_deactivate_when_no_license(self, manager):
        # Should not error even if nothing to deactivate
        result = manager.deactivate()
        # Returns True or False, but should not crash
        assert isinstance(result, bool)


# ==================================================================
# LicenseManager.revalidate()
# ==================================================================

class TestLicenseRevalidate:
    @pytest.mark.asyncio
    async def test_revalidate_no_cache_returns_trial(self, manager):
        result = await manager.revalidate()
        assert result.status == LicenseStatus.TRIAL

    @pytest.mark.asyncio
    async def test_revalidate_success_updates_last_validated(self, manager):
        old_time = time.time() - 86400 * 10  # 10 days ago
        manager._save_cache({
            "key": "test-key",
            "status": "active",
            "last_validated": old_time,
            "customer_email": "x@x.com",
        })

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "valid": True,
            "error": None,
            "license_key": {"status": "active", "expires_at": None},
            "meta": {"customer_email": "x@x.com", "product_name": "Velqua"},
        }

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await manager.revalidate()

        assert result.success is True
        assert result.status == LicenseStatus.ACTIVE
        # last_validated should now be recent
        assert manager._cached["last_validated"] > old_time

    @pytest.mark.asyncio
    async def test_revalidate_network_error_falls_back_to_cache(self, manager):
        """On network failure during revalidation, fall back to cached status."""
        manager._save_cache({
            "key": "test-key",
            "status": "active",
            "last_validated": time.time(),  # recently validated
            "customer_email": "x@x.com",
        })

        import httpx as httpx_mod
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.side_effect = httpx_mod.ConnectError("offline")
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await manager.revalidate()

        # Falls back to check() — should still be ACTIVE from cache
        assert result.status == LicenseStatus.ACTIVE


# ==================================================================
# LicenseManager properties
# ==================================================================

class TestLicenseProperties:
    def test_is_active_trial(self, manager):
        assert manager.is_active is True  # Trial counts as active

    def test_is_active_with_license(self, manager):
        manager._save_cache({
            "key": "k",
            "status": "active",
            "last_validated": time.time(),
        })
        assert manager.is_active is True

    def test_is_trial(self, manager):
        assert manager.is_trial is True

    def test_is_not_trial_when_licensed(self, manager):
        manager._save_cache({
            "key": "k",
            "status": "active",
            "last_validated": time.time(),
        })
        assert manager.is_trial is False


# ==================================================================
# License API Endpoints
# ==================================================================

class TestLicenseEndpoints:
    """Test the /license/* API routes via TestClient."""

    @pytest.fixture(scope="class")
    def client(self):
        from backend.server import app
        from fastapi.testclient import TestClient
        with TestClient(app) as c:
            yield c

    def test_license_status(self, client):
        r = client.get("/license/status")
        assert r.status_code == 200
        data = r.json()
        assert "status" in data
        assert "is_active" in data
        assert "is_trial" in data

    def test_activate_empty_key(self, client):
        r = client.post("/license/activate", json={"key": ""})
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is False

    def test_deactivate(self, client):
        r = client.post("/license/deactivate")
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is True

    def test_revalidate_endpoint(self, client):
        r = client.post("/license/revalidate")
        assert r.status_code == 200
        data = r.json()
        assert "status" in data


# ==================================================================
# Sandbox integration tests (skipped without LEMONSQUEEZY_TEST_KEY)
# ==================================================================

@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("LEMONSQUEEZY_TEST_KEY"),
    reason="Set LEMONSQUEEZY_TEST_KEY env var to run against LemonSqueezy sandbox",
)
class TestLemonSqueezyContract:
    """
    Contract tests against the real LemonSqueezy sandbox API.
    Verify our parsing matches what the API actually returns.

    Set LEMONSQUEEZY_TEST_KEY to a valid sandbox key to run.
    Set LEMONSQUEEZY_INVALID_KEY to an invalid key (optional, defaults to 'invalid-key').
    """

    @pytest.fixture
    def manager(self, tmp_path):
        return LicenseManager(tmp_path)

    @pytest.mark.asyncio
    async def test_valid_key_returns_active(self, manager):
        key = os.environ["LEMONSQUEEZY_TEST_KEY"]
        result = await manager.activate(key)
        assert result.success is True
        assert result.status == LicenseStatus.ACTIVE
        # Verify the API actually returns customer_email in meta
        assert "@" in result.customer_email or result.customer_email == ""

    @pytest.mark.asyncio
    async def test_invalid_key_returns_invalid(self, manager):
        key = os.environ.get("LEMONSQUEEZY_INVALID_KEY", "invalid-test-key-12345")
        result = await manager.activate(key)
        assert result.success is False
        assert result.status in (LicenseStatus.INVALID, LicenseStatus.EXPIRED)

    @pytest.mark.asyncio
    async def test_response_has_expected_shape(self, manager):
        """Smoke test: the API response structure hasn't changed."""
        import httpx
        from backend.license import LEMONSQUEEZY_API
        key = os.environ["LEMONSQUEEZY_TEST_KEY"]

        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.post(
                f"{LEMONSQUEEZY_API}/licenses/validate",
                json={"license_key": key},
                headers={"Accept": "application/json"},
            )

        assert r.status_code == 200
        data = r.json()
        # These top-level fields must always exist
        assert "valid" in data
        assert "meta" in data
        assert "license_key" in data
        # customer_email must be in meta (not license_key)
        assert "customer_email" in data["meta"]
