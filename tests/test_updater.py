"""
Tests for the auto-update version checker.

Tests cover: version parsing, version comparison, update check with mocked HTTP.
"""
import os
import tempfile
import importlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Temp DB setup
_tmpdir = tempfile.mkdtemp()
os.environ["VELQUA_DB_PATH"] = os.path.join(_tmpdir, "test_updater.db")

import backend.config
importlib.reload(backend.config)

from backend.updater import (
    _parse_version,
    is_newer,
    check_for_updates,
    UpdateInfo,
)


class TestVersionParsing:
    def test_simple_version(self):
        assert _parse_version("2.0.0") == (2, 0, 0, 4)

    def test_alpha_version(self):
        result = _parse_version("2.0.0-alpha.2")
        assert result[3] == 1  # alpha = 1

    def test_beta_version(self):
        result = _parse_version("2.0.0-beta.1")
        assert result[3] == 2  # beta = 2

    def test_rc_version(self):
        result = _parse_version("2.0.0-rc.1")
        assert result[3] == 3  # rc = 3

    def test_strip_v_prefix(self):
        assert _parse_version("v2.0.0") == _parse_version("2.0.0")

    def test_release_sorts_after_prerelease(self):
        assert _parse_version("2.0.0") > _parse_version("2.0.0-alpha.1")
        assert _parse_version("2.0.0") > _parse_version("2.0.0-beta.1")
        assert _parse_version("2.0.0") > _parse_version("2.0.0-rc.1")


class TestIsNewer:
    def test_newer_version(self):
        assert is_newer("2.1.0", "2.0.0") is True

    def test_same_version(self):
        assert is_newer("2.0.0", "2.0.0") is False

    def test_older_version(self):
        assert is_newer("1.9.0", "2.0.0") is False

    def test_alpha_to_release(self):
        assert is_newer("2.0.0", "2.0.0-alpha.2") is True

    def test_alpha_to_beta(self):
        assert is_newer("2.0.0-beta.1", "2.0.0-alpha.2") is True

    def test_patch_bump(self):
        assert is_newer("2.0.1", "2.0.0") is True

    def test_major_bump(self):
        assert is_newer("3.0.0", "2.9.9") is True

    def test_invalid_versions(self):
        # Should not crash on bad input
        assert is_newer("not.a.version", "also.bad") is False


class TestCheckForUpdates:
    @pytest.mark.asyncio
    async def test_update_available(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "tag_name": "v99.0.0",
            "html_url": "https://github.com/velqua/velqua/releases/v99.0.0",
            "body": "Big update!",
        }

        with patch("httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            info = await check_for_updates()

        assert info.update_available is True
        assert info.latest_version == "99.0.0"
        assert info.release_url == "https://github.com/velqua/velqua/releases/v99.0.0"
        assert info.release_notes == "Big update!"
        assert info.error == ""

    @pytest.mark.asyncio
    async def test_no_update(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "tag_name": "v0.0.1",
            "html_url": "",
            "body": "",
        }

        with patch("httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            info = await check_for_updates()

        assert info.update_available is False

    @pytest.mark.asyncio
    async def test_api_error(self):
        mock_response = MagicMock()
        mock_response.status_code = 403

        with patch("httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            info = await check_for_updates()

        assert info.update_available is False
        assert "403" in info.error

    @pytest.mark.asyncio
    async def test_network_error(self):
        import httpx as httpx_mod

        with patch("httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.get.side_effect = httpx_mod.ConnectError("No internet")
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            info = await check_for_updates()

        assert info.update_available is False
        assert "internet" in info.error.lower() or "reach" in info.error.lower()

    @pytest.mark.asyncio
    async def test_no_tag_in_response(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {}

        with patch("httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            info = await check_for_updates()

        assert info.update_available is False
        assert "tag" in info.error.lower()


class TestUpdateCheckEndpoint:
    """Test the /update/check API route via TestClient."""

    @pytest.fixture(scope="class")
    def client(self):
        from backend.server import app
        from fastapi.testclient import TestClient
        with TestClient(app) as c:
            yield c

    def test_update_check_returns_200(self, client):
        with patch("backend.updater.check_for_updates") as mock_check:
            mock_check.return_value = UpdateInfo(
                update_available=False,
                current_version="2.0.0-alpha.2",
                latest_version="2.0.0-alpha.2",
            )
            r = client.get("/update/check")
        assert r.status_code == 200
        data = r.json()
        assert "update_available" in data
        assert "current_version" in data


class TestIsNewerEdgeCases:
    """Cover is_newer() exception path (line 68-69)."""

    def test_is_newer_malformed_version_returns_false(self):
        """is_newer() returns False when _parse_version raises (malformed version)."""
        with patch("backend.updater._parse_version", side_effect=ValueError("bad version")):
            result = is_newer("not-a-version", "1.0.0")
        assert result is False


class TestCheckForUpdatesExceptions:
    """Cover check_for_updates() exception paths (lines 127-129)."""

    @pytest.mark.asyncio
    async def test_check_for_updates_general_exception(self):
        """General exception (not ConnectError) returns error UpdateInfo."""
        with patch("backend.updater.httpx.AsyncClient") as mock_cls:
            mock_instance = AsyncMock()
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            mock_instance.get.side_effect = RuntimeError("unexpected error")
            mock_cls.return_value = mock_instance

            result = await check_for_updates("http://example.com/releases")

        assert result.update_available is False
        assert "unexpected error" in result.error
