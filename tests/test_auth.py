"""
Tests for auth middleware.

Tests the auth middleware logic using a standalone FastAPI app with the
middleware injected directly. Does NOT set VELQUA_AUTH_TOKEN env var to
avoid leaking state into other test modules.
"""
import os
import tempfile
import importlib

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse as StarletteJSONResponse

from anamnesis import Anamnesis
from backend.routes import register_routes
from backend.routes._shared import init_shared

# Use a temp DB (conftest.py handles sys.path)
_tmpdir = tempfile.mkdtemp()
os.environ["VELQUA_DB_PATH"] = os.path.join(_tmpdir, "test_auth.db")

import backend.config
importlib.reload(backend.config)
from backend.config import VelquaConfig as Config

TEST_TOKEN = "test-secret-token-12345"


def _create_auth_app():
    """Build a fresh FastAPI app with auth middleware enabled."""
    app = FastAPI(title="Velqua Auth Test")

    class AuthMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            # Health, root, and static files are always public
            if request.url.path in ("/health", "/") or request.url.path.startswith("/static"):
                return await call_next(request)
            auth = request.headers.get("Authorization", "")
            if auth != f"Bearer {TEST_TOKEN}":
                return StarletteJSONResponse(
                    status_code=401,
                    content={"detail": "Invalid or missing auth token"},
                )
            return await call_next(request)

    app.add_middleware(AuthMiddleware)

    mem = Anamnesis(str(Config.DB_PATH))
    init_shared(mem)
    register_routes(app)

    @app.get("/")
    async def root():
        return {"status": "ok"}

    return app


_auth_app = _create_auth_app()

AUTH_HEADER = {"Authorization": f"Bearer {TEST_TOKEN}"}
BAD_HEADER = {"Authorization": "Bearer wrong-token"}


@pytest.fixture
def client():
    return TestClient(_auth_app)


class TestAuthMiddleware:
    """Test that auth middleware protects endpoints correctly."""

    def test_health_is_public(self, client):
        """Health endpoint should work without auth."""
        r = client.get("/health")
        assert r.status_code == 200

    def test_root_is_public(self, client):
        """Root endpoint should work without auth."""
        r = client.get("/")
        assert r.status_code == 200

    def test_protected_endpoint_no_token(self, client):
        """API endpoints should return 401 without auth token."""
        r = client.get("/facts/list")
        assert r.status_code == 401
        assert "auth token" in r.json()["detail"].lower()

    def test_protected_endpoint_wrong_token(self, client):
        """API endpoints should return 401 with wrong token."""
        r = client.get("/facts/list", headers=BAD_HEADER)
        assert r.status_code == 401

    def test_protected_endpoint_correct_token(self, client):
        """API endpoints should work with correct auth token."""
        r = client.get("/facts/list", headers=AUTH_HEADER)
        assert r.status_code == 200

    def test_search_requires_auth(self, client):
        """Search endpoint requires auth."""
        r = client.get("/facts/search?q=test")
        assert r.status_code == 401

        r = client.get("/facts/search?q=test", headers=AUTH_HEADER)
        assert r.status_code == 200

    def test_review_requires_auth(self, client):
        """Review endpoint requires auth."""
        r = client.get("/review/pending")
        assert r.status_code == 401

        r = client.get("/review/pending", headers=AUTH_HEADER)
        assert r.status_code == 200

    def test_import_requires_auth(self, client):
        """Import history endpoint requires auth."""
        r = client.get("/import/history")
        assert r.status_code == 401

        r = client.get("/import/history", headers=AUTH_HEADER)
        assert r.status_code == 200

    def test_backup_requires_auth(self, client):
        """Backup endpoint requires auth."""
        r = client.post("/backup/create")
        assert r.status_code == 401

    def test_export_requires_auth(self, client):
        """Export endpoint requires auth."""
        r = client.get("/export/facts")
        assert r.status_code == 401

        r = client.get("/export/facts", headers=AUTH_HEADER)
        assert r.status_code == 200
