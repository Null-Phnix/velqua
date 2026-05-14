#!/usr/bin/env python3
"""
Velqua FastAPI Server — app factory and startup.

All route handlers live in backend/routes/. This file wires up
middleware, static files, and the shared memory instance.
"""
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# Add both backend dir (for anamnesis) and project root (for backend.xxx) to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from anamnesis import Anamnesis
from backend import __version__
from backend.config import VelquaConfig as Config
from backend.logging_config import setup_logging, get_logger
from backend.routes import register_routes
from backend.routes._shared import init_shared

# Initialize logging
setup_logging(level=Config.LOG_LEVEL)
logger = get_logger("server")

app = FastAPI(title="Velqua Memory API", version=__version__)

# Create data/logs directories
Config.ensure_directories()

# CORS — restricted to configured localhost origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=Config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# License check middleware — blocks API when license is expired (beyond grace).
# Trial mode and active licenses pass through. Exempts license/health/static/root.
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse as StarletteJSONResponse


class LicenseMiddleware(BaseHTTPMiddleware):
    """Return 402 Payment Required when the license is expired."""

    # Paths that are always accessible regardless of license status
    EXEMPT_PREFIXES = ("/license", "/health", "/static", "/favicon")
    EXEMPT_EXACT = ("/", "/health")

    async def dispatch(self, request, call_next):
        path = request.url.path

        # Exempt routes
        if path in self.EXEMPT_EXACT or any(path.startswith(p) for p in self.EXEMPT_PREFIXES):
            return await call_next(request)

        # Check license
        try:
            from backend.routes.license import get_license_manager
            manager = get_license_manager()
            if not manager.is_active:
                return StarletteJSONResponse(
                    status_code=402,
                    content={
                        "detail": "License expired. Please re-activate.",
                        "license_status": "expired",
                    },
                )
        except Exception:
            # If license check fails, allow through (don't brick the app)
            pass

        return await call_next(request)


app.add_middleware(LicenseMiddleware)

# Optional auth middleware — only active when VELQUA_AUTH_TOKEN is set
if Config.AUTH_TOKEN:
    class AuthMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            # Health, root, and static files are always public
            if request.url.path in ("/health", "/") or request.url.path.startswith("/static"):
                return await call_next(request)
            auth = request.headers.get("Authorization", "")
            if auth != f"Bearer {Config.AUTH_TOKEN}":
                return StarletteJSONResponse(
                    status_code=401,
                    content={"detail": "Invalid or missing auth token"},
                )
            return await call_next(request)

    app.add_middleware(AuthMiddleware)
    logger.info("Auth token enabled — API requires Bearer token")

# Initialize the shared Anamnesis instance
memory = Anamnesis(str(Config.DB_PATH))
init_shared(memory)

# Mount static files (serves index.html, styles.css, app.js)
STATIC_DIR = Path(__file__).parent.parent / "src"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def root():
    """Serve the web UI."""
    return FileResponse(str(STATIC_DIR / "index.html"))


# Register all route modules
register_routes(app)


def main():
    """Entry point for the `velqua` CLI command."""
    logger.info("Starting Velqua API server...")
    logger.info("Database: %s", Config.DB_PATH)
    logger.info("Listening on: http://%s:%d", Config.HOST, Config.PORT)
    logger.info("Config: %s", Config.get_summary())

    uvicorn.run(
        app,
        host=Config.HOST,
        port=Config.PORT,
        log_level=Config.LOG_LEVEL.lower(),
    )


if __name__ == "__main__":
    main()
