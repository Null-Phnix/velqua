"""
Route modules for the Velqua API.

Each module defines a FastAPI APIRouter for a specific domain:
  - facts: CRUD, search, merge, bulk operations, tags, types, feedback
  - imports: smart import, chatgpt import, fact JSON import/export
  - review: pending fact approval/rejection queue
  - backup: database backup, restore, export
  - system: health check, import history, contradiction detection
  - settings: provider configuration, API key management, app settings
  - license: license activation, status check, deactivation
  - mesh: multi-agent coordination (registry, shared memory, noteboard, websocket)
  - activity: chronological feed of system events
  - graph: fact-to-fact relationship detection and queries
"""
from fastapi import APIRouter

from backend.routes.facts import router as facts_router
from backend.routes.imports import router as imports_router
from backend.routes.review import router as review_router
from backend.routes.backup import router as backup_router
from backend.routes.system import router as system_router
from backend.routes.settings import router as settings_router
from backend.routes.license import router as license_router
from backend.routes.mesh import router as mesh_router
from backend.routes.episodes import router as episodes_router
from backend.routes.activity import router as activity_router
from backend.routes.graph import router as graph_router


def register_routes(app):
    """Mount all route modules onto the FastAPI app."""
    app.include_router(facts_router)
    app.include_router(imports_router)
    app.include_router(review_router)
    app.include_router(backup_router)
    app.include_router(system_router)
    app.include_router(settings_router)
    app.include_router(license_router)
    app.include_router(mesh_router)
    app.include_router(episodes_router)
    app.include_router(activity_router)
    app.include_router(graph_router)
