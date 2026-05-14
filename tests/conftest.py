"""Shared test configuration for Velqua tests."""
import os
import sys
import warnings
from pathlib import Path

import pytest

# Ensure backend modules are importable
VELQUA_ROOT = Path(__file__).parent.parent
BACKEND_DIR = VELQUA_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(VELQUA_ROOT))


@pytest.fixture
def temp_db(tmp_path):
    """Create a temporary database path and set env var before server import."""
    db_path = str(tmp_path / "test_velqua.db")
    os.environ["VELQUA_DB_PATH"] = db_path
    yield db_path
    os.environ.pop("VELQUA_DB_PATH", None)


@pytest.fixture(autouse=True, scope="session")
def _close_mesh_db():
    """Close the mesh thread-local connection at the end of the test session."""
    yield
    try:
        from backend.mesh.db import close_conn
        close_conn()
    except Exception:
        pass


@pytest.fixture(autouse=True, scope="session")
def _close_anamnesis_backend():
    """Close the Anamnesis SQLite backend at session end."""
    yield
    try:
        from backend.proxy import memory
        if hasattr(memory, "semantic") and hasattr(memory.semantic, "store"):
            memory.semantic.store.close()
    except Exception:
        pass


def pytest_configure(config):
    """Suppress GC cleanup warnings that fire during test teardown."""
    # Direct ResourceWarnings from sqlite3 and asyncio
    warnings.filterwarnings("ignore", message=r"unclosed database", category=ResourceWarning)
    warnings.filterwarnings("ignore", message=r"unclosed event loop", category=ResourceWarning)
    warnings.filterwarnings("ignore", message=r"unclosed.*socket", category=ResourceWarning)
    # Pytest wraps unraisable ResourceWarnings in PytestUnraisableExceptionWarning
    config.addinivalue_line(
        "filterwarnings",
        "ignore::pytest.PytestUnraisableExceptionWarning",
    )
    # Python 3.13 AsyncMock GC warnings — mock coroutines created internally
    # by unittest.mock that are never awaited during test teardown
    warnings.filterwarnings(
        "ignore",
        message=r"coroutine 'AsyncMockMixin\._execute_mock_call' was never awaited",
        category=RuntimeWarning,
    )
