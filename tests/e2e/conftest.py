"""
E2E test fixtures — start the Velqua server as a real subprocess,
wait for it to be ready, then tear it down after all tests.
"""
import os
import subprocess
import sys
import tempfile
import time
import httpx
import pytest


SERVER_PORT = 9999  # Isolated port so E2E tests don't clash with dev server
SERVER_URL = f"http://127.0.0.1:{SERVER_PORT}"

# Use a temp DB so E2E tests don't touch the real data
_tmpdir = tempfile.mkdtemp(prefix="velqua_e2e_")
_db_path = os.path.join(_tmpdir, "e2e_test.db")


@pytest.fixture(scope="session")
def velqua_url():
    """Start the Velqua server, yield its URL, stop it after all tests."""
    env = os.environ.copy()
    env["VELQUA_PORT"] = str(SERVER_PORT)
    env["VELQUA_DB_PATH"] = _db_path
    env["VELQUA_PROXY_PORT"] = "19435"  # Use a different proxy port too

    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.server:app",
         "--host", "127.0.0.1", "--port", str(SERVER_PORT),
         "--log-level", "error"],
        env=env,
        cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Wait up to 10 seconds for the server to come up
    for _ in range(50):
        try:
            r = httpx.get(f"{SERVER_URL}/health", timeout=1)
            if r.status_code == 200:
                break
        except Exception:
            pass
        time.sleep(0.2)
    else:
        proc.terminate()
        pytest.fail(f"Velqua server failed to start on port {SERVER_PORT}")

    yield SERVER_URL

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


@pytest.fixture(scope="session")
def browser_context_args():
    """Playwright browser context args — skip HTTPS, allow localhost."""
    return {"ignore_https_errors": True}


@pytest.fixture
def page(page, velqua_url):
    """Navigate to the app and dismiss onboarding before each test."""
    page.goto(velqua_url)
    # Dismiss the setup wizard if it appears
    page.evaluate("() => { localStorage.setItem('velqua_onboarding_done', 'true'); }")
    page.reload()
    page.wait_for_load_state("networkidle")
    return page
