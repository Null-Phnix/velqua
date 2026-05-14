#!/usr/bin/env python3
"""
Velqua Desktop — native window using pywebview.

Wraps the Velqua web UI in a native OS window using the system webview.
One process, one language (Python), no Electron/Chromium bloat.

Falls back to:
1. System tray mode (if pywebview unavailable but pystray is)
2. Console mode (if neither is available)

Usage:
    python backend/desktop.py             # pywebview window
    python backend/desktop.py --no-gui    # console mode (servers only)
"""
import argparse
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend import __version__
from backend.config import VelquaConfig as Config
from backend.logging_config import setup_logging, get_logger

setup_logging(level=Config.LOG_LEVEL)
logger = get_logger("desktop")


def run_server_thread():
    """Run the FastAPI server in a background thread."""
    import uvicorn
    from backend.server import app

    uvicorn.run(
        app,
        host=Config.HOST,
        port=Config.PORT,
        log_level=Config.LOG_LEVEL.lower(),
    )


def run_proxy_thread():
    """Run the memory proxy in a background thread."""
    import uvicorn
    from backend.proxy import app as proxy_app

    uvicorn.run(
        proxy_app,
        host=Config.HOST,
        port=Config.PROXY_PORT,
        log_level=Config.LOG_LEVEL.lower(),
    )


def start_backend():
    """Start both server and proxy in daemon threads."""
    server_thread = threading.Thread(target=run_server_thread, daemon=True)
    proxy_thread = threading.Thread(target=run_proxy_thread, daemon=True)
    server_thread.start()
    proxy_thread.start()

    logger.info("Backend started")
    logger.info("  API: http://%s:%d", Config.HOST, Config.PORT)
    logger.info("  Proxy: http://%s:%d", Config.HOST, Config.PROXY_PORT)

    return server_thread, proxy_thread


def wait_for_server(timeout: float = 10.0) -> bool:
    """Wait for the server to start accepting connections."""
    import httpx

    start = time.time()
    while time.time() - start < timeout:
        try:
            r = httpx.get(f"http://{Config.HOST}:{Config.PORT}/health", timeout=1)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.2)
    return False


class VelquaAPI:
    """
    Python API exposed to JavaScript via window.pywebview.api.

    These methods are callable from the web UI when running in
    desktop mode. They provide native OS capabilities that a
    browser tab can't do (file dialogs, version info, quit).
    """

    def get_version(self) -> str:
        """Return the current Velqua version."""
        return __version__

    def is_desktop(self) -> bool:
        """Return True when running in desktop mode (vs browser)."""
        return True

    def open_file_dialog(self) -> str | None:
        """Open a native file picker for JSON files."""
        try:
            import webview
            window = webview.windows[0] if webview.windows else None
            if window:
                result = window.create_file_dialog(
                    webview.OPEN_DIALOG,
                    file_types=("JSON files (*.json)",),
                )
                if result and len(result) > 0:
                    return result[0]
        except Exception as e:
            logger.warning("File dialog failed: %s", e)
        return None

    def quit(self) -> None:
        """Exit the application."""
        try:
            import webview
            for window in webview.windows:
                window.destroy()
        except Exception:
            pass
        logger.info("Velqua shutting down")
        sys.exit(0)


def run_desktop():
    """Run Velqua in a native pywebview window."""
    try:
        import webview
    except ImportError:
        logger.warning("pywebview not installed. Run: pip install pywebview")
        logger.info("Falling back to tray/console mode...")
        run_fallback()
        return

    # Start backend services
    start_backend()

    # Wait for server before opening window
    if not wait_for_server():
        logger.error("Server failed to start within 10 seconds")
        sys.exit(1)

    logger.info("Opening Velqua desktop window")

    api = VelquaAPI()
    window = webview.create_window(
        "Velqua",
        f"http://{Config.HOST}:{Config.PORT}",
        width=1100,
        height=750,
        min_size=(800, 600),
        js_api=api,
        text_select=True,
    )

    # webview.start() blocks until all windows are closed
    webview.start()
    logger.info("Desktop window closed. Shutting down.")


def run_fallback():
    """Fall back to tray mode or console mode."""
    try:
        from backend.tray import run_with_tray
        run_with_tray()
    except ImportError:
        run_console()


def run_console():
    """Run servers in console mode (no GUI)."""
    server_thread, proxy_thread = start_backend()

    logger.info("Velqua running in console mode")
    logger.info("Open http://localhost:%d in your browser", Config.PORT)
    logger.info("Press Ctrl+C to stop")

    try:
        server_thread.join()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        sys.exit(0)


def main():
    """Entry point for the `velqua-desktop` CLI command."""
    parser = argparse.ArgumentParser(description="Velqua Desktop")
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Run in console mode without a window",
    )
    args = parser.parse_args()

    if args.no_gui:
        run_console()
    else:
        run_desktop()


if __name__ == "__main__":
    main()
