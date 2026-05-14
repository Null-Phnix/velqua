#!/usr/bin/env python3
"""
Velqua System Tray - Background mode with system tray icon.

Optional dependency: pystray, Pillow
Install: pip install pystray Pillow

Falls back to console-only mode if pystray is not available.
"""
import threading
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from backend import __version__
from backend.config import VelquaConfig as Config
from backend.logging_config import setup_logging, get_logger

setup_logging(level=Config.LOG_LEVEL)
logger = get_logger("tray")


def create_icon_image():
    """Create a simple tray icon (blue circle with V)."""
    try:
        from PIL import Image, ImageDraw, ImageFont

        img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.ellipse([4, 4, 60, 60], fill=(102, 126, 234, 255))
        try:
            font = ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSans-Bold.ttf", 36)
        except OSError:
            font = ImageFont.load_default()
        draw.text((18, 10), "V", fill=(255, 255, 255, 255), font=font)
        return img
    except ImportError:
        return None


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
    """Run the proxy in a background thread."""
    import uvicorn
    from backend.proxy import app as proxy_app

    uvicorn.run(
        proxy_app,
        host=Config.HOST,
        port=Config.PROXY_PORT,
        log_level=Config.LOG_LEVEL.lower(),
    )


def run_with_tray():
    """Run Velqua with system tray icon."""
    try:
        import pystray
    except ImportError:
        logger.warning("pystray not installed. Run: pip install pystray Pillow")
        logger.info("Starting in console mode instead...")
        run_console_mode()
        return

    icon_image = create_icon_image()
    if icon_image is None:
        logger.warning("Pillow not installed for tray icon. Using console mode.")
        run_console_mode()
        return

    # Start servers in background threads
    server_thread = threading.Thread(target=run_server_thread, daemon=True)
    proxy_thread = threading.Thread(target=run_proxy_thread, daemon=True)
    server_thread.start()
    proxy_thread.start()

    logger.info("Velqua running in system tray")
    logger.info("API: http://%s:%d", Config.HOST, Config.PORT)
    logger.info("Proxy: http://%s:%d", Config.HOST, Config.PROXY_PORT)

    def open_browser(icon, item):
        import webbrowser
        webbrowser.open(f"http://localhost:{Config.PORT}")

    def quit_app(icon, item):
        icon.stop()
        logger.info("Velqua shutting down")
        sys.exit(0)

    menu = pystray.Menu(
        pystray.MenuItem(
            f"Velqua v{__version__} - Port {Config.PORT}",
            None,
            enabled=False,
        ),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Open Dashboard", open_browser, default=True),
        pystray.MenuItem("Quit", quit_app),
    )

    icon = pystray.Icon("velqua", icon_image, "Velqua Memory", menu)
    icon.run()


def run_console_mode():
    """Run both server and proxy without system tray."""
    server_thread = threading.Thread(target=run_server_thread, daemon=True)
    proxy_thread = threading.Thread(target=run_proxy_thread, daemon=True)

    server_thread.start()
    proxy_thread.start()

    logger.info("Velqua running in console mode")
    logger.info("API: http://%s:%d", Config.HOST, Config.PORT)
    logger.info("Proxy: http://%s:%d", Config.HOST, Config.PROXY_PORT)
    logger.info("Press Ctrl+C to stop")

    try:
        server_thread.join()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        sys.exit(0)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Velqua Memory System")
    parser.add_argument(
        "--no-tray",
        action="store_true",
        help="Run in console mode without system tray",
    )
    args = parser.parse_args()

    if args.no_tray:
        run_console_mode()
    else:
        run_with_tray()
