#!/usr/bin/env python3
"""
Build Velqua macOS app bundle (.app in .dmg).

Steps:
  1. PyInstaller creates a .app bundle
  2. hdiutil packages it into a .dmg disk image

Requirements:
  pip install pyinstaller

Usage:
  python packaging/build_macos.py
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"
APP_NAME = "Velqua"
APP_BUNDLE = DIST / f"{APP_NAME}.app"
DMG_PATH = DIST / f"{APP_NAME}.dmg"


def check_deps():
    try:
        import PyInstaller
        print(f"PyInstaller: {PyInstaller.__version__}")
    except ImportError:
        print("PyInstaller not installed. Run: pip install pyinstaller")
        sys.exit(1)


def run_pyinstaller():
    print("=== Step 1: PyInstaller .app bundle ===")
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm",
        "--windowed",
        f"--name={APP_NAME}",
        f"--distpath={DIST}",
        "--add-data=src:src",
        "--add-data=backend/anamnesis:backend/anamnesis",
        "--hidden-import=uvicorn.logging",
        "--hidden-import=uvicorn.protocols.http",
        "--hidden-import=uvicorn.protocols.http.auto",
        "--hidden-import=uvicorn.protocols.http.h11_impl",
        "--hidden-import=uvicorn.protocols.websockets",
        "--hidden-import=uvicorn.protocols.websockets.auto",
        "--hidden-import=uvicorn.lifespan",
        "--hidden-import=uvicorn.lifespan.on",
        "--hidden-import=uvicorn.lifespan.off",
        "--hidden-import=fastapi",
        "--hidden-import=starlette",
        "--hidden-import=httpx",
        "--hidden-import=pydantic",
        "--hidden-import=anyio._backends._asyncio",
        "--hidden-import=backend.server",
        "--hidden-import=backend.proxy",
        "--hidden-import=backend.config",
        "--hidden-import=backend.logging_config",
        "--hidden-import=backend.auto_learner",
        "--hidden-import=backend.validators",
        "--hidden-import=backend.file_detector",
        "--hidden-import=backend.keystore",
        "--hidden-import=backend.license",
        "--hidden-import=backend.desktop",
        "--hidden-import=backend.tray",
        "--hidden-import=backend.providers",
        "--hidden-import=backend.providers.base",
        "--hidden-import=backend.providers.ollama",
        "--hidden-import=backend.providers.openai_compat",
        "--hidden-import=backend.providers.anthropic",
        "--hidden-import=backend.routes",
        "--hidden-import=backend.routes.settings",
        "--hidden-import=backend.routes.license",
        "--hidden-import=backend.updater",
        "--hidden-import=webview",
        "--hidden-import=cryptography",
        "--hidden-import=cryptography.fernet",
        "--exclude-module=matplotlib",
        "--exclude-module=tkinter",
        str(ROOT / "backend" / "desktop.py"),
    ]

    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        print("PyInstaller failed.")
        sys.exit(1)

    if APP_BUNDLE.exists():
        print(f"App bundle: {APP_BUNDLE}")
    else:
        print("Build completed but .app not found at expected path.")
        sys.exit(1)


def create_dmg():
    print("\n=== Step 2: Create .dmg ===")

    if DMG_PATH.exists():
        DMG_PATH.unlink()

    staging = DIST / "dmg_staging"
    staging.mkdir(exist_ok=True)

    # Copy .app to staging
    subprocess.run(["cp", "-R", str(APP_BUNDLE), str(staging / f"{APP_NAME}.app")])

    # Create symlink to /Applications for drag-install
    apps_link = staging / "Applications"
    if not apps_link.exists():
        apps_link.symlink_to("/Applications")

    # Create DMG
    result = subprocess.run([
        "hdiutil", "create",
        "-volname", APP_NAME,
        "-srcfolder", str(staging),
        "-ov",
        "-format", "UDZO",
        str(DMG_PATH),
    ])

    # Cleanup staging
    subprocess.run(["rm", "-rf", str(staging)])

    if result.returncode == 0 and DMG_PATH.exists():
        size_mb = DMG_PATH.stat().st_size / (1024 * 1024)
        print(f"DMG: {DMG_PATH} ({size_mb:.1f} MB)")
    else:
        print("DMG creation failed. The .app bundle is still usable.")
        print(f"  App: {APP_BUNDLE}")


def main():
    check_deps()
    run_pyinstaller()
    create_dmg()


if __name__ == "__main__":
    main()
