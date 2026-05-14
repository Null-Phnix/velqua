#!/usr/bin/env python3
"""
Build Velqua Linux AppImage.

Steps:
  1. PyInstaller creates a single directory bundle
  2. Package into an AppImage using appimagetool

Requirements:
  pip install pyinstaller
  wget https://github.com/AppImage/appimagetool/releases/latest/download/appimagetool-x86_64.AppImage

Usage:
  python packaging/build_linux.py
"""
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"
APP_DIR = DIST / "Velqua.AppDir"


def check_deps():
    try:
        import PyInstaller
        print(f"PyInstaller: {PyInstaller.__version__}")
    except ImportError:
        print("PyInstaller not installed. Run: pip install pyinstaller")
        sys.exit(1)


def run_pyinstaller():
    print("=== Step 1: PyInstaller bundle ===")
    bundle_dir = DIST / "velqua-linux"
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm",
        "--onedir",
        "--name=velqua",
        f"--distpath={bundle_dir}",
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

    print(f"Bundle: {bundle_dir / 'velqua'}")
    return bundle_dir / "velqua"


def create_appdir(bundle_path):
    """Structure the PyInstaller output as an AppDir."""
    print("\n=== Step 2: Create AppDir structure ===")

    if APP_DIR.exists():
        subprocess.run(["rm", "-rf", str(APP_DIR)])
    APP_DIR.mkdir(parents=True)

    usr_bin = APP_DIR / "usr" / "bin"
    usr_bin.mkdir(parents=True)

    # Move bundle contents into usr/bin
    for item in bundle_path.iterdir():
        subprocess.run(["cp", "-a", str(item), str(usr_bin / item.name)])

    # AppRun — entry point
    apprun = APP_DIR / "AppRun"
    apprun.write_text(textwrap.dedent("""\
    #!/bin/bash
    SELF="$(readlink -f "$0")"
    HERE="${SELF%/*}"
    exec "${HERE}/usr/bin/velqua" "$@"
    """))
    apprun.chmod(0o755)

    # Desktop file
    desktop = APP_DIR / "velqua.desktop"
    desktop.write_text(textwrap.dedent("""\
    [Desktop Entry]
    Name=Velqua
    Exec=velqua
    Icon=velqua
    Type=Application
    Categories=Development;
    Comment=Memory proxy for local and cloud AI
    Terminal=false
    """))

    # Minimal icon (1x1 PNG placeholder — replace with real icon)
    icon_path = APP_DIR / "velqua.png"
    if not icon_path.exists():
        # Minimal valid 1x1 purple PNG
        import base64
        png_data = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
            "+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        )
        icon_path.write_bytes(png_data)

    print(f"AppDir: {APP_DIR}")


def create_appimage():
    """Run appimagetool to create the final AppImage."""
    print("\n=== Step 3: Create AppImage ===")

    appimage_path = DIST / "Velqua-x86_64.AppImage"

    # Try to find appimagetool
    tool = None
    for candidate in ["appimagetool", "appimagetool-x86_64.AppImage"]:
        result = subprocess.run(["which", candidate], capture_output=True, text=True)
        if result.returncode == 0:
            tool = candidate
            break

    if tool is None:
        print("appimagetool not found.")
        print("Download from: https://github.com/AppImage/appimagetool/releases")
        print(f"\nThe AppDir is still usable: {APP_DIR}/AppRun")
        return

    result = subprocess.run(
        [tool, str(APP_DIR), str(appimage_path)],
        env={**__import__("os").environ, "ARCH": "x86_64"},
    )

    if result.returncode == 0 and appimage_path.exists():
        appimage_path.chmod(0o755)
        size_mb = appimage_path.stat().st_size / (1024 * 1024)
        print(f"AppImage: {appimage_path} ({size_mb:.1f} MB)")
    else:
        print("appimagetool failed. The AppDir is still usable.")
        print(f"  Run: {APP_DIR}/AppRun")


def main():
    check_deps()
    bundle_path = run_pyinstaller()
    create_appdir(bundle_path)
    create_appimage()


if __name__ == "__main__":
    main()
