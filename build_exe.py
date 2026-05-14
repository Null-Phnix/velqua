#!/usr/bin/env python3
"""
Build Velqua as a single executable using PyInstaller.

Usage:
    pip install pyinstaller
    python build_exe.py

Output: dist/velqua (or dist/velqua.exe on Windows)
"""
import subprocess
import sys
import platform
from pathlib import Path

ROOT = Path(__file__).parent


def build():
    """Build Velqua executable."""
    print("Building Velqua executable...")

    # Check PyInstaller is installed
    try:
        import PyInstaller
        print(f"PyInstaller version: {PyInstaller.__version__}")
    except ImportError:
        print("PyInstaller not installed. Run: pip install pyinstaller")
        sys.exit(1)

    # Determine platform-specific settings
    is_windows = platform.system() == "Windows"
    separator = ";" if is_windows else ":"

    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--onefile",
        "--name=velqua",
        # Include data files
        f"--add-data=src{separator}src",
        f"--add-data=backend/anamnesis{separator}backend/anamnesis",
        # Hidden imports (FastAPI needs these)
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
        # Backend modules
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
        # Provider modules
        "--hidden-import=backend.providers",
        "--hidden-import=backend.providers.base",
        "--hidden-import=backend.providers.ollama",
        "--hidden-import=backend.providers.openai_compat",
        "--hidden-import=backend.providers.anthropic",
        # Route modules
        "--hidden-import=backend.routes",
        "--hidden-import=backend.routes.settings",
        "--hidden-import=backend.routes.license",
        "--hidden-import=backend.updater",
        # Desktop window
        "--hidden-import=webview",
        # Encryption
        "--hidden-import=cryptography",
        "--hidden-import=cryptography.fernet",
        # Exclude heavy optional deps to keep size manageable
        "--exclude-module=matplotlib",
        "--exclude-module=tkinter",
        # Entry point — desktop.py with fallback to tray/console
        str(ROOT / "backend" / "desktop.py"),
    ]

    print(f"Command: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, cwd=str(ROOT))

    if result.returncode == 0:
        exe_name = "velqua.exe" if is_windows else "velqua"
        exe_path = ROOT / "dist" / exe_name
        if exe_path.exists():
            size_mb = exe_path.stat().st_size / (1024 * 1024)
            print(f"\nBuild successful!")
            print(f"Executable: {exe_path}")
            print(f"Size: {size_mb:.1f} MB")
        else:
            print("\nBuild may have succeeded but executable not found at expected path.")
    else:
        print(f"\nBuild failed with exit code {result.returncode}")
        sys.exit(1)


if __name__ == "__main__":
    build()
