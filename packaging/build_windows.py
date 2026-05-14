#!/usr/bin/env python3
"""
Build Velqua Windows installer (.exe) using PyInstaller + Inno Setup.

Steps:
  1. PyInstaller bundles Velqua into a single directory (--onedir for Inno Setup)
  2. Inno Setup compiles the directory into a setup .exe installer

Requirements:
  pip install pyinstaller
  choco install innosetup  (or download from https://jrsoftware.org/isinfo.php)

Usage:
  python packaging/build_windows.py
"""
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"
BUILD_DIR = DIST / "velqua-win"
INSTALLER_DIR = DIST / "installer"


def check_deps():
    try:
        import PyInstaller
        print(f"PyInstaller: {PyInstaller.__version__}")
    except ImportError:
        print("PyInstaller not installed. Run: pip install pyinstaller")
        sys.exit(1)


def run_pyinstaller():
    print("=== Step 1: PyInstaller bundle ===")
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm",
        "--onedir",
        "--name=velqua",
        f"--distpath={BUILD_DIR}",
        "--add-data=src;src",
        "--add-data=backend/anamnesis;backend/anamnesis",
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

    print(f"Bundle created at: {BUILD_DIR / 'velqua'}")


def generate_inno_script():
    """Generate an Inno Setup .iss script for the installer."""
    version = "2.0.0"
    try:
        from backend import __version__
        version = __version__.replace("-alpha", "a").replace("-beta", "b")
    except Exception:
        pass

    iss = textwrap.dedent(f"""\
    [Setup]
    AppName=Velqua
    AppVersion={version}
    AppPublisher=Velqua
    DefaultDirName={{autopf}}\\Velqua
    DefaultGroupName=Velqua
    UninstallDisplayIcon={{app}}\\velqua.exe
    OutputDir={INSTALLER_DIR}
    OutputBaseFilename=velqua_setup
    Compression=lzma2
    SolidCompression=yes
    ArchitecturesAllowed=x64compatible
    ArchitecturesInstallIn64BitMode=x64compatible

    [Files]
    Source: "{BUILD_DIR / 'velqua' / '*'}"; DestDir: "{{app}}"; Flags: ignoreversion recursesubdirs

    [Icons]
    Name: "{{group}}\\Velqua"; Filename: "{{app}}\\velqua.exe"
    Name: "{{commondesktop}}\\Velqua"; Filename: "{{app}}\\velqua.exe"; Tasks: desktopicon

    [Tasks]
    Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional icons:"

    [Run]
    Filename: "{{app}}\\velqua.exe"; Description: "Launch Velqua"; Flags: nowait postinstall skipifsilent
    """)

    INSTALLER_DIR.mkdir(parents=True, exist_ok=True)
    iss_path = ROOT / "packaging" / "velqua.iss"
    iss_path.write_text(iss)
    print(f"Inno Setup script: {iss_path}")
    return iss_path


def run_inno_setup(iss_path):
    print("\n=== Step 2: Inno Setup installer ===")
    iscc = "ISCC"  # Must be on PATH (Inno Setup compiler)

    result = subprocess.run([iscc, str(iss_path)], capture_output=True, text=True)
    if result.returncode == 0:
        installer = INSTALLER_DIR / "velqua_setup.exe"
        if installer.exists():
            size_mb = installer.stat().st_size / (1024 * 1024)
            print(f"Installer: {installer} ({size_mb:.1f} MB)")
        else:
            print("Inno Setup completed but installer not found at expected path.")
    else:
        print("Inno Setup not found or failed. The PyInstaller bundle is still usable.")
        print(f"  Bundle: {BUILD_DIR / 'velqua'}")
        if result.stderr:
            print(f"  Error: {result.stderr[:200]}")
        print("\nTo create installer manually:")
        print(f"  1. Install Inno Setup from https://jrsoftware.org/isinfo.php")
        print(f"  2. Open {ROOT / 'packaging' / 'velqua.iss'} in Inno Setup")
        print(f"  3. Click Build > Compile")


def main():
    check_deps()
    run_pyinstaller()
    iss_path = generate_inno_script()
    run_inno_setup(iss_path)


if __name__ == "__main__":
    main()
