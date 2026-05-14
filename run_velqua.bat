@echo off
echo Starting Velqua API server...

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found. Please install Python 3.9+
    exit /b 1
)

REM Check if dependencies are installed
python -c "import fastapi" >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies...
    pip install -r requirements.txt
)

REM Kill any existing server on port 8765
for /f "tokens=5" %%a in ('netstat -aon ^| find ":8765" ^| find "LISTENING"') do (
    echo Stopping existing server (PID %%a)...
    taskkill /F /PID %%a >nul 2>&1
)

REM Start server
echo Server starting on http://localhost:8765
echo Press Ctrl+C to stop
cd velqua
python backend/server.py
