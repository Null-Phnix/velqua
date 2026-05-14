Write-Host "Starting Velqua API server..." -ForegroundColor Cyan

# Check Python
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "Error: Python not found. Please install Python 3.9+" -ForegroundColor Red
    exit 1
}

# Check dependencies
try {
    python -c "import fastapi" 2>$null
} catch {
    Write-Host "Installing dependencies..." -ForegroundColor Yellow
    pip install -r requirements.txt
}

# Kill existing server
$existing = Get-NetTCPConnection -LocalPort 8765 -ErrorAction SilentlyContinue
if ($existing) {
    $pid = $existing.OwningProcess
    Write-Host "Stopping existing server (PID $pid)..." -ForegroundColor Yellow
    Stop-Process -Id $pid -Force
}

# Start server
Write-Host "Server starting on http://localhost:8765" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop" -ForegroundColor Yellow
cd velqua
python backend/server.py
