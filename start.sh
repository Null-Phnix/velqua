#!/bin/bash
# Velqua Startup Script

echo "🚀 Starting Velqua Memory Proxy..."
echo ""

# Check Python
if ! command -v python &> /dev/null; then
    echo "❌ Python not found. Please install Python 3.9+"
    exit 1
fi

# Check dependencies
python -c "import fastapi, uvicorn, httpx" 2>/dev/null || {
    echo "📦 Installing dependencies..."
    pip install fastapi uvicorn httpx
}

# Start backend server
echo "🔧 Starting API server on port 8765..."
python backend/server.py > logs/server.log 2>&1 &
SERVER_PID=$!
echo $SERVER_PID > /tmp/velqua-server.pid

# Start proxy
echo "🔧 Starting Ollama proxy on port 11435..."
python backend/proxy.py > logs/proxy.log 2>&1 &
PROXY_PID=$!
echo $PROXY_PID > /tmp/velqua-proxy.pid

# Wait for startup
sleep 2

# Check status
if curl -s http://localhost:8765/health > /dev/null; then
    echo "✅ API server running (PID: $SERVER_PID)"
else
    echo "❌ API server failed to start"
    cat logs/server.log
    exit 1
fi

if curl -s http://localhost:11435/ > /dev/null; then
    echo "✅ Proxy running (PID: $PROXY_PID)"
else
    echo "❌ Proxy failed to start"
    cat logs/proxy.log
    exit 1
fi

echo ""
echo "🎉 Velqua is running!"
echo ""
echo "📊 Web UI:    http://localhost:8765/static/index.html"
echo "🔌 API Docs:  http://localhost:8765/docs"
echo "🔀 Proxy:     http://localhost:11435"
echo ""
echo "💡 Point your Ollama apps to localhost:11435 to enable memory"
echo ""
echo "Press Ctrl+C to stop..."

# Wait for interrupt
trap "kill $SERVER_PID $PROXY_PID 2>/dev/null; exit" INT
wait
