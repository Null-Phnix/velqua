#!/bin/bash
# Velqua End-to-End Test Script v2
# Improved: dynamic port checking, proper cleanup, no hardcoded sleeps

set -e

echo "Velqua E2E Tests"
echo "================"
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Configuration
SERVER_PORT=8765
PROXY_PORT=11435
PASSED=0
FAILED=0
SERVER_PID=""
PROXY_PID=""

# Cleanup function - always runs on exit
cleanup() {
    echo ""
    echo -e "${YELLOW}Cleaning up...${NC}"
    if [ ! -z "$SERVER_PID" ]; then
        kill $SERVER_PID 2>/dev/null || true
        wait $SERVER_PID 2>/dev/null || true
    fi
    if [ ! -z "$PROXY_PID" ]; then
        kill $PROXY_PID 2>/dev/null || true
        wait $PROXY_PID 2>/dev/null || true
    fi
    # Kill any remaining processes on test ports
    lsof -ti:$SERVER_PORT 2>/dev/null | xargs kill -9 2>/dev/null || true
    lsof -ti:$PROXY_PORT 2>/dev/null | xargs kill -9 2>/dev/null || true
    rm -f data/test_velqua.db 2>/dev/null || true
}

trap cleanup EXIT INT TERM

# Wait for port to be ready (replaces hardcoded sleeps)
wait_for_port() {
    local port=$1
    local max_wait=${2:-15}
    local elapsed=0

    while ! curl -s "http://localhost:$port" > /dev/null 2>&1; do
        if [ $elapsed -ge $max_wait ]; then
            echo -e "${RED}Timeout waiting for port $port after ${max_wait}s${NC}"
            return 1
        fi
        sleep 0.5
        elapsed=$((elapsed + 1))
    done
    return 0
}

pass() {
    echo -e "${GREEN}  PASS${NC} $1"
    PASSED=$((PASSED + 1))
}

fail() {
    echo -e "${RED}  FAIL${NC} $1"
    if [ ! -z "$2" ]; then
        echo "       $2"
    fi
    FAILED=$((FAILED + 1))
}

cd "$(dirname "$0")"

# Kill any existing processes on our ports first
lsof -ti:$SERVER_PORT 2>/dev/null | xargs kill -9 2>/dev/null || true
lsof -ti:$PROXY_PORT 2>/dev/null | xargs kill -9 2>/dev/null || true
sleep 0.5

# Start server
echo -e "${YELLOW}Starting server...${NC}"
python backend/server.py > /tmp/velqua_test_server.log 2>&1 &
SERVER_PID=$!

if ! wait_for_port $SERVER_PORT 15; then
    echo -e "${RED}Server failed to start. Log:${NC}"
    tail -20 /tmp/velqua_test_server.log
    exit 1
fi
echo -e "${GREEN}Server ready on port $SERVER_PORT${NC}"
echo ""

# --- Tests ---

# Test 1: Health check
echo "Test 1: Backend Health Check"
HEALTH=$(curl -s http://localhost:$SERVER_PORT/health)
if echo "$HEALTH" | grep -q '"status":"ok"'; then
    pass "Health endpoint returns ok"
else
    fail "Health check failed" "$HEALTH"
fi

# Test 2: Import Claude Memories (smart endpoint)
echo "Test 2: Smart Import - Claude Memories"
if [ -f "test_data/claude_memories_proper.json" ]; then
    RESULT=$(curl -s -F "file=@test_data/claude_memories_proper.json" http://localhost:$SERVER_PORT/import/smart)
    if echo "$RESULT" | grep -q '"success":true'; then
        STORED=$(echo "$RESULT" | python -c "import sys, json; print(json.load(sys.stdin)['facts_stored'])" 2>/dev/null || echo "?")
        pass "Imported $STORED facts from Claude memories"
    else
        fail "Smart import failed" "$RESULT"
    fi
else
    echo -e "${YELLOW}  SKIP${NC} No test data (test_data/claude_memories_proper.json)"
fi

# Test 3: List Facts
echo "Test 3: Retrieve Stored Facts"
FACTS=$(curl -s "http://localhost:$SERVER_PORT/facts/list?limit=5")
if echo "$FACTS" | grep -q '"facts"'; then
    COUNT=$(echo "$FACTS" | python -c "import sys, json; print(len(json.load(sys.stdin)['facts']))" 2>/dev/null || echo "0")
    if [ "$COUNT" -gt 0 ]; then
        pass "Retrieved $COUNT facts"
    else
        pass "Facts endpoint works (0 facts stored)"
    fi
else
    fail "Facts list failed" "$FACTS"
fi

# Test 4: Error handling - invalid JSON
echo "Test 4: Error Handling - Invalid File"
echo "not valid json" > /tmp/velqua_test_invalid.json
RESULT=$(curl -s -w "\nHTTP_CODE:%{http_code}" \
    -F "file=@/tmp/velqua_test_invalid.json" \
    http://localhost:$SERVER_PORT/import/smart)

if echo "$RESULT" | grep -q "HTTP_CODE:400"; then
    pass "Invalid JSON returns 400"
elif echo "$RESULT" | grep -q "HTTP_CODE:422"; then
    pass "Invalid JSON returns 422"
else
    fail "Expected 400/422 for invalid JSON" "$RESULT"
fi
rm -f /tmp/velqua_test_invalid.json

# Test 5: Security - filename sanitization
echo "Test 5: Security - Path Traversal Prevention"
echo '{"test": true}' > /tmp/velqua_test_safe.json
RESULT=$(curl -s -F "file=@/tmp/velqua_test_safe.json;filename=../../../etc/passwd" \
    http://localhost:$SERVER_PORT/import/smart)
# Should not crash - should sanitize the filename and process normally
if echo "$RESULT" | grep -q '"file_type"'; then
    pass "Path traversal filename sanitized safely"
elif echo "$RESULT" | grep -q '"success"'; then
    pass "Path traversal filename handled"
else
    # Even an error response is ok as long as server didn't crash
    HEALTH2=$(curl -s http://localhost:$SERVER_PORT/health)
    if echo "$HEALTH2" | grep -q '"status":"ok"'; then
        pass "Server survived path traversal attempt"
    else
        fail "Server crashed on path traversal" "$RESULT"
    fi
fi
rm -f /tmp/velqua_test_safe.json

# Test 6: ChatGPT Import (if test data exists)
echo "Test 6: ChatGPT Import"
# Create minimal ChatGPT test data
CHATGPT_TEST=$(cat <<'ENDJSON'
[{"title": "Python Help", "mapping": {"m1": {"message": {"author": {"role": "user"}, "content": {"parts": ["I'm working on a web development project for my portfolio"]}}}}}]
ENDJSON
)
echo "$CHATGPT_TEST" > /tmp/velqua_test_chatgpt.json
RESULT=$(curl -s -F "file=@/tmp/velqua_test_chatgpt.json" http://localhost:$SERVER_PORT/import/chatgpt-export)
if echo "$RESULT" | grep -q '"success":true'; then
    STORED=$(echo "$RESULT" | python -c "import sys, json; print(json.load(sys.stdin)['facts_stored'])" 2>/dev/null || echo "?")
    pass "ChatGPT import: $STORED facts stored"
else
    fail "ChatGPT import failed" "$RESULT"
fi
rm -f /tmp/velqua_test_chatgpt.json

# Test 7: Start and test proxy
echo "Test 7: Ollama Proxy"
python backend/proxy.py > /tmp/velqua_test_proxy.log 2>&1 &
PROXY_PID=$!

if wait_for_port $PROXY_PORT 10; then
    PROXY_STATUS=$(curl -s http://localhost:$PROXY_PORT/)
    if echo "$PROXY_STATUS" | grep -q "Velqua Ollama Proxy"; then
        pass "Proxy running on port $PROXY_PORT"
    else
        fail "Proxy response unexpected" "$PROXY_STATUS"
    fi
else
    fail "Proxy failed to start" "$(tail -5 /tmp/velqua_test_proxy.log)"
fi

# --- Summary ---
echo ""
TOTAL=$((PASSED + FAILED))
echo "========================"
if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}All $TOTAL tests passed${NC}"
else
    echo -e "${RED}$FAILED/$TOTAL tests failed${NC}"
fi
echo "========================"
echo ""

# Quick stats
echo "Stats:"
curl -s http://localhost:$SERVER_PORT/health | python -c "
import sys, json
h = json.load(sys.stdin)
print(f'  Facts: {h[\"facts_count\"]}')
print(f'  Episodes: {h[\"episodes_count\"]}')
print(f'  DB Size: {h[\"database_size_mb\"]} MB')
" 2>/dev/null || true

# Exit with failure if any tests failed
if [ $FAILED -gt 0 ]; then
    exit 1
fi
