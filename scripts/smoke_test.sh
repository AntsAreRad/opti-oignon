#!/usr/bin/env bash
# =============================================================================
# Opti-Oignon -- Smoke Test: End-to-End API Validation (v1.9.0)
# =============================================================================
# Starts the FastAPI backend, runs a sequence of curl tests, then cleans up.
#
# Usage:
#   bash scripts/smoke_test.sh
#   bash scripts/smoke_test.sh --no-start   # Skip backend start (already running)
#
# Prerequisites:
#   - Python dependencies installed (pip install -e .)
#   - Port 8199 available (uses non-standard port to avoid conflicts)
# =============================================================================

set -u

BOLD="\033[1m"
GREEN="\033[92m"
RED="\033[91m"
YELLOW="\033[93m"
BLUE="\033[94m"
RESET="\033[0m"

PORT=8199
BASE="http://localhost:${PORT}"
PASSED=0
FAILED=0
SERVER_PID=""
NO_START=false

# Parse arguments
for arg in "$@"; do
    case "$arg" in
        --no-start) NO_START=true ;;
    esac
done

step()  { echo -e "\n${BOLD}${BLUE}[>] $1${RESET}"; }
ok()    { echo -e "  ${GREEN}[PASS] $1${RESET}"; PASSED=$((PASSED + 1)); }
fail()  { echo -e "  ${RED}[FAIL] $1${RESET}"; FAILED=$((FAILED + 1)); }
info()  { echo -e "  ${YELLOW}[INFO] $1${RESET}"; }

cleanup() {
    if [ "$NO_START" = false ] && [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        info "Stopping backend (PID $SERVER_PID)..."
        kill "$SERVER_PID" 2>/dev/null
        wait "$SERVER_PID" 2>/dev/null
    fi
}
trap cleanup EXIT

echo -e "${BOLD}============================================="
echo "  Opti-Oignon Smoke Test v1.9.0"
echo -e "=============================================${RESET}"

# -------------------------------------------------
# Start backend (unless --no-start)
# -------------------------------------------------
if [ "$NO_START" = false ]; then
    step "Starting FastAPI backend on port ${PORT}..."

    python3 -m uvicorn opti_oignon.api.app:app \
        --host 127.0.0.1 --port "$PORT" --log-level warning &
    SERVER_PID=$!

    RETRIES=30
    while [ $RETRIES -gt 0 ]; do
        if curl -sf "${BASE}/api/health" > /dev/null 2>&1; then
            break
        fi
        sleep 0.5
        RETRIES=$((RETRIES - 1))
    done

    if [ $RETRIES -eq 0 ]; then
        fail "Backend did not start within 15 seconds"
        exit 1
    fi
    ok "Backend started (PID $SERVER_PID)"
else
    step "Skipping backend start (--no-start)"
    PORT="${SMOKE_TEST_PORT:-8000}"
    BASE="http://localhost:${PORT}"
    info "Using port ${PORT}"
fi

# -------------------------------------------------
# Test 1: Health check
# -------------------------------------------------
step "GET /api/health"
RESP=$(curl -sf "${BASE}/api/health" 2>&1)
if echo "$RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); assert d['status']=='ok'" 2>/dev/null; then
    ok "Health check: status=ok"
else
    fail "Health check: unexpected response"
fi

# -------------------------------------------------
# Test 2: Version check
# -------------------------------------------------
step "Version check"
VERSION=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('version',''))" 2>/dev/null)
if [ "$VERSION" = "1.9.0" ]; then
    ok "Version: $VERSION"
else
    fail "Version: expected 1.9.0, got $VERSION"
fi

# -------------------------------------------------
# Test 3: Module availability
# -------------------------------------------------
step "Module availability"
MOD_COUNT=$(echo "$RESP" | python3 -c "import sys,json; print(len(json.load(sys.stdin).get('modules',{})))" 2>/dev/null)
if [ "$MOD_COUNT" -ge 30 ] 2>/dev/null; then
    ok "Module count: $MOD_COUNT (>= 30)"
else
    fail "Module count: expected >= 30, got $MOD_COUNT"
fi

# -------------------------------------------------
# Test 4: Create conversation
# -------------------------------------------------
step "POST /api/conversations"
RESP=$(curl -sf -X POST "${BASE}/api/conversations" \
    -H "Content-Type: application/json" \
    -d '{"title": "Smoke Test v1.9.0"}' 2>&1)
CONV_ID=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])" 2>/dev/null)
if [ -n "$CONV_ID" ] && [ "$CONV_ID" != "null" ]; then
    ok "Created conversation: $CONV_ID"
else
    fail "Create conversation: no ID returned"
    CONV_ID=""
fi

# -------------------------------------------------
# Test 5: Get conversation
# -------------------------------------------------
if [ -n "$CONV_ID" ]; then
    step "GET /api/conversations/${CONV_ID}"
    HTTP_CODE=$(curl -sf -o /dev/null -w "%{http_code}" "${BASE}/api/conversations/${CONV_ID}" 2>&1)
    if [ "$HTTP_CODE" = "200" ]; then
        ok "Get conversation: 200"
    else
        fail "Get conversation: HTTP $HTTP_CODE"
    fi
fi

# -------------------------------------------------
# Test 6: List models
# -------------------------------------------------
step "GET /api/models"
RESP=$(curl -sf "${BASE}/api/models" 2>&1)
if echo "$RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); assert 'models' in d" 2>/dev/null; then
    MODEL_COUNT=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['count'])" 2>/dev/null)
    ok "Models endpoint: ${MODEL_COUNT:-0} models"
else
    fail "Models endpoint: unexpected response"
fi

# -------------------------------------------------
# Test 7: List presets
# -------------------------------------------------
step "GET /api/presets"
HTTP_CODE=$(curl -sf -o /dev/null -w "%{http_code}" "${BASE}/api/presets" 2>&1)
if [ "$HTTP_CODE" = "200" ]; then
    ok "Presets endpoint: 200"
else
    fail "Presets endpoint: HTTP $HTTP_CODE"
fi

# -------------------------------------------------
# Test 8: System presets list
# -------------------------------------------------
step "GET /api/system-presets/list"
RESP=$(curl -sf "${BASE}/api/system-presets/list" 2>&1)
HTTP_CODE=$(curl -sf -o /dev/null -w "%{http_code}" "${BASE}/api/system-presets/list" 2>&1)
if [ "$HTTP_CODE" = "200" ]; then
    PRESET_COUNT=$(echo "$RESP" | python3 -c "import sys,json; print(len(json.load(sys.stdin).get('presets',[])))" 2>/dev/null)
    ok "System presets: $PRESET_COUNT presets"
elif [ "$HTTP_CODE" = "503" ]; then
    info "System presets module not available (503)"
    PASSED=$((PASSED + 1))
else
    fail "System presets: HTTP $HTTP_CODE"
fi

# -------------------------------------------------
# Test 9: System presets detect
# -------------------------------------------------
step "GET /api/system-presets/detect"
HTTP_CODE=$(curl -sf -o /dev/null -w "%{http_code}" "${BASE}/api/system-presets/detect" 2>&1)
if [ "$HTTP_CODE" = "200" ] || [ "$HTTP_CODE" = "503" ]; then
    ok "System presets detect: HTTP $HTTP_CODE"
else
    fail "System presets detect: HTTP $HTTP_CODE"
fi

# -------------------------------------------------
# Test 10: Onboarding state
# -------------------------------------------------
step "GET /api/system-presets/onboarding"
HTTP_CODE=$(curl -sf -o /dev/null -w "%{http_code}" "${BASE}/api/system-presets/onboarding" 2>&1)
if [ "$HTTP_CODE" = "200" ] || [ "$HTTP_CODE" = "503" ]; then
    ok "Onboarding state: HTTP $HTTP_CODE"
else
    fail "Onboarding state: HTTP $HTTP_CODE"
fi

# -------------------------------------------------
# Test 11: Health dashboard
# -------------------------------------------------
step "GET /api/health/dashboard"
RESP=$(curl -sf "${BASE}/api/health/dashboard" 2>&1)
if echo "$RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); assert 'modules' in d" 2>/dev/null; then
    ok "Health dashboard: modules present"
else
    fail "Health dashboard: unexpected response"
fi

# -------------------------------------------------
# Test 12: Smart routing config
# -------------------------------------------------
step "GET /api/smart-routing/config"
HTTP_CODE=$(curl -sf -o /dev/null -w "%{http_code}" "${BASE}/api/smart-routing/config" 2>&1)
if [ "$HTTP_CODE" = "200" ]; then
    ok "Smart routing config: 200"
else
    fail "Smart routing config: HTTP $HTTP_CODE"
fi

# -------------------------------------------------
# Test 13: Model health
# -------------------------------------------------
step "GET /api/smart-routing/model-health"
HTTP_CODE=$(curl -sf -o /dev/null -w "%{http_code}" "${BASE}/api/smart-routing/model-health" 2>&1)
if [ "$HTTP_CODE" = "200" ]; then
    ok "Model health: 200"
else
    fail "Model health: HTTP $HTTP_CODE"
fi

# -------------------------------------------------
# Test 14: Feedback endpoint
# -------------------------------------------------
step "GET /api/feedback/list"
HTTP_CODE=$(curl -sf -o /dev/null -w "%{http_code}" "${BASE}/api/feedback/list" 2>&1)
if [ "$HTTP_CODE" = "200" ]; then
    ok "Feedback list: 200"
else
    fail "Feedback list: HTTP $HTTP_CODE"
fi

# -------------------------------------------------
# Test 15: Analytics overview
# -------------------------------------------------
step "GET /api/feedback/analytics/overview"
HTTP_CODE=$(curl -sf -o /dev/null -w "%{http_code}" "${BASE}/api/feedback/analytics/overview" 2>&1)
if [ "$HTTP_CODE" = "200" ]; then
    ok "Analytics overview: 200"
else
    fail "Analytics overview: HTTP $HTTP_CODE"
fi

# -------------------------------------------------
# Test 16: Projects list
# -------------------------------------------------
step "GET /api/projects"
HTTP_CODE=$(curl -sf -o /dev/null -w "%{http_code}" "${BASE}/api/projects" 2>&1)
if [ "$HTTP_CODE" = "200" ]; then
    ok "Projects list: 200"
else
    fail "Projects list: HTTP $HTTP_CODE"
fi

# -------------------------------------------------
# Test 17: Pipelines list
# -------------------------------------------------
step "GET /api/pipelines"
HTTP_CODE=$(curl -sf -o /dev/null -w "%{http_code}" "${BASE}/api/pipelines" 2>&1)
if [ "$HTTP_CODE" = "200" ]; then
    ok "Pipelines list: 200"
else
    fail "Pipelines list: HTTP $HTTP_CODE"
fi

# -------------------------------------------------
# Test 18: Benchmark suites
# -------------------------------------------------
step "GET /api/benchmark/suites"
HTTP_CODE=$(curl -sf -o /dev/null -w "%{http_code}" "${BASE}/api/benchmark/suites" 2>&1)
if [ "$HTTP_CODE" = "200" ]; then
    ok "Benchmark suites: 200"
else
    fail "Benchmark suites: HTTP $HTTP_CODE"
fi

# -------------------------------------------------
# Test 19: Search proxy status
# -------------------------------------------------
step "GET /api/search/proxy-status"
HTTP_CODE=$(curl -sf -o /dev/null -w "%{http_code}" "${BASE}/api/search/proxy-status" 2>&1)
if [ "$HTTP_CODE" = "200" ]; then
    ok "Search proxy status: 200"
else
    fail "Search proxy status: HTTP $HTTP_CODE"
fi

# -------------------------------------------------
# Test 20: Coding agent status
# -------------------------------------------------
step "GET /api/coding/status"
HTTP_CODE=$(curl -sf -o /dev/null -w "%{http_code}" "${BASE}/api/coding/status" 2>&1)
if [ "$HTTP_CODE" = "200" ]; then
    ok "Coding agent status: 200"
else
    fail "Coding agent status: HTTP $HTTP_CODE"
fi

# -------------------------------------------------
# Test 21: Sandbox status
# -------------------------------------------------
step "GET /api/sandbox"
HTTP_CODE=$(curl -sf -o /dev/null -w "%{http_code}" "${BASE}/api/sandbox" 2>&1)
if [ "$HTTP_CODE" = "200" ]; then
    ok "Sandbox status: 200"
else
    fail "Sandbox status: HTTP $HTTP_CODE"
fi

# -------------------------------------------------
# Test 22: Performance summary
# -------------------------------------------------
step "GET /api/performance/summary"
HTTP_CODE=$(curl -sf -o /dev/null -w "%{http_code}" "${BASE}/api/performance/summary" 2>&1)
if [ "$HTTP_CODE" = "200" ]; then
    ok "Performance summary: 200"
else
    fail "Performance summary: HTTP $HTTP_CODE"
fi

# -------------------------------------------------
# Test 23: Delete conversation (cleanup)
# -------------------------------------------------
if [ -n "$CONV_ID" ]; then
    step "DELETE /api/conversations/${CONV_ID}"
    HTTP_CODE=$(curl -sf -o /dev/null -w "%{http_code}" -X DELETE "${BASE}/api/conversations/${CONV_ID}" 2>&1)
    if [ "$HTTP_CODE" = "204" ] || [ "$HTTP_CODE" = "200" ]; then
        ok "Delete conversation: $HTTP_CODE"
    else
        fail "Delete conversation: HTTP $HTTP_CODE"
    fi
fi

# -------------------------------------------------
# Results
# -------------------------------------------------
echo ""
echo -e "${BOLD}============================================="
echo "  Results: $PASSED passed, $FAILED failed"
echo -e "=============================================${RESET}"

if [ "$FAILED" -gt 0 ]; then
    exit 1
fi
exit 0
