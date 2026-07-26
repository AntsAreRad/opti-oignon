#!/usr/bin/env bash
# ============================================================================
# run_e2e.sh — Opti-Oignon Playwright E2E test runner
# Frontend E2E Tests
#
# Usage:
#   ./scripts/run_e2e.sh              # default: headless chromium
#   ./scripts/run_e2e.sh --headed     # visible browser
#   ./scripts/run_e2e.sh --ui         # interactive Playwright UI
#   CI=1 ./scripts/run_e2e.sh         # CI mode (retries, single worker)
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
FRONTEND_DIR="$PROJECT_ROOT/frontend"

echo "=== Opti-Oignon E2E Tests ==="
echo "Project root: $PROJECT_ROOT"
echo "Frontend dir: $FRONTEND_DIR"

# ── Check prerequisites ────────────────────────────────────────────────────
if ! command -v node &>/dev/null; then
    echo "ERROR: Node.js is required. Install it first."
    exit 1
fi

if [ ! -d "$FRONTEND_DIR/node_modules" ]; then
    echo ">> Installing frontend dependencies..."
    (cd "$FRONTEND_DIR" && npm install)
fi

if [ ! -d "$FRONTEND_DIR/node_modules/@playwright" ]; then
    echo "ERROR: @playwright/test not installed. Run: cd frontend && npm install"
    exit 1
fi

# ── Ensure Playwright browsers are installed ───────────────────────────────
echo ">> Checking Playwright browsers..."
(cd "$FRONTEND_DIR" && npx playwright install chromium --with-deps 2>/dev/null) || {
    echo "WARNING: Could not install Playwright browsers automatically."
    echo "Run manually: cd frontend && npx playwright install chromium --with-deps"
}

# ── Run E2E tests ──────────────────────────────────────────────────────────
echo ""
echo ">> Running Playwright E2E tests..."
(cd "$FRONTEND_DIR" && npx playwright test "$@")

EXIT_CODE=$?

# ── Report ─────────────────────────────────────────────────────────────────
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=== E2E Tests PASSED ==="
else
    echo ""
    echo "=== E2E Tests FAILED (exit code: $EXIT_CODE) ==="
    echo "View report: cd frontend && npx playwright show-report ../tests/e2e/playwright-report"
fi

exit $EXIT_CODE
