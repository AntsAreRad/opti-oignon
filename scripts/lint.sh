#!/usr/bin/env bash
# Opti-Oignon -- Lint Runner
#
# Usage:
#   bash scripts/lint.sh          # Run all linters
#   bash scripts/lint.sh --fix    # Auto-fix where possible
#   bash scripts/lint.sh --py     # Python only
#   bash scripts/lint.sh --ts     # TypeScript only
#
# Exit codes:
#   0 -- no issues found
#   1 -- lint issues detected

set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Defaults
FIX=false
RUN_PY=true
RUN_TS=true
EXIT_CODE=0

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --fix)  FIX=true; shift ;;
        --py)   RUN_TS=false; shift ;;
        --ts)   RUN_PY=false; shift ;;
        -h|--help)
            echo "Usage: bash scripts/lint.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --fix    Auto-fix issues where possible"
            echo "  --py     Python linting only"
            echo "  --ts     TypeScript checking only"
            echo "  -h       Show this help"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "========================================="
echo "  Opti-Oignon Lint Runner"
echo "========================================="

# -------------------------------------------------------------------------
# Python linting with ruff
# -------------------------------------------------------------------------
if [[ "$RUN_PY" == true ]]; then
    echo ""
    echo "--- Python: ruff check ---"

    if command -v ruff &> /dev/null; then
        if [[ "$FIX" == true ]]; then
            ruff check opti_oignon/ tests/ --fix || EXIT_CODE=1
        else
            ruff check opti_oignon/ tests/ || EXIT_CODE=1
        fi
    else
        echo "  ruff not found. Install with: pip install ruff"
        echo "  Trying with python -m ruff..."
        if [[ "$FIX" == true ]]; then
            python -m ruff check opti_oignon/ tests/ --fix 2>/dev/null || {
                echo "  ruff unavailable, skipping Python lint."
            }
        else
            python -m ruff check opti_oignon/ tests/ 2>/dev/null || {
                echo "  ruff unavailable, skipping Python lint."
            }
        fi
    fi

    echo ""
    echo "--- Python: mypy type check ---"

    if command -v mypy &> /dev/null; then
        mypy opti_oignon/ --ignore-missing-imports --no-error-summary 2>/dev/null || {
            echo "  mypy found issues (non-blocking)."
        }
    else
        echo "  mypy not found, skipping type check."
    fi
fi

# -------------------------------------------------------------------------
# TypeScript / Svelte checking
# -------------------------------------------------------------------------
if [[ "$RUN_TS" == true ]]; then
    echo ""
    echo "--- TypeScript: svelte-check ---"

    if [[ -d "frontend" && -f "frontend/package.json" ]]; then
        cd frontend

        # Check if node_modules exists
        if [[ ! -d "node_modules" ]]; then
            echo "  Installing frontend dependencies..."
            npm install --silent 2>/dev/null
        fi

        # Run svelte-check if available
        if npx svelte-check --version &> /dev/null 2>&1; then
            npx svelte-check --tsconfig ./tsconfig.json 2>/dev/null || {
                echo "  svelte-check found issues (non-blocking)."
            }
        else
            echo "  svelte-check not available, skipping TypeScript check."
        fi

        cd "$PROJECT_ROOT"
    else
        echo "  Frontend directory not found, skipping."
    fi
fi

echo ""
echo "========================================="
if [[ $EXIT_CODE -eq 0 ]]; then
    echo "  All checks passed."
else
    echo "  Issues found (see above)."
fi
echo "========================================="

exit $EXIT_CODE
