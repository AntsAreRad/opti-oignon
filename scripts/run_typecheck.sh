#!/usr/bin/env bash
# scripts/run_typecheck.sh — CI-ready mypy type checking with baseline gate
# Companion to the CI typecheck job.
#
# Usage:
#   ./scripts/run_typecheck.sh          # Run mypy, fail if errors exceed baseline
#   ./scripts/run_typecheck.sh --update  # Update baseline to current error count
#   ./scripts/run_typecheck.sh --ci      # CI mode (no colors, plain output)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BASELINE_FILE="$PROJECT_ROOT/mypy_baseline.json"

cd "$PROJECT_ROOT"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
CI_MODE=false
UPDATE_MODE=false
for arg in "$@"; do
    case "$arg" in
        --ci)     CI_MODE=true ;;
        --update) UPDATE_MODE=true ;;
    esac
done

# ---------------------------------------------------------------------------
# Colors (disabled in CI mode)
# ---------------------------------------------------------------------------
if [[ "$CI_MODE" == true ]]; then
    RED=''
    GREEN=''
    YELLOW=''
    CYAN=''
    NC=''
else
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    CYAN='\033[0;36m'
    NC='\033[0m'
fi

# ---------------------------------------------------------------------------
# Read baseline
# ---------------------------------------------------------------------------
if [ ! -f "$BASELINE_FILE" ]; then
    echo -e "${YELLOW}WARNING: No baseline file found at $BASELINE_FILE${NC}"
    echo "Run with --update to create one."
    BASELINE=9999
else
    BASELINE=$(python3 -c "import json; print(json.load(open('$BASELINE_FILE'))['mypy_baseline_errors'])")
fi

# ---------------------------------------------------------------------------
# Run mypy
# ---------------------------------------------------------------------------
echo -e "${CYAN}Running mypy on opti_oignon/...${NC}"
echo ""

MYPY_ARGS=("opti_oignon/")
if [[ -f "pyproject.toml" ]]; then
    MYPY_ARGS+=("--config-file" "pyproject.toml")
fi
MYPY_ARGS+=("--ignore-missing-imports")

MYPY_OUTPUT=$(mypy "${MYPY_ARGS[@]}" 2>&1) || true

# Extract error count from "Found N errors in M files"
ERROR_COUNT=$(echo "$MYPY_OUTPUT" | grep -oP 'Found \K\d+(?= error)' || echo "0")

# If mypy says "Success", error count is 0
if echo "$MYPY_OUTPUT" | grep -q "Success: no issues found"; then
    ERROR_COUNT=0
fi

# ---------------------------------------------------------------------------
# Update mode
# ---------------------------------------------------------------------------
if [ "$UPDATE_MODE" = true ]; then
    echo "$MYPY_OUTPUT" | tail -5
    echo ""
    CHECKED=$(echo "$MYPY_OUTPUT" | grep -oP 'checked \K\d+' || echo "?")
    python3 -c "
import json
data = {
    'mypy_baseline_errors': $ERROR_COUNT,
    'date': '$(date -I)',
    'version': '$(python3 -c "exec(open('opti_oignon/__version__.py').read()); print(__version__)")',
    'checked_files': $CHECKED,
    'notes': 'Updated by run_typecheck.sh --update'
}
with open('$BASELINE_FILE', 'w') as f:
    json.dump(data, f, indent=2)
    f.write('\n')
"
    echo -e "${GREEN}Baseline updated: $ERROR_COUNT errors${NC}"
    exit 0
fi

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
echo "=========================================="
echo -e " mypy type check summary"
echo "=========================================="
echo -e " Errors found:    ${CYAN}$ERROR_COUNT${NC}"
echo -e " Baseline:        ${CYAN}$BASELINE${NC}"

if [ "$ERROR_COUNT" -le "$BASELINE" ]; then
    DELTA=$((BASELINE - ERROR_COUNT))
    echo -e " Status:          ${GREEN}PASS${NC} ($DELTA below baseline)"
    echo "=========================================="
    echo ""
    if [ "$DELTA" -ge 10 ] && [ "$BASELINE" -ne 9999 ]; then
        echo -e "${YELLOW}TIP: You reduced errors by $DELTA. Consider updating baseline:${NC}"
        echo "  ./scripts/run_typecheck.sh --update"
    fi
    exit 0
else
    DELTA=$((ERROR_COUNT - BASELINE))
    echo -e " Status:          ${RED}FAIL${NC} ($DELTA above baseline)"
    echo "=========================================="
    echo ""
    echo -e "${RED}New type errors introduced! Fix them or update the baseline.${NC}"
    echo ""
    echo "Recent errors:"
    echo "$MYPY_OUTPUT" | grep "error:" | tail -20
    exit 1
fi
