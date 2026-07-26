#!/usr/bin/env bash
# Opti-Oignon — Coverage Gate Script
#
# Runs the full test suite with coverage measurement and enforces
# minimum coverage thresholds.
#
# Usage:
#   bash scripts/run_coverage.sh              # Run with default gates
#   bash scripts/run_coverage.sh --html       # Also generate HTML report
#   bash scripts/run_coverage.sh --json       # Also generate JSON report
#   bash scripts/run_coverage.sh --baseline   # Print baseline and exit
#   bash scripts/run_coverage.sh --no-gate    # Run without fail-under checks
#
# Exit codes:
#   0 — all tests passed AND coverage gates met
#   1 — test failures
#   2 — coverage below minimum threshold
#   3 — setup error

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# -------------------------------------------------------------------------
# Coverage thresholds
# -------------------------------------------------------------------------
# Overall minimum (set after baseline measurement)
OVERALL_FAIL_UNDER=30

# Per-file minimums for security-critical modules (target: 80%)
# These are enforced individually after the main run.
declare -A SECURITY_MODULE_THRESHOLDS
SECURITY_MODULE_THRESHOLDS=(
    ["opti_oignon/auth.py"]=30
    ["opti_oignon/auth_2fa.py"]=35
    ["opti_oignon/encryption.py"]=60
    ["opti_oignon/db_encryption.py"]=40
    ["opti_oignon/sandbox_manager.py"]=0
    ["opti_oignon/security_mode.py"]=50
    ["opti_oignon/pqc_signatures.py"]=0
    ["opti_oignon/signed_audit_log.py"]=80
    ["opti_oignon/session_fingerprint.py"]=80
)

# -------------------------------------------------------------------------
# Defaults
# -------------------------------------------------------------------------
HTML_REPORT=false
JSON_REPORT=false
SHOW_BASELINE=false
GATE_ENABLED=true
VERBOSE=false
CI_MODE=false

# -------------------------------------------------------------------------
# Parse arguments
# -------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --html)      HTML_REPORT=true; shift ;;
        --json)      JSON_REPORT=true; shift ;;
        --baseline)  SHOW_BASELINE=true; shift ;;
        --no-gate)   GATE_ENABLED=false; shift ;;
        --verbose)   VERBOSE=true; shift ;;
        --ci)        CI_MODE=true; JSON_REPORT=true; shift ;;
        -h|--help)
            echo "Usage: bash scripts/run_coverage.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --html       Generate HTML report in htmlcov/"
            echo "  --json       Generate JSON report (coverage.json)"
            echo "  --baseline   Print baseline info and exit"
            echo "  --no-gate    Run without fail-under checks"
            echo "  --ci         CI mode: JSON output, no colors, no headers"
            echo "  --verbose    Verbose pytest output"
            echo "  -h, --help   Show this help"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 3 ;;
    esac
done

# -------------------------------------------------------------------------
# Baseline display
# -------------------------------------------------------------------------
if [[ "$SHOW_BASELINE" == true ]]; then
    if [[ -f "$PROJECT_ROOT/coverage_baseline.json" ]]; then
        echo "=== Coverage Baseline ==="
        python3 -c "
import json
with open('coverage_baseline.json') as f:
    d = json.load(f)
for k, v in d.items():
    print(f'  {k}: {v}')
"
    else
        echo "No baseline file found (coverage_baseline.json)"
    fi
    exit 0
fi

# -------------------------------------------------------------------------
# Build pytest arguments
# -------------------------------------------------------------------------
PYTEST_ARGS=("tests/")
PYTEST_ARGS+=("--ignore=tests/test_live_v130.py")
PYTEST_ARGS+=("--cov=opti_oignon")
PYTEST_ARGS+=("--cov-config=.coveragerc")
PYTEST_ARGS+=("--cov-report=term-missing:skip-covered")
PYTEST_ARGS+=("--cov-fail-under=$OVERALL_FAIL_UNDER")

if [[ "$HTML_REPORT" == true ]]; then
    PYTEST_ARGS+=("--cov-report=html:htmlcov")
fi

if [[ "$JSON_REPORT" == true ]]; then
    PYTEST_ARGS+=("--cov-report=json:coverage.json")
fi

# Always produce JSON for per-module gate checking
if [[ "$GATE_ENABLED" == true ]]; then
    PYTEST_ARGS+=("--cov-report=json:coverage.json")
fi

if [[ "$VERBOSE" == true ]]; then
    PYTEST_ARGS+=("-v" "-s")
else
    PYTEST_ARGS+=("-q")
fi

PYTEST_ARGS+=("--tb=no")

# -------------------------------------------------------------------------
# Header
# -------------------------------------------------------------------------
if [[ "$CI_MODE" != true ]]; then
    echo "========================================="
    echo "  Opti-Oignon Coverage Gate"
    echo "========================================="
    echo "  Overall fail-under:  ${OVERALL_FAIL_UNDER}%"
    echo "  Gate enabled:        $GATE_ENABLED"
    echo "  HTML report:         $HTML_REPORT"
    echo "  JSON report:         $JSON_REPORT"
    echo "========================================="
    echo ""
fi

# -------------------------------------------------------------------------
# Run tests with coverage
# -------------------------------------------------------------------------
set +e
python3 -m pytest "${PYTEST_ARGS[@]}"
TEST_EXIT=$?
set -e

if [[ $TEST_EXIT -ne 0 ]]; then
    echo ""
    echo "FAIL: Tests or overall coverage gate failed (exit code $TEST_EXIT)"
    exit 1
fi

# -------------------------------------------------------------------------
# Per-module security gate (requires JSON output)
# -------------------------------------------------------------------------
if [[ "$GATE_ENABLED" == true && -f "coverage.json" ]]; then
    echo ""
    echo "=== Security Module Coverage Gates ==="

    GATE_FAILED=false
    python3 -c "
import json, sys

with open('coverage.json') as f:
    data = json.load(f)

thresholds = {
$(for mod in "${!SECURITY_MODULE_THRESHOLDS[@]}"; do
    echo "    '$mod': ${SECURITY_MODULE_THRESHOLDS[$mod]},"
done)
}

failed = False
for module, minimum in sorted(thresholds.items()):
    if module in data['files']:
        pct = data['files'][module]['summary']['percent_covered']
        status = 'PASS' if pct >= minimum else 'FAIL'
        if status == 'FAIL':
            failed = True
        print(f'  {status}  {pct:5.1f}% >= {minimum}%  {module}')
    else:
        print(f'  SKIP  (not measured)  {module}')

if failed:
    print()
    print('FAIL: One or more security modules below threshold')
    sys.exit(2)
else:
    print()
    print('All security module gates passed.')
" || {
        echo ""
        echo "FAIL: Security module coverage gate failed"
        exit 2
    }
fi

# -------------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------------
echo ""
echo "========================================="
echo "  Coverage gates: ALL PASSED"
echo "========================================="

if [[ "$HTML_REPORT" == true ]]; then
    echo "  HTML report: htmlcov/index.html"
fi

exit 0
