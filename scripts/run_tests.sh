#!/usr/bin/env bash
# Opti-Oignon -- Test Runner with Coverage Report
#
# Usage:
#   bash scripts/run_tests.sh              # Run all tests
#   bash scripts/run_tests.sh --coverage   # Run with coverage report
#   bash scripts/run_tests.sh --module X   # Run a specific test module
#   bash scripts/run_tests.sh --quick      # Skip slow integration tests
#   bash scripts/run_tests.sh --ci         # CI mode (no colors, JSON coverage)
#
# Exit codes:
#   0 -- all tests passed
#   1 -- test failures detected
#   2 -- setup error

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Defaults
COVERAGE=false
MODULE=""
QUICK=false
VERBOSE=false
CI_MODE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --coverage) COVERAGE=true; shift ;;
        --module)   MODULE="$2"; shift 2 ;;
        --quick)    QUICK=true; shift ;;
        --verbose)  VERBOSE=true; shift ;;
        --ci)       CI_MODE=true; shift ;;
        -v)         VERBOSE=true; shift ;;
        -h|--help)
            echo "Usage: bash scripts/run_tests.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --coverage    Generate HTML coverage report in htmlcov/"
            echo "  --module X    Run only tests/test_X.py"
            echo "  --quick       Skip live tests and slow integration tests"
            echo "  --ci          CI mode: no colors, JUnit XML, JSON coverage"
            echo "  --verbose     Verbose pytest output (-v -s)"
            echo "  -h, --help    Show this help"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 2 ;;
    esac
done

# Build pytest arguments
PYTEST_ARGS=("tests/")
PYTEST_ARGS+=("--ignore=tests/test_live_v130.py")

if [[ -n "$MODULE" ]]; then
    PYTEST_ARGS=("tests/test_${MODULE}.py")
fi

if [[ "$QUICK" == true ]]; then
    PYTEST_ARGS+=("--ignore=tests/test_live_v130.py")
    PYTEST_ARGS+=("-x")  # Stop on first failure for quick runs
fi

if [[ "$VERBOSE" == true ]]; then
    PYTEST_ARGS+=("-v" "-s")
else
    PYTEST_ARGS+=("-q")
fi

# CI-specific flags
if [[ "$CI_MODE" == true ]]; then
    PYTEST_ARGS+=("--no-header" "--tb=short" "-p" "no:cacheprovider")
    PYTEST_ARGS+=("--junitxml=test-results.xml")
    export NO_COLOR=1
fi

if [[ "$COVERAGE" == true ]]; then
    # Check if pytest-cov is available
    if ! python -m pytest --co -q --collect-only 2>/dev/null | head -1 > /dev/null; then
        echo "Installing pytest-cov..."
        pip install pytest-cov --break-system-packages -q 2>/dev/null || pip install pytest-cov -q
    fi
    PYTEST_ARGS+=(
        "--cov=opti_oignon"
        "--cov-report=term-missing"
    )
    if [[ "$CI_MODE" == true ]]; then
        # CI: JSON report for badge, skip HTML
        PYTEST_ARGS+=("--cov-report=json:coverage.json")
    else
        PYTEST_ARGS+=("--cov-report=html:htmlcov")
    fi
fi

if [[ "$CI_MODE" != true ]]; then
    echo "========================================="
    echo "  Opti-Oignon Test Runner"
    echo "========================================="
    echo "  Project root: $PROJECT_ROOT"
    echo "  Coverage:     $COVERAGE"
    echo "  Module:       ${MODULE:-all}"
    echo "  Quick:        $QUICK"
    echo "  Verbose:      $VERBOSE"
    echo "========================================="
    echo ""
fi

# Run tests
python -m pytest "${PYTEST_ARGS[@]}"
EXIT_CODE=$?

if [[ "$COVERAGE" == true && $EXIT_CODE -eq 0 ]]; then
    if [[ "$CI_MODE" == true ]]; then
        echo "Coverage JSON: coverage.json"
    else
        echo ""
        echo "Coverage report generated: htmlcov/index.html"
    fi
fi

exit $EXIT_CODE
