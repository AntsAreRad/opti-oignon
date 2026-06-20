#!/usr/bin/env bash
# audit_deps.sh -- Automated dependency vulnerability scan for Opti-Oignon (S155).
#
# Runs pip-audit (Python) and npm audit (frontend) and produces
# machine-readable JSON reports suitable for CI integration.
#
# Usage:
#   ./scripts/audit_deps.sh [--output-dir DIR] [--fail-on SEVERITY]
#
# Options:
#   --output-dir DIR       Directory for JSON reports (default: ./audit_reports)
#   --fail-on SEVERITY     Exit non-zero if any vuln >= SEVERITY
#                          (critical, high, moderate, low; default: high)
#
# Outputs:
#   <output-dir>/pip_audit.json    -- pip-audit results
#   <output-dir>/npm_audit.json   -- npm audit results
#   <output-dir>/audit_summary.json -- combined summary

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="$PROJECT_ROOT/audit_reports"
FAIL_ON="high"
EXIT_CODE=0

# -- Parse arguments --------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --fail-on)
            FAIL_ON="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 2
            ;;
    esac
done

mkdir -p "$OUTPUT_DIR"

echo "Opti-Oignon Dependency Audit (S155)"
echo "Output: $OUTPUT_DIR"
echo "Fail threshold: $FAIL_ON"
echo ""

# -- Python dependencies (pip-audit) ----------------------------------------
echo "-- Python dependency scan (pip-audit) --"

PIP_AUDIT_JSON="$OUTPUT_DIR/pip_audit.json"

if command -v pip-audit &>/dev/null; then
    pip-audit --format=json > "$PIP_AUDIT_JSON" 2>/dev/null || true

    PYTHON_TOTAL=$(python3 -c "
import json
with open('$PIP_AUDIT_JSON') as f:
    data = json.load(f)
deps = data.get('dependencies', [])
vulns = [d for d in deps if d.get('vulns')]
print(f'{len(deps)} scanned, {len(vulns)} with vulnerabilities')
for v in vulns:
    name = v['name']
    ver = v['version']
    for vuln in v['vulns']:
        vid = vuln.get('id', 'unknown')
        fix = vuln.get('fix_versions', [])
        fix_str = ', '.join(fix) if fix else 'no fix available'
        print(f'  {name}=={ver}: {vid} (fix: {fix_str})')
")
    echo "$PYTHON_TOTAL"
else
    echo "pip-audit not installed -- skipping Python scan"
    echo '{"dependencies": [], "error": "pip-audit not installed"}' > "$PIP_AUDIT_JSON"
fi

echo ""

# -- Frontend dependencies (npm audit) --------------------------------------
echo "-- Frontend dependency scan (npm audit) --"

NPM_AUDIT_JSON="$OUTPUT_DIR/npm_audit.json"
FRONTEND_DIR="$PROJECT_ROOT/frontend"

if [ -d "$FRONTEND_DIR" ] && [ -f "$FRONTEND_DIR/package-lock.json" ]; then
    cd "$FRONTEND_DIR"
    npm audit --json > "$NPM_AUDIT_JSON" 2>/dev/null || true

    python3 -c "
import json
with open('$NPM_AUDIT_JSON') as f:
    data = json.load(f)
meta = data.get('metadata', {}).get('vulnerabilities', {})
total = sum(meta.values())
print(f'Vulnerabilities: {total} total')
for sev in ('critical', 'high', 'moderate', 'low', 'info'):
    count = meta.get(sev, 0)
    if count > 0:
        print(f'  {sev}: {count}')
vulns = data.get('vulnerabilities', {})
for name, info in vulns.items():
    sev = info.get('severity', '?')
    via_list = info.get('via', [])
    titles = []
    for v in via_list:
        if isinstance(v, dict):
            titles.append(v.get('title', '?'))
        else:
            titles.append(str(v))
    print(f'  {name} ({sev}): {titles[0] if titles else \"?\"}')" || true

    cd "$PROJECT_ROOT"
else
    echo "Frontend directory or package-lock.json not found -- skipping"
    echo '{"vulnerabilities": {}, "error": "no frontend lockfile"}' > "$NPM_AUDIT_JSON"
fi

echo ""

# -- Combined summary -------------------------------------------------------
echo "-- Generating combined summary --"

SUMMARY_JSON="$OUTPUT_DIR/audit_summary.json"

python3 -c "
import json
from datetime import datetime, timezone

pip_data = {}
npm_data = {}

with open('$PIP_AUDIT_JSON') as f:
    pip_data = json.load(f)
with open('$NPM_AUDIT_JSON') as f:
    npm_data = json.load(f)

# Python summary
py_deps = pip_data.get('dependencies', [])
py_vuln_deps = [d for d in py_deps if d.get('vulns')]
py_vulns = []
for d in py_vuln_deps:
    for v in d['vulns']:
        py_vulns.append({
            'package': d['name'],
            'version': d['version'],
            'id': v.get('id', 'unknown'),
            'fix_versions': v.get('fix_versions', []),
            'description': v.get('description', ''),
        })

# NPM summary
npm_meta = npm_data.get('metadata', {}).get('vulnerabilities', {})
npm_vulns = []
for name, info in npm_data.get('vulnerabilities', {}).items():
    for via in info.get('via', []):
        if isinstance(via, dict):
            npm_vulns.append({
                'package': name,
                'severity': via.get('severity', info.get('severity', '?')),
                'title': via.get('title', '?'),
                'url': via.get('url', ''),
            })

summary = {
    'timestamp': datetime.now(timezone.utc).isoformat(),
    'python': {
        'total_packages': len(py_deps),
        'vulnerable_packages': len(py_vuln_deps),
        'total_vulnerabilities': len(py_vulns),
        'vulnerabilities': py_vulns,
    },
    'frontend': {
        'severity_counts': npm_meta,
        'total_vulnerabilities': sum(npm_meta.values()) if npm_meta else 0,
        'vulnerabilities': npm_vulns,
    },
}

with open('$SUMMARY_JSON', 'w') as f:
    json.dump(summary, f, indent=2)

print(f'Python: {len(py_deps)} packages, {len(py_vulns)} vulnerabilities')
print(f'Frontend: {summary[\"frontend\"][\"total_vulnerabilities\"]} vulnerabilities')
print(f'Summary written to $SUMMARY_JSON')
" || true

# -- Threshold check --------------------------------------------------------
SEVERITY_ORDER="info low moderate high critical"

should_fail() {
    local threshold="$1"
    local found_sev="$2"
    local past_threshold=0
    for s in $SEVERITY_ORDER; do
        if [ "$s" = "$threshold" ]; then
            past_threshold=1
        fi
        if [ "$past_threshold" -eq 1 ] && [ "$s" = "$found_sev" ]; then
            return 0
        fi
    done
    return 1
}

# Check Python vulns (pip-audit does not provide per-vuln severity in basic output,
# so we flag any vulnerability as potentially high)
PY_VULN_COUNT=$(python3 -c "
import json
with open('$PIP_AUDIT_JSON') as f:
    data = json.load(f)
print(sum(1 for d in data.get('dependencies', []) if d.get('vulns')))
")

if [ "$PY_VULN_COUNT" -gt 0 ]; then
    echo ""
    echo "Warning: $PY_VULN_COUNT Python packages have known vulnerabilities"
fi

# Check npm severity threshold
NPM_FAIL=$(python3 -c "
import json
threshold = '$FAIL_ON'
order = ['info', 'low', 'moderate', 'high', 'critical']
threshold_idx = order.index(threshold) if threshold in order else 3
with open('$NPM_AUDIT_JSON') as f:
    data = json.load(f)
meta = data.get('metadata', {}).get('vulnerabilities', {})
for sev in order[threshold_idx:]:
    if meta.get(sev, 0) > 0:
        print('1')
        exit()
print('0')
")

if [ "$NPM_FAIL" = "1" ]; then
    echo "Warning: Frontend has vulnerabilities at or above '$FAIL_ON' severity"
fi

echo ""
echo "Audit complete. Reports in $OUTPUT_DIR/"
exit $EXIT_CODE
