#!/usr/bin/env bash
# Graded check ladder. Cheapest tier first; a red tier stops the run.
# Usage: ladder.sh [t0|t1|t2|t3|t4|t5|all|stopgate]
# Every tier reports PASS, FAIL, or SKIP with a named reason. A tier is never
# silently absent: a missing tool is a named skip, not a pass.
set -uo pipefail
cd "${CLAUDE_PROJECT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}" || exit 1

TIER="${1:-all}"
JUNIT="${JUNIT_OUT:-${TMPDIR:-/tmp}/oo_junit.xml}"
rc_total=0

say()  { printf '%s\n' "$*"; }
head_() { printf '\n=== %s ===\n' "$*"; }
pass() { printf '  PASS  %s\n' "$*"; }
fail() { printf '  FAIL  %s\n' "$*"; rc_total=1; }
skip() { printf '  SKIP  %s\n' "$*"; }

purge() { find . -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null; }

t0() {
  head_ "t0  syntax, imports, lint"
  if python3 - <<'PY'
import ast,pathlib,sys
bad=[]
for p in list(pathlib.Path('opti_oignon').rglob('*.py'))+list(pathlib.Path('tests').rglob('*.py'))+list(pathlib.Path('scripts').rglob('*.py')):
    try: ast.parse(p.read_text(errors='ignore'))
    except SyntaxError as e: bad.append(f"{p}:{e.lineno}")
print("parse errors:", len(bad))
[print("   ",b) for b in bad[:10]]
sys.exit(1 if bad else 0)
PY
  then pass "every tracked Python file parses"; else fail "syntax errors above"; fi

  if command -v ruff >/dev/null 2>&1; then
    out=$(ruff check . --output-format=concise 2>/dev/null); rc=$?
    n=$(printf '%s\n' "$out" | grep -cE ':[0-9]+:[0-9]+:')
    dbt="${RUFF_DEBT:-35}"
    # Proven capable: a non-zero exit with a zero count means the probe is blind.
    if [ "$rc" -ne 0 ] && [ "$n" -eq 0 ]; then
      fail "lint probe is blind: ruff exited ${rc} while the probe counted 0"
    elif [ "$n" -le "$dbt" ]; then pass "ruff ${n} <= recorded debt ${dbt}"
    else fail "ruff ${n} > recorded debt ${dbt} (new lint introduced)"; fi
  else skip "ruff not installed"; fi
}

t1() {
  head_ "t1  contracts"
  purge
  if PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:randomly --junitxml="$JUNIT" >${TMPDIR:-/tmp}/oo_pytest.txt 2>&1; then :; fi
  if [ -f "$JUNIT" ]; then
    python3 - "$JUNIT" <<'PY'
import sys,xml.etree.ElementTree as ET
r=ET.parse(sys.argv[1]).getroot()
s=r if r.tag=='testsuite' else r.find('testsuite')
g=lambda k:int(s.get(k,0))
tot,f,e,sk=g('tests'),g('failures'),g('errors'),g('skipped')
print(f"  junitxml: {tot} collected / {f} failed / {e} errors / {sk} skipped")
sys.exit(1 if (f or e) else 0)
PY
    if [ $? -eq 0 ]; then pass "junitxml is the authority; no failures, no errors"
    else fail "see ${TMPDIR:-/tmp}/oo_pytest.txt"; grep -E '^FAILED|^ERROR' ${TMPDIR:-/tmp}/oo_pytest.txt | head -12; fi
  else fail "no junitxml produced - the sweep did not run"; fi
}

t2() {
  head_ "t2  guards"
  n=0
  for g in .github/scripts/*_guard.py; do
    [ -e "$g" ] || continue
    n=$((n+1))
    if out=$(python3 "$g" 2>&1); then pass "$(basename "$g")"
    else fail "$(basename "$g") -> $(printf '%s' "$out" | tail -1 | cut -c1-140)"; fi
  done
  [ "$n" -gt 0 ] || skip "no guard found under .github/scripts/"
  printf '  %d guard(s) run\n' "$n"
}

t3() {
  head_ "t3  directed mutation"
  say "  Manual tier, driven by the blade skill: for each new contract, apply its"
  say "  blade, observe red, restore byte-exact, confirm the checksum."
  if [ -f .claude/state/blades.md ]; then
    pending=$(grep -cE '^[[:space:]]*- \[ \]' .claude/state/blades.md 2>/dev/null | head -1)
    pending=${pending:-0}
    if [ "$pending" -eq 0 ]; then pass "no blade left unproven"
    else fail "$pending blade(s) still unproven in .claude/state/blades.md"; fi
  else skip "no blade register yet (created by the open-block skill)"; fi
}

t4() {
  head_ "t4  property and fuzz"
  if python3 -c "import hypothesis" 2>/dev/null; then
    if [ -d tests/property ]; then
      if PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:randomly tests/property >${TMPDIR:-/tmp}/oo_prop.txt 2>&1
      then pass "property suite"; else fail "property suite - see ${TMPDIR:-/tmp}/oo_prop.txt"; fi
    else skip "tests/property/ does not exist yet"; fi
  else skip "hypothesis not installed"; fi
}

t5() {
  head_ "t5  adversarial"
  if [ -d tests/adversarial ]; then
    if PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:randomly tests/adversarial >${TMPDIR:-/tmp}/oo_adv.txt 2>&1
    then pass "adversarial suite"; else fail "adversarial suite - see ${TMPDIR:-/tmp}/oo_adv.txt"; fi
  else skip "tests/adversarial/ does not exist yet"; fi
}


case "$TIER" in
  t0) t0 ;;
  t1) t1 ;;
  t2) t2 ;;
  t3) t3 ;;
  t4) t4 ;;
  t5) t5 ;;
  all) t0; t2; t1; t3; t4; t5 ;;
  *) say "unknown tier: $TIER"; exit 64 ;;
esac

printf '\n=== ladder result: %s ===\n' "$([ $rc_total -eq 0 ] && echo GREEN || echo RED)"
exit $rc_total
