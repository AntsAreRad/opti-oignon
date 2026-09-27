#!/usr/bin/env bash
# Graded check ladder. Cheapest tier first; a red tier stops the run.
# Usage: ladder.sh [t0|t1|t2|t3|t4|t5|all|life|frontend|stopgate]
# Every tier reports PASS, FAIL, or SKIP with a named reason. A tier is never
# silently absent: a missing tool is a named skip, not a pass.
# "life" is not part of "all": it runs alone, and exits 3 when it is owed.
# "frontend" runs t1's frontend step alone.
set -uo pipefail
cd "${CLAUDE_PROJECT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}" || exit 1

TIER="${1:-all}"
JUNIT="${JUNIT_OUT:-${TMPDIR:-/tmp}/oo_junit.xml}"
rc_total=0
owed=0

say()  { printf '%s\n' "$*"; }
head_() { printf '\n=== %s ===\n' "$*"; }
pass() { printf '  PASS  %s\n' "$*"; }
fail() { printf '  FAIL  %s\n' "$*"; rc_total=1; }
skip() { printf '  SKIP  %s\n' "$*"; }

purge() { find . -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null; }

t0() {
  head_ "t0  syntax, imports, lint"
  # Every Python file git tracks, and every new one it does not ignore; the
  # data places are never opened. An empty list is a failure, not a pass.
  if python3 - <<'PY'
import ast,os,subprocess,sys
listed=subprocess.run(["git","ls-files","--cached","--others","--exclude-standard","*.py"],capture_output=True,text=True,check=True).stdout.split("\n")
files=[p for p in listed if p and os.path.isfile(p) and not p.startswith(("data/","opti_oignon/data/"))]
bad=[]
for p in files:
    try: ast.parse(open(p,encoding="utf-8",errors="ignore").read())
    except SyntaxError as e: bad.append(f"{p}:{e.lineno}")
print(f"parse errors: {len(bad)} in {len(files)} Python file(s)")
[print("   ",b) for b in bad[:10]]
sys.exit(1 if bad or not files else 0)
PY
  then pass "every tracked or new Python file parses"; else fail "syntax errors above, or no file found"; fi

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
    # What the data firewall kept off the real places, and what it does not reach.
    grep -m1 '^data firewall:' ${TMPDIR:-/tmp}/oo_pytest.txt | sed 's/^/  /'
    budgets
  else fail "no junitxml produced - the sweep did not run"
    grep -m1 'data firewall' ${TMPDIR:-/tmp}/oo_pytest.txt | sed 's/^/    /'; fi
  engine_rust
  frontend_step
}

# Every componion and frontend contract carries a time budget (BUDGET_S in its
# suite), read back from the junit file: over budget, or without a budget, is
# named. The frontend suites are the test_ui_* suites and every suite that
# imports the frontend helper (tests/_frontend.py). Any other suite that
# declares BUDGET_S is held to it the same way, every one of its contracts.
budgets() {
  if python3 - "$JUNIT" <<'PY'
import ast, pathlib, sys, xml.etree.ElementTree as ET

def declared(path):
    found = {}
    for node in ast.walk(ast.parse(path.read_text(encoding="ascii"))):
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "BUDGET_S" for t in node.targets):
            found.update(ast.literal_eval(node.value))
    return found

def uses_helper(path):
    text = path.read_text(encoding="utf-8", errors="replace")
    if "_frontend" not in text:
        return False
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.ImportFrom) and node.module == "_frontend":
            return True
        if isinstance(node, ast.Import) and any(alias.name == "_frontend" for alias in node.names):
            return True
    return False

tests = pathlib.Path("tests")
componion = sorted(tests.glob("test_allium_*_contracts.py"))
frontend = sorted({*tests.glob("test_ui_*_contracts.py"), *(p for p in tests.glob("test_*.py") if uses_helper(p))})
modules = {f"tests.{path.stem}" for path in frontend}
others = sorted(p for p in tests.glob("test_*.py") if p not in componion and p not in frontend
                and "BUDGET_S" in p.read_text(encoding="utf-8", errors="replace") and declared(p))
declaring = {f"tests.{path.stem}" for path in others}
families = (
    ("componion", componion, lambda cls: cls.startswith("tests.test_allium_")),
    ("frontend", frontend, lambda cls: cls in modules or cls.rsplit(".", 1)[0] in modules),
    ("declared", others, lambda cls: cls in declaring or cls.rsplit(".", 1)[0] in declaring),
)
cases = list(ET.parse(sys.argv[1]).getroot().iter("testcase"))
red = False
for label, suites, member in families:
    budgets = {}
    for path in suites:
        budgets.update(declared(path))
    seen = over = missing = 0
    for case in cases:
        if not member(case.get("classname", "")):
            continue
        seen += 1
        name, took = case.get("name"), float(case.get("time", 0))
        if name not in budgets:
            missing += 1
            print(f"    no budget: {name}")
        elif took > budgets[name]:
            over += 1
            print(f"    over budget: {name} {took:.2f}s > {budgets[name]}s")
    print(f"  {label} budgets: {len(suites)} suite(s), {seen} contract(s) read, {over} over, {missing} without a budget")
    red = red or bool(over or missing or (suites and not seen))
if (tests / "_frontend.py").is_file() and not frontend:
    print("    the frontend helper exists and no suite was found to use it: the frontend budgets read nothing")
    red = True
sys.exit(1 if red else 0)
PY
  then pass "every componion, frontend and declared contract within its time budget"; else fail "time budgets (above)"; fi
}

# The frontend's lint and production build, both on a copy under $TMPDIR made
# by tests/_frontend.py (frontend_copy: the listed files, the dependencies
# linked one package at a time), so nothing is written in the tree or beside
# the installed packages. eslint must lint at least one component and one
# module under src/ and report 0 errors; vite build must exit 0 and write
# build/index.html. The reports live in the run's own scratch directory, so
# two ladders running at once never read each other's; it is removed when
# the step passes and kept, and named, when it fails. Without the
# dependencies the step is owed, never a pass.
frontend_step() {
  if [ ! -d frontend/node_modules ]; then
    skip "OWED: frontend/node_modules absent"; return 0
  fi
  local scratch copy rc red=0
  if ! scratch=$(mktemp -d "${TMPDIR:-/tmp}/oo_frontend.XXXXXX"); then fail "frontend: no scratch directory"; return; fi
  if ! copy=$(PYTHONDONTWRITEBYTECODE=1 python3 -B -c 'import sys; sys.path.insert(0, "tests"); import _frontend; print(_frontend.frontend_copy(sys.argv[1]))' "$scratch" 2>"$scratch/copy.txt"); then
    fail "frontend: the copy failed -> $scratch/copy.txt"; return
  fi
  (cd "$copy" && node_modules/.bin/eslint . --format json) >"$scratch/eslint.json" 2>"$scratch/eslint.txt"
  if python3 - "$scratch/eslint.json" "$copy" <<'PY'
import json, sys
try:
    with open(sys.argv[1], encoding="utf-8") as handle:
        results = json.load(handle)
except (OSError, ValueError) as exc:
    print(f"    eslint wrote no report ({exc})")
    sys.exit(1)
errors = sum(result.get("errorCount", 0) for result in results)
warnings = sum(result.get("warningCount", 0) for result in results)
source = sys.argv[2] + "/src/"
kinds = {}
for result in results:
    path = result.get("filePath", "")
    if path.startswith(source):
        kind = path.rsplit(".", 1)[-1]
        kinds[kind] = kinds.get(kind, 0) + 1
shown_kinds = ", ".join(f"{count} .{kind}" for kind, count in sorted(kinds.items())) or "none"
print(f"  eslint: {len(results)} file(s) linted ({shown_kinds} under src/), {errors} error(s), {warnings} warning(s)")
shown = 0
for result in results:
    for message in result.get("messages", []):
        if message.get("severity") == 2 and shown < 10:
            shown += 1
            where = result.get("filePath", "").replace(sys.argv[2] + "/", "frontend/")
            print(f"    {where}:{message.get('line')} {message.get('ruleId')} {message.get('message')}")
if not kinds.get("svelte") or not kinds.get("ts"):
    print("    eslint linted no component or no module under src/: its 0 errors mean nothing")
    sys.exit(1)
sys.exit(1 if errors else 0)
PY
  then pass "eslint: 0 errors"; else fail "eslint -> $scratch/eslint.json"; red=1; fi
  (cd "$copy" && node_modules/.bin/vite build) >"$scratch/build.txt" 2>&1; rc=$?
  if [ "$rc" -ne 0 ]; then
    fail "vite build exited $rc -> $scratch/build.txt"; red=1
    { grep -A2 -m1 'error during build' "$scratch/build.txt" || tail -5 "$scratch/build.txt"; } | sed 's/^/    /'
  elif [ ! -f "$copy/build/index.html" ]; then
    fail "vite build exited 0 and wrote no build/index.html -> $scratch/build.txt"; red=1
  else pass "vite build: exit 0, build/index.html written"; fi
  if [ "$red" -eq 0 ]; then rm -rf "$scratch"; else say "  frontend reports kept in $scratch"; fi
}

# The componion engine's own tests, and clippy's proof that its arithmetic is
# explicit. A missing tool is owed, never a pass.
engine_rust() {
  if ! command -v cargo >/dev/null 2>&1; then
    skip "OWED: cargo is not installed; rust/allium's own tests and clippy did not run"; return 0
  fi
  if (cd rust/allium && cargo test --locked --quiet >${TMPDIR:-/tmp}/oo_allium_test.txt 2>&1); then pass "cargo test (rust/allium)"
  else fail "cargo test (rust/allium) -> ${TMPDIR:-/tmp}/oo_allium_test.txt"; fi
  if cargo clippy --version >/dev/null 2>&1; then
    if (cd rust/allium && cargo clippy --locked --quiet -- -D warnings >${TMPDIR:-/tmp}/oo_allium_clippy.txt 2>&1); then
      pass "cargo clippy (rust/allium): the arithmetic is explicit"
    else fail "cargo clippy (rust/allium) -> ${TMPDIR:-/tmp}/oo_allium_clippy.txt"; fi
  else skip "OWED: cargo clippy is not installed; rust/allium's explicit arithmetic is unproven here"; fi
}

t2() {
  head_ "t2  guards"
  n=0
  # Each guard runs with the data places mirrored from HEAD, as CI sees them:
  # a guard that imports the application opens the stores it declares.
  for g in .github/scripts/*_guard.py; do
    [ -e "$g" ] || continue
    n=$((n+1))
    if out=$(python3 tests/_guard_mirrored.py "$g" 2>&1); then pass "$(basename "$g")"
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

# The life tier: the componion's ten-year lives on the native core, in
# tests/life, which the t1 sweep ignores. Exactly AK6 and AQ9 must be
# collected; a skip means the native core is absent or stale, which is owed
# (exit 3) and never a pass; a failure, an error or a contract that ran
# over its budget is a failure, and is reported as one even beside a skip.
life() {
  head_ "life  ten-year lives on the native core"
  purge
  local junit="${TMPDIR:-/tmp}/oo_junit_life.xml"
  rm -f "$junit"
  PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:randomly tests/life --junitxml="$junit" \
    >${TMPDIR:-/tmp}/oo_pytest_life.txt 2>&1
  if [ ! -f "$junit" ]; then fail "no junitxml produced - the life tier did not run"; return; fi
  python3 - "$junit" <<'PY'
import ast, pathlib, sys, xml.etree.ElementTree as ET
budgets = {}
suite = pathlib.Path("tests/life/test_allium_life_contracts.py")
for node in ast.walk(ast.parse(suite.read_text(encoding="ascii"))):
    if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "BUDGET_S" for t in node.targets):
        budgets.update(ast.literal_eval(node.value))
cases = list(ET.parse(sys.argv[1]).getroot().iter("testcase"))
names = sorted(case.get("name") for case in cases)
print(f"  junitxml: {len(cases)} collected: {', '.join(names) or 'none'}")
ids = sorted(name.split("_")[1] for name in names)
if names != sorted(budgets) or ids != ["ak6", "aq9"]:
    print("    the life tier collects exactly AK6 and AQ9, each with its budget")
    sys.exit(1)
failed = [case.get("name") for case in cases if case.find("failure") is not None or case.find("error") is not None]
skipped = [case.get("name") for case in cases if case.find("skipped") is not None]
# A contract that ran is held to its budget even when the other was skipped.
over = [(case.get("name"), float(case.get("time", 0))) for case in cases
        if case.get("name") not in skipped and float(case.get("time", 0)) > budgets[case.get("name")]]
if failed:
    print("    failed: " + ", ".join(failed))
for name, took in over:
    print(f"    over budget: {name} {took:.2f}s > {budgets[name]}s")
if failed or over:
    sys.exit(1)
if skipped:
    sys.exit(3)
for case in cases:
    print(f"    {case.get('name')} {float(case.get('time', 0)):.2f}s of {budgets[case.get('name')]}s")
sys.exit(0)
PY
  case $? in
    0) pass "the life tier ran on the native core, within its budgets" ;;
    3) say "  OWED: the native core is not built here or is stale (scripts/build_oo_core.sh); the life tier did not run"
       owed=1 ;;
    *) fail "the life tier -> ${TMPDIR:-/tmp}/oo_pytest_life.txt" ;;
  esac
}

case "$TIER" in
  t0) t0 ;;
  t1) t1 ;;
  t2) t2 ;;
  t3) t3 ;;
  t4) t4 ;;
  t5) t5 ;;
  all) t0; t2; t1; t3; t4; t5 ;;
  life) life ;;
  frontend) head_ "frontend  lint and build (also run by t1)"; frontend_step ;;
  *) say "unknown tier: $TIER"; exit 64 ;;
esac

if [ "$rc_total" -eq 0 ] && [ "$owed" -eq 1 ]; then
  printf '\n=== ladder result: OWED ===\n'
  exit 3
fi
printf '\n=== ladder result: %s ===\n' "$([ $rc_total -eq 0 ] && echo GREEN || echo RED)"
exit $rc_total
