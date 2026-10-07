#!/usr/bin/env bash
# Graded check ladder. Cheapest tier first; a red tier stops the run.
# Usage: ladder.sh [t0|t1|t2|t3|t4|t5|all|life|frontend|stopgate]
# Every tier reports PASS, FAIL, or SKIP with a named reason. A tier is never
# silently absent: a missing tool is a named skip, not a pass.
# "life" is not part of "all": it runs alone, and exits 3 when it is owed.
# "frontend" runs t1's frontend step alone.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

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
  # The blade register is local maintainer state, never tracked: its path
  # comes from the local git config key oo.bladeRegister.
  blades=$(git config --get oo.bladeRegister 2>/dev/null)
  if [ -n "$blades" ] && [ -f "$blades" ]; then
    pending=$(grep -cE '^[[:space:]]*- \[ \]' "$blades" 2>/dev/null | head -1)
    pending=${pending:-0}
    if [ "$pending" -eq 0 ]; then pass "no blade left unproven"
    else fail "$pending blade(s) still unproven in $blades"; fi
    blade_census "$blades"
  else skip "no blade register (set git config oo.bladeRegister <path>)"; fi
}

# The unchecked lines say nothing of a contract nobody wrote a line for. The
# census lists the contracts in the tree - the tests pytest collects under the
# selection rule's own names and directories, the Rust tests, the front-end
# tests - and holds every one the change adds against HEAD to a checked line
# naming it: by its bare name when no other contract shares it, by its path
# otherwise. Deselected, ignored or skipped ones are exempt and named; a
# front-end test runs outside the ladder, so its blade is owed to the machine.
# A census that finds no Python contract, misses a Rust test or cannot read a
# file fails: it could not see. What the census cannot read of the selection
# rule exempts nothing, so it only ever asks for more. The data places are
# never opened.
blade_census() {
  python3 - "$1" <<'PY'
import ast,fnmatch,os,re,shlex,subprocess,sys
def git(*a): return subprocess.run(["git",*a],capture_output=True,text=True,encoding="utf-8",errors="replace",check=True).stdout
def say(s): print("  "+s)
def show(lines):
    for x in lines[:20]: say("  "+x)
    if len(lines)>20: say(f"  ... and {len(lines)-20} more")
DATA=("data/","opti_oignon/data/")
try:
    import tomllib
    with open("pyproject.toml","rb") as f: ini=tomllib.load(f).get("tool",{}).get("pytest",{}).get("ini_options",{})
except (ImportError,OSError,ValueError): ini={}
def opt(k,d):
    v=ini.get(k,d); return v.split() if isinstance(v,str) else [str(x) for x in v]
FILES,CLASSES,FUNCS=opt("python_files",["test_*.py","*_test.py"]),opt("python_classes",["Test"]),opt("python_functions",["test"])
NOREC=opt("norecursedirs",["*.egg",".*","_darcs","build","CVS","dist","node_modules","venv","{arch}"])
add=ini.get("addopts","")
args=[a for x in ([add] if isinstance(add,str) else add) for a in shlex.split(str(x))]
rule={"--ignore":[],"--ignore-glob":[],"--deselect":[]}
for k,a in enumerate(args):
    for o in rule:
        if a==o and k+1<len(args): rule[o].append(args[k+1])
        elif a.startswith(o+"="): rule[o].append(a[len(o)+1:])
IGN=[os.path.relpath(p) for p in rule["--ignore"]]
DES=[os.path.relpath(p.split("::")[0])+p[len(p.split("::")[0]):] for p in rule["--deselect"]]
def why_py(i,ps):
    p=i.split("::")[0]
    if any(p==g or p.startswith(g+"/") for g in IGN): return "--ignore"
    if any(fnmatch.fnmatch(p,os.path.relpath(g)) for g in rule["--ignore-glob"]): return "--ignore-glob"
    if any(i==d or i.startswith(d+"::") for d in DES if "[" not in d): return "--deselect"
    if ps and all(f"{i}[{x}]" in DES for x in ps): return "--deselect"
def named(n,pats): return any(n.startswith(p) or (any(c in p for c in "*?[") and fnmatch.fnmatch(n,p)) for p in pats)
def kind(p):
    if p.startswith(DATA): return None
    d,b=p.split("/")[:-1],p.split("/")[-1]
    if b.endswith(".py"):
        return "py" if any(fnmatch.fnmatch(b,g) for g in FILES) and not any(fnmatch.fnmatch(x,n) for x in d for n in NOREC) else None
    if b.endswith(".rs"): return "rs"
    if re.search(r"\.(spec|test)\.[cm]?[jt]sx?$",b) and "node_modules" not in d: return "js"
def mark(f):
    v=getattr(f,"value",None)
    return f.attr if isinstance(f,ast.Attribute) and (isinstance(v,ast.Attribute) and v.attr=="mark" or isinstance(v,ast.Name) and v.id=="mark") else None
def skip(ds): return any(mark(d)=="skip" or isinstance(d,ast.Call) and mark(d.func)=="skip" for d in ds)
def pids(n):
    # The ids pytest gives the cases of one parametrize over literal values;
    # None when they cannot be known for sure, and None exempts nothing.
    ps=[d for d in n.decorator_list if isinstance(d,ast.Call) and mark(d.func)=="parametrize"]
    if len(ps)!=1 or len(ps[0].args)!=2 or ps[0].keywords: return None
    try: names,vals=ast.literal_eval(ps[0].args[0]),ast.literal_eval(ps[0].args[1])
    except (ValueError,TypeError,SyntaxError,MemoryError,RecursionError): return None
    names=[s.strip() for s in names.split(",")] if isinstance(names,str) else names
    if not isinstance(names,(list,tuple)) or not isinstance(vals,(list,tuple)) or not vals: return None
    one=lambda v:v if isinstance(v,str) and v and v.isascii() and v.isprintable() else str(v) if v is None or isinstance(v,(bool,int,float)) else None
    out=[]
    for v in vals:
        v=[v] if len(names)==1 else v
        if not isinstance(v,(list,tuple)) or len(v)!=len(names) or None in [one(x) for x in v]: return None
        out.append("-".join(one(x) for x in v))
    return out if len(set(out))==len(out) else None
def py(src):
    # What pytest collects: functions under the rule's names, in classes under
    # its names or in any unittest case, through every block that is not itself
    # a function; a skip mark on the function, its class or its module is kept.
    tree,out=ast.parse(src),[]
    mod=any(isinstance(n,ast.Assign) and any(getattr(t,"id","")=="pytestmark" for t in n.targets)
            and skip(n.value.elts if isinstance(n.value,(ast.List,ast.Tuple)) else [n.value]) for n in tree.body)
    def walk(body,pre,fn,sk):
        for n in body:
            if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)):
                if fn(n.name): out.append((pre+n.name,sk or skip(n.decorator_list),pids(n)))
            elif isinstance(n,ast.ClassDef):
                tc=any((getattr(b,"attr","") or getattr(b,"id","")).endswith("TestCase") for b in n.bases)
                if tc or named(n.name,CLASSES):
                    walk(n.body,pre+n.name+"::",(lambda s:s.startswith("test")) if tc else (lambda s:named(s,FUNCS)),sk or skip(n.decorator_list))
            else:
                for _,v in ast.iter_fields(n):
                    if isinstance(v,list):
                        walk([x for x in v if isinstance(x,ast.stmt)],pre,fn,sk)
                        for x in v:
                            if isinstance(x,(ast.excepthandler,ast.match_case)): walk(x.body,pre,fn,sk)
    walk(tree.body,"",lambda s:named(s,FUNCS),mod)
    return out,[n.name for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))]
RA=re.compile(r"#\[\s*(?:\w+\s*::\s*)*test\s*\]")
RF=re.compile(r"((?:#\[[^\]]*\]\s*)+)(?:pub(?:\s*\([^)]*\))?\s+)?(?:(?:const|async|unsafe)\s+)*(?:extern\s+\"[^\"]*\"\s+)?fn\s+(\w+)")
def rs(src):
    src=re.sub(r"//[^\n]*","",re.sub(r"/\*.*?\*/","",src,flags=re.S))
    return [(m.group(2),bool(re.search(r"#\[\s*ignore\b",m.group(1)))) for m in RF.finditer(src) if RA.search(m.group(1))],len(RA.findall(src))
JT=re.compile(r"(?<![\w.$])(?:test|it)(?:\.(only|skip|fixme|fail|todo))?\s*\(\s*(['\"`])((?:\\.|(?!\2)[^\\])*)\2")
def ids(p,src):
    k=kind(p)
    if k=="py": c,d=py(src); return [(f"{p}::{n}","skip" if s else None,ps) for n,s,ps in c],d
    if k=="rs": return [(f"{p}::{n}","#[ignore]" if g else None,None) for n,g in rs(src)[0]],[]
    return [(f"{p}::{m.group(3)}","skip" if m.group(1) in ("skip","fixme","todo") else None,None) for m in JT.finditer(src)],[]
try:
    tree,bad,defs,raw,got={},[],set(),0,0
    for p in git("ls-files","-z","--cached","--others","--exclude-standard").split("\0"):
        k=p and kind(p)
        if not k or not os.path.isfile(p): continue
        try:
            src=open(p,encoding="utf-8").read()
            found,d=ids(p,src); defs.update(f"{p}::{n}" for n in d)
            if k=="rs": raw+=rs(src)[1]
        except (OSError,UnicodeDecodeError,SyntaxError,ValueError) as e:
            bad.append(f"{p}: {type(e).__name__}"); continue
        got+=len(found) if k=="rs" else 0
        tree.update((i,(k,w,ps)) for i,w,ps in found)
    head=set(git("ls-tree","-r","-z","--name-only","HEAD").split("\0"))
    changed=set(git("diff","--name-only","--no-renames","-z","HEAD").split("\0"))|set(git("ls-files","-z","--others","--exclude-standard").split("\0"))
    added=[]
    for p in sorted(x for x in changed if x and kind(x)):
        try: before={i for i,_,_ in ids(p,git("show",f"HEAD:{p}"))[0]} if p in head else set()
        except (SyntaxError,ValueError) as e:
            bad.append(f"HEAD:{p}: {type(e).__name__}"); continue
        added+=sorted(i for i in tree if i.startswith(p+"::") and i not in before)
    suf={}
    for i in tree:
        q=i.split("::")
        for j in range(len(q)): s="::".join(q[j:]); suf[s]=suf.get(s,0)+1
    known=suf.keys()|defs|{d.split("::")[-1] for d in defs}
    covers,lost=set(),[]
    for line in open(sys.argv[1],encoding="utf-8",errors="replace"):
        m=re.match(r"\s*- \[[xX]\] (.*)",line)
        if not m: continue
        h=m.group(1).split(" - ")[0].strip()
        c={h}|{h[:k] for k,ch in enumerate(h) if ch=="["}
        covers|={x for x in c if suf.get(x)==1}
        if not c&known: lost.append(h)
    cov,exm,unc,owe=[],[],[],[]
    for i in added:
        k,why,ps=tree[i]
        why=why or (why_py(i,ps) if k=="py" else None)
        q=i.split("::")
        if why: exm.append(f"exempt ({why}): {i}")
        elif any("::".join(q[j:]) in covers for j in range(len(q))): cov.append(i)
        elif k=="js": owe.append(i)
        else: unc.append(i)
    say(f"census: {len(tree)} contract(s) in the tree, {len(added)} added against HEAD; {len(lost)} checked line(s) name nothing in the tree ({len(set(lost))} name(s))")
    rc=0
    if bad: say(f"FAIL  the census could not read {len(bad)} file(s)"); show([f"unreadable: {b}" for b in bad]); rc=1
    if not any(k=="py" for k,*_ in tree.values()): say("FAIL  the census is blind: no Python contract found in the tree"); rc=1
    if got<raw: say(f"FAIL  the census is blind to Rust tests: {raw} test attribute(s), {got} test function(s) found"); rc=1
    show(exm)
    if unc: say(f"FAIL  {len(unc)} contract(s) added against HEAD have no checked blade line"); show([f"uncovered: {i}" for i in unc]); rc=1
    if owe:
        say(f"SKIP  OWED: {len(owe)} front-end test(s) added against HEAD run outside the ladder; their blades are owed to the machine")
        show([f"owed: {i}" for i in owe])
    if not rc: say("PASS  "+(f"every contract added against HEAD is accounted for ({len(added)} added: {len(cov)} covered, {len(exm)} exempt, {len(owe)} owed)" if added else "no contract added against HEAD"))
    sys.exit(rc or (3 if owe else 0))
except (subprocess.CalledProcessError,OSError) as e:
    say(f"FAIL  the census could not run: {e}"); sys.exit(1)
PY
  case $? in
    0) ;;
    3) owed=1 ;;
    *) rc_total=1 ;;
  esac
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
