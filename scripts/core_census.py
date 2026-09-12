#!/usr/bin/env python3
"""Static census of the package: what each module pulls, before the core is cut.

The core/packs cut needs a number, not a feeling: which modules the
resident core must hold, and what each of them drags in. This instrument
reads every module's syntax tree and never executes the package. For each
module it records the lines, the project imports it names at module scope
(eager -- a guarded import at module scope is still eager, the interpreter
runs it) and inside functions (lazy), the transitive closure of each, the
third-party imports it names outside the standard library, the database
files it names, and the direct client sites the registry-funnel guard
counts. It maps every router the API includes to the module that defines
it and the closure that module pulls.

Every figure carries ``source: "static"``: an import graph is not a
runtime measurement. A census that finds no module is an error, never a
result. Run from the repository root::

    python3 scripts/core_census.py                 # the table
    python3 scripts/core_census.py --json out.json  # the whole report
"""

import argparse
import ast
import json
import re
import sys
from pathlib import Path

_DB = re.compile(r"[\w./-]+\.db\b")
_STDLIB = set(getattr(sys, "stdlib_module_names", ())) | {"__future__"}
_GUARD = Path(__file__).resolve().parent.parent / ".github" / "scripts" / "registry_funnel_guard.py"


def _count_sites():
    """The funnel guard's own counter, so the census and the guard agree."""
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("registry_funnel_guard", str(_GUARD))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.count_sites
    except Exception:  # noqa: BLE001 - the guard may be absent in a fixture tree
        return lambda text: 0


def _modules(root):
    root = Path(root).resolve()
    package = root.name
    out = {}
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root).with_suffix("")
        parts = [package, *rel.parts]
        if parts[-1] == "__init__":
            parts = parts[:-1]
        out[".".join(parts)] = path
    return package, out


def _resolve(name, level, module_name, is_package, known):
    """The project module an import statement names, or None when it is not one."""
    if level:
        base = module_name.split(".") if is_package else module_name.split(".")[:-1]
        base = base[: len(base) - (level - 1)] if level > 1 else base
        target = ".".join(base + ([name] if name else []))
    else:
        target = name
    return target if target in known else None


def _imports(tree, module_name, is_package, known, package):
    eager_project, lazy_project = [], []
    eager_third, lazy_third = [], []

    def record(node, lazy):
        project = lazy_project if lazy else eager_project
        third = lazy_third if lazy else eager_third
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if alias.name == package or alias.name.startswith(package + "."):
                    target = _resolve(alias.name, 0, module_name, is_package, known)
                    if target:
                        project.append(target)
                elif top not in _STDLIB:
                    third.append(top)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level or base == package or base.startswith(package + "."):
                target = _resolve(base, node.level, module_name, is_package, known)
                if target:
                    project.append(target)
                # ``from pkg.sub import name``: name may itself be a module.
                for alias in node.names:
                    child = _resolve((base + "." if base else "") + alias.name, node.level, module_name, is_package, known)
                    if child:
                        project.append(child)
            elif base:
                top = base.split(".")[0]
                if top not in _STDLIB:
                    third.append(top)

    def walk(body, lazy):
        for node in body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                record(node, lazy)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                walk(getattr(node, "body", []), True)
            elif isinstance(node, ast.ClassDef):
                walk(node.body, lazy)
            else:
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        walk(child.body, True)
                    elif hasattr(child, "body") and isinstance(getattr(child, "body"), list):
                        walk(child.body, lazy)
                        for attr in ("orelse", "finalbody", "handlers"):
                            extra = getattr(child, attr, None)
                            if isinstance(extra, list):
                                walk([h for h in extra if not isinstance(h, ast.ExceptHandler)], lazy)
                                walk([s for h in extra if isinstance(h, ast.ExceptHandler) for s in h.body], lazy)
                    elif isinstance(child, (ast.Import, ast.ImportFrom)):
                        record(child, lazy)

    walk(tree.body, False)
    # Top-level try/except and if blocks are module scope: their imports are eager.
    dedupe = lambda items: sorted(set(items))  # noqa: E731
    eager_project = [m for m in dedupe(eager_project) if m != module_name]
    lazy_project = [m for m in dedupe(lazy_project) if m != module_name and m not in eager_project]
    return eager_project, lazy_project, dedupe(eager_third), [t for t in dedupe(lazy_third) if t not in eager_third]


def _closure(start, edges):
    seen, stack = set(), list(edges.get(start, []))
    while stack:
        node = stack.pop()
        if node in seen or node == start:
            continue
        seen.add(node)
        stack.extend(edges.get(node, []))
    return sorted(seen)


def _api_map(root, package, modules, report_modules):
    app_path = Path(root) / "api" / "app.py"
    if not app_path.exists():
        return {"routers": [], "app_closure_eager": 0}
    text = app_path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    app_name = f"{package}.api.app"
    origin = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            target = _resolve(node.module or "", node.level, app_name, False, modules)
            for alias in node.names:
                local = alias.asname or alias.name
                child = _resolve(((node.module or "") + "." if node.module else "") + alias.name, node.level, app_name, False, modules)
                origin[local] = child or target
    routers = []
    for m in re.finditer(r"include_router\(\s*([A-Za-z_][\w.]*)", text):
        name = m.group(1)
        module = origin.get(name.split(".")[0], app_name)
        routers.append({
            "name": name,
            "module": module,
            "closure_eager": len(report_modules.get(module, {}).get("closure_eager", [])),
        })
    return {"routers": routers, "app_closure_eager": len(report_modules.get(app_name, {}).get("closure_eager", []))}


def census(root):
    """The report for the package at ``root``. Static; refuses an empty tree."""
    package, modules = _modules(root)
    if not modules:
        raise ValueError(f"no module under {root}: a census of nothing is an error")
    count_sites = _count_sites()
    parsed, report = {}, {}
    for name, path in modules.items():
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text)
        is_package = path.name == "__init__.py"
        eager, lazy, heavy_eager, heavy_lazy = _imports(tree, name, is_package, modules, package)
        databases = sorted({m.group(0) for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str) for m in _DB.finditer(node.value)})
        parsed[name] = (eager, lazy)
        report[name] = {
            "path": str(path.relative_to(Path(root).resolve().parent)),
            "lines": text.count("\n") + (0 if text.endswith("\n") or not text else 1),
            "imports_eager": eager,
            "imports_lazy": lazy,
            "heavy_eager": heavy_eager,
            "heavy_lazy": heavy_lazy,
            "databases": databases,
            "direct_client_sites": int(count_sites(text)),
        }
    eager_edges = {n: e for n, (e, _l) in parsed.items()}
    all_edges = {n: e + l for n, (e, l) in parsed.items()}
    for name in report:
        report[name]["closure_eager"] = _closure(name, eager_edges)
        report[name]["closure_all"] = _closure(name, all_edges)
    ordered = {name: report[name] for name in sorted(report)}
    return {
        "source": "static",
        "package": package,
        "totals": {"modules": len(ordered), "lines": sum(m["lines"] for m in ordered.values())},
        "modules": ordered,
        "api": _api_map(root, package, modules, ordered),
    }


def _table(report, limit):
    rows = sorted(report["modules"].items(), key=lambda kv: (-len(kv[1]["closure_eager"]), kv[0]))
    print(f"source: static  package: {report['package']}  modules: {report['totals']['modules']}  lines: {report['totals']['lines']}")
    print(f"{'module':52} {'lines':>6} {'eager':>6} {'all':>6} {'sites':>5}  heavy (eager)")
    for name, m in rows[:limit]:
        print(f"{name:52} {m['lines']:6d} {len(m['closure_eager']):6d} {len(m['closure_all']):6d} {m['direct_client_sites']:5d}  {','.join(m['heavy_eager'])}")
    print(f"api: {len(report['api']['routers'])} routers included; app eager closure {report['api']['app_closure_eager']} modules")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=str(Path(__file__).resolve().parent.parent / "opti_oignon"))
    parser.add_argument("--json", default=None, help="write the whole report here")
    parser.add_argument("--limit", type=int, default=40)
    args = parser.parse_args(argv)
    report = census(args.root)
    _table(report, args.limit)
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1, sort_keys=True), encoding="utf-8")
        print(f"report written: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
