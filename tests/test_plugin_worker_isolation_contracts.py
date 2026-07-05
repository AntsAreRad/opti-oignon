#!/usr/bin/env python3
"""Contracts for the host-package boundary of the plugin worker.

The worker runs plugin code behind a process boundary with a minimal,
secret-free environment: no PYTHONPATH, no encryption keys, no search
credentials. A plugin importing the host package at module level makes
the initialization handshake hostage to whatever the host install
layout drags in -- on the field this stalled four plugins past the
startup timeout ("failed initialization handshake: Timed out reading
from plugin socket"). These contracts pin the deterministic boundary:

  * Contract 1 -- inside the worker's module loading, importing the
    host package (top level or any submodule) raises ImportError
    immediately -- no filesystem-layout dependence, no wait.
  * Contract 2 -- the boundary is surgical: the standard library stays
    fully importable (pathlib included -- the in-process sandbox blocks
    it, the worker does not: here the process IS the isolation), and
    the guard is idempotent.
  * Contract 3 -- the four field-logged plugins (fact-checker,
    github-connector, scratchpad, task-extractor) load under the
    boundary with their standalone fallbacks engaged, instead of
    hanging on the host stack.
  * Contract 4 -- the real initialization path (the worker server's
    initialize handler) completes fast on an entry that reaches for the
    host package, answering ok instead of timing out.

Local-only (the public distribution ships no tests). Runs under pytest or
the __main__ runner. The worker module is loaded from file (stdlib-only);
the harness hides any host modules cached by the test process so the
worker's pristine import world is reproduced faithfully. Delivered
plugins are copied to a temporary directory so initialization side
effects never touch the tree.
"""

import contextlib
import importlib.util
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_OO = _REPO / "opti_oignon"
_PLUGINS_DIR = _OO / "plugins"


def _load_worker():
    """Load plugin_worker.py from file under a private module name."""
    name = "_plugin_worker_isolation_contract"
    saved = sys.modules.get(name)
    spec = importlib.util.spec_from_file_location(name, _OO / "plugin_worker.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)

    def restore():
        if saved is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = saved

    return mod, restore


@contextlib.contextmanager
def _pristine_worker_world(worker_mod):
    """Reproduce the worker's import world in the test process.

    The worker process never has host modules cached; the test process
    does. Hide them for the duration, and remove any guard instance the
    production code installed so the host process is left untouched.
    """
    saved = {
        k: sys.modules.pop(k)
        for k in list(sys.modules)
        if k == "opti_oignon" or k.startswith("opti_oignon.")
    }
    # Force the ADVERSE layout: the host package IS importable on sys.path,
    # so only the deliberate guard -- never an accident of the install
    # layout -- can be what refuses it.
    path_added = str(_REPO) not in sys.path
    if path_added:
        sys.path.insert(0, str(_REPO))
    try:
        yield
    finally:
        if path_added:
            with contextlib.suppress(ValueError):
                sys.path.remove(str(_REPO))
        guard_cls = getattr(worker_mod, "_HostPackageGuard", None)
        if guard_cls is not None:
            sys.meta_path[:] = [
                f for f in sys.meta_path if not isinstance(f, guard_cls)
            ]
        for key in list(sys.modules):
            if key == "opti_oignon" or key.startswith("opti_oignon."):
                sys.modules.pop(key, None)
        for key in list(sys.modules):
            if key.startswith("_oo_worker_plugin_"):
                sys.modules.pop(key, None)
        sys.modules.update(saved)


def _write_entry(tmp: Path, body: str) -> Path:
    plugin_dir = tmp / "probe"
    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "entry_point.py").write_text(body)
    return plugin_dir


_PROBE_ENTRY = '''
import time as _time

_t0 = _time.perf_counter()
try:
    import opti_oignon
    HOST_TOP = "succeeded"
except ImportError as exc:
    HOST_TOP = f"blocked: {exc}"
TOP_SECONDS = _time.perf_counter() - _t0

_t1 = _time.perf_counter()
try:
    from opti_oignon.db_utils import safe_connect
    HOST_SUB = "succeeded"
except ImportError as exc:
    HOST_SUB = f"blocked: {exc}"
SUB_SECONDS = _time.perf_counter() - _t1

import json as _json
import pathlib as _pathlib
import sqlite3 as _sqlite3
STDLIB_OK = all((
    _json.dumps({"x": 1}) == '{"x": 1}',
    _pathlib.Path(".").name is not None,
    _sqlite3.sqlite_version is not None,
))
'''


# ---------------------------------------------------------------------------
# Contract 1 -- host imports fail immediately inside the worker loading
# ---------------------------------------------------------------------------
def test_c1_host_package_import_fails_fast():
    worker, restore = _load_worker()
    try:
        with _pristine_worker_world(worker), \
                tempfile.TemporaryDirectory() as tmp:
            plugin_dir = _write_entry(Path(tmp), _PROBE_ENTRY)
            module = worker.load_plugin_module(
                "probe", str(plugin_dir), "entry_point.py",
            )
            assert "isolation boundary" in str(module.HOST_TOP), (
                "the refusal must come from the deliberate worker guard, "
                f"not from the install layout: {module.HOST_TOP}"
            )
            assert "isolation boundary" in str(module.HOST_SUB), (
                "the refusal must come from the deliberate worker guard, "
                f"not from the install layout: {module.HOST_SUB}"
            )
            assert module.TOP_SECONDS < 0.5, (
                f"the refusal must be immediate: {module.TOP_SECONDS:.3f}s"
            )
            assert module.SUB_SECONDS < 0.5, (
                f"the refusal must be immediate: {module.SUB_SECONDS:.3f}s"
            )
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract 2 -- surgical boundary: stdlib untouched, guard idempotent
# ---------------------------------------------------------------------------
def test_c2_stdlib_untouched_and_guard_idempotent():
    worker, restore = _load_worker()
    try:
        with _pristine_worker_world(worker), \
                tempfile.TemporaryDirectory() as tmp:
            plugin_dir = _write_entry(Path(tmp), _PROBE_ENTRY)
            module = worker.load_plugin_module(
                "probe", str(plugin_dir), "entry_point.py",
            )
            assert module.STDLIB_OK is True, (
                "the boundary must not touch the standard library "
                "(pathlib stays importable in the worker)"
            )
            # Idempotence: a second load must not stack guard instances.
            guard_cls = getattr(worker, "_HostPackageGuard", None)
            assert guard_cls is not None, "guard class must exist"
            sys.modules.pop("_oo_worker_plugin_probe", None)
            worker.load_plugin_module(
                "probe", str(plugin_dir), "entry_point.py",
            )
            count = sum(
                1 for f in sys.meta_path if isinstance(f, guard_cls)
            )
            assert count == 1, f"guard must install once, found {count}"
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract 3 -- the four field-logged plugins load with fallbacks engaged
# ---------------------------------------------------------------------------
def test_c3_field_plugins_load_with_fallbacks():
    worker, restore = _load_worker()
    failures: list[str] = []
    try:
        with _pristine_worker_world(worker), \
                tempfile.TemporaryDirectory() as tmp:
            for name, folder in (
                ("fact-checker", "fact-checker"),
                ("github-connector", "github-connector"),
                ("scratchpad", "scratchpad"),
                ("task-extractor", "task-extractor"),
            ):
                src = _PLUGINS_DIR / folder
                work_dir = Path(tmp) / folder
                shutil.copytree(src, work_dir)
                started = time.perf_counter()
                try:
                    module = worker.load_plugin_module(
                        name, str(work_dir), "entry_point.py",
                    )
                except Exception as exc:
                    failures.append(
                        f"{name}: load failed: {type(exc).__name__}: {exc}"
                    )
                    continue
                elapsed = time.perf_counter() - started
                if elapsed > 2.0:
                    failures.append(f"{name}: load took {elapsed:.2f}s")
                if name == "fact-checker":
                    if getattr(module, "_WEB_SEARCH_AVAILABLE", None) is not False:
                        failures.append(
                            f"{name}: web search fallback not engaged"
                        )
                else:
                    fallback = getattr(module, "_safe_connect", None)
                    if getattr(fallback, "__name__", "") != "<lambda>":
                        failures.append(
                            f"{name}: sqlite fallback not engaged "
                            f"({fallback!r})"
                        )
        assert not failures, (
            "field plugins did not engage their fallbacks:\n  "
            + "\n  ".join(failures)
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract 4 -- the real initialize handler answers ok, fast
# ---------------------------------------------------------------------------
def test_c4_initialize_handler_completes_fast():
    worker, restore = _load_worker()
    try:
        with _pristine_worker_world(worker), \
                tempfile.TemporaryDirectory() as tmp:
            plugin_dir = _write_entry(Path(tmp), _PROBE_ENTRY)
            server = worker.PluginWorkerServer(
                plugin_name="probe",
                plugin_dir=str(plugin_dir),
                entry_point="entry_point.py",
                socket_path=str(Path(tmp) / "unused.sock"),
                hmac_key=b"\x00" * 32,
            )
            started = time.perf_counter()
            response = server._handle_initialize({}, "rid-1")
            elapsed = time.perf_counter() - started
            assert "result" in response, response
            assert response["result"].get("status") == "ok", response
            assert elapsed < 2.0, (
                f"initialize must not stall on the host stack: {elapsed:.2f}s"
            )
    finally:
        restore()


# ---------------------------------------------------------------------------
# Runner (pytest picks up the test_ functions; direct execution works too)
# ---------------------------------------------------------------------------
def _main(argv: list[str]) -> int:
    names = sorted(n for n in globals() if n.startswith("test_"))
    selected = [
        n for n in names if not argv or any(fragment in n for fragment in argv)
    ]
    failures = 0
    for name in selected:
        try:
            globals()[name]()
        except Exception as exc:
            failures += 1
            print(f"FAIL {name}: {exc.__class__.__name__}: {exc}")
            traceback.print_exc()
        else:
            print(f"PASS {name}")
    print(f"{len(selected) - failures}/{len(selected)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
