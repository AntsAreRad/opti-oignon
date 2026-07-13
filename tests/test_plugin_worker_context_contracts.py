#!/usr/bin/env python3
"""Contracts for the hook context handed to plugins by the worker.

Plugin hook callbacks are written against the host-side context object
(``.data`` / ``.config`` / ``.metadata`` plus ``get``/``set``). The RPC
proxy serializes that object faithfully onto the wire, but the worker
handed the raw wire dict to the callbacks -- every attribute access
raised AttributeError, and even ``ctx.get("message")`` silently read
the wrong level (the envelope instead of the data). These contracts pin
the worker-side reconstruction:

  * Contract 1 -- the callback receives an OBJECT exposing the seven
    context fields, ``get`` reads the data dict, ``set`` writes a LOCAL
    view only (the wire payload is never mutated; propagation happens
    through the returned dict, exactly as in-process), and the returned
    dict passes through unchanged.
  * Contract 2 -- a minimal wire payload still yields a working
    context: absent fields default (hook name from the call, plugin
    name from the loaded module, empty dicts) instead of crashing.
  * Contract 3 -- every delivered plugin loads in a worker-faithful
    harness (host package unavailable, fallbacks engaged) and executes
    every hook its manifest declares without raising -- the exact
    failure the field logs showed ("'dict' object has no attribute
    'config'/'data'") cannot reappear.
  * Contract 4 -- the worker envelope semantics are pinned as shipped:
    a hook returning None yields the ok envelope, a missing handler
    yields the no_handler envelope.

Local-only (the public distribution ships no tests). Runs under pytest or
the __main__ runner. The worker module is loaded from file (it is
stdlib-only); delivered plugins are copied to a temporary directory so
their initialization side effects (SQLite files) never touch the tree.
"""

import contextlib
import importlib.util
import shutil
import sys
import tempfile
import traceback
import types
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parent.parent
_OO = _REPO / "opti_oignon"
_PLUGINS_DIR = _OO / "plugins"


def _load_worker():
    """Load plugin_worker.py from file under a private module name.

    The worker seals itself off from the host package by inserting a
    ``_HostPackageGuard`` at the head of ``sys.meta_path`` (plugin_worker.py:176)
    and never takes it back off -- correct in production, where the worker OWNS
    its process. Loaded in-process by this suite, that guard outlives the clause
    and refuses ``opti_oignon.*`` for every later suite in the runner, with an
    honest-looking ImportError that has nothing to do with the code under test.
    Restoring the module cache alone does not reach it. The meta path is
    therefore snapshotted whole and put back whole.
    """
    name = "_plugin_worker_under_contract"
    saved = sys.modules.get(name)
    saved_meta_path = list(sys.meta_path)
    spec = importlib.util.spec_from_file_location(name, _OO / "plugin_worker.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)

    def restore():
        sys.meta_path[:] = saved_meta_path
        if saved is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = saved

    return mod, restore


class _HostBlocker:
    """Worker-faithful import world: the host package is unavailable."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] == "opti_oignon":
            raise ImportError(
                f"'{fullname}' unavailable in the worker-faithful harness"
            )
        return None


@contextlib.contextmanager
def _worker_faithful_imports():
    """Hide any cached host modules and block fresh host imports."""
    saved = {
        k: sys.modules.pop(k)
        for k in list(sys.modules)
        if k == "opti_oignon" or k.startswith("opti_oignon.")
    }
    blocker = _HostBlocker()
    sys.meta_path.insert(0, blocker)
    try:
        yield
    finally:
        sys.meta_path.remove(blocker)
        for key in list(sys.modules):
            if key == "opti_oignon" or key.startswith("opti_oignon."):
                sys.modules.pop(key, None)
        sys.modules.update(saved)


def _purge_worker_plugin_modules():
    for key in list(sys.modules):
        if key.startswith("_oo_worker_plugin_"):
            sys.modules.pop(key, None)


def _superset_wire(hook_name: str, plugin_name: str) -> dict[str, Any]:
    """One representative wire payload covering every hook's needs."""
    return {
        "hook_name": hook_name,
        "plugin_name": plugin_name,
        "conversation_id": "conv-contract",
        "model": "contract-model",
        "data": {
            "message": "please review the plan and compute 2 + 2",
            "response": (
                "The result is 4. TODO: verify the sources.\n"
                "```python\nprint(2 + 2)\n```"
            ),
            "model": "contract-model",
            "duration_ms": 12.5,
            "tokens_in": 10,
            "tokens_out": 20,
            "tool_name": "execute_code",
            "arguments": {"code": "print(2 + 2)"},
            "result": "4",
            "success": True,
            "conversation_id": "conv-contract",
        },
        "config": {},
        "metadata": {"turn": 1},
    }


# ---------------------------------------------------------------------------
# Contract 1 -- object surface, get/set semantics, return passthrough
# ---------------------------------------------------------------------------
def test_c1_context_object_surface_and_local_set():
    worker, restore = _load_worker()
    try:
        captured: dict[str, Any] = {}

        def hook(ctx):
            captured["is_dict"] = isinstance(ctx, dict)
            captured["hook_name"] = ctx.hook_name
            captured["plugin_name"] = ctx.plugin_name
            captured["conversation_id"] = ctx.conversation_id
            captured["model"] = ctx.model
            captured["config"] = dict(ctx.config)
            captured["metadata"] = dict(ctx.metadata)
            captured["get_message"] = ctx.get("message")
            captured["get_default"] = ctx.get("absent-key", "fallback")
            ctx.set("local_key", "local_value")
            captured["after_set"] = ctx.get("local_key")
            return {"reply": "annotated"}

        module = types.ModuleType("probe_plugin")
        module.HOOKS = {"post_inference": hook}
        wire = _superset_wire("post_inference", "probe-plugin")

        result = worker.execute_hook(module, "post_inference", wire)

        assert captured.get("is_dict") is False, (
            f"the callback must receive a context object, not a dict: {captured}"
        )
        assert captured["hook_name"] == "post_inference"
        assert captured["plugin_name"] == "probe-plugin"
        assert captured["conversation_id"] == "conv-contract"
        assert captured["model"] == "contract-model"
        assert captured["metadata"] == {"turn": 1}
        assert captured["get_message"] == (
            "please review the plan and compute 2 + 2"
        ), "get() must read the DATA level, not the wire envelope"
        assert captured["get_default"] == "fallback"
        assert captured["after_set"] == "local_value"
        # HK-01 fidelity: set() writes a LOCAL view; the wire payload the
        # host serialized is never mutated by the plugin.
        assert "local_key" not in wire["data"], (
            "set() must not write through to the wire payload"
        )
        assert result == {"reply": "annotated"}, result
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract 2 -- minimal wire payload still yields a working context
# ---------------------------------------------------------------------------
def test_c2_minimal_wire_defaults():
    worker, restore = _load_worker()
    try:
        captured: dict[str, Any] = {}

        def hook(ctx):
            captured["hook_name"] = ctx.hook_name
            captured["plugin_name"] = ctx.plugin_name
            captured["conversation_id"] = ctx.conversation_id
            captured["model"] = ctx.model
            captured["data"] = dict(ctx.data)
            captured["config"] = dict(ctx.config)
            captured["metadata"] = dict(ctx.metadata)
            return None

        module = types.ModuleType("minimal_plugin")
        module.__plugin_name__ = "minimal-plugin"
        module.HOOKS = {"tool_call": hook}

        worker.execute_hook(module, "tool_call", {"data": {"x": 1}})

        assert captured["hook_name"] == "tool_call"
        assert captured["plugin_name"] == "minimal-plugin", (
            "the plugin name must default to the loaded module's name"
        )
        assert captured["conversation_id"] is None
        assert captured["model"] is None
        assert captured["data"] == {"x": 1}
        assert captured["config"] == {}
        assert captured["metadata"] == {}
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract 3 -- every delivered plugin runs every declared hook
# ---------------------------------------------------------------------------
def test_c3_all_delivered_plugins_execute_declared_hooks():
    import yaml

    worker, restore = _load_worker()
    failures: list[str] = []
    try:
        plugin_dirs = sorted(
            d for d in _PLUGINS_DIR.iterdir()
            if d.is_dir() and (d / "manifest.yaml").exists()
        )
        assert plugin_dirs, f"no delivered plugins found under {_PLUGINS_DIR}"

        with tempfile.TemporaryDirectory() as tmp:
            with _worker_faithful_imports():
                for src in plugin_dirs:
                    with open(src / "manifest.yaml", encoding="utf-8") as fh:
                        manifest = yaml.safe_load(fh)
                    name = str(manifest.get("name") or src.name)
                    entry = str(manifest.get("entry_point") or "entry_point.py")
                    hooks = [str(h) for h in (manifest.get("hooks") or [])]

                    work_dir = Path(tmp) / src.name
                    shutil.copytree(src, work_dir)

                    try:
                        module = worker.load_plugin_module(
                            name, str(work_dir), entry,
                        )
                        init_fn = getattr(module, "init", None)
                        if callable(init_fn):
                            init_fn()
                    except Exception as exc:
                        failures.append(
                            f"{name}: load/init failed: "
                            f"{type(exc).__name__}: {exc}"
                        )
                        continue

                    for hook_name in hooks:
                        wire = _superset_wire(hook_name, name)
                        try:
                            result = worker.execute_hook(
                                module, hook_name, wire,
                            )
                        except Exception as exc:
                            failures.append(
                                f"{name}/{hook_name}: "
                                f"{type(exc).__name__}: {exc}"
                            )
                            continue
                        if not isinstance(result, dict):
                            failures.append(
                                f"{name}/{hook_name}: non-dict result "
                                f"{type(result).__name__}"
                            )

        assert not failures, (
            "delivered plugins failed under the worker context:\n  "
            + "\n  ".join(failures)
        )
    finally:
        _purge_worker_plugin_modules()
        restore()


# ---------------------------------------------------------------------------
# Contract 4 -- worker envelope semantics pinned as shipped
# ---------------------------------------------------------------------------
def test_c4_envelopes_pinned():
    worker, restore = _load_worker()
    try:
        module = types.ModuleType("envelope_plugin")
        module.HOOKS = {"post_inference": lambda ctx: None}

        ok = worker.execute_hook(
            module, "post_inference", _superset_wire("post_inference", "e"),
        )
        assert ok == {"status": "ok"}, ok

        missing = worker.execute_hook(
            module, "tool_call", _superset_wire("tool_call", "e"),
        )
        assert missing == {"status": "no_handler", "hook_name": "tool_call"}, (
            missing
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
