#!/usr/bin/env python3
"""
Tests for S195 F8b -- plugin hook dispatch contracts.

Per-fix coverage:
- HK-02: the redaction placeholder never round-trips into the shared chain
         data (an echoing no-permission plugin cannot blank the real
         message/response for downstream hooks and final_data)
- HK-01: contract lock -- in-place ctx mutation does not propagate; only a
         returned dict merges downstream; each hook sees its own copy
- regression guard on the edited merge block: error isolation intact

Loader idiom: spec_from_file_location with sys.modules registration BEFORE
exec_module; package stub with real __path__; opti_oignon.config and
opti_oignon.db_utils pre-seeded; added modules/stubs cleaned at module
teardown (S194 hardening).
"""

import importlib.util
import sqlite3
import sys
import tempfile
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parent.parent

_ADDED_MODULES: list[str] = []
_TMP_DATA_DIR = tempfile.mkdtemp(prefix="oo_s195_f8b_")


def _seed_stub(name: str, mod: ModuleType) -> None:
    if name not in sys.modules:
        sys.modules[name] = mod
        _ADDED_MODULES.append(name)


def _seed_common_stubs() -> None:
    if "opti_oignon" not in sys.modules:
        pkg = ModuleType("opti_oignon")
        pkg.__path__ = [str(ROOT / "opti_oignon")]
        _seed_stub("opti_oignon", pkg)
    if "opti_oignon.config" not in sys.modules:
        cfg = ModuleType("opti_oignon.config")
        cfg.DATA_DIR = _TMP_DATA_DIR
        _seed_stub("opti_oignon.config", cfg)
    if "opti_oignon.db_utils" not in sys.modules:
        dbu = ModuleType("opti_oignon.db_utils")
        dbu.safe_connect = lambda p, **kw: sqlite3.connect(str(p), **kw)
        _seed_stub("opti_oignon.db_utils", dbu)


def _load(name: str, relpath: str) -> ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    _seed_common_stubs()
    spec = importlib.util.spec_from_file_location(name, ROOT / relpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # register BEFORE exec_module (3.12+ dataclasses)
    _ADDED_MODULES.append(name)
    spec.loader.exec_module(mod)
    return mod


manifest_mod = _load(
    "opti_oignon.plugin_manifest", "opti_oignon/plugin_manifest.py",
)
hooks_mod = _load(
    "opti_oignon.plugin_hooks", "opti_oignon/plugin_hooks.py",
)

HookManager = hooks_mod.HookManager
HookContext = hooks_mod.HookContext
REDACTED_PLACEHOLDER = hooks_mod.REDACTED_PLACEHOLDER
PluginManifest = manifest_mod.PluginManifest


@pytest.fixture(scope="module", autouse=True)
def _cleanup_added_modules():
    yield
    for name in _ADDED_MODULES:
        sys.modules.pop(name, None)


@pytest.fixture
def mgr():
    return HookManager()


def _manifest(name: str, permissions: list | None = None) -> "PluginManifest":
    return PluginManifest.from_dict({
        "name": name,
        "version": "1.0.0",
        "author": "tester",
        "description": "test plugin",
        "entry_point": "entry_point.py",
        "permissions": permissions or [],
    })


@pytest.fixture
def singleton_registry():
    """The module-level registry used by the permission lookup."""
    reg = manifest_mod.plugin_registry
    if reg is None:
        pytest.skip("plugin_registry singleton unavailable in this container")
    return reg


# ---------------------------------------------------------------------------
# HK-02 -- placeholder must not round-trip into the chain
# ---------------------------------------------------------------------------

class TestHK02RedactionPoisoning:
    def test_redacted_echo_does_not_blank_message(self, mgr):
        """An unregistered (zero-permission) plugin echoing its redacted view
        must not replace the real message in final_data -- this is the exact
        pre_inference seam scenario (routes_chat replaces the prompt with
        final_data['message'])."""
        def echo(ctx):
            return dict(ctx.data)  # echoes message=REDACTED_PLACEHOLDER

        mgr.register("pre_inference", "spy-plugin", echo)
        report = mgr.execute(
            "pre_inference",
            data={"message": "real prompt", "model": "m"},
            redact_sensitive=True,
        )

        assert report.successful == 1
        assert report.final_data["message"] == "real prompt"
        assert REDACTED_PLACEHOLDER not in report.final_data.values()

    def test_redacted_plugin_view_is_actually_redacted(self, mgr):
        seen = {}

        def capture(ctx):
            seen.update(ctx.data)
            return None

        mgr.register("post_inference", "spy-plugin", capture)
        mgr.execute(
            "post_inference",
            data={"response": "secret answer", "model": "m"},
            redact_sensitive=True,
        )
        assert seen["response"] == REDACTED_PLACEHOLDER
        assert seen["model"] == "m"

    def test_redacted_plugin_can_still_add_nonsensitive_keys(self, mgr):
        def annotate(ctx):
            return {"tag": "seen", "message": ctx.data["message"]}

        mgr.register("pre_inference", "spy-plugin", annotate)
        report = mgr.execute(
            "pre_inference",
            data={"message": "real prompt"},
            redact_sensitive=True,
        )
        assert report.final_data["tag"] == "seen"
        assert report.final_data["message"] == "real prompt"

    def test_downstream_hook_still_sees_real_value(self, mgr):
        downstream = {}

        def echo(ctx):
            return dict(ctx.data)

        def consumer(ctx):
            downstream.update(ctx.data)
            return None

        mgr.register("pre_inference", "spy-plugin", echo, priority=10)
        mgr.register("pre_inference", "late-plugin", consumer, priority=200)
        mgr.execute(
            "pre_inference",
            data={"message": "real prompt"},
            redact_sensitive=True,
        )
        # late-plugin is also unregistered -> its VIEW is redacted, but the
        # chain data it was redacted FROM must still hold the real value.
        assert downstream["message"] == REDACTED_PLACEHOLDER

    def test_privileged_plugin_can_modify_message(
        self, mgr, singleton_registry,
    ):
        """The filter must not break legitimate modification by a plugin
        holding inference_content."""
        name = "privileged-hk02"
        singleton_registry.register(
            _manifest(name, permissions=["inference_content"]),
            "/tmp/privileged-hk02",
        )
        try:
            def enhance(ctx):
                return {"message": "[ENH] " + ctx.data["message"]}

            mgr.register("pre_inference", name, enhance)
            report = mgr.execute(
                "pre_inference",
                data={"message": "real prompt"},
                redact_sensitive=True,
            )
            assert report.final_data["message"] == "[ENH] real prompt"
        finally:
            singleton_registry.unregister(name)

    def test_no_redaction_path_unchanged(self, mgr):
        def enhance(ctx):
            return {"message": "[ENHANCED] " + ctx.data.get("message", "")}

        mgr.register("pre_inference", "any", enhance)
        report = mgr.execute("pre_inference", data={"message": "Hello"})
        assert report.final_data["message"] == "[ENHANCED] Hello"


# ---------------------------------------------------------------------------
# HK-01 -- contract lock: only returned dicts propagate
# ---------------------------------------------------------------------------

class TestHK01ReturnContract:
    def test_inplace_set_does_not_propagate(self, mgr):
        def mutator(ctx):
            ctx.set("flag", True)
            return None

        mgr.register("pre_inference", "p1", mutator)
        report = mgr.execute("pre_inference", data={"message": "x"})
        assert "flag" not in report.final_data

    def test_returned_dict_propagates(self, mgr):
        mgr.register("pre_inference", "p1", lambda ctx: {"flag2": True})
        report = mgr.execute("pre_inference", data={})
        assert report.final_data["flag2"] is True

    def test_each_hook_receives_its_own_copy(self, mgr):
        seen_by_b = {}

        def hook_a(ctx):
            ctx.data["a"] = 99  # local scribble, returns nothing
            return None

        def hook_b(ctx):
            seen_by_b.update(ctx.data)
            return None

        mgr.register("pre_inference", "pa", hook_a, priority=10)
        mgr.register("pre_inference", "pb", hook_b, priority=20)
        mgr.execute("pre_inference", data={"a": 1})
        assert seen_by_b["a"] == 1

    def test_docstrings_state_return_contract(self):
        src = (ROOT / "opti_oignon" / "plugin_hooks.py").read_text()
        idx = src.index("class HookContext")
        chunk = src[idx:idx + 900]
        assert "RETURN a dict" in chunk
        assert "does NOT propagate" in src


# ---------------------------------------------------------------------------
# Regression guard on the edited merge block
# ---------------------------------------------------------------------------

class TestErrorIsolationIntact:
    def test_failing_hook_does_not_break_chain(self, mgr):
        def boom(ctx):
            raise RuntimeError("kaput")

        mgr.register("pre_inference", "bad", boom, priority=10)
        mgr.register("pre_inference", "good", lambda ctx: {"ok": 1}, priority=20)
        report = mgr.execute("pre_inference", data={})
        assert report.failed == 1
        assert report.successful == 1
        assert report.final_data["ok"] == 1
        assert any(r.error and "kaput" in r.error for r in report.results)
