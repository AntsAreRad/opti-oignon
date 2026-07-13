#!/usr/bin/env python3
"""Mode-exposure contracts for the agent tool set.

The agent's tool registry assembles a per-mode tool set through the mode
allowlist, and the skills tool is a state-mutation tool: it exists in the
frictionless mode and is structurally absent from the isolated one. This
suite pins the exposure seam:

  * MX1 -- the isolated mode never exposes the state-mutation or network
    tools (``manage_skills`` included), only allowlisted schemas appear, and
    the skills-consult guidance never reaches its system prompt;
  * MX2 -- an unknown or unresolvable mode resolves to the isolated mode
    (fail-secure), never to the frictionless one;
  * MX3 -- when ``manage_skills`` IS exposed, the system prompt section
    carries the consult-before-domain-work guidance;
  * MX4 -- handlers are attached only for exposed, non-sandboxed tools (a
    sandboxed tool carries a schema but never a direct handler), in both
    modes.

Loads the allowlists, skills, and tools modules in isolation. Local-only.
Runs under pytest or the __main__ runner.
"""

import importlib.util
import sys
import types
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_AGENT = _REPO / "opti_oignon" / "agent"

_MODULES = ("allowlists", "skills", "tools")
# security_mode is snapshotted and BLOCKED (None entry) too: the live-mode
# clause needs the lateral import to fail inside the isolation window. A bare
# eviction is not enough -- a residual real module in a shared process, or an
# editable-install meta-path finder that resolves the name without consulting
# the stand-in package path, would both let the import succeed. A None entry
# in sys.modules raises ImportError before any finder runs, deterministically.
_KEYS = ("opti_oignon", "opti_oignon.agent", "opti_oignon.security_mode") + tuple(
    f"opti_oignon.agent.{m}" for m in _MODULES
)


def _load():
    saved = {k: sys.modules.get(k) for k in _KEYS}

    root = types.ModuleType("opti_oignon")
    root.__path__ = []
    agent = types.ModuleType("opti_oignon.agent")
    agent.__path__ = []
    sys.modules["opti_oignon"] = root
    sys.modules["opti_oignon.agent"] = agent
    sys.modules["opti_oignon.security_mode"] = None  # block the lateral import

    loaded = {}
    for m in _MODULES:
        full = f"opti_oignon.agent.{m}"
        spec = importlib.util.spec_from_file_location(full, _AGENT / f"{m}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[full] = mod
        setattr(agent, m, mod)
        spec.loader.exec_module(mod)
        loaded[m] = mod

    loaded["skills"]._sync_publish_skill = lambda *a, **k: None
    loaded["skills"]._audit = lambda *a, **k: None

    def restore():
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    return loaded["allowlists"], loaded["tools"], restore


_GUIDANCE_MARK = "consult the skill registry"
_FORBIDDEN_IN_ISOLATION = frozenset(
    {"manage_skills", "manage_memory", "manage_notes", "web_search"}
)


def test_mx1_isolated_mode_never_exposes_state_or_network_tools():
    allowlists, tools, restore = _load()
    try:
        ts = tools.ToolRegistry().build(allowlists.MODE_BULBE)
        names = set(ts.names)
        assert ts.mode == allowlists.MODE_BULBE
        assert names.isdisjoint(_FORBIDDEN_IN_ISOLATION)
        # Only allowlisted schemas appear, and the sandboxed core is present.
        assert names <= set(allowlists.BULBE_ALLOWLIST)
        assert "bash" in names and "view" in names
        # The skills-consult guidance never reaches the isolated prompt.
        assert _GUIDANCE_MARK not in ts.system_prompt_section()
    finally:
        restore()


def test_mx2_unknown_or_unresolvable_mode_resolves_to_isolation():
    allowlists, tools, restore = _load()
    try:
        assert tools._resolve_mode("weird-mode") == allowlists.MODE_BULBE
        ts = tools.ToolRegistry().build("weird-mode")
        assert ts.mode == allowlists.MODE_BULBE
        # With no security-mode source resolvable (isolated load), a live
        # resolution is fail-secure to the isolated mode as well.
        assert allowlists.current_mode() == allowlists.MODE_BULBE
        assert tools.ToolRegistry().build(None).mode == allowlists.MODE_BULBE
    finally:
        restore()


def test_mx3_exposed_skills_tool_carries_the_consult_guidance():
    allowlists, tools, restore = _load()
    try:
        ts = tools.ToolRegistry().build(allowlists.MODE_DAILY)
        assert "manage_skills" in ts.names
        section = ts.system_prompt_section()
        assert section.startswith("Tools available in this mode")
        assert _GUIDANCE_MARK in section
    finally:
        restore()


def test_mx4_handlers_only_for_exposed_non_sandboxed_tools():
    allowlists, tools, restore = _load()
    try:
        registry = tools.ToolRegistry()
        for mode in (allowlists.MODE_DAILY, allowlists.MODE_BULBE):
            ts = registry.build(mode)
            names = set(ts.names)
            handlers = set(ts.tool_handlers)
            assert handlers <= names  # never a handler for an unexposed tool
            sandboxed = {s.name for s in ts.schemas if s.sandboxed}
            assert handlers.isdisjoint(sandboxed)  # sandboxed: schema only
            assert "todo" in handlers  # per-run session state, both modes
    finally:
        restore()


if __name__ == "__main__":
    _failures = 0
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            try:
                _fn()
                print(f"PASS {_name}")
            except Exception as _e:  # noqa: BLE001
                _failures += 1
                print(f"FAIL {_name}: {_e!r}")
    print(f"\n{'OK' if _failures == 0 else str(_failures) + ' FAILED'}")
    sys.exit(1 if _failures else 0)
