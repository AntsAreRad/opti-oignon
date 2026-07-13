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

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402

_MODULES = ("allowlists", "skills", "tools")


def _load():
    """Load the agent tool set in isolation; returns (allowlists, tools, restore).

    ``security_mode`` is declared absent, and the window PROVES it absent
    before any target runs. MX2 asserts what the tool set resolves to when no
    mode source can be reached at all: that absence IS the condition of the
    clause. A window that merely hopes for it -- by standing in a parent
    package whose path is empty, say -- lets the live module resolve behind
    the test's back wherever a finder answers on the module name, and MX2 then
    reports on the running mode rather than on the fail-secure path it names.
    """
    loaded, restore = isolate(
        targets={
            f"opti_oignon.agent.{m}": source("agent", f"{m}.py")
            for m in _MODULES
        },
        blocked=("opti_oignon.security_mode",),
        packages=("opti_oignon.agent",),
    )
    skills = loaded["opti_oignon.agent.skills"]
    skills._sync_publish_skill = lambda *a, **k: None
    skills._audit = lambda *a, **k: None
    return (
        loaded["opti_oignon.agent.allowlists"],
        loaded["opti_oignon.agent.tools"],
        restore,
    )


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
